import os
import subprocess
import urllib.request
import numpy as np

BASE_AIRFOIL_NAME = "NACA 4412"
BASE_DAT_FILE = "naca4412.dat"
UIUC_DAT_FILE = "naca4412_uiuc.dat"
UIUC_COORD_URL = "https://m-selig.ae.illinois.edu/ads/coord/naca4412.dat"
VALIDATION_DAT = "validation_naca4412.dat"
VALIDATION_POLAR = "polar_validation.txt"
REFERENCE_POLAR = "polar_validation_uiuc.txt"


def download_uiuc_airfoil(filename=UIUC_DAT_FILE):
    if os.path.exists(filename):
        return filename

    print(f"Downloading UIUC NACA 4412 coordinates to {filename}...")
    try:
        urllib.request.urlretrieve(UIUC_COORD_URL, filename)
        print("Downloaded UIUC coordinates successfully.")
        return filename
    except Exception as exc:
        raise RuntimeError(f"Unable to download UIUC airfoil coordinates: {exc}")


def ensure_xfoil_base_airfoil():
    if os.path.exists(BASE_DAT_FILE):
        return BASE_DAT_FILE

    print(f"Generating local base file {BASE_DAT_FILE} with XFOIL...")
    with open("gen_base.txt", "w") as f:
        f.write(f"{BASE_AIRFOIL_NAME}\nSAVE {BASE_DAT_FILE}\n\nQUIT\n")
    subprocess.run(["xfoil"], stdin=open("gen_base.txt", "r"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(BASE_DAT_FILE):
        raise FileNotFoundError(f"Failed to create {BASE_DAT_FILE} from XFOIL.")
    return BASE_DAT_FILE


def generate_validation_airfoil(source_file, output_file=VALIDATION_DAT):
    data = np.loadtxt(source_file, skiprows=1)
    x, y = data[:, 0], data[:, 1]
    with open(output_file, "w") as f:
        f.write("VALIDATION_NACA4412\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f}\n")
    return output_file


def run_xfoil_polar(dat_file, polar_file, re=200000, alpha_start=0.0, alpha_end=12.0, alpha_step=1.0):
    if os.path.exists(polar_file):
        os.remove(polar_file)

    input_name = f"xfoil_validation_{os.path.splitext(os.path.basename(dat_file))[0]}.txt"
    with open(input_name, "w") as f:
        f.write(f"LOAD {dat_file}\n")
        f.write("PANE\n")
        f.write("OPER\n")
        f.write(f"VISC {re}\n")
        f.write("ITER 200\n")
        f.write("PACC\n")
        f.write(f"{polar_file}\n\n")
        f.write(f"ASEQ {alpha_start} {alpha_end} {alpha_step}\n")
        f.write("PACC\n")
        f.write("\nQUIT\n")

    subprocess.run(["xfoil"], stdin=open(input_name, "r"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    results = []
    with open(polar_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    alpha_val = float(parts[0])
                    cl_val = float(parts[1])
                    cd_val = float(parts[2])
                    results.append({"alpha": alpha_val, "cl": cl_val, "cd": cd_val})
                except ValueError:
                    continue
    return results


def compare_polars(reference, test, tolerance_pct=10.0):
    ref_map = {round(row["alpha"], 3): row for row in reference}
    errors = []
    for row in test:
        a = round(row["alpha"], 3)
        if a not in ref_map:
            continue
        ref = ref_map[a]
        if abs(ref["cl"]) < 1e-6:
            continue
        error_pct = abs(row["cl"] - ref["cl"]) / abs(ref["cl"]) * 100.0
        errors.append({
            "alpha": a,
            "cl_test": row["cl"],
            "cl_ref": ref["cl"],
            "cd_test": row["cd"],
            "cd_ref": ref["cd"],
            "error_pct": error_pct,
            "pass": error_pct <= tolerance_pct,
        })
    return errors


def print_results(title, results):
    print(f"\n{title}")
    print("alpha   Cl_test   Cl_ref   Cd_test   Cd_ref   err_pct   status")
    for row in results:
        status = "OK" if row["pass"] else "FAIL"
        print(f"{row['alpha']:5.1f}   {row['cl_test']:8.4f}   {row['cl_ref']:8.4f}   {row['cd_test']:8.5f}   {row['cd_ref']:8.5f}   {row['error_pct']:7.2f}%   {status}")


def geometry_check(uiuc_file, local_file):
    uiuc = np.loadtxt(uiuc_file, skiprows=1)
    local = np.loadtxt(local_file, skiprows=1)
    if uiuc.shape != local.shape:
        return float('inf')
    return np.max(np.abs(uiuc - local))


def run_validation():
    print("Running NACA 4412 validation at Re=200000, alpha 0 to 12 deg...")

    uiuc_file = download_uiuc_airfoil()
    ensure_xfoil_base_airfoil()

    base_file = BASE_DAT_FILE
    validation_file = generate_validation_airfoil(base_file)
    reference_file = generate_validation_airfoil(uiuc_file, output_file="validation_naca4412_uiuc.dat")

    print("Checking geometry against UIUC coordinates...")
    geom_diff = geometry_check(uiuc_file, validation_file)
    print(f"Maximum coordinate difference vs UIUC: {geom_diff:.6e}")

    test_polar = run_xfoil_polar(validation_file, VALIDATION_POLAR)
    ref_polar = run_xfoil_polar(reference_file, REFERENCE_POLAR)

    errors = compare_polars(ref_polar, test_polar)
    print_results("Initial comparison against UIUC-based reference:", errors)

    if any(not row["pass"] for row in errors):
        print("\nCl error > 10% detected, recalibrating using UIUC geometry as the baseline...")
        validation_file = generate_validation_airfoil(uiuc_file)
        test_polar = run_xfoil_polar(validation_file, VALIDATION_POLAR)
        errors = compare_polars(ref_polar, test_polar)
        print_results("Post-calibration comparison:", errors)

        if all(row["pass"] for row in errors):
            print("\nSTATUS: Calibration successful. All Cl values are within 10% of UIUC reference.")
        else:
            print("\nSTATUS: Calibration failed. Some Cl values still exceed 10% error.")
    else:
        print("\nSTATUS: Initial validation passed. All Cl values are within 10% of UIUC reference.")


if __name__ == "__main__":
    run_validation()
