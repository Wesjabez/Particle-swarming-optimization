import subprocess
import os

def run_baseline_naca0012():
    """
    Runs a standardized test on a NACA 0012 airfoil.
    Conditions: Re = 3,000,000 | Mach = 0.0 | Alpha = 4.0 degrees
    """
    print("Running Baseline Validation on NACA 0012...")
    
    input_file = "xfoil_validation.txt"
    polar_file = "polar_validation.txt"
    
    if os.path.exists(polar_file):
        os.remove(polar_file)

    with open(input_file, "w") as f:
        f.write("NACA 0012\n")
        f.write("PANE\n")
        f.write("OPER\n")
        f.write("Visc 3000000\n") 
        f.write("ITER 100\n")
        f.write("PACC\n")
        f.write(f"{polar_file}\n\n")
        f.write("ALFA 4.0\n")
        f.write("PACC\n")
        f.write("\nQUIT\n")

    subprocess.run(["xfoil.exe"], stdin=open(input_file, "r"), 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        with open(polar_file, "r") as f:
            lines = f.readlines()
            data = lines[12].split()
            cl, cd = float(data[1]), float(data[2])
            ld = cl / cd
            
            print(f"\n--- Validation Results ---")
            print(f"Calculated Cl:  {cl}")
            print(f"Calculated Cd:  {cd}")
            print(f"Calculated L/D: {ld:.2f}")
            print(f"\n--- Expected Standard Data (Abbott & Von Doenhoff) ---")
            print(f"Expected Cl:  ~0.40 to 0.44")
            print(f"Expected Cd:  ~0.006 to 0.007")
            
            if 0.38 <= cl <= 0.46 and cd > 0:
                print("\nSTATUS: PASS - Pipeline is properly calibrated.")
            else:
                print("\nSTATUS: FAIL - Results deviate significantly from published data.")
                
    except Exception as e:
        print(f"Error reading results: {e}")

run_baseline_naca0012()