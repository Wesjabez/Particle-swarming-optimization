import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def load_airfoil_coordinates(filename):
    """
    Load airfoil coordinates from a .dat file.
    Returns x, y arrays (upper and lower surfaces combined).
    """
    try:
        data = np.loadtxt(filename, skiprows=1)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error loading airfoil from {filename}: {e}")
        return None, None

def plot_base_vs_morphed(base_filename, morphed_filename, regime_name, deflection_angle):
    """
    Plots the base NACA 4212 and morphed airfoil side-by-side for comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Load base airfoil
    x_base, y_base = load_airfoil_coordinates(base_filename)
    if x_base is None:
        print(f"Could not load base airfoil from {base_filename}")
        return
    
    # Load morphed airfoil
    x_morph, y_morph = load_airfoil_coordinates(morphed_filename)
    if x_morph is None:
        print(f"Could not load morphed airfoil from {morphed_filename}")
        return
    
    # Plot base airfoil (left)
    ax1.fill(x_base, y_base, alpha=0.3, color='blue', label='Base NACA 4212')
    ax1.plot(x_base, y_base, 'b-', linewidth=2.5)
    ax1.set_xlabel("Chord Position (x/c)", fontsize=11)
    ax1.set_ylabel("Thickness (y/c)", fontsize=11)
    ax1.set_title("Base NACA 4212", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axis('equal')
    ax1.legend(fontsize=10)
    
    # Plot morphed airfoil (right)
    ax2.fill(x_morph, y_morph, alpha=0.3, color='red', label=f'Morphed (δ={deflection_angle:.2f}°)')
    ax2.plot(x_morph, y_morph, 'r-', linewidth=2.5)
    ax2.set_xlabel("Chord Position (x/c)", fontsize=11)
    ax2.set_ylabel("Thickness (y/c)", fontsize=11)
    ax2.set_title(f"Optimized for {regime_name}", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.axis('equal')
    ax2.legend(fontsize=10)
    
    # Overall title
    fig.suptitle(f"Airfoil Morphing Comparison: {regime_name} (Deflection: {deflection_angle:.2f}°)", 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    safe_name = regime_name.replace(" ", "_").replace("-", "_").lower()
    filename = f"airfoil_comparison_{safe_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved comparison visualization: {filename}")

def plot_overlaid_airfoils(base_filename, morphed_filename, regime_name, deflection_angle, ld_value):
    """
    Plots base and morphed airfoils overlaid on the same plot for direct comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load base airfoil
    x_base, y_base = load_airfoil_coordinates(base_filename)
    if x_base is None:
        return
    
    # Load morphed airfoil
    x_morph, y_morph = load_airfoil_coordinates(morphed_filename)
    if x_morph is None:
        return
    
    # Plot both
    ax.plot(x_base, y_base, 'b-', linewidth=2.5, label='Base NACA 4212', alpha=0.8)
    ax.fill(x_base, y_base, alpha=0.15, color='blue')
    
    ax.plot(x_morph, y_morph, 'r-', linewidth=2.5, label=f'Optimized Morphed (δ={deflection_angle:.2f}°)', alpha=0.8)
    ax.fill(x_morph, y_morph, alpha=0.15, color='red')
    
    ax.set_xlabel("Chord Position (x/c)", fontsize=12)
    ax.set_ylabel("Thickness (y/c)", fontsize=12)
    ax.set_title(f"Regime: {regime_name} | Optimized L/D: {ld_value:.2f}", 
                 fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axis('equal')
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    
    safe_name = regime_name.replace(" ", "_").replace("-", "_").lower()
    filename = f"airfoil_overlay_{safe_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved overlay visualization: {filename}")

def plot_optimized_airfoil(m, p, t, regime_name, chord=1.0):
    """
    Plots a NACA 4-digit airfoil and saves it as a PNG file.
    """
    x = np.linspace(0, 1, 100)
    
    yc = np.where(x < p,
                  (m / p**2) * (2 * p * x - x**2),
                  (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x - x**2))
    
    yt = 5 * t * chord * (0.2969 * np.sqrt(x) - 0.1260 * x - 
                          0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    dyc_dx = np.where(x < p,
                      (2 * m / p**2) * (p - x),
                      (2 * m / (1 - p)**2) * (p - x))
    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    plt.figure(figsize=(10, 3)) 
    plt.plot(xu, yu, 'b-', linewidth=2, label='Upper Surface')
    plt.plot(xl, yl, 'r-', linewidth=2, label='Lower Surface')
    plt.plot(x, yc, 'k--', linewidth=1, label='Mean Camber Line')
    
    plt.title(f"Optimized Wing: {regime_name}\nCamber: {m*100:.1f}%, Position: {p*100:.1f}%, Thickness: {t*100:.1f}%")
    plt.xlabel("Chord Fraction (x/c)")
    plt.ylabel("Thickness (y/c)")
    plt.axis('equal') 
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    safe_name = regime_name.replace(" ", "_").lower()
    filename = f"{safe_name}_wing_geometry.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f" -> Saved visualization: {filename}")

if __name__ == "__main__":
    csv_filename = "morphing_lookup_table.csv"

    # Check if the PSO script has been run first
    if not os.path.exists(csv_filename):
        print(f"Error: Could not find '{csv_filename}'. Please run pso.py first!")
    else:
        print(f"Reading morphing lookup data from {csv_filename}...\n")
        
        with open(csv_filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                regime = row["Regime"]
                angle = float(row["Deflection_Angle"])
                ld = float(row["L_D"])
                print(f"Regime: {regime} | Deflection Angle: {angle:.2f} | L/D: {ld:.2f}")
        
        print("\nPlotting functions are available in this module; use them from another script after generating the morphed DAT files.")