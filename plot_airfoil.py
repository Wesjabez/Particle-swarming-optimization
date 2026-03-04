import numpy as np
import matplotlib.pyplot as plt
import csv
import os

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

# --- STANDALONE EXECUTION BLOCK ---
csv_filename = "morphing_lookup_table.csv"

# Check if the PSO script has been run first
if not os.path.exists(csv_filename):
    print(f"Error: Could not find '{csv_filename}'. Please run pso.py first!")
else:
    print(f"Reading aerodynamic data from {csv_filename}...\n")
    
    # Open and read the CSV file
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            regime = row["Regime"]
            # Convert strings from CSV back into math floats
            m = float(row["Max_camber"])
            p = float(row["Camber_Position"])
            t = float(row["Thickness"])
            
            print(f"Generating plot for: {regime}")
            plot_optimized_airfoil(m, p, t, regime)
            
    print("\nPipeline complete! All wing geometries have been plotted and saved.")