import numpy as np
import os
import subprocess
import csv
import shutil
import matplotlib.pyplot as plt
from plot_airfoil import plot_base_vs_morphed, plot_overlaid_airfoils

BASE_AIRFOIL_NAME = "NACA 4412"
BASE_DAT_FILE = "naca4412.dat"

# each particle represents each airfoil geometry
class Particle:
    def __init__(self, bounds):
        
        # the properties for each airfoil geometry
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.random.uniform(-1, 1, len(bounds))
        self.pbest_pos = np.copy(self.position)
        self.pbest_value = float('-inf') 

def generate_smooth_airfoil(deflection_angle, filename="current_morphed.dat"):
    """
    Bypasses XFOIL's internal geometry engine by mathematically calculating 
    a smooth, continuous flexural curve using NumPy.
    """
    # 1. Ensure we have the base NACA 4412 coordinates to work with
    if not os.path.exists(BASE_DAT_FILE):
        with open("gen_base.txt", "w") as f:
            f.write(f"{BASE_AIRFOIL_NAME}\nSAVE {BASE_DAT_FILE}\n\nQUIT\n")
        subprocess.run(["xfoil"], stdin=open("gen_base.txt", "r"), stdout=subprocess.DEVNULL)

    # 2. Load the base coordinates
    data = np.loadtxt(BASE_DAT_FILE, skiprows=1)
    x, y = data[:, 0], data[:, 1]
    new_y = np.copy(y)
    
    TE_START = 0.65 # 130mm hinge line
    
    # 3. Apply the smooth sinusoidal deformation to the flap
    for i in range(len(x)):
        if x[i] > TE_START:
            # Simple rotation of points aft of the hinge line
            dx = x[i] - TE_START
            angle_rad = np.radians(-deflection_angle)
            new_y[i] = y[i] + dx * np.tan(angle_rad)
    # 4. Save the ultra-smooth geometry for XFOIL to read
    with open(filename, "w") as f:
        f.write("MORPHED_PROFILE\n")
        for i in range(len(x)): 
            f.write(f"{x[i]:.6f} {new_y[i]:.6f}\n")
            
    return filename


class WingPSO:
    def __init__(self, num_particles, bounds, iterations, mach, re, alpha):
        self.num_particles = num_particles
        self.bounds = np.array(bounds)
        self.iterations = iterations
        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.gbest_pos = np.zeros(len(bounds))
        self.gbest_value = float('-inf')
        self.gbest_positions = []

        #defines the flight regime
        self.mach = mach
        self.re = re
        self.alpha = alpha




    def fitness_function(self, params):
        """
        Evaluates the aerodynamic efficiency (L/D) of an airfoil using XFOIL.
        """
        deflection_angle = params[0]
        
        input_file = "xfoil_input.txt"
        polar_file = "polar_output.txt"
        
        if os.path.exists(polar_file):
            os.remove(polar_file)
        
        dat_file = "current_morphed.dat"
        generate_smooth_airfoil(deflection_angle, filename=dat_file)

        # XFOIL interaction
        with open(input_file, "w") as f:
    
            
            f.write(f"LOAD {dat_file}\n")
            f.write("PANE\n")
            f.write("OPER\n")
            f.write(f"VISC {self.re}\n")
            f.write("ITER 300\n")
            f.write("PACC\n")
            f.write(f"{polar_file}\n")
            f.write("\n")
            
            f.write(f"ASEQ 0 {self.alpha} 0.5\n")
                
            f.write("PACC\n")
            f.write("\nQUIT\n")

        try:
            
            subprocess.run(["xfoil"], 
                           stdin=open(input_file, "r"), 
                           stdout=open("xfoil_debug.log", "w"), # Hides the messy terminal output
                           stderr=subprocess.STDOUT, 
                           timeout=15) 
        except subprocess.TimeoutExpired:
            return 0.0 

        try:
            with open(polar_file, "r") as f:
                lines = f.readlines()
            
            # Log how many data lines we actually got
            data_lines = [l for l in lines[12:] if len(l.split()) >= 3]
            print(f"  [DEBUG] Polar file has {len(data_lines)} data lines")
            
            for line in reversed(lines):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        cl = float(parts[1])
                        cd = float(parts[2])
                        if 0.0001 < cd < 0.2:
                            ld = cl / cd
                            if ld < 200:
                                return ld
                    except ValueError:
                        continue
            return 0.0
        except Exception as e:
            print(f"  [DEBUG] Parser exception: {e}")
            return 0.0

        try:
            with open(polar_file, "r") as f:
                lines = f.readlines()
                
                max_ld = 0.0
                # XFOIL data always starts at line 13 (index 12)
                for line in lines[12:]:
                    data = line.split()
                    if len(data) >= 3:
                        cl = float(data[1])
                        cd = float(data[2])
                        
                        # Filter out extreme drag and failed convergence artifacts
                        if cd > 0.0001 and cd < 0.2:
                            ld = cl / cd
                            if ld > max_ld and ld < 200:
                                max_ld = ld
                
                return max_ld
                
        except Exception:
            return 0.0


    
    def optimize(self, w=0.5, c1=1.5, c2=2.0):
        convergence_history = []
        for i in range(self.iterations):

 
            for p in self.swarm:
                
                current_fitness = self.fitness_function(p.position)

                #Update Personal Best
                if current_fitness > p.pbest_value:
                    p.pbest_value = current_fitness
                    p.pbest_pos = np.copy(p.position)

                # Update Global Best
                if current_fitness > self.gbest_value:
                    self.gbest_value = current_fitness
                    self.gbest_pos = np.copy(p.position)
                    self.gbest_positions.append((self.gbest_pos, self.gbest_value))

            convergence_history.append(self.gbest_value)

            # Update Velocity and Position
            for p in self.swarm:
                r1, r2 = np.random.rand(), np.random.rand()
                
                #Equation for particle velocity and position
                p.velocity = (w * p.velocity + 
                             c1 * r1 * (p.pbest_pos - p.position) + 
                             c2 * r2 * (self.gbest_pos - p.position))
                
                p.position += p.velocity

                # 5. Boundary Constraints 
                p.position = np.clip(p.position, self.bounds[:, 0], self.bounds[:, 1])

        return self.gbest_pos, self.gbest_value, convergence_history

# dummy parameters we pass to the wing pso class, 
flight_regimes = {
    "Takeoff": 
    {
        "mach": 0.2,
        "re": 500000,
        "alpha": 8.0,
        "bounds":[(0,15)]
    },
    "cruise":
    {
        "mach":0.5,
        "re":200000,
        "alpha":2.0,
        "bounds":[(-2,6)]
    },
    "High-speed dash":
    {
        "mach":0.6,
        "re":300000,
        "alpha":0.3,
        "bounds":[(-10,2)]
    },
    "landing":
    {
        "mach":0.1,
        "re":600000,
        "alpha":10.0,
        "bounds":[(5,20)]
    }
}

# The translation function (ensure this is placed BEFORE the loop)
def angle_to_duty(deflection_angle):
    min_angle, max_angle = -15.0, 25.0
    min_duty, max_duty = 40, 115

    # Linear interpolation
    duty = min_duty + ((deflection_angle - min_angle) / (max_angle - min_angle)) * (max_duty - min_duty)
    
    # Clip it to ensure we NEVER break the physical servo
    return int(np.clip(duty, min_duty, max_duty))

morphing_lookup_table = {}

print("Starting Multi-Regime Morphing Wing Optimization..\n")

# Performing iterations for each of the flight regimes
for regime_name, conditions in flight_regimes.items():
    print(f"--- Optimization for {regime_name} ---")

    optimizer = WingPSO (
        num_particles= 20,
        bounds = conditions["bounds"],
        iterations = 50,
        mach = conditions["mach"],
        re = conditions["re"],
        alpha =  conditions["alpha"] 
    )

    best_shape, best_ld, convergence = optimizer.optimize()
    
    # best_shape is an array like [12.45]. We extract the actual number.
    best_angle = best_shape[0]

    morphing_lookup_table[regime_name] = {
        "Angle": best_angle,
        "L/D": best_ld
    }

    print(f"Optimal Deflection Angle: {best_angle:.2f} degrees")
    print(f"Maximized L/D: {best_ld:.2f}")
    # Plot convergence for this regime
    plt.figure()
    plt.plot(convergence)
    plt.xlabel("Iterations")
    plt.ylabel("Best L/D Value")
    plt.title(f"Convergence Curve for {regime_name}")
    safe_name = regime_name.replace(" ", "_").replace("-", "_").lower()
    plt.savefig(f"convergence_{safe_name}.png")
    plt.close()

    # Generate the final morphed airfoil with optimal deflection angle
    final_morphed_file = f"final_morphed_{safe_name}.dat"
    generate_smooth_airfoil(best_angle, filename=final_morphed_file)
    
    # Plot comparisons: base NACA 4412 vs final optimized morphed airfoil
    print(f"Generating airfoil comparison visualizations for {regime_name}...")
    plot_base_vs_morphed(BASE_DAT_FILE, final_morphed_file, regime_name, best_angle)
    plot_overlaid_airfoils(BASE_DAT_FILE, final_morphed_file, regime_name, best_angle, best_ld)
    print()

    # --- HARDWARE ACTUATION ---
    target_duty = angle_to_duty(best_angle)
    print(f">> Actuating servo to duty cycle {target_duty} for {regime_name}...\n")
    
    try:
        hardware_cmd = f"import machine; machine.PWM(machine.Pin(4), freq=50).duty({target_duty})"
        
        # Pointing directly to the mpremote inside your local virtual environment
        venv_mpremote = "/home/wesley/Documents/pso/Particle-swarming-optimization/env/bin/mpremote"
        
        print(f">> Triggering hardware via local venv...")
        
        # Clean execution with no system path overrides needed
        subprocess.run(
            [venv_mpremote, "connect", "/dev/ttyUSB0", "exec", hardware_cmd], 
            check=True
        )
        print(">> Hardware Actuation Successful!\n")
            
    except Exception as e:
        # If it fails, we just print the error and move to the next flight regime
        print(f">> Hardware actuation failed: {e}\n")

print("Optimization complete!")
            
    


# Save the 1-Dimensional results to CSV
csv_filename = "morphing_lookup_table.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Regime", "Deflection_Angle", "L_D"])

    for regime, data in morphing_lookup_table.items():
        angle = data['Angle']
        ld = data['L/D']
        writer.writerow([regime, angle, ld])

print(f"Success. Lookup table saved to {csv_filename}")

