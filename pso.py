import numpy as np
import os
import subprocess

class Particle:
    def __init__(self, bounds):
        # Position: [Max Camber, Camber Position, Thickness]
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.random.uniform(-1, 1, len(bounds))
        self.pbest_pos = np.copy(self.position)
        self.pbest_value = float('-inf') # Initializing for maximization (L/D ratio)

class WingPSO:
    def __init__(self, num_particles, bounds, iterations):
        self.num_particles = num_particles
        self.bounds = np.array(bounds)
        self.iterations = iterations
        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.gbest_pos = np.zeros(len(bounds))
        self.gbest_value = float('-inf')

    def fitness_function(self, params):
        """
        Evaluates the aerodynamic efficiency (L/D) of an airfoil using XFOIL.
        """
        m_camber, p_pos, thickness = params
        
        # 1. Convert PSO parameters to a NACA 4-digit string
        # Example: [0.02, 0.4, 0.12] becomes "2412"
        d1 = int(round(m_camber * 100))
        d2 = int(round(p_pos * 10))
        d34 = int(round(thickness * 100))
        naca_code = f"{d1}{d2}{d34:02d}"
        
        # File names for the current evaluation
        input_file = "xfoil_input.txt"
        polar_file = "polar_output.txt"
        
        # Clean up old polar file if it exists to prevent reading old data
        if os.path.exists(polar_file):
            os.remove(polar_file)

        # 2. Write the command sequence for XFOIL
        # This simulates typing commands directly into the XFOIL terminal
        with open(input_file, "w") as f:
            f.write(f"NACA {naca_code}\n")
            f.write("PANE\n")          # Smooth the paneling (important for optimization)
            f.write("OPER\n")
            f.write("Visc 500000\n")   # Set Reynolds number (adjust for your flight regime)
            f.write("Mach 0.2\n")      # Set Mach number 
            f.write("ITER 100\n")      # Increase iteration limit for stubborn shapes
            f.write("PACC\n")          # Start accumulating polar data
            f.write(f"{polar_file}\n") # File to save polar data
            f.write("\n")              # No dump file
            f.write("ALFA 4.0\n")      # Run at an Angle of Attack of 4 degrees
            f.write("PACC\n")          # Stop accumulating
            f.write("\nQUIT\n")

        # 3. Execute XFOIL silently
        # Ensure xfoil.exe is in the exact same folder as this Python script
        try:
            subprocess.run(["xfoil.exe"], 
                           stdin=open(input_file, "r"), 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL, 
                           timeout=5) # 5-second timeout in case XFOIL hangs
        except subprocess.TimeoutExpired:
            return 0.0 # If it hangs, it's a bad shape. Return zero fitness.

        # 4. Parse the results
        # XFOIL writes the Cl and Cd data to the polar_file
        try:
            with open(polar_file, "r") as f:
                lines = f.readlines()
                
                # The actual data usually starts on line 13 in an XFOIL polar file
                # If the file is shorter than 13 lines, XFOIL failed to converge
                if len(lines) < 13:
                    return 0.0 
                    
                data = lines[12].split()
                cl = float(data[1])
                cd = float(data[2])
                
                # Guard against division by zero or negative drag
                if cd <= 0:
                    return 0.0
                    
                # Our objective is to maximize the L/D ratio
                return cl / cd 
                
        except Exception:
            # If anything goes wrong (file missing, bad format), penalize the particle
            return 0.0

    def optimize(self, w=0.5, c1=1.5, c2=2.0):
        for i in range(self.iterations):
            for p in self.swarm:
                # 1. Evaluate Fitness
                current_fitness = self.fitness_function(p.position)

                # 2. Update Personal Best
                if current_fitness > p.pbest_value:
                    p.pbest_value = current_fitness
                    p.pbest_pos = np.copy(p.position)

                # 3. Update Global Best
                if current_fitness > self.gbest_value:
                    self.gbest_value = current_fitness
                    self.gbest_pos = np.copy(p.position)

            # 4. Update Velocity and Position
            for p in self.swarm:
                r1, r2 = np.random.rand(), np.random.rand()
                
                # The Core PSO Equation
                p.velocity = (w * p.velocity + 
                             c1 * r1 * (p.pbest_pos - p.position) + 
                             c2 * r2 * (self.gbest_pos - p.position))
                
                p.position += p.velocity

                # 5. Boundary Constraints (Keep airfoil parameters realistic)
                p.position = np.clip(p.position, self.bounds[:, 0], self.bounds[:, 1])

        return self.gbest_pos, self.gbest_value

# Usage for a Cruise Regime
# Bounds: Camber (0-6%), Position (20-50%), Thickness (8-15%)
wing_bounds = [(0, 0.06), (0.2, 0.5), (0.08, 0.15)]
optimizer = WingPSO(num_particles=30, bounds=wing_bounds, iterations=10)
best_shape, best_ld = optimizer.optimize()

print(f"Optimal Wing Geometry: {best_shape}")
print(f"Maximized L/D: {best_ld}")