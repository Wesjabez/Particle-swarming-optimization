import numpy as np
import os
import subprocess
import csv

# each particle represents each airfoil geometry
class Particle:
    def __init__(self, bounds):
        
        # the properties for each airfoil geometry
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.random.uniform(-1, 1, len(bounds))
        self.pbest_pos = np.copy(self.position)
        self.pbest_value = float('-inf') 

class WingPSO:
    def __init__(self, num_particles, bounds, iterations, mach, re, alpha):
        self.num_particles = num_particles
        self.bounds = np.array(bounds)
        self.iterations = iterations
        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.gbest_pos = np.zeros(len(bounds))
        self.gbest_value = float('-inf')

        #defines the flight regime
        self.mach = mach
        self.re = re
        self.alpha = alpha


    def fitness_function(self, params):
        """
        Evaluates the aerodynamic efficiency (L/D) of an airfoil using XFOIL.
        """
        m_camber, p_pos, thickness = params
        
        #  Converting PSO parameters to a NACA 4-digit string
    
        d1 = int(round(m_camber * 100))
        d2 = int(round(p_pos * 10))
        d34 = int(round(thickness * 100))
        naca_code = f"{d1}{d2}{d34:02d}"
        
    
        input_file = "xfoil_input.txt"
        polar_file = "polar_output.txt"
        
    
        if os.path.exists(polar_file):
            os.remove(polar_file)

        # XFOIL interaction to provide results of cl and cd
        with open(input_file, "w") as f:
            f.write(f"NACA {naca_code}\n")
            f.write("PANE\n")          
            f.write("OPER\n")
            f.write(f"Visc {self.re}\n")  
            f.write(f"Mach {self.mach}\n")     
            f.write("ITER 100\n")     
            f.write("PACC\n")          
            f.write(f"{polar_file}\n")
            f.write("\n")              
            f.write(f"ALFA {self.alpha}\n")      
            f.write("PACC\n")          
            f.write("\nQUIT\n")

        
        
        try:
            subprocess.run(["xfoil.exe"], 
                           stdin=open(input_file, "r"), 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL, 
                           timeout=5) 
        except subprocess.TimeoutExpired:
            return 0.0 

        
        try:
            with open(polar_file, "r") as f:
                lines = f.readlines()
                
            
                if len(lines) < 13:
                    return 0.0 
                    
                data = lines[12].split()
                cl = float(data[1])
                cd = float(data[2])
                
                ld_ratio = cl/cd
                if cd <= 0:
                    return 0.0
                if cd > 0.2: # massive drag
                    return 0.0
                
                if ld_ratio > 200:
                    return 0.0
                    
            
                return ld_ratio
                
        except Exception:
            
            return 0.0

    def optimize(self, w=0.5, c1=1.5, c2=2.0):
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

        return self.gbest_pos, self.gbest_value

# dummy parameters we pass to the wing pso class, 
flight_regimes = {
    "Takeoff": 
    {
        "mach": 0.2,
        "re": 500000,
        "alpha": 8.0,
        "bounds":[(0.02,0.08),(0.2,0.5),(0.10,0.15)]
    },
    "cruise":
    {
        "mach":0.6,
        "re":200000,
        "alpha":2.0,
        "bounds":[(0.0,0.04),(0.3,0.5),(0.08,0.12)]
    },
    "High-speed dash":
    {
        "mach":0.8,
        "re":300000,
        "alpha":0.3,
        "bounds":[(0.02,0.08),(0.2,0.5),(0.10,0.15)]
    },
    "landing":
    {
        "mach":0.1,
        "re":600000,
        "alpha":10.0,
        "bounds":[(0.04, 0.08), (0.2, 0.5), (0.10, 0.15)]
    }
}

morphing_lookup_table = {}

print("starting Multi-Regime Morphing Wing Optimization..\n")

# performing iterations for each of the hard coded flight regimes
for regime_name, conditions in flight_regimes.items():
    print(f"--- Optimization for {regime_name} ---")

    optimizer = WingPSO (
        num_particles= 20,
        bounds = conditions["bounds"],
        iterations = 5,
        mach = conditions["mach"],
        re = conditions["re"],
        alpha =  conditions["alpha"] 
    )

    best_shape, best_ld = optimizer.optimize()

    morphing_lookup_table[regime_name] = {
        "shape": best_shape,
        "L/D": best_ld
    }

    print(f"Optimal Geometry [camber, Position, Thickness]: {best_shape}")
    print(f"Maximized L/D: {best_ld:.2}\n")

print("Optimization complete!")
for regime, data in morphing_lookup_table.items():
    print(f"{regime}: {data['shape']} at L/D = {data['L/D']:.2f}")

# we store the results in a csv format which can be parsed to other softwares

csv_filename = "morphing_lookup_table.csv"

with open(csv_filename, mode = 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Regime", "Max_camber", "Camber_Position", "Thickness","L_D"])

    for regime, data in morphing_lookup_table.items():
        shape = data['shape']
        ld = data['L/D']

        print(f"{regime}: {shape} at L/D = {ld:.2f}")

        writer.writerow([regime,shape[0],shape[1],shape[2],ld])
print(f"success. lookup table saved to {csv_filename}")
