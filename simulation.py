import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# --- Simulation Parameters ---
n_particles = 30
n_iterations = 100
dimensions = 2

# PSO Parameters
w = 0.5   # Inertia
c1 = 1.5  # Cognitive (Personal Best)
c2 = 1.5  # Social (Global Best)

# The "Food" location (the objective we want to minimize distance to)
food_location = np.array([5.0, 5.0])

# --- Objective Function ---
# Calculates the distance from a particle to the food
def objective_function(position):
    return np.sum((position - food_location)**2)

# --- Initialization ---
# Random initial positions and velocities between 0 and 10
positions = np.random.uniform(0, 10, (n_particles, dimensions))
velocities = np.random.uniform(-1, 1, (n_particles, dimensions))

# Track personal and global bests
pbest_positions = np.copy(positions)
pbest_scores = np.array([objective_function(p) for p in positions])

gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index]

# --- Setup the Plot ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("PSO: Flock of Birds Locating Food")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

# Draw the food
food_plot = ax.scatter(*food_location, color='red', marker='*', s=300, label='Food')
# Draw the birds
particles_plot = ax.scatter(positions[:, 0], positions[:, 1], color='blue', marker='v', s=50, label='Birds')

ax.legend()

# --- Animation Update Function ---
def update(frame):
    global positions, velocities, pbest_positions, pbest_scores, gbest_position
    
    for i in range(n_particles):
        # Generate random numbers
        r1 = np.random.rand(dimensions)
        r2 = np.random.rand(dimensions)
        
        # Update Velocity
        cognitive_velocity = c1 * r1 * (pbest_positions[i] - positions[i])
        social_velocity = c2 * r2 * (gbest_position - positions[i])
        velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity
        
        # Update Position
        positions[i] = positions[i] + velocities[i]
        
        # Evaluate new position
        current_score = objective_function(positions[i])
        
        # Update Personal Best
        if current_score < pbest_scores[i]:
            pbest_scores[i] = current_score
            pbest_positions[i] = positions[i]
            
    # Update Global Best
    best_particle_idx = np.argmin(pbest_scores)
    if pbest_scores[best_particle_idx] < objective_function(gbest_position):
        gbest_position = pbest_positions[best_particle_idx]
        
    # Update the plot points
    particles_plot.set_offsets(positions)
    return particles_plot,

# --- Create and Show Animation ---
ani = animation.FuncAnimation(fig, update, frames=n_iterations, interval=200, blit=False)

plt.show()

# Optional: Save as an MP4 or GIF for your presentation
ani.save('pso_birds.gif', writer=PillowWriter(fps=5))