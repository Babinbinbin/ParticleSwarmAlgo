import numpy as np
import matplotlib.pyplot as plt

# Define the fitness function
def fitness_function(position):
    x, y = position
    return (x - 3.14)**2 + (y - 2.72)**2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)

# Define obstacle penalty function
def obstacle_penalty(position, obstacles):
    penalty = 0
    for obs in obstacles:
        center, radius = obs
        distance = np.linalg.norm(position - center)
        if distance < radius:
            penalty += 1e3 * (radius - distance)  # Add a large penalty if inside the obstacle
    return penalty

# Combined fitness function with obstacle penalties
def combined_fitness(position, obstacles):
    return fitness_function(position) + obstacle_penalty(position, obstacles)

# Initialize parameters
num_particles = 30
num_iterations = 100
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5
bounds = [-10, 10]

# Define obstacles (center and radius)
obstacles = [np.array([[2, 2], 1.5]), np.array([[-3, -3], 2]), np.array([[6, -4], 1])]

# Initialize particles and velocities
particles = np.random.uniform(bounds[0], bounds[1], (num_particles, 2))
velocities = np.zeros_like(particles)
best_particle_positions = np.copy(particles)
best_particle_scores = np.array([combined_fitness(p, obstacles) for p in particles])
global_best_position = best_particle_positions[np.argmin(best_particle_scores)]
global_best_score = np.min(best_particle_scores)

# Visualization setup
fig, ax = plt.subplots()
X, Y = np.meshgrid(np.linspace(bounds[0], bounds[1], 100), np.linspace(bounds[0], bounds[1], 100))
Z = fitness_function([X, Y])
ax.contourf(X, Y, Z, levels=50, cmap="viridis")
for obs in obstacles:
    circle = plt.Circle(obs[0], obs[1], color='red', alpha=0.3)
    ax.add_artist(circle)
scat = ax.scatter(particles[:, 0], particles[:, 1], color="white")

# PSO loop
for iteration in range(num_iterations):
    for i, particle in enumerate(particles):
        # Update velocity
        r1, r2 = np.random.rand(2)
        velocities[i] = (inertia_weight * velocities[i] +
                         cognitive_coeff * r1 * (best_particle_positions[i] - particle) +
                         social_coeff * r2 * (global_best_position - particle))

        # Update position
        particles[i] += velocities[i]
        particles[i] = np.clip(particles[i], bounds[0], bounds[1])

        # Evaluate fitness with obstacle penalties
        score = combined_fitness(particles[i], obstacles)
        if score < best_particle_scores[i]:
            best_particle_scores[i] = score
            best_particle_positions[i] = particles[i]

        if score < global_best_score:
            global_best_score = score
            global_best_position = particles[i]

    # Update visualization
    scat.set_offsets(particles)
    plt.pause(0.1)

plt.show()
