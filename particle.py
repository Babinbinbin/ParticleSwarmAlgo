import numpy as np
import matplotlib.pyplot as plt

# Define the maze as a grid using numpy.
# 0 represents a free path, 1 represents a wall.
maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
])
maze_height, maze_width = maze.shape

# Define start and goal positions (using x-y coordinates)
start = np.array([0, 0], dtype=float)
goal = np.array([maze_width - 1, maze_height - 1], dtype=float)

# Number of intermediate waypoints (not including start and goal)
n_waypoints = 3
dimensions = n_waypoints * 2  # each waypoint has an x and y coordinate

# PSO hyperparameters
n_particles = 30
n_iterations = 100
w = 0.5     # inertia weight
c1 = 1.0    # cognitive parameter
c2 = 1.0    # social parameter

# Define bounds for each waypoint coordinate.
lower_bound = 0
upper_bound_x = maze_width - 1
upper_bound_y = maze_height - 1

# Initialize particles randomly within the allowed range for each intermediate waypoint.
particles = np.zeros((n_particles, dimensions))
for i in range(n_particles):
    for j in range(n_waypoints):
        particles[i, 2 * j] = np.random.uniform(lower_bound, upper_bound_x)   # x coordinate
        particles[i, 2 * j + 1] = np.random.uniform(lower_bound, upper_bound_y) # y coordinate

# Random initial velocities
velocities = np.random.uniform(-1, 1, (n_particles, dimensions))

# Personal best positions and their fitness scores
pbest_positions = particles.copy()
pbest_scores = np.full(n_particles, np.inf)

# Global best initialization
gbest_position = None
gbest_score = np.inf

def check_collision(A, B, maze, samples=20):
    """
    Checks if the straight line between A and B collides with a wall in the maze.
    Samples several points along the line and returns True if any sampled point is a wall.
    """
    for t in np.linspace(0, 1, samples):
        point = A + t * (B - A)
        x, y = int(round(point[0])), int(round(point[1]))
        # Check boundaries first.
        if x < 0 or x >= maze_width or y < 0 or y >= maze_height:
            return True
        if maze[y, x] == 1:  # note: maze is indexed as [row, column]
            return True
    return False

def fitness_function(candidate):
    """
    Given a candidate (a flat vector for intermediate waypoints), constructs the full path
    (start -> waypoints -> goal) and computes its total Euclidean distance.
    If any segment collides with a wall, a large penalty is added.
    """
    # Build the path: start + waypoints + goal.
    path = [start]
    for i in range(n_waypoints):
        x = candidate[2 * i]
        y = candidate[2 * i + 1]
        path.append(np.array([x, y]))
    path.append(goal)
    
    total_distance = 0
    penalty = 0
    for i in range(len(path) - 1):
        A = path[i]
        B = path[i + 1]
        dist = np.linalg.norm(B - A)
        total_distance += dist
        if check_collision(A, B, maze):
            penalty += 1000  # heavy penalty for collision with wall
    return total_distance + penalty, path

# Main PSO loop
for it in range(n_iterations):
    for i in range(n_particles):
        score, _ = fitness_function(particles[i])
        # Update personal best if the current solution is better.
        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = particles[i].copy()
        # Update global best if necessary.
        if score < gbest_score:
            gbest_score = score
            gbest_position = particles[i].copy()
    
    # Update each particle's velocity and position based on PSO update rules.
    for i in range(n_particles):
        r1 = np.random.rand(dimensions)
        r2 = np.random.rand(dimensions)
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest_positions[i] - particles[i]) +
                         c2 * r2 * (gbest_position - particles[i]))
        particles[i] += velocities[i]
        # Enforce the bounds on each coordinate.
        for j in range(n_waypoints):
            particles[i, 2 * j] = np.clip(particles[i, 2 * j], lower_bound, upper_bound_x)
            particles[i, 2 * j + 1] = np.clip(particles[i, 2 * j + 1], lower_bound, upper_bound_y)
    
    if it % 10 == 0:
        print(f"Iteration {it}, Best Fitness: {gbest_score:.2f}")

# Retrieve the best path from the global best candidate.
_, best_path = fitness_function(gbest_position)

# Visualize the maze and the best path found.
plt.figure(figsize=(6, 6))
plt.imshow(maze, cmap='binary', origin='upper')
plt.title("SWARMMMMM")
# Convert the list of path points to an array for easier plotting.
path_coords = np.array(best_path)
plt.plot(path_coords[:, 0], path_coords[:, 1], 'r.-', linewidth=2, markersize=8)
plt.plot(start[0], start[1], 'go', markersize=10, label="Start")
plt.plot(goal[0], goal[1], 'bo', markersize=10, label="Goal")
plt.legend()
# Invert the y-axis to match the maze's indexing (row 0 at top).
plt.gca().invert_yaxis()
plt.show()
