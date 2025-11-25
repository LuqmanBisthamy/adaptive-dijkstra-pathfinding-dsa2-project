import tkinter as tk
from queue import PriorityQueue
import time
import random

# Define constants for grid size
GRID_SIZE = 20  # Number of rows and columns in the grid
CELL_SIZE = 20  # Size of each cell in pixels

# Define the Manhattan heuristic
def manhattan_heuristic(node, goal):
    """Calculate Manhattan distance as the heuristic.
    Used for direct paths in sparse grids."""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

# Define the Euclidean heuristic
def euclidean_heuristic(node, goal):
    """Calculate Euclidean distance as the heuristic.
    Used for diagonal paths in dense grids."""
    return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2) ** 0.5

# Define the dynamic heuristic function
def dynamic_heuristic(node, goal, grid):
    """Switch heuristics dynamically based on grid conditions.
    Manhattan is used for fewer obstacles; Euclidean is used for dense obstacles."""
    obstacle_count = sum(
        1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= node[0] + dx < GRID_SIZE and 0 <= node[1] + dy < GRID_SIZE and grid[node[0] + dx][node[1] + dy] == 1
    )
    if obstacle_count < 2:
        return manhattan_heuristic(node, goal)
    else:
        return euclidean_heuristic(node, goal)

# Adaptive Heuristic Dijkstra Algorithm
def adaptive_dijkstra(grid, start, goal):
    """Modified Dijkstra's algorithm with dynamic heuristic adaptation."""
    rows, cols = len(grid), len(grid[0])
    visited = set()  # Track visited nodes
    pq = PriorityQueue()  # Priority queue to prioritize nodes
    pq.put((0, start))  # Add start node with cost 0
    costs = {start: 0}  # Track cumulative costs
    came_from = {}  # Track path information

    while not pq.empty():
        current_cost, current = pq.get()

        if current in visited:
            continue

        visited.add(current)

        if current == goal:  # Goal found; reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, costs[goal], len(visited)

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                new_cost = costs[current] + 1  # Cost to reach neighbor
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    priority = new_cost + dynamic_heuristic(neighbor, goal, grid)
                    pq.put((priority, neighbor))
                    came_from[neighbor] = current

    return [], float('inf'), len(visited)  # No path found

# Basic Dijkstra's Algorithm
def dijkstra(grid, start, goal):
    """Basic Dijkstra's algorithm."""
    rows, cols = len(grid), len(grid[0])
    visited = set()  # Track visited nodes
    pq = PriorityQueue()  # Priority queue to prioritize nodes
    pq.put((0, start))  # Add start node with cost 0
    costs = {start: 0}  # Track cumulative costs
    came_from = {}  # Track path information

    while not pq.empty():
        current_cost, current = pq.get()

        if current in visited:
            continue

        visited.add(current)

        if current == goal:  # Goal found; reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, costs[goal], len(visited)

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                new_cost = costs[current] + 1  # Cost to reach neighbor
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    pq.put((new_cost, neighbor))
                    came_from[neighbor] = current

    return [], float('inf'), len(visited)  # No path found

class PathfindingApp:
    def __init__(self, master):
        """Initialize the GUI application."""
        self.master = master
        self.master.title("Multi-Heuristic Adaptive Pathfinding")

        # Initialize grid, start, and goal positions
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.start = (1, 1)  # Start node
        self.goal = (18, 18)  # Goal node

        # Generate the maze
        self.generate_maze()

        # Create canvas for visualization
        self.canvas = tk.Canvas(self.master, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)  # Set goal on mouse click

        # Create information and control frames
        self.info_frame = tk.Frame(self.master)
        self.info_frame.pack()

        # Info and results labels
        self.info_label = tk.Label(self.info_frame, text="Click on the grid to set the goal location.")
        self.info_label.pack()

        self.basic_results = tk.Label(self.info_frame, text="Basic Dijkstra: Not Run")
        self.basic_results.pack()

        self.adaptive_results = tk.Label(self.info_frame, text="Adaptive Heuristic: Not Run")
        self.adaptive_results.pack()

        self.comparison_results = tk.Label(self.info_frame, text="Comparison: Not Available")
        self.comparison_results.pack()

        # Create control buttons
        self.control_frame = tk.Frame(self.master)
        self.control_frame.pack()

        self.run_basic_btn = tk.Button(self.control_frame, text="Run Basic Dijkstra", command=self.run_basic)
        self.run_basic_btn.grid(row=0, column=0, padx=5, pady=5)

        self.run_adaptive_btn = tk.Button(self.control_frame, text="Run Adaptive Heuristic", command=self.run_adaptive)
        self.run_adaptive_btn.grid(row=0, column=1, padx=5, pady=5)

        self.compare_btn = tk.Button(self.control_frame, text="Compare Results", command=self.compare_results)
        self.compare_btn.grid(row=0, column=2, padx=5, pady=5)

        self.reset_btn = tk.Button(self.control_frame, text="Reset", command=self.reset)
        self.reset_btn.grid(row=0, column=3, padx=5, pady=5)

        # Data storage for results
        self.basic_data = None
        self.adaptive_data = None

        # Draw initial grid
        self.draw_grid()

    def generate_maze(self):
        """Generate a random maze with a guaranteed path."""
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if random.random() < 0.3 and (i, j) not in [self.start, self.goal]:
                    self.grid[i][j] = 1  # Mark as obstacle

    def draw_grid(self):
        """Visualize the grid on the canvas."""
        self.canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                color = "white"  # Default for empty cells
                if self.grid[i][j] == 1:
                    color = "black"  # Obstacles
                elif (i, j) == self.start:
                    color = "green"  # Start node
                elif (i, j) == self.goal:
                    color = "gold"  # Goal node
                self.canvas.create_rectangle(j * CELL_SIZE, i * CELL_SIZE, (j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE, fill=color, outline="gray")

    def handle_click(self, event):
        """Set a new goal position on mouse click."""
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.grid[row][col] == 0:
            self.goal = (row, col)
            self.info_label.config(text=f"Goal set to: {self.goal}")
            self.draw_grid()

    def run_basic(self):
        """Run the Basic Dijkstra algorithm."""
        if self.start and self.goal:
            start_time = time.time()
            path, cost, nodes_explored = dijkstra(self.grid, self.start, self.goal)
            execution_time = time.time() - start_time
            self.basic_data = (cost, nodes_explored, execution_time, path)
            self.display_results(path, cost, nodes_explored, execution_time, "Basic Dijkstra", self.basic_results, "orange")

    def run_adaptive(self):
        """Run the Adaptive Heuristic Dijkstra algorithm."""
        if self.start and self.goal:
            start_time = time.time()
            path, cost, nodes_explored = adaptive_dijkstra(self.grid, self.start, self.goal)
            execution_time = time.time() - start_time
            self.adaptive_data = (cost, nodes_explored, execution_time, path)
            self.display_results(path, cost, nodes_explored, execution_time, "Adaptive Heuristic", self.adaptive_results, "blue")

    def display_results(self, path, cost, nodes_explored, execution_time, algorithm_name, result_label, path_color):
        """Display the results of the algorithm and highlight the path."""
        for i, j in path:
            self.canvas.create_rectangle(j * CELL_SIZE, i * CELL_SIZE, (j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE, fill=path_color, outline="gray")
        result_label.config(text=f"{algorithm_name}: Path Cost: {cost} | Nodes Explored: {nodes_explored} | Time: {execution_time:.4f}s")

    def compare_results(self):
        """Compare the results of both algorithms."""
        if self.basic_data and self.adaptive_data:
            basic_cost, basic_nodes, basic_time, basic_path = self.basic_data
            adaptive_cost, adaptive_nodes, adaptive_time, adaptive_path = self.adaptive_data

            # Highlight both paths
            self.draw_grid()
            for i, j in basic_path:
                self.canvas.create_rectangle(j * CELL_SIZE, i * CELL_SIZE, (j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE, fill="orange", outline="gray")
            for i, j in adaptive_path:
                self.canvas.create_rectangle(j * CELL_SIZE, i * CELL_SIZE, (j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE, fill="blue", outline="gray")

            # Display comparison results
            comparison = (
                f"Comparison:\n"
                f"Basic Dijkstra - Nodes Explored: {basic_nodes}, Time: {basic_time:.4f}s\n"
                f"Adaptive Heuristic - Nodes Explored: {adaptive_nodes}, Time: {adaptive_time:.4f}s\n"
                f"Efficiency: {'Adaptive Heuristic is more efficient' if adaptive_nodes < basic_nodes else 'Basic Dijkstra is more efficient'}"
            )
            self.comparison_results.config(text=comparison)

    def reset(self):
        """Reset the grid and results."""
        self.generate_maze()
        self.basic_data = None
        self.adaptive_data = None
        self.basic_results.config(text="Basic Dijkstra: Not Run")
        self.adaptive_results.config(text="Adaptive Heuristic: Not Run")
        self.comparison_results.config(text="Comparison: Not Available")
        self.info_label.config(text="Click on the grid to set the goal location.")
        self.draw_grid()

if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingApp(root)
    root.mainloop()
