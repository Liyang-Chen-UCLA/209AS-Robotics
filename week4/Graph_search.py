import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import pickle


class PRMPlanner:
    def __init__(self, y_max=10, v_max=5, y_steps=20, v_steps=50, a_max=1.0, A=0.1, tol=0.05):
        # Initialize parameters for the PRM planner
        self.y_max = y_max  # Maximum y-axis value
        self.v_max = v_max  # Maximum velocity value
        self.y_steps = y_steps  # Number of steps for y-axis
        self.v_steps = v_steps  # Number of steps for velocity
        self.a_max = a_max  # Maximum acceleration
        self.A = A  # Constant used in the equations
        self.tol = tol  # Tolerance for stopping condition
        self.graph = None  # Placeholder for the graph

    def connection_check(self, y0, v0, y2, v2):
        # Check if there is a valid connection between two states (y0, v0) and (y2, v2)
        f0 = y2 - y0 - 2 * v0 + self.A * np.sin(y0)  # Compute the first factor f0
        f1 = v2 - v0 - f0 + self.A * (np.sin(y0) + np.sin(y0 + v0))  # Compute the second factor f1
        return -1 <= f0 <= 1 and -1 <= f1 <= 1  # Return True if both factors are within bounds

    def build_graph(self, num_samples=950):
        # Build the probabilistic roadmap (PRM) graph
        try:
            # Attempt to load a pre-built graph from file
            with open(f"G_{num_samples}", 'rb') as f:
                self.graph = pickle.load(f)
            print('Build graph successfully!')

        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            # Create a new directed graph if loading fails
            G = nx.DiGraph()

            # Generate Gaussian-distributed samples for nodes
            V_gauss = [(random.gauss(0, self.y_max / 6), random.gauss(0, self.v_max / 6)) for _ in range(50)]
            # Generate uniformly distributed samples for nodes
            V_uniform = [(random.uniform(-self.y_max, self.y_max), random.uniform(-self.v_max, self.v_max)) for _ in range(num_samples)]
            V = V_gauss + V_uniform + [(0, 0)]  # Combine all nodes with the origin
            G.add_nodes_from(V)

            # Add edges between valid connections
            for v in V:
                for u in V:
                    if self.connection_check(v[0], v[1], u[0], u[1]):
                        y0 = v[0]
                        v0 = v[1]
                        y2 = u[0]
                        v2 = u[1]
                        f0 = y2 - y0 - 2 * v0 + self.A * np.sin(y0)  # Calculate f0
                        f1 = v2 - v0 - f0 + self.A * (np.sin(y0) + np.sin(y0 + v0))  # Calculate f1
                        G.add_edge(v, u, f0=f0, f1=f1)  # Add edge with calculated factors

            self.graph = G  # Save the graph
            with open(f"G_{num_samples}", 'wb') as f:
                pickle.dump(self.graph, f)  # Save the graph to a file

    def visualize_graph(self):
        # Visualize the constructed PRM graph
        plt.figure(figsize=(8, 6))

        # Plot nodes as blue dots
        for node in self.graph.nodes():
            plt.scatter(node[0], node[1], color='blue', s=50)

        # Plot edges as gray lines
        for edge in self.graph.edges():
            start, end = edge
            plt.plot([start[0], end[0]], [start[1], end[1]], color='gray', alpha=0.05)

        plt.xlim(-self.y_max, self.y_max)  # Set x-axis limits
        plt.ylim(-self.v_max, self.v_max)  # Set y-axis limits
        plt.grid(True, linestyle='--', alpha=0.5)  # Add grid
        plt.xticks(np.linspace(-self.y_max, self.y_max, num=10))
        plt.yticks(np.linspace(-self.v_max, self.v_max, num=10))
        plt.xlabel('Y-axis')
        plt.ylabel('V-axis')
        plt.title('Probabilistic Roadmap (PRM)')
        # plt.show()  # Uncomment to display the graph

    def find_nearest_node(self, state):
        # Find the nearest node in the graph to the given state
        min_dist = float('inf')  # Initialize minimum distance as infinity
        nearest_node = None  # Placeholder for the nearest node
        for node in self.graph.nodes():
            dist = np.sqrt((node[0] - state[0])**2 + (node[1] - state[1])**2)  # Compute Euclidean distance
            if dist < min_dist:  # Update nearest node if distance is smaller
                min_dist = dist
                nearest_node = node
        return nearest_node

    def find_shortest_path(self, s0, sf):
        # Find the nearest nodes to the start (s0) and goal (sf) states in the graph
        start_node = self.find_nearest_node(s0)

        # Use Dijkstra's algorithm to find the shortest path
        try:
            shortest_path = nx.shortest_path(self.graph, source=start_node, target=sf)
        except nx.NetworkXNoPath:
            print("No path found between the given states.")
            return None, []

        # Generate the control sequence for the path
        control_sequence = []
        for i in range(len(shortest_path) - 1):
            y0, v0 = shortest_path[i]
            y2, v2 = shortest_path[i + 1]
            f0 = y2 - y0 - 2 * v0 + self.A * np.sin(y0)  # Calculate f0
            f1 = v2 - v0 - f0 + self.A * (np.sin(y0) + np.sin(y0 + v0))  # Calculate f1
            control_sequence.append((f0, f1))

        return shortest_path, control_sequence

    def visualize_shortest_path(self, s0, sf):
        # Visualize the shortest path found in the PRM graph
        shortest_path, control_sequence = self.find_shortest_path(s0, sf)
        if not shortest_path:
            print("Unable to visualize the shortest path.")
            return

        # self.visualize_graph()

        # Plot the shortest path using green arrows
        path_nodes = np.array(shortest_path)
        for i in range(len(path_nodes) - 1):
            y_start, v_start = path_nodes[i]
            y_end, v_end = path_nodes[i + 1]
            
            # Draw arrows using quiver
            plt.quiver(y_start, v_start, 
                       y_end - y_start, v_end - v_start, 
                       angles='xy', scale_units='xy', scale=1, 
                       color='green', width=0.005, linewidth=1, zorder=5)

        # Plot the shortest path nodes as green dots
        for node in shortest_path:
            plt.scatter(node[0], node[1], color='green', s=50, zorder=4)

        plt.title('Probabilistic Roadmap with Shortest Path')
        plt.show()

    def find_best_action(self, s0):
        # Find the best action from the current state s0
        shortest_path, control_sequence = self.find_shortest_path(s0, (0, 0))

        if not shortest_path:
            print("Unable to visualize the shortest path.")
            return None
        
        u = self.find_nearest_node(s0)  # Nearest node to s0
        v = shortest_path[1]  # Next node in the shortest path
        
        edge_data = self.graph.get_edge_data(u, v)  # Get edge data for control inputs

        return edge_data.get('f0'), edge_data.get('f1')

    def particle_motion_simulation(self, s0, num_steps=100):
        # Simulate particle motion starting from initial state s0
        y, v = s0  # Initialize position and velocity
        pos = [y]  # List to store positions
        speed = [v]  # List to store speeds
        input = []  # List to store control inputs

        print(f"Initial state: y={y}, v={v}")

        # Perform the particle motion simulation
        for t in range(num_steps):
            best_action = self.find_best_action((y, v))  # Query the best action
            if not best_action:
                print("No valid action found. Simulation stopped.")
                break

            f0, f1 = best_action  # Unpack the control inputs
            
            # Update state based on the particle motion equations
            a = f0 - self.A * np.sin(y)
            y = y + v
            v = v + a
            pos.append(y)
            speed.append(v)
            input.append(f0)

            if y**2 <= self.tol and v**2 <= self.tol:  # Check stopping condition
                break

            a = f1 - self.A * np.sin(y)
            y = y + v
            v = v + a
            pos.append(y)
            speed.append(v)
            input.append(f1)

            if y**2 <= self.tol and v**2 <= self.tol:  # Check stopping condition
                break

        sim_history = {
            'position': pos,
            'speed': speed,
            'input': input
        }
        print(f"Len: {len(pos)}")

        # self.visualize_simulation(sim_history)  # Visualize the simulation results
        return sim_history

    def visualize_simulation(self, sim_history):
        """
        Visualize the results of the particle motion simulation,
        including changes in position, speed, and control input over time.
        """
        pos = sim_history['position']  # Extract positions
        speed = sim_history['speed']  # Extract speeds
        inputs = sim_history['input']  # Extract inputs
        time_steps = range(len(pos))  # Time steps

        # Create three separate subplots
        plt.figure(figsize=(12, 6))

        # Plot position over time
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, pos, label='Position (y)', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Position (y)')
        plt.title('Position Over Time')
        plt.grid(True)

        # Plot speed over time
        plt.subplot(3, 1, 2)
        plt.plot(time_steps, speed, label='Speed (v)', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Speed (v)')
        plt.title('Speed Over Time')
        plt.grid(True)

        # Plot control input over time
        plt.subplot(3, 1, 3)
        plt.plot(range(len(inputs)), inputs, label='Control Input (f)', color='red')
        plt.xlabel('Time Step')
        plt.ylabel('Control Input (f)')
        plt.title('Control Input Over Time')
        plt.grid(True)

        plt.tight_layout()  # Adjust layout
        plt.show()


prm = PRMPlanner()
s0 = (np.random.uniform(-prm.y_max, prm.y_max), np.random.uniform(-prm.v_max, prm.v_max))  # Random initial state
sf = (0, 0)  # Goal state
s0 = (6, 2)
prm.build_graph()  # Build the graph

# prm.visualize_graph()
# prm.visualize_shortest_path(s0, sf)  # Uncomment to visualize the shortest path

sim_nums = 15
reached_nums = 0
for _ in range(sim_nums):
    s0 = (np.random.uniform(-prm.y_max, prm.y_max), np.random.uniform(-prm.v_max, prm.v_max))  # Random initial state
    sim_history = prm.particle_motion_simulation(s0)
    if len(sim_history['position']) < 200:
        reached_nums += 1
print(f"Reached rate: {reached_nums/sim_nums}")
