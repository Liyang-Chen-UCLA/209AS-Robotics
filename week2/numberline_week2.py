import numpy as np
import matplotlib.pyplot as plt
import random
import pickle  # For saving/loading policy and value function
import seaborn as sns

class MDPSystem:
    def __init__(self, y_max=10, v_max=5, A=1.5, gamma=0.9, pw=0.01, pc=0.01, c=-0.1):
        # Initialize the parameters
        self.y_max = y_max
        self.v_max = v_max
        self.A = A
        self.gamma = gamma
        self.pw = pw  # Probability of speed wobble
        self.pc = pc  # Probability of crashing
        self.c = c      # fuel cost
        self.actions = [-1, 0, 1]  # Possible forces
        self.optimal_state = (0, 0)  # Goal state (y=0, v=0)
        self.V = np.zeros((2 * y_max + 1, 2 * v_max + 1))  # Initialize value function
        self.policy = np.zeros((2 * self.y_max + 1, 2 * self.v_max + 1), dtype=int)
        self.V_v = None
        self.policy_v = None
        self.V_p = None
        self.policy_p = None

        self.load_flag = False

    # Define the reward function
    def reward(self, y, v, a):
        if y == 0:
            distance_reward = 1
        else:
            distance_reward = 1/(y**2)

        speed_reward = 0
        if np.abs(y) <= 2:
            if v == 0:
                speed_reward = 1
            else:
                speed_reward = 1/(v**2)

        if a == 0:
            fuel_reward = 0
        else:
            fuel_reward = self.c

        if y == 0 and v == 0:
            time_reward = 100  # Reward for reaching the goal
        else:
            time_reward = -2  # Step penalty

        reward = distance_reward + speed_reward + fuel_reward + time_reward
        return reward

    # Transition function: returns next (y, v) given current state and action
    def transition(self, y, v, fi):
        current_pw = np.abs(v) / self.v_max * self.pw
        speed_wobble = np.random.choice([-1, 0, 1], p=[current_pw/2, 1-current_pw, current_pw/2])
        current_pc = np.abs(v) / self.v_max * self.pc
        new_y = y + v

        if np.random.rand() <= current_pc:
            new_v = 0
        else:
            f_phi = self.A * np.sin(2 * np.pi * y / self.y_max)
            a = int(fi + f_phi)
            new_v = v + a + speed_wobble

        new_y = max(min(new_y, self.y_max), -self.y_max)
        new_v = max(min(new_v, self.v_max), -self.v_max)

        return new_y, new_v

    # Value iteration algorithm with stopping condition, policy extraction, and saving
    def value_iteration(self, threshold=0.01, max_iterations=500, policy_filename='policy_v.pkl', value_filename='value_v.pkl'):
        total_rewards = []  
        optimal_reached = False

        for iteration in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)
            total_reward = 0
            print(f"Iteration {iteration+1}:")

            for y in range(-self.y_max, self.y_max+1):
                for v in range(-self.v_max, self.v_max+1):
                    max_value = -float('inf')
                    best_action = None
                    for a in self.actions:
                        new_y, new_v = self.transition(y, v, a)
                        r = self.reward(new_y, new_v, a)
                        value = r + self.gamma * self.V[new_y + self.y_max][new_v + self.v_max]
                        if value > max_value:
                            max_value = value
                            best_action = a  # Extract the best action

                    new_V[y + self.y_max][v + self.v_max] = max_value
                    self.policy[y + self.y_max][v + self.v_max] = best_action  # Update policy
                    delta = max(delta, abs(self.V[y + self.y_max][v + self.v_max] - max_value))
                    total_reward += max_value

            if new_V[self.optimal_state[0] + self.y_max][self.optimal_state[1] + self.v_max] >= 100:
                optimal_reached = True

            self.V = new_V
            total_rewards.append(total_reward)

            if delta < threshold and optimal_reached:
                print(f"Optimal state reached after {iteration} iterations.")
                break

        # Save policy and value function after value iteration completes
        with open(policy_filename, 'wb') as pf:
            pickle.dump(self.policy, pf)
        with open(value_filename, 'wb') as vf:
            pickle.dump(self.V, vf)
        print("Policy and value function saved successfully.")

        return self.V, self.policy, total_rewards    

    def policy_iteration(self, max_iterations=500, threshold=0.01, policy_filename='policy_p.pkl', value_filename='value_p.pkl'):
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}: Policy Evaluation and Improvement")
            delta = 0
            policy_stable = True

            # Policy evaluation and improvement in one loop
            new_V = np.copy(self.V)
            for y in range(-self.y_max, self.y_max + 1):
                for v in range(-self.v_max, self.v_max + 1):
                    old_action = self.policy[y + self.y_max][v + self.v_max]
                    best_action = None
                    best_value = -float('inf')

                    # Evaluate for each action
                    for a in self.actions:
                        new_y, new_v = self.transition(y, v, a)
                        r = self.reward(new_y, new_v, a)
                        value = r + self.gamma * self.V[new_y + self.y_max][new_v + self.v_max]

                        if value > best_value:
                            best_value = value
                            best_action = a

                    # Update value function
                    new_V[y + self.y_max][v + self.v_max] = best_value
                    delta = max(delta, abs(self.V[y + self.y_max][v + self.v_max] - best_value))

                    # Update policy
                    self.policy[y + self.y_max][v + self.v_max] = best_action
                    if old_action != best_action:
                        policy_stable = False

            self.V = new_V

            # Check convergence
            if policy_stable:
                print("Policy is stable, stopping policy iteration.")
                break

            if delta < threshold:
                print("Value function converged, stopping evaluation.")
                break

        # Save policy and value function
        with open(policy_filename, 'wb') as pf:
            pickle.dump(self.policy, pf)
        with open(value_filename, 'wb') as vf:
            pickle.dump(self.V, vf)
        print("Policy and value function saved successfully.")

        return self.V, self.policy

    # Load policy and value function from specific files
    def load_policy_and_value(self, policy_v='policy_v.pkl', value_v='value_v.pkl', policy_p='policy_p.pkl', value_p='value_p.pkl'):
        try:
            with open(policy_v, 'rb') as pv_file:
                self.policy_v = pickle.load(pv_file)
            with open(value_v, 'rb') as vv_file:
                self.V_v = pickle.load(vv_file)
            print("Policy and value function (v) loaded successfully.")
            
            with open(policy_p, 'rb') as pp_file:
                self.policy_p = pickle.load(pp_file)
            with open(value_p, 'rb') as vp_file:
                self.V_p = pickle.load(vp_file)
            print("Policy and value function (p) loaded successfully.")

            self.V = self.V_v
            self.policy = self.policy_v
            self.load_flag = True
        except FileNotFoundError:
            print("One or more of the specified files not found.")

    def simulate(self, initial_state, max_steps=100):
        simulations = {}
        for policy_name in ['policy_v', 'policy_p']:
            y, v = initial_state
            total_reward = 0
            state_history = []
            action_history = []

            # Select the appropriate policy for the simulation
            policy = getattr(self, policy_name)

            for step in range(max_steps):
                state_history.append((y, v))
                action = policy[y + self.y_max][v + self.v_max]
                action_history.append(action)
                new_y, new_v = self.transition(y, v, action)
                total_reward += self.reward(new_y, new_v, action)
                y, v = new_y, new_v

                if (y, v) == (0, 0):
                    print(f"Reached optimal state in {policy_name}!")
                    break

            state_history.append((y, v))
            action_history.append(None)

            simulations[policy_name] = {
                'state_history': state_history,
                'action_history': action_history,
                'total_reward': total_reward
            }

        # 自动调用可视化函数
        self.visualize(simulations)
        return simulations

    def visualize(self, simulations):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        for i, policy_name in enumerate(['policy_v', 'policy_p']):
            state_history = simulations[policy_name]['state_history']
            action_history = simulations[policy_name]['action_history']
            steps = list(range(len(state_history)))

            y_vals = [state[0] for state in state_history]
            v_vals = [state[1] for state in state_history]

            # Subplot 1: y (Position) over steps
            axes[i, 0].plot(steps, y_vals, label='y (Position)')
            # axes[i, 0].set_xlabel('Steps')
            axes[i, 0].set_ylabel('y (Position)')
            axes[i, 0].set_title(f'Position (y) over Time ({policy_name})')
            axes[i, 0].grid(True)
            axes[i, 0].legend()

            # Subplot 2: v (Velocity) over steps
            axes[i, 1].plot(steps, v_vals, label='v (Velocity)', color='orange')
            # axes[i, 1].set_xlabel('Steps')
            axes[i, 1].set_ylabel('v (Velocity)')
            axes[i, 1].set_title(f'Velocity (v) over Time ({policy_name})')
            axes[i, 1].grid(True)
            axes[i, 1].legend()

            # Subplot 3: Actions over steps
            axes[i, 2].step(steps[:-1], action_history[:-1], label='Action', where='post')
            # axes[i, 2].set_xlabel('Steps')
            axes[i, 2].set_ylabel('Action')
            axes[i, 2].set_title(f'Actions over Time ({policy_name})')
            axes[i, 2].grid(True)
            axes[i, 2].legend()

        # # Plot value function heatmaps
        fig, heatmap_axes = plt.subplots(1, 2, figsize=(12, 6))

        sns.heatmap(self.V_v, cmap='viridis', ax=heatmap_axes[0], cbar_kws={'label': 'Value'})
        heatmap_axes[0].set_title('Value Function Heatmap (value iteration)')
        heatmap_axes[0].set_xlabel('Velocity')
        heatmap_axes[0].set_ylabel('Position')

        sns.heatmap(self.V_p, cmap='viridis', ax=heatmap_axes[1], cbar_kws={'label': 'Value'})
        heatmap_axes[1].set_title('Value Function Heatmap (policy iteration)')
        heatmap_axes[1].set_xlabel('Velocity')
        heatmap_axes[1].set_ylabel('Position')

        plt.tight_layout()
        plt.show()



# Example usage
mdp = MDPSystem()

# Simulate with a random initial state and optionally update the policy
initial_state = (random.randint(-mdp.y_max, mdp.y_max), random.randint(-mdp.v_max, mdp.v_max))

# Load the policy and value function if they exist
mdp.load_policy_and_value()

mdp.simulate(initial_state)


