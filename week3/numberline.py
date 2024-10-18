import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pickle  # For saving/loading policy and value function
import seaborn as sns

class MDPSystem:
    def __init__(self, y_max=10, v_max=5, A=1.5, gamma=0.9, pw=0.01, pc=0.01, c=-0.1, h=[20, 10, 10], k=1):
        # Initialize the parameters
        self.y_max = y_max
        self.v_max = v_max
        self.A = A
        self.gamma = gamma
        self.pw = pw  # Probability of speed wobble
        self.pc = pc  # Probability of crashing
        self.c = c      # fuel cost
        self.h_y = h[0] # position resolution
        self.h_v = h[1] # velocity resolution
        self.h_a = h[2] # action resolution
        self.k = k      # k-NN
        
        self.optimal_state = (0, 0)  # Goal state (y=0, v=0)    TODO: consider moving optimal_state
        self.actions = np.linspace(-1, 1, self.h_a)  # Possible forces
        self.V = np.zeros((self.h_y + 1, self.h_v + 1))  # Initialize value function
        self.policy = np.zeros((self.h_y + 1, self.h_v + 1))    # Initialize policy function

        
        # 生成离散的状态空间
        y_space = np.linspace(-self.y_max, self.y_max, self.h_y)
        v_space = np.linspace(-self.v_max, self.v_max, self.h_v)
        y_grid, v_grid = np.meshgrid(y_space, v_space)
        self.state_space = np.column_stack((y_grid.ravel(), v_grid.ravel()))
        self.y_space = list(np.linspace(-self.y_max, self.y_max, self.h_y + 1))
        self.v_space = list(np.linspace(-self.v_max, self.v_max, self.h_v + 1))

        self.load_flag = False

    def config_text(self):
        return f"_k={self.k}_hy={self.h_y}_hv={self.h_v}"

    # Define the reward function
    def reward(self, y, v, a):
        if y == 0:
            distance_reward = 1
        else:
            distance_reward = 5/(y**2)

        speed_reward = 0
        if np.abs(y) <= 2:
            if v == 0:
                speed_reward = 1
            else:
                speed_reward = 1/(v**2)

        if np.isclose(y, 0, atol=self.y_max / self.h_y) and np.isclose(v, 0, atol=self.v_max / self.h_v):
            time_reward = 100  # Reward for reaching the goal
        else:
            time_reward = -2  # Step penalty

        reward = distance_reward + speed_reward + time_reward
        return reward

    # Transition function: returns next (y, v) given current state and action
    def transition(self, y, v, f_i):
        current_pw = np.abs(v) / self.v_max * self.pw
        speed_wobble = np.random.normal(0, (0.01 * v)**2)

        current_pc = np.abs(v) / self.v_max * self.pc
        # new_y = y + v

        f_phi = self.A * np.sin(2 * np.pi * y / self.y_max)
        a = f_i + f_phi
        # new_v = v + a + speed_wobble

        # new_y = max(min(new_y, self.y_max), -self.y_max)
        # new_v = max(min(new_v, self.v_max), -self.v_max)

        p1 = (1 - current_pw) * (1 - current_pc)
        p2 = current_pw * (1 - current_pc)
        p3 = current_pc

        new_y = np.clip(y + v, -self.y_max, self.y_max)
        new_v1 = np.clip(v + a, -self.v_max, self.v_max)
        new_v2 = np.clip(v + a + speed_wobble, -self.v_max, self.v_max)
        
        return (p1, p2, p3), new_y, (new_v1, new_v2, 0)
    
    def search_NN(self, y, v):
        min_distance = float('inf')  # 初始化最小距离为无穷大
        nearest_index = (-1, -1)     # 初始化最近的索引

        for i in range(len(self.y_space)):
            for j in range(len(self.v_space)):  # 假设所有行的列数相同
                y_point = self.y_space[i]
                v_point = self.v_space[j]
                distance = math.sqrt((y - y_point)**2 + (v - v_point)**2)
                
                # 找到最小距离并记录索引
                if distance < min_distance:
                    min_distance = distance
                    nearest_index = (i, j)
        return nearest_index
    
    def calculate_V(self, y, v):
        if self.k == 1:
            idx_y, idx_v = self.search_NN(y, v)
            return self.V[idx_y][idx_v]
        else:
            distances = []  # 存储所有点的索引、距离和 V 值
    
            # 遍历所有点，计算与 (y, v) 的距离，并保存到 distances 列表中
            for i in range(len(self.y_space)):
                for j in range(len(self.v_space)):
                    y_point = self.y_space[i]
                    v_point = self.v_space[j]
                    distance = math.sqrt((y - y_point)**2 + (v - v_point)**2)
                    
                    # 如果点正好是目标点，避免除以零，使用一个很小的值替代距离为 0 的情况
                    if distance == 0:
                        distance = 0.01

                    distances.append(((i, j), distance, self.V[i][j]))

            # 按距离从小到大排序
            distances.sort(key=lambda x: x[1])

            # 选择最近的 k 个点
            nearest_k = distances[:self.k]

            # 使用距离的倒数作为初步权重
            weights = [1 / d[1] for d in nearest_k]
            
            # 归一化权重，使它们的和为 1
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # 计算加权 V 值之和
            weighted_sum = sum(w * d[2] for w, d in zip(normalized_weights, nearest_k))
            
            return weighted_sum



    # Value iteration algorithm with stopping condition, policy extraction, and saving
    def value_iteration(self, threshold=0.01, max_iterations=100):
        total_rewards = []  
        optimal_reached = False

        for iteration in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)
            total_reward = 0
            if iteration % 10 == 0:
                print(f"Iteration {iteration+1}:")

            for i, y in enumerate(self.y_space):
                for j, v in enumerate(self.v_space):
                    max_value = -float('inf')
                    best_action = None
                    for a in self.actions:
                        (p1, p2, p3), new_y, (new_v1, new_v2, new_v3) = self.transition(y, v, a)
                        r = p1 * self.reward(new_y, new_v1, a) + p2 * self.reward(new_y, new_v2, a) + p3 * self.reward(new_y, new_v3, a)
                        
                        # value = r + Gamma{ p1*V(s1) + p1*V(s2) + p1*V(s3) }
                        # V(si) = V(e) -- N.N.  /   V(si)=sigma(wk * V(ek))
                        value = r + self.gamma * (p1 * self.calculate_V(new_y, new_v1) + 
                                                  p2 * self.calculate_V(new_y, new_v2) + 
                                                  p3 * self.calculate_V(new_y, new_v3))
                        if value > max_value:
                            max_value = value
                            best_action = a  # Extract the best action

                    new_V[i][j] = max_value
                    self.policy[i][j] = best_action  # Update policy
                    delta = max(delta, abs(self.V[i][j] - max_value))
                    total_reward += max_value

            # if new_V[self.optimal_state[0] + self.y_max][self.optimal_state[1] + self.v_max] >= 100:
            #     optimal_reached = True

            self.V = new_V
            total_rewards.append(total_reward)

            # if delta < threshold and optimal_reached:
            if delta < threshold:
                print(f"Optimal state reached after {iteration} iterations.")
                break

        # Extract Policy
        for i, y in enumerate(self.y_space):
            for i, v in enumerate(self.v_space):
                max_value = -float('inf')
                best_action = None
                for a in self.actions:
                    (p1, p2, p3), new_y, (new_v1, new_v2, new_v3) = self.transition(y, v, a)
                    r = p1 * self.reward(new_y, new_v1, a) + p2 * self.reward(new_y, new_v2, a) + p3 * self.reward(new_y, new_v3, a)
                    
                    # value = r + Gamma{ p1*V(s1) + p1*V(s2) + p1*V(s3) }
                    # V(si) = V(e) -- N.N.  /   V(si)=sigma(wk*V(ek))
                    value = r + self.gamma * (p1 * self.calculate_V(new_y, new_v1) + 
                                                  p2 * self.calculate_V(new_y, new_v2) + 
                                                  p3 * self.calculate_V(new_y, new_v3))
                    if value > max_value:
                        max_value = value
                        best_action = a  # Extract the best action
                self.policy[i][j] = best_action  # Update policy


        # Save policy and value function after value iteration completes
        config = self.config_text()
        with open("policy" + config + ".pkl", 'wb') as pf:
            pickle.dump(self.policy, pf)
        with open("value" + config + ".pkl", 'wb') as vf:
            pickle.dump(self.V, vf)
        print("Policy and value function saved successfully.")

        return self.V, self.policy, total_rewards    

    # Load policy and value function from specific files
    def load_policy_and_value(self):
        try:
            config = self.config_text()
            with open("policy" + config + ".pkl", 'rb') as pv_file:
                self.policy = pickle.load(pv_file)
            with open("value" + config + ".pkl", 'rb') as vv_file:
                self.V = pickle.load(vv_file)
            print("Policy and value function loaded successfully.")
            self.load_flag = True
        except FileNotFoundError:
            print("One or more of the specified files not found.")

    def simulate(self, initial_state, max_steps=100, vis=True):
        simulations = {}
        y, v = initial_state
        total_reward = 0
        state_history = []
        state_history.append((y, v))
        action_history = []


        for step in range(max_steps):
            if self.k == 1:
                idx_y, idx_v = self.search_NN(y, v)
                action = self.policy[idx_y][idx_v]
            else:
                max_value = -float('inf')
                action = None
                for a in self.actions:
                    (p1, p2, p3), new_y, (new_v1, new_v2, new_v3) = self.transition(y, v, a)
                    r = p1 * self.reward(new_y, new_v1, a) + p2 * self.reward(new_y, new_v2, a) + p3 * self.reward(new_y, new_v3, a)
                    
                    # value = r + Gamma{ p1*V(s1) + p1*V(s2) + p1*V(s3) }
                    # V(si) = V(e) -- N.N.  /   V(si)=sigma(wk*V(ek))
                    value = r + self.gamma * (p1 * self.calculate_V(new_y, new_v1) + 
                                                  p2 * self.calculate_V(new_y, new_v2) + 
                                                  p3 * self.calculate_V(new_y, new_v3))
                    if value > max_value:
                        max_value = value
                        action = a  # Extract the best action


            (p1, p2, p3), new_y, (new_v1, new_v2, new_v3) = self.transition(y, v, action)
            events = ['p1', 'p2', 'p3']
            probabilities = [p1, p2, p3]
            selected_event = random.choices(events, probabilities)[0]
            if selected_event == 'p1':
                new_v = new_v1
            elif selected_event == 'p2':
                new_v = new_v2
            else:
                new_v = new_v3
            
            total_reward += self.reward(new_y, new_v, action)
            y, v = new_y, new_v
            state_history.append((y, v))
            action_history.append(action)

            if np.isclose(y, 0, atol=self.y_max / self.h_y) and np.isclose(v, 0, atol=self.v_max / self.h_v):
                break

        simulations = {
            'state_history': state_history,
            'action_history': action_history,
            'total_reward': total_reward
        }

        if vis == True:
            self.visualize(simulations)
        return simulations

    def visualize(self, simulations):
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))

        state_history = simulations['state_history']
        action_history = simulations['action_history']
        steps = list(range(len(state_history)))

        y_vals = [state[0] for state in state_history]
        v_vals = [state[1] for state in state_history]

        # Subplot 1: y (Position) over steps
        axes[0].plot(steps, y_vals, label='y (Position)')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('y (Position)')
        axes[0].set_title('Position (y) over Time')
        axes[0].grid(True)
        axes[0].legend()

        # Subplot 2: v (Velocity) over steps
        axes[1].plot(steps, v_vals, label='v (Velocity)', color='orange')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('v (Velocity)')
        axes[1].set_title('Velocity (v) over Time')
        axes[1].grid(True)
        axes[1].legend()

        # Subplot 3: Actions over steps
        axes[2].step(steps[:-1], action_history, label='Action', where='post')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Action')
        axes[2].set_title('Actions over Time')
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()

        # # Plot value function heatmaps
        # plt.figure(figsize=(6, 6))  # 设置图的大小

        # sns.heatmap(self.V, cmap='viridis', cbar_kws={'label': 'Value'})  # 绘制热图
        # plt.title('Value Function Heatmap (value iteration)')
        # plt.xlabel('Velocity')
        # plt.ylabel('Position')

        
        plt.show()


def test_MDP_statistics():
    ks = [1, 4]
    hs = [[20, 10, 10], [10, 10, 10], [10, 5, 10]]
    results = {}

    for k in ks:
        for h in hs:
            mdp = MDPSystem(k=k, h=h)
            mdp.load_policy_and_value()

            sim_times = 200
            optimal_reached = 0
            total_steps = 0
            
            for _ in range(sim_times):
                initial_state = (random.randint(-mdp.y_max, mdp.y_max), random.randint(-mdp.v_max, mdp.v_max))
                simulates = mdp.simulate(initial_state, vis=False)
                state_history = simulates['state_history']
                if len(state_history) >= 99:
                    continue
                optimal_reached += 1
                total_steps += len(state_history)
            
            reached_rate = optimal_reached / sim_times
            avg_steps = total_steps / optimal_reached
            results[(k, tuple(h))] = {
                'reached_rate': reached_rate,
                'avg_steps': avg_steps
            }

    print(results)

            
def test_MDP():
    # Example usage
    mdp = MDPSystem(k=4, h=[10, 10, 10])

    # Simulate with a random initial state and optionally update the policy
    initial_state = (random.randint(-mdp.y_max, mdp.y_max), random.randint(-mdp.v_max, mdp.v_max))

    # Load the policy and value function if they exist
    mdp.load_policy_and_value()
    # mdp.value_iteration()

    mdp.simulate(initial_state)


test_MDP_statistics()
