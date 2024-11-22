# main.py

import numpy as np
import matplotlib.pyplot as plt
from numberline_env import NumberlineEnv, LinearSystemParams
from q_network import QLearning, QNetworkParams
from typing import List, Tuple
import torch
import time
from tqdm import tqdm

def run_episode(
    env: NumberlineEnv,
    agent: QLearning,
    epsilon: float,
    max_steps: int,
    training: bool = True
) -> Tuple[float, List[np.ndarray], List[float], List[float]]:
    """
    Run a single episode
    
    Returns:
        tuple: (total_reward, states, actions, rewards)
    """
    state = env.reset()
    total_reward = 0
    states = [state.copy()]
    actions = []
    rewards = []
    
    for step in range(max_steps):
        # Select action
        action = agent.select_action(state, epsilon if training else 0.0)
        
        # Take step in environment
        next_state, reward, done = env.step(action)
        
        # Save trajectory info
        states.append(next_state.copy())
        actions.append(action)
        rewards.append(reward)
        
        # Save transition to replay buffer if training
        if training:
            agent.save_transition(state, action, reward, next_state, done)
            # Train agent
            loss = agent.train()
        
        total_reward += reward
        state = next_state
        
        if done:
            break
            
    return total_reward, states, actions, rewards

def train_agent(
    num_episodes: int = 1000,
    max_steps: int = 200,
    eval_interval: int = 50,
    render_interval: int = 100
) -> Tuple[List[float], List[float]]:
    """
    Train the agent
    
    Returns:
        tuple: (training_rewards, eval_rewards)
    """
    # Initialize environment and agent
    env = NumberlineEnv()
    # q_params = QNetworkParams(
    #     hidden_dim=128,
    #     learning_rate=3e-4,
    #     batch_size=128
    # )
    # agent = QLearning(q_params)
    agent = QLearning()
    # Training metrics
    training_rewards = []
    eval_rewards = []
    best_eval_reward = -np.inf
    
    # Exploration parameters
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.9
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        # Run training episode
        episode_reward, states, actions, rewards = run_episode(
            env, agent, epsilon, max_steps, training=True
        )
        training_rewards.append(episode_reward)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Evaluate agent
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env)
            eval_rewards.append(eval_reward)
            
            # Update progress bar
            pbar.set_postfix({
                'Episode Reward': f"{episode_reward:.2f}",
                'Eval Reward': f"{eval_reward:.2f}",
                'Epsilon': f"{epsilon:.2f}"
            })
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(agent.q_network.state_dict(), 'best_model.pth')
        
        # Visualize trajectory
        if (episode + 1) % render_interval == 0:
            visualize_trajectory(states, actions, rewards, episode + 1)
        
        agent.increment_episode()

    return training_rewards, eval_rewards

def evaluate_agent(
    agent: QLearning,
    env: NumberlineEnv,
    num_episodes: int = 5,
    max_steps: int = 50
) -> float:
    """Evaluate the agent's performance"""
    eval_rewards = []
    
    for _ in range(num_episodes):
        episode_reward, _, _, _ = run_episode(
            env, agent, epsilon=0.0, max_steps=max_steps, training=False
        )
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)

def visualize_trajectory(
    states: List[np.ndarray],
    actions: List[float],
    rewards: List[float],
    episode: int
):
    """Visualize the trajectory of the agent"""
    states = np.array(states)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Episode {episode} Trajectory')
    
    # Plot position vs time
    axs[0, 0].plot(states[:, 0], label='Position')
    axs[0, 0].set_title('Position vs Time')
    axs[0, 0].set_xlabel('Step')
    axs[0, 0].set_ylabel('Position')
    axs[0, 0].grid(True)
    
    # Plot velocity vs time
    axs[0, 1].plot(states[:, 1], label='Velocity')
    axs[0, 1].set_title('Velocity vs Time')
    axs[0, 1].set_xlabel('Step')
    axs[0, 1].set_ylabel('Velocity')
    axs[0, 1].grid(True)
    
    # Plot actions vs time
    axs[1, 0].plot(actions, label='Action')
    axs[1, 0].set_title('Action vs Time')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('Action')
    axs[1, 0].grid(True)
    
    # Plot rewards vs time
    axs[1, 1].plot(rewards, label='Reward')
    axs[1, 1].set_title('Reward vs Time')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('Reward')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'trajectory_{episode}.png')
    plt.close()

def plot_learning_curves(
    training_rewards: List[float],
    eval_rewards: List[float],
    eval_interval: int
):
    """Plot the learning curves"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot training rewards
    ax1.plot(training_rewards)
    ax1.set_title('Training Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot evaluation rewards
    eval_episodes = np.arange(eval_interval, len(training_rewards) + 1, eval_interval)
    ax2.plot(eval_episodes, eval_rewards)
    ax2.set_title('Evaluation Rewards')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Training parameters
    NUM_EPISODES = 1000
    MAX_STEPS = 50
    EVAL_INTERVAL = 100
    RENDER_INTERVAL = 100
    
    # Train agent
    print("Starting training...")
    start_time = time.time()
    
    training_rewards, eval_rewards = train_agent(
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        eval_interval=EVAL_INTERVAL,
        render_interval=RENDER_INTERVAL
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot learning curves
    plot_learning_curves(training_rewards, eval_rewards, EVAL_INTERVAL)
    
    # Load and evaluate best model
    env = NumberlineEnv()
    agent = QLearning()
    agent.q_network.load_state_dict(torch.load('best_model.pth'))
    
    final_eval_reward = evaluate_agent(agent, env, num_episodes=10)
    print(f"\nFinal evaluation reward: {final_eval_reward:.2f}")
    
    # Run and visualize one final trajectory
    _, states, actions, rewards = run_episode(
        env, agent, epsilon=0.0, max_steps=MAX_STEPS, training=False
    )
    visualize_trajectory(states, actions, rewards, "final")