# q_network.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
from collections import deque
import random

@dataclass
class QNetworkParams:
    """Parameters for Q-Network"""
    state_dim: int = 2           # dimension of state (position, velocity)
    action_dim: int = 1          # dimension of action (force)
    hidden_dim: int = 64         # dimension of hidden layers
    learning_rate: float = 1e-3  # learning rate for gradient descent
    gamma: float = 0.99         # discount factor for future rewards
    tau: float = 0.005          # soft update parameter for target network
    batch_size: int = 64        # batch size for training
    buffer_size: int = 100000   # replay buffer size

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        # Flatten state and next_state if they are 2D arrays
        if isinstance(state, np.ndarray):
            state = state.flatten()
        if isinstance(next_state, np.ndarray):
            next_state = next_state.flatten()
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions"""
        transitions = random.sample(self.buffer, batch_size)
        batch = list(zip(*transitions))
        return (np.array(batch[0]), np.array(batch[1]), 
                np.array(batch[2]), np.array(batch[3]), 
                np.array(batch[4]))
    
    def __len__(self) -> int:
        return len(self.buffer)

class QNetwork(nn.Module):
    """Neural network for Q-function approximation"""
    def __init__(self, params: Optional[QNetworkParams] = None):
        super().__init__()
        self.params = params if params is not None else QNetworkParams()
        
        # Input layer: state and action concatenated
        input_dim = self.params.state_dim + self.params.action_dim
        
        # Neural network architecture
        self.layer1 = nn.Linear(input_dim, self.params.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.params.hidden_dim)
        
        self.layer2 = nn.Linear(self.params.hidden_dim, self.params.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.params.hidden_dim)
        
        self.layer3 = nn.Linear(self.params.hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network"""
        # Ensure state has correct shape [batch_size, state_dim]
        if state.dim() == 3:
            state = state.squeeze(2)
        
        x = torch.cat([state, action], dim=1)
        
        x = self.layer1(x)
        if x.shape[0] > 1:  # Only use batch norm for batch size > 1
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.layer2(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = F.relu(x)
        
        q_value = self.layer3(x)
        return q_value

class QLearning:
    """Q-Learning implementation with neural network approximation"""
    def __init__(self, params: Optional[QNetworkParams] = None):
        self.params = params if params is not None else QNetworkParams()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create Q networks
        self.q_network = QNetwork(self.params).to(self.device)
        self.target_network = QNetwork(self.params).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=self.params.learning_rate
        )
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(self.params.buffer_size)
        
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> float:
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.uniform(-1, 1)
        
        # Ensure state has correct shape [1, state_dim]
        state = state.flatten()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Sample actions and find best one
        sampled_actions = torch.FloatTensor(
            np.random.uniform(-1, 1, size=(1, 100))
        ).to(self.device)
        
        with torch.no_grad():
            q_values = torch.zeros(100).to(self.device)
            for i in range(100):
                action = sampled_actions[0, i:i+1].unsqueeze(0)
                q_value = self.q_network(state_tensor, action)
                q_values[i] = q_value
            
            best_action_idx = torch.argmax(q_values)
            best_action = sampled_actions[0, best_action_idx].item()
            
        return best_action
    
    def train(self):
        """Update Q-network using a batch from replay buffer"""
        if len(self.replay_buffer) < self.params.batch_size:
            return 0.0
            
        # Sample a batch
        batch = self.replay_buffer.sample(self.params.batch_size)
        states, actions, rewards, next_states, dones = [
            torch.FloatTensor(x).to(self.device) for x in batch
        ]
        
        # Reshape tensors
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        
        # Current Q values
        current_q = self.q_network(states, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            # Sample actions for next states
            sampled_actions = torch.FloatTensor(
                np.random.uniform(-1, 1, size=(len(next_states), 100))
            ).to(self.device)
            
            next_q_values = torch.zeros(
                (len(next_states), 100)
            ).to(self.device)
            
            for i in range(100):
                action_sample = sampled_actions[:, i:i+1]
                q_value = self.target_network(next_states, action_sample)
                next_q_values[:, i] = q_value.squeeze()
            
            # Take max over sampled actions
            next_q = torch.max(next_q_values, dim=1, keepdim=True)[0]
            
            # Compute target Q value
            target_q = rewards + (1 - dones) * self.params.gamma * next_q
        
        # Compute loss (mean over batch)
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(
            self.target_network.parameters(), 
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.params.tau * param.data + 
                (1.0 - self.params.tau) * target_param.data
            )
        
        return loss.item()
    
    def save_transition(
        self, 
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Save transition to replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)