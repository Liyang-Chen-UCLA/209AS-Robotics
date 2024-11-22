# numberline_env.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class LinearSystemParams:
    """Parameters for the linear system"""
    dt: float = 1.0       # time step
    mass: float = 1.0     # mass of particle
    sigma_d: float = 0.1  # velocity disturbance coefficient
    
    # Reward function parameters
    position_weight: float = 1.0     # weight for position error in reward
    velocity_weight: float = 0.5     # weight for velocity error in reward
    control_weight: float = 0.1      # weight for control effort in reward
    goal_threshold: float = 0.1      # threshold for considering goal reached

class NumberlineEnv:
    """
    Linear system dynamics for particle on a numberline
    State vector x = [position; velocity]
    """
    def __init__(self, params: Optional[LinearSystemParams] = None):
        self.params = params if params is not None else LinearSystemParams()
        
        # Define system matrices
        # State transition matrix A
        self.A = np.array([
            [1.0, self.params.dt],  # position update
            [0.0, 1.0]              # velocity update
        ])
        
        # Input matrix B
        self.B = np.array([
            [0.0],                          
            [self.params.dt/self.params.mass]  
        ])
        
        # Process noise covariance matrix Q
        self.Q = np.array([
            [0.0, 0.0],
            [0.0, (self.params.sigma_d ** 2)]  
        ])
        
        # Current state
        self.state = np.zeros((2, 1))
        
    def compute_reward(self, state: np.ndarray, action: float) -> Tuple[float, bool]:
        """Compute reward based on current state and action"""
        position = state[0, 0]
        velocity = state[1, 0]

        # Check if goal is reached
        done = (abs(position) < self.params.goal_threshold and 
                abs(velocity) < self.params.goal_threshold)

        if done:
            return 10.0, True

        reward = -abs(position) - 0.1*abs(action)
        return reward, done
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset the system state"""
        if initial_state is not None:
            self.state = initial_state.reshape((2, 1))
        else:
            self.state = np.array([[np.random.uniform(-20.0, 20.0)],
                                 [np.random.uniform(-5.0, 5.0)]])
        return self.state.copy()
    
    def step(self, u: float) -> Tuple[np.ndarray, float, bool]:
        """
        Take one step in the system
        
        Args:
            u: input force (scalar)
            
        Returns:
            tuple: (next_state, reward, done)
        """
        # Ensure input is in correct shape and bounded
        u = np.clip(np.array([[u]]), -1, 1)
        
        # Save current state for reward computation
        current_state = self.state.copy()
        
        # Generate process noise
        w = np.random.multivariate_normal(
            mean=[0, 0], 
            cov=self.Q
        ).reshape((2, 1))
        
        # Update state
        self.state = self.A @ self.state + self.B @ u + w
        
        # Compute reward and done flag
        reward, done = self.compute_reward(current_state, u[0, 0])
        
        return self.state.copy(), reward, done
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.state.copy()