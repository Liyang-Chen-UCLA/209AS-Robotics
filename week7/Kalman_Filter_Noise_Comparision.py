import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

class KalmanFilter:
    def __init__(self, F, H, Q, R, init_state, init_cov):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.state_est = init_state
        self.P = init_cov

    def predict(self, u=np.zeros((2, 1))):
        self.state_est = self.F @ self.state_est + u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, observation):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.state_est = self.state_est + K @ (observation - self.H @ self.state_est)
        self.P = (np.eye(self.F.shape[0]) - K @ self.H) @ self.P

class LQRController:
    def __init__(self, F, G, Q, R):
        self.F = F
        self.G = G
        self.Q = Q
        self.R = R
        # Solve the discrete-time algebraic Riccati equation
        self.P = solve_discrete_are(F, G, Q, R)
        # Compute the optimal gain matrix K
        self.K = np.linalg.inv(G.T @ self.P @ G + R) @ (G.T @ self.P @ F)

    def control(self, state):
        return -self.K @ state


def simulate_with_lqr(F, H, Q_func, R_func, init_state, init_cov, num_steps, target_state, lqr):
    true_state = init_state.copy()
    kf = KalmanFilter(F, H, Q_func(true_state[1, 0]), R_func(true_state[1, 0]), init_state, init_cov)
    states, estimates, observations = [true_state.flatten()], [kf.state_est.flatten()], []

    for _ in range(num_steps):
        u = lqr.control(kf.state_est - target_state)
        process_noise = np.random.multivariate_normal([0, 0], Q_func(true_state[1, 0])).reshape(-1, 1)
        true_state = F @ true_state + process_noise + np.array([[0], u[0]])
        observation_noise = np.random.multivariate_normal([0, 0], R_func(true_state[1, 0])).reshape(-1, 1)
        observation = H @ true_state + observation_noise
        kf.predict(np.array([[0], u[0]]))
        kf.update(observation)
        
        states.append(true_state.flatten())
        estimates.append(kf.state_est.flatten())
        observations.append(observation.flatten())  # Record observations

    return np.array(states), np.array(estimates), np.array(observations)


# Noise functions
def simple_process_noise(velocity):
    """Amplified process noise with constant variance for velocity."""
    return np.array([[0, 0], [0, 0.05]])

def complex_process_noise(velocity):
    """Amplified process noise where variance scales with the velocity."""
    return np.array([[0, 0], [0, 0.1 * abs(velocity)]])

def fixed_observation_noise(_):
    """Amplified fixed observation noise, independent of velocity."""
    return np.array([[0.1, 0], [0, 0]])

def velocity_observation_noise(velocity):
    """Amplified observation noise that increases with velocity."""
    return np.array([[1e-4 + 0.1 * abs(velocity), 0], [0, 1e-4]])


# System parameters
F = np.array([[1, 1], [0, 1]])
G = np.array([[0], [1]])
H = np.eye(2)
Q_lqr = np.diag([1, 1])
R_lqr = np.array([[100]])
lqr = LQRController(F, G, Q_lqr, R_lqr)

init_state = np.array([[-5], [0]])
init_cov = np.eye(2) * 0.1
target_state = np.array([[5], [0]])
num_steps = 25

# Simple noise setup
states_simple, estimates_simple, observations_simple = simulate_with_lqr(
    F, H, simple_process_noise, fixed_observation_noise, init_state, init_cov, num_steps, target_state, lqr
)

# Complex noise setup
states_complex, estimates_complex, observations_complex = simulate_with_lqr(
    F, H, complex_process_noise, velocity_observation_noise, init_state, init_cov, num_steps, target_state, lqr
)

# Plotting
plt.figure(figsize=(10, 6))

# Position comparison
plt.subplot(2, 1, 1)
plt.plot(states_simple[:, 0], 'b--', label='True Pos (Additive Noise)')
plt.plot(estimates_simple[:, 0], 'b-', label='Est. Pos (Additive Noise)')
plt.plot(observations_simple[:, 0], 'bx', label='Obs. Pos (Additive Noise)', markersize=4)
plt.plot(states_complex[:, 0], 'r--', label='True Pos (Multiplicative Noise)')
plt.plot(estimates_complex[:, 0], 'r-', label='Est. Pos (Multiplicative Noise)')
plt.plot(observations_complex[:, 0], 'rx', label='Obs. Pos (Multiplicative Noise)', markersize=4)
plt.title('Position Comparison')
plt.legend()

# Velocity comparison
plt.subplot(2, 1, 2)
plt.plot(states_simple[:, 1], 'b--', label='True Vel (Additive Noise)')
plt.plot(estimates_simple[:, 1], 'b-', label='Est. Vel (Additive Noise)')
plt.plot(observations_simple[:, 1], 'bx', label='Obs. Vel (Additive Noise)', markersize=4)
plt.plot(states_complex[:, 1], 'r--', label='True Vel (Multiplicative Noise)')
plt.plot(estimates_complex[:, 1], 'r-', label='Est. Vel (Multiplicative Noise)')
plt.plot(observations_complex[:, 1], 'rx', label='Obs. Vel (Multiplicative Noise)', markersize=4)
plt.title('Velocity Comparison')
plt.legend()

plt.tight_layout()
plt.show()

