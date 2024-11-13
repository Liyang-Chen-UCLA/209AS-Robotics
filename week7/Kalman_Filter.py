import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, F, H, Q, R, init_state, init_cov):
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Observation noise covariance
        self.state_est = init_state  # Initial state estimate
        self.P = init_cov  # Initial estimation error covariance

    def predict(self, u=np.zeros((2, 1))):
        self.state_est = self.F @ self.state_est + u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, observation):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.state_est = self.state_est + K @ (observation - self.H @ self.state_est)
        self.P = (np.eye(self.F.shape[0]) - K @ self.H) @ self.P

def simulate_single_sensor(F, H, Q, R, init_state, init_cov, num_steps):
    true_state = init_state.copy()
    kf = KalmanFilter(F, H, Q, R, init_state, init_cov)
    
    states = []
    estimates = []
    measurements = []
    
    for _ in range(num_steps):
        process_noise = np.random.multivariate_normal([0, 0], Q).reshape(-1, 1)
        observation_noise = np.random.multivariate_normal([0], R).reshape(-1, 1)
        true_state = F @ true_state + process_noise
        observation = H @ true_state + observation_noise
        
        kf.predict()
        kf.update(observation)
        
        states.append(true_state.flatten())
        estimates.append(kf.state_est.flatten())
        measurements.append(observation.flatten())
        
    return np.array(states), np.array(estimates), np.array(measurements)

def plot_results(states, estimates, measurements, title="Simulation"):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(states[:, 0], label="True Position")
    plt.plot(estimates[:, 0], label="Estimated Position")
    plt.title(f'{title}: Position')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(states[:, 1], label="True Velocity")
    plt.plot(estimates[:, 1], label="Estimated Velocity")
    plt.title(f'{title}: Velocity')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(states[:, 1], label="True Velocity")
    plt.plot(estimates[:, 1], label="Estimated Velocity")
    plt.scatter(range(len(measurements)), measurements, label="Measurements", c='r', s=10, alpha=0.5)
    plt.title(f'{title}: Velocity with Measurement Noise')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Define constants
F = np.array([[1, 1], [0, 1]])  # State transition matrix
H = np.array([[1, 0]])  # Observation matrix
Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise covariance
init_state = np.array([[10], [0]])  # Initial state
init_cov = np.eye(2) * 0.1  # Initial state covariance

# Simulate for the noise in positions and process
# R = np.array([[1]])  # Observation noise
# states, estimates, measurements = simulate_single_sensor(F, H, Q, R, init_state, init_cov, 50)
# plot_results(states, estimates, measurements, title="Noise in Position Process")

# Scenario: Compare Sensors Ra and Rb for Position and Velocity Estimates
R_a = np.array([[0.5**2, 0], [0, 0.5**2]])
R_b = np.array([[0.5**2, 0.2], [0.2, 0.2]])
H_new = np.array([1, 1])
states_a, estimates_a, _ = simulate_single_sensor(F, H_new, Q, R_a, init_state, init_cov, 50)
states_b, estimates_b, _ = simulate_single_sensor(F, H_new, Q, R_b, init_state, init_cov, 50)

plot_results(states_a, estimates_a, np.zeros((50, 1)), title="Sensor with Ra Uncorrelated Noise")
plot_results(states_b, estimates_b, np.zeros((50, 1)), title="Sensor with Rb Correlated Noise")
