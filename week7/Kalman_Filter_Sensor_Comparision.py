import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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

from scipy.linalg import solve_discrete_are

class LQRController:
    def __init__(self, F, G, Q, R):
        self.F = F
        self.G = G
        self.Q = Q
        self.R = R
        self.P = solve_discrete_are(F, G, Q, R)
        self.K = np.linalg.inv(G.T @ self.P @ G + R) @ (G.T @ self.P @ F)

    def control(self, state):
        return -self.K @ state

# Define LQR parameters
F = np.array([[1, 1], [0, 1]])
G = np.array([[0], [1]])
Q_lqr = np.diag([1, 1])
R_lqr = np.array([[0.1]])

lqr = LQRController(F, G, Q_lqr, R_lqr)

def simulate_with_lqr(F, H, Q, R, init_state, init_cov, num_steps, lqr):
    true_state = init_state.copy()
    kf = KalmanFilter(F, H, Q, R, init_state, init_cov)
    states = [true_state.flatten()]
    estimates = [kf.state_est.flatten()]

    for _ in range(num_steps):
        u = lqr.control(kf.state_est)
        process_noise = np.random.multivariate_normal([0, 0], Q).reshape(-1, 1)
        true_state = F @ true_state + G @ u + process_noise
        observation_noise = np.random.multivariate_normal([0, 0], R).reshape(-1, 1)
        observation = H @ true_state + observation_noise
        kf.predict(G @ u)
        kf.update(observation)
        states.append(true_state.flatten())
        estimates.append(kf.state_est.flatten())

    return np.array(states), np.array(estimates)

def create_gif(states_a, estimates_a, states_b, estimates_b, filename="Ra-Rb.gif"):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    x_max = len(states_a)

    def update(frame):
        ax[0].clear()
        ax[1].clear()
        
        ax[0].plot(states_a[:frame, 0], label="True Pos (Ra)", linestyle='dotted', color='blue')
        ax[0].plot(estimates_a[:frame, 0], label="Estimated Pos (Ra)", color='blue')
        ax[0].plot(states_b[:frame, 0], label="True Pos (Rb)", linestyle='dotted', color='red')
        ax[0].plot(estimates_b[:frame, 0], label="Estimated Pos (Rb)", color='red')
        ax[0].set_title("Position Comparison")
        ax[0].legend()
        ax[0].set_xlim(0, x_max)
        ax[0].set_ylim(min(states_a[:, 0].min(), states_b[:, 0].min()) - 1, 
                       max(states_a[:, 0].max(), states_b[:, 0].max()) + 1)

        ax[1].plot(states_a[:frame, 1], label="True Vel (Ra)", linestyle='dotted', color='blue')
        ax[1].plot(estimates_a[:frame, 1], label="Estimated Vel (Ra)", color='blue')
        ax[1].plot(states_b[:frame, 1], label="True Vel (Rb)", linestyle='dotted', color='red')
        ax[1].plot(estimates_b[:frame, 1], label="Estimated Vel (Rb)", color='red')
        ax[1].set_title("Velocity Comparison")
        ax[1].legend()
        ax[1].set_xlim(0, x_max)
        ax[1].set_ylim(min(states_a[:, 1].min(), states_b[:, 1].min()) - 1, 
                       max(states_a[:, 1].max(), states_b[:, 1].max()) + 1)


    ani = FuncAnimation(fig, update, frames=len(states_a), repeat=False)
    ani.save(filename, writer=PillowWriter(fps=10))

# Define constants
H = np.eye(2)
Q = np.array([[0.1, 0], [0, 0.1]])
init_state = np.array([[10], [0]])
init_cov = np.eye(2) * 0.1
R_a = np.array([[0.25**2, 0], [0, 0.25**2]])  # Larger diagonal elements, no correlation
R_b = np.array([[0.2**2, 0.05], [0.05, 0.2**2]])  # Smaller diagonal elements, nonzero off-diagonal elements

states_a_lqr, estimates_a_lqr = simulate_with_lqr(F, H, Q, R_a, init_state, init_cov, 50, lqr)
states_b_lqr, estimates_b_lqr = simulate_with_lqr(F, H, Q, R_b, init_state, init_cov, 50, lqr)

create_gif(states_a_lqr, estimates_a_lqr, states_b_lqr, estimates_b_lqr)
