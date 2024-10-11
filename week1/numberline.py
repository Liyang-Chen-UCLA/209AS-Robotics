import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
time_steps = 100  # Number of time steps
m = 1  # Mass of the ball
y = np.zeros(time_steps + 1)  # Position of the ball
v = np.zeros(time_steps + 1)  # Velocity of the ball
fi = np.random.uniform(-1, 1, time_steps)  # Random external force in [-1, 1]

# Initial conditions
y[0] = 0  # Initial position
v[0] = 0  # Initial velocity

# Simulation loop with Gaussian noise and potential field force f_phi
for t in range(time_steps):
    f_phi = np.cos(y[t])  # Potential field force based on position y[t]
    nv = np.random.normal(0, (0.01 * v[t])**2)  # Gaussian noise
    v[t + 1] = v[t] + (1 / m) * (fi[t] + f_phi) + nv  # Update velocity with f_phi and noise
    y[t + 1] = y[t] + v[t]  # Update position

# Plot results
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot position over time
ax[0].plot(range(time_steps + 1), y, label='Position (y)')
ax[0].set_title('Position of the Ball Over Time (with Potential Field)')
ax[0].set_xlabel('Time Step')
ax[0].set_ylabel('Position (y)')
ax[0].grid(True)
ax[0].legend()

# Plot velocity over time
ax[1].plot(range(time_steps + 1), v, label='Velocity (v)', color='orange')
ax[1].set_title('Velocity of the Ball Over Time (with Potential Field)')
ax[1].set_xlabel('Time Step')
ax[1].set_ylabel('Velocity (v)')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()
