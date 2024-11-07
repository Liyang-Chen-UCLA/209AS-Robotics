import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import stats

class ParticleFilterSystem:
    def __init__(self, n_particles=500):
        # System matrices
        self.A = np.array([[1, 1],
                          [0, 1]])
        self.B = np.array([[0],
                          [1]])
        
        # Noise parameters
        self.sigma_d = 0.1  # Process noise
        self.sigma_n = 0.5  # Measurement noise
        
        # Particle filter parameters
        self.n_particles = n_particles
        self.particles = None
        self.weights = None
        
        # LQR parameters
        self.Q = np.diag([1.0, 1.0])
        self.R = np.array([[1.0]])
        self.K = self.compute_lqr_gain()
    
    def compute_lqr_gain(self):
        """Compute LQR gain matrix"""
        P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        return K
    
    def initialize_particles(self, initial_state=None, state_range=None):
        """Initialize particles either around known state or uniformly"""
        if initial_state is not None:
            # Known initial state
            self.particles = np.random.normal(
                loc=initial_state,
                scale=0.1,
                size=(self.n_particles, 2)
            )
        else:
            # Unknown initial state
            if state_range is None:
                state_range = [(-10, 10), (-5, 5)]
            self.particles = np.random.uniform(
                low=[r[0] for r in state_range],
                high=[r[1] for r in state_range],
                size=(self.n_particles, 2)
            )
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def predict_particles(self, action):
        """Predict step of particle filter"""
        for i in range(self.n_particles):
            # Apply system dynamics
            self.particles[i] = self.A @ self.particles[i] + self.B @ np.array([action])
            # Add process noise
            noise_v = np.random.normal(0, self.sigma_d * abs(self.particles[i, 1]))
            self.particles[i, 1] += noise_v
    
    def update_particles(self, measurement):
        """Update step of particle filter"""
        for i in range(self.n_particles):
            # Compute likelihood
            pred_measurement = self.particles[i, 0]
            likelihood = stats.norm.pdf(
                measurement,
                pred_measurement,
                self.sigma_n * abs(self.particles[i, 1])
            )
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
        # Resample if needed
        self.resample_if_needed()
    
    def resample_if_needed(self):
        """Resample particles if effective sample size is too low"""
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 2:
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            
            # Generate random sampling positions
            positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
            
            # Resample particles
            new_particles = np.zeros_like(self.particles)
            i, j = 0, 0
            while i < self.n_particles:
                if positions[i] < cumsum[j]:
                    new_particles[i] = self.particles[j]
                    i += 1
                else:
                    j += 1
            
            self.particles = new_particles
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def get_state_estimate(self):
        """Get weighted mean of particles"""
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def get_control_input(self, method='mean_state'):
        """
        Compute control input using either mean state or particle-based approach
        """
        if method == 'mean_state':
            # Use mean state for control
            mean_state = self.get_state_estimate()
            u = float(-self.K @ mean_state)
        else:
            # Apply controller to each particle and average
            individual_controls = [-float(self.K @ p) for p in self.particles]
            u = np.average(individual_controls, weights=self.weights)
        
        return np.clip(u, -1, 1)

def simulate_system(pf_system, true_initial_state, n_steps=100, control_method='mean_state', known_initial_state=True):
    """Run complete simulation"""
    # Initialize true state and history
    true_state = true_initial_state
    true_states = [true_state.copy()]
    measurements = []
    estimates = []
    controls = []
    
    # Initialize particles
    if known_initial_state:
        pf_system.initialize_particles(initial_state=true_initial_state)
    else:
        pf_system.initialize_particles()
    
    estimates.append(pf_system.get_state_estimate())
    
    # Simulation loop
    for t in range(n_steps):
        # Get control input
        u = pf_system.get_control_input(method=control_method)
        controls.append(u)
        
        # Update true state
        noise_v = np.random.normal(0, pf_system.sigma_d * abs(true_state[1]))
        true_state = pf_system.A @ true_state + pf_system.B @ np.array([u]) + np.array([0, noise_v])
        true_states.append(true_state.copy())
        
        # Generate measurement
        measurement = true_state[0] + np.random.normal(0, pf_system.sigma_n * abs(true_state[1]))
        measurements.append(measurement)
        
        # Update particle filter
        pf_system.predict_particles(u)
        pf_system.update_particles(measurement)
        estimates.append(pf_system.get_state_estimate())
    
    return np.array(true_states), np.array(measurements), np.array(estimates), np.array(controls)

def plot_results(true_states, measurements, estimates, controls, title):
    """Plot simulation results"""
    plt.figure(figsize=(15, 10))
    
    # Position and velocity plot
    plt.subplot(2, 1, 1)
    t = np.arange(len(true_states))
    plt.plot(t, true_states[:, 0], 'b-', label='True Position')
    plt.plot(t, true_states[:, 1], 'g-', label='True Velocity')
    plt.plot(t, estimates[:, 0], 'b--', label='Estimated Position')
    plt.plot(t, estimates[:, 1], 'g--', label='Estimated Velocity')
    plt.scatter(t[1:], measurements, color='r', alpha=0.3, label='Measurements')
    plt.grid(True)
    plt.legend()
    plt.title(f'{title} - States')
    
    # Control input plot
    plt.subplot(2, 1, 2)
    plt.plot(t[:-1], controls, 'r-', label='Control Input')
    plt.grid(True)
    plt.legend()
    plt.title('Control Input')
    
    plt.tight_layout()
    plt.show()

def main():
    # System parameters
    n_particles = 1000
    n_steps = 100
    true_initial_state = np.array([5.0, 0.0])
    
    # Create system
    pf_system = ParticleFilterSystem(n_particles=n_particles)
    
    # Test cases
    test_cases = [
        ('Known Initial State - Mean Control', True, 'mean_state'),
        ('Unknown Initial State - Mean Control', False, 'mean_state'),
        ('Known Initial State - Particle Control', True, 'particle'),
        ('Unknown Initial State - Particle Control', False, 'particle')
    ]
    
    # Run simulations
    for title, known_init, control_method in test_cases:
        print(f"\nRunning simulation: {title}")
        true_states, measurements, estimates, controls = simulate_system(
            pf_system,
            true_initial_state,
            n_steps=n_steps,
            control_method=control_method,
            known_initial_state=known_init
        )
        
        # Calculate performance metrics
        position_rmse = np.sqrt(np.mean((true_states[1:, 0] - estimates[1:, 0])**2))
        velocity_rmse = np.sqrt(np.mean((true_states[1:, 1] - estimates[1:, 1])**2))
        control_effort = np.mean(np.abs(controls))
        
        print(f"Position RMSE: {position_rmse:.3f}")
        print(f"Velocity RMSE: {velocity_rmse:.3f}")
        print(f"Average Control Effort: {control_effort:.3f}")
        
        plot_results(true_states, measurements, estimates, controls, title)

if __name__ == "__main__":
    np.random.seed(42)
    main()