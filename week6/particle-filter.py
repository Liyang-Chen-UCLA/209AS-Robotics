import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import stats
from matplotlib.animation import FuncAnimation, PillowWriter

class ParticleFilterSystem:
    def __init__(self, n_particles=500, init_type='known', control_type='state_mean'):
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
        
        # Strategy parameters
        self.init_type = init_type
        self.control_type = control_type
        
        # LQR parameters
        self.Q = np.diag([3.0, 3.0])
        self.R = np.array([[2.0]])
        self.K = self.compute_lqr_gain()
    
    def compute_lqr_gain(self):
        """Compute LQR gain matrix"""
        P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        return K
    
    def initialize_particles(self, initial_state=None):
        """Initialize particles based on initialization type"""
        if self.init_type == 'known' and initial_state is not None:
            self.particles = np.random.normal(
                loc=initial_state,
                scale=0.1,
                size=(self.n_particles, 2)
            )
        else:  # unknown initial state
            self.particles = np.random.uniform(
                low=[-10, -5],
                high=[10, 5],
                size=(self.n_particles, 2)
            )
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def predict_particles(self, action):
        """Predict step of particle filter"""
        for i in range(self.n_particles):
            self.particles[i] = self.A @ self.particles[i] + self.B.flatten() * action
            noise = np.random.normal(0, self.sigma_d, 2)
            self.particles[i] += noise
    
    def update_particles(self, measurement_y, measurement_v):
        """Update step of particle filter with both position and velocity measurements"""
        for i in range(self.n_particles):
            likelihood_y = stats.norm.pdf(
                measurement_y,
                self.particles[i, 0],
                self.sigma_n
            )
            likelihood_v = stats.norm.pdf(
                measurement_v,
                self.particles[i, 1],
                self.sigma_n
            )
            self.weights[i] *= likelihood_y * likelihood_v
        
        self.weights = np.clip(self.weights, 1e-300, None)
        self.weights /= np.sum(self.weights)
        self.resample_if_needed()
    
    def resample_if_needed(self):
        """Resample particles if effective sample size is too low"""
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 2:
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            
            positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
            
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
    
    def get_control_input(self):
        """Compute control input based on control strategy"""
        if self.control_type == 'state_mean':
            # Use mean state for control
            mean_state = self.get_state_estimate()
            return float(-self.K[0] @ mean_state)
        else:  # control_mean
            # Compute control for each particle and average
            individual_controls = -self.K[0] @ self.particles.T
            return np.average(individual_controls, weights=self.weights)
        

def create_and_save_animation(init_type, control_type, title, n_steps=100):
    """Create animation and save as GIF"""
    
    class StateHolder:
        def __init__(self):
            self.true_state = np.array([5.0, 2.0])
            self.control_history = []
            self.true_position_history = []
            self.true_velocity_history = []
            self.est_position_history = []
            self.est_velocity_history = []
            self.time_steps = []
    
    # Initialize system and state
    pf_system = ParticleFilterSystem(
        n_particles=1000,
        init_type=init_type,
        control_type=control_type
    )
    state = StateHolder()
    
    # Initialize particles
    if init_type == 'known':
        pf_system.initialize_particles(state.true_state)
    else:
        pf_system.initialize_particles()
    
    # Create figure with subplots
    fig, (ax_phase, ax_history) = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3, top=0.85)
    fig.suptitle(title, fontsize=12)
    
    # Setup phase space plot
    scatter = ax_phase.scatter([], [], c=[], cmap='viridis', alpha=0.5)
    true_point, = ax_phase.plot([], [], 'r*', markersize=15, label='True State')
    estimate_point, = ax_phase.plot([], [], 'g*', markersize=15, label='Estimate')
    
    ax_phase.set_xlim(-10, 10)
    ax_phase.set_ylim(-5, 5)
    ax_phase.set_xlabel('Position')
    ax_phase.set_ylabel('Velocity')
    ax_phase.grid(True)
    ax_phase.legend()
    ax_phase.set_title('Phase Space')
    
    # Setup history plot
    control_line, = ax_history.plot([], [], 'b-', label='Control')
    true_pos_line, = ax_history.plot([], [], 'r-', label='True Position')
    true_vel_line, = ax_history.plot([], [], 'g-', label='True Velocity')
    est_pos_line, = ax_history.plot([], [], 'r--', label='Est Position')
    est_vel_line, = ax_history.plot([], [], 'g--', label='Est Velocity')
    
    ax_history.set_xlim(0, n_steps)
    ax_history.set_ylim(-5, 5)
    ax_history.set_xlabel('Time Step')
    ax_history.set_ylabel('Value')
    ax_history.grid(True)
    ax_history.legend()
    ax_history.set_title('Time History')

    measurement_y = state.true_state[0] + np.random.normal(0, pf_system.sigma_n)
    measurement_v = state.true_state[1] + np.random.normal(0, pf_system.sigma_n)
    state.time_steps.append(0)
    state.true_position_history.append(state.true_state[0])
    state.true_velocity_history.append(state.true_state[1])
    estimate = pf_system.get_state_estimate()
    state.est_position_history.append(estimate[0])
    state.est_velocity_history.append(estimate[1])
    
    def update(frame):
        if frame > 0:
            # Get control input
            u = pf_system.get_control_input()
            state.control_history.append(u)
            
            # Update true state
            state.true_state = pf_system.A @ state.true_state + pf_system.B.flatten() * u
            state.true_state += np.random.normal(0, pf_system.sigma_d, 2)
            
            # Generate measurements
            measurement_y = state.true_state[0] + np.random.normal(0, pf_system.sigma_n)
            measurement_v = state.true_state[1] + np.random.normal(0, pf_system.sigma_n)
            
            # Update particle filter
            pf_system.predict_particles(u)
            pf_system.update_particles(measurement_y, measurement_v)
            
            # Update histories
            state.time_steps.append(frame)
            state.true_position_history.append(state.true_state[0])
            state.true_velocity_history.append(state.true_state[1])
            estimate = pf_system.get_state_estimate()
            state.est_position_history.append(estimate[0])
            state.est_velocity_history.append(estimate[1])
        
        # Update phase space plot
        scatter.set_offsets(pf_system.particles)
        scatter.set_array(pf_system.weights)
        true_point.set_data([state.true_state[0]], [state.true_state[1]])
        estimate = pf_system.get_state_estimate()
        estimate_point.set_data([estimate[0]], [estimate[1]])
        
        # Update history plots
        if len(state.control_history) > 0:
            time_steps_control = state.time_steps[1:len(state.control_history) + 1]
            control_line.set_data(time_steps_control, state.control_history)
            
        true_pos_line.set_data(state.time_steps, state.true_position_history)
        true_vel_line.set_data(state.time_steps, state.true_velocity_history)
        est_pos_line.set_data(state.time_steps, state.est_position_history)
        est_vel_line.set_data(state.time_steps, state.est_velocity_history)
        
        # Add frame number to title
        fig.suptitle(f"{title}\nFrame: {frame}/{n_steps-1}", fontsize=12)
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_steps, interval=100)
    
    # Save as GIF
    filename = f"particle_filter_{init_type}_{control_type}.gif"
    writer = PillowWriter(fps=10)
    print(f"Saving animation to {filename}...")
    anim.save(filename, writer=writer)
    
    plt.close()
    print(f"Saved animation to {filename}")

def main():
    np.random.seed(42)
    n_steps = 41
    
    # Run each configuration separately and save as GIF
    configurations = [
        ('known', 'state_mean', 'Known Initial State, State-Average Control'),
        ('known', 'control_mean', 'Known Initial State, Control-Average'),
        ('unknown', 'state_mean', 'Unknown Initial State, State-Average Control'),
        ('unknown', 'control_mean', 'Unknown Initial State, Control-Average')
    ]
    
    for init_type, control_type, title in configurations:
        print(f"\nProcessing: {title}")
        create_and_save_animation(init_type, control_type, title, n_steps)

if __name__ == "__main__":
    main()