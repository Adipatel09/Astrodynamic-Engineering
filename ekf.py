import numpy as np
from scipy.linalg import block_diag
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ExtendedKalmanFilter:
    def __init__(self, initial_state=None):
        # State vector: [x, y, z, vx, vy, vz, clock_bias, clock_drift]
        self.state_dim = 8
        self.measurement_dim = 4  # x, y, z, clock measurements
        
        # Initialize state vector
        if initial_state is not None:
            self.x = initial_state
        else:
            self.x = np.zeros(self.state_dim)
        
        # Initialize state covariance matrix
        self.P = np.eye(self.state_dim) * 1000  # Large initial uncertainty
        
        # Earth's gravitational constant (m³/s²)
        self.mu = 3.986004418e14
        
        # Process noise parameters
        self.q_pos = 1.0  # Increased position process noise
        self.q_vel = 1.0  # Increased velocity process noise
        self.q_clock = 0.1  # Clock process noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * 10.0  # Increased measurement noise
    
    def state_transition(self, state, dt):
        """
        Nonlinear state transition function
        """
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        cb, cd = state[6:8]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        r3 = r**3
        
        # Acceleration due to gravity
        ax = -self.mu * x / r3
        ay = -self.mu * y / r3
        az = -self.mu * z / r3
        
        # New state
        new_state = np.zeros_like(state)
        new_state[0] = x + vx*dt + 0.5*ax*dt**2
        new_state[1] = y + vy*dt + 0.5*ay*dt**2
        new_state[2] = z + vz*dt + 0.5*az*dt**2
        new_state[3] = vx + ax*dt
        new_state[4] = vy + ay*dt
        new_state[5] = vz + az*dt
        new_state[6] = cb + cd*dt
        new_state[7] = cd
        
        return new_state
    
    def calculate_jacobian(self, state, dt):
        """
        Calculate Jacobian matrix of state transition function
        """
        x, y, z = state[0:3]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        r3 = r**3
        r5 = r**5
        
        # Initialize Jacobian
        F = np.zeros((self.state_dim, self.state_dim))
        
        # Position derivatives
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Velocity derivatives
        F[3:6, 0:3] = np.array([
            [-self.mu/r3 + 3*self.mu*x**2/r5, 3*self.mu*x*y/r5, 3*self.mu*x*z/r5],
            [3*self.mu*x*y/r5, -self.mu/r3 + 3*self.mu*y**2/r5, 3*self.mu*y*z/r5],
            [3*self.mu*x*z/r5, 3*self.mu*y*z/r5, -self.mu/r3 + 3*self.mu*z**2/r5]
        ]) * dt
        F[3:6, 3:6] = np.eye(3)
        
        # Clock states
        F[6, 7] = dt
        F[6:8, 6:8] += np.eye(2)
        
        return F
    
    def predict(self, dt):
        """
        Predict step of EKF using orbital dynamics
        """
        # Predict state using nonlinear model
        self.x = self.state_transition(self.x, dt)
        
        # Calculate Jacobian
        F = self.calculate_jacobian(self.x, dt)
        
        # Process noise covariance
        Q = np.zeros((self.state_dim, self.state_dim))
        Q[0:3, 0:3] = np.eye(3) * self.q_pos * dt**2
        Q[3:6, 3:6] = np.eye(3) * self.q_vel * dt
        Q[6:8, 6:8] = np.array([[self.q_clock * dt**3/3, self.q_clock * dt**2/2],
                               [self.q_clock * dt**2/2, self.q_clock * dt]])
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
    
    def update(self, measurement):
        """
        Update step of EKF
        """
        # Measurement matrix (linear for this case)
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Position measurements
        H[3, 6] = 1  # Clock bias measurement
        
        # Innovation
        y = measurement - H @ self.x
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

def process_gps_data(filename):
    """
    Process GPS measurement data
    """
    # Read CSV file
    df = pd.read_csv(filename)
    
    # Get first measurement to initialize EKF
    first_group = df[df['time'] == df['time'].iloc[0]]
    initial_state = np.zeros(8)
    
    # Initialize position from first measurement
    for _, row in first_group.iterrows():
        if row['ECEF'] == 'x':
            initial_state[0] = row['position']
            initial_state[3] = row['velocity']
        elif row['ECEF'] == 'y':
            initial_state[1] = row['position']
            initial_state[4] = row['velocity']
        elif row['ECEF'] == 'z':
            initial_state[2] = row['position']
            initial_state[5] = row['velocity']
        initial_state[6] = row['clock']  # Clock bias
        initial_state[7] = row['dclock']  # Clock drift
    
    # Initialize EKF with first measurement
    ekf = ExtendedKalmanFilter(initial_state)
    
    # Initialize results storage
    results = []
    prev_time = None
    
    # Group measurements by timestamp
    grouped = df.groupby('time')
    
    for time, group in grouped:
        # Convert time string to datetime
        current_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        
        # Calculate dt (time step)
        if prev_time is None:
            dt = 30  # Initial assumption of 30 seconds
        else:
            dt = (current_time - prev_time).total_seconds()
        
        # Predict step
        ekf.predict(dt)
        
        # Prepare measurement vector
        measurement = np.zeros(4)
        for _, row in group.iterrows():
            if row['ECEF'] == 'x':
                measurement[0] = row['position']
            elif row['ECEF'] == 'y':
                measurement[1] = row['position']
            elif row['ECEF'] == 'z':
                measurement[2] = row['position']
            measurement[3] = row['clock']
        
        # Update step
        ekf.update(measurement)
        
        # Store results
        results.append({
            'time': time,
            'x_est': ekf.x[0],
            'y_est': ekf.x[1],
            'z_est': ekf.x[2],
            'vx_est': ekf.x[3],
            'vy_est': ekf.x[4],
            'vz_est': ekf.x[5],
            'clock_bias': ekf.x[6],
            'clock_drift': ekf.x[7]
        })
        
        prev_time = current_time
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def plot_results(results_df):
    """
    Plot the EKF results
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(results_df['x_est'], results_df['y_est'], results_df['z_est'])
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    
    # Position plots
    times = pd.to_datetime(results_df['time'])
    
    # X position
    ax2 = fig.add_subplot(222)
    ax2.plot(times, results_df['x_est'])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X Position (m)')
    ax2.set_title('X Position vs Time')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Velocity plots
    ax3 = fig.add_subplot(223)
    ax3.plot(times, results_df['vx_est'], label='VX')
    ax3.plot(times, results_df['vy_est'], label='VY')
    ax3.plot(times, results_df['vz_est'], label='VZ')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Components vs Time')
    ax3.legend()
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Clock states
    ax4 = fig.add_subplot(224)
    ax4.plot(times, results_df['clock_bias'], label='Clock Bias')
    ax4.plot(times, results_df['clock_drift'], label='Clock Drift')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Clock States')
    ax4.set_title('Clock States vs Time')
    ax4.legend()
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Process GPS measurements
    results = process_gps_data('GPS meas.csv')
    
    # Save results to CSV
    results.to_csv('ekf_results.csv', index=False)
    
    # Plot results
    fig = plot_results(results)
    
    # Save plot
    plt.savefig('ekf_results.png', dpi=300, bbox_inches='tight')
    
    print("EKF processing complete. Results saved to 'ekf_results.csv'")
    print("Plots saved to 'ekf_results.png'")
    
    # Show plot (optional)
    plt.show() 