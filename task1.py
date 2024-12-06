import numpy as np
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
        self.P = np.diag([
            1.0, 1.0, 1.0,  # Position uncertainty (m²)
            10.0, 10.0, 10.0,     # Velocity uncertainty (m²/s²)
            1e-6, 1e-8           # Clock bias and drift uncertainty
        ])
        
        # Earth's gravitational constant (m³/s²)
        self.mu = 3.986004418e14
        
        # Process noise parameters
        self.q_pos = 1e2  # Position process noise
        self.q_vel = 1e3  # Velocity process noise
        self.q_clock = 1  # Clock process noise
        
        # Measurement noise
        self.R = np.diag([
            1.0, 1.0, 1.0,  # Position measurement noise (m²)
            1e-8            # Clock measurement noise
        ])
    
    def state_transition(self, state, dt):
        """
        Nonlinear state transition using orbital dynamics
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
        Predict step using orbital dynamics
        """
        # Predict state
        self.x = self.state_transition(self.x, dt)
        
        # Calculate Jacobian
        F = self.calculate_jacobian(self.x, dt)
        
        # Process noise
        Q = np.zeros((self.state_dim, self.state_dim))
        Q[0:3, 0:3] = np.eye(3) * self.q_pos * dt**3 / 3
        Q[3:6, 3:6] = np.eye(3) * self.q_vel * dt
        Q[6:8, 6:8] = np.array([[self.q_clock * dt, 0],
                               [0, self.q_clock * dt**2]])
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
    
    def update(self, measurement):
        """
        Update step
        """
        # Measurement matrix
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

def filter_extreme_data(df):
    """
    Filter out data points beyond ±1e9
    """
    filtered_df = df.copy()
    threshold = 1e9
    filtered_df = filtered_df[abs(filtered_df['position']) < threshold]
    return filtered_df

def process_and_filter_data(df):
    """
    Process and filter GPS data
    """
    # Filter extreme values
    filtered_df = filter_extreme_data(df)
    
    # Initialize EKF with first measurement
    first_group = filtered_df[filtered_df['time'] == filtered_df['time'].iloc[0]]
    initial_state = np.zeros(8)
    
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
        initial_state[6] = row['clock']
        initial_state[7] = row['dclock']
    
    # Initialize EKF
    ekf = ExtendedKalmanFilter(initial_state)
    
    # Process measurements
    results = []
    prev_time = None
    
    for time, group in filtered_df.groupby('time'):
        current_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        
        if prev_time is None:
            dt = 30
        else:
            dt = (current_time - prev_time).total_seconds()
        
        # Predict
        ekf.predict(dt)
        
        # Prepare measurement
        measurement = np.zeros(4)
        for _, row in group.iterrows():
            if row['ECEF'] == 'x':
                measurement[0] = row['position']
            elif row['ECEF'] == 'y':
                measurement[1] = row['position']
            elif row['ECEF'] == 'z':
                measurement[2] = row['position']
            measurement[3] = row['clock']
        
        # Update
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
    
    return pd.DataFrame(results)

def plot_results(raw_df, filtered_results):
    """
    Plot raw and filtered results
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot only filtered raw data
    filtered_raw_df = filter_extreme_data(raw_df)
    grouped = filtered_raw_df.groupby('time')
    x_raw, y_raw, z_raw = [], [], []
    for time, group in grouped:
        x = group[group['ECEF'] == 'x']['position'].values
        y = group[group['ECEF'] == 'y']['position'].values
        z = group[group['ECEF'] == 'z']['position'].values
        if len(x) > 0 and len(y) > 0 and len(z) > 0:
            x_raw.append(x[0])
            y_raw.append(y[0])
            z_raw.append(z[0])
    
    ax1.scatter(x_raw, y_raw, z_raw, c='gray', s=10, alpha=0.3, label='Filtered Raw Data')
    ax1.scatter(filtered_results['x_est'], filtered_results['y_est'], 
                filtered_results['z_est'], c='b', s=20, alpha=0.6, 
                label='EKF Estimate')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # Position plots
    times = pd.to_datetime(filtered_results['time'])
    
    # X position
    ax2 = fig.add_subplot(222)
    ax2.scatter(times, filtered_results['x_est'], c='b', s=20, alpha=0.6, 
                label='X Position')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X Position (m)')
    ax2.set_title('X Position vs Time')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Velocity
    ax3 = fig.add_subplot(223)
    ax3.scatter(times, filtered_results['vx_est'], c='r', s=20, alpha=0.6, 
                label='VX')
    ax3.scatter(times, filtered_results['vy_est'], c='g', s=20, alpha=0.6, 
                label='VY')
    ax3.scatter(times, filtered_results['vz_est'], c='b', s=20, alpha=0.6, 
                label='VZ')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Components')
    ax3.legend()
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Clock states
    ax4 = fig.add_subplot(224)
    ax4.scatter(times, filtered_results['clock_bias'], c='r', s=20, alpha=0.6, 
                label='Clock Bias')
    ax4.scatter(times, filtered_results['clock_drift'], c='b', s=20, alpha=0.6, 
                label='Clock Drift')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Clock States')
    ax4.set_title('Clock States vs Time')
    ax4.legend()
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def main():
    # Read data
    df = pd.read_csv('GPS meas.csv')
    
    # Process and filter data
    filtered_results = process_and_filter_data(df)
    
    # Plot results
    fig = plot_results(df, filtered_results)
    
    # Save results
    filtered_results.to_csv('ekf_filtered_results.csv', index=False)
    plt.savefig('ekf_filtered_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()