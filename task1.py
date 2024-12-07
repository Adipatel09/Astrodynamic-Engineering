import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ExtendedKalmanFilter:
    def __init__(self, initial_state=None, dt=1800):
        # State vector: [x, y, z, vx, vy, vz, clock_bias, clock_drift]
        self.state_dim = 6
        self.measurement_dim = 6  # x, y, z, clock measurements
        
        # Initialize state vector
        if initial_state is not None:
            self.x = initial_state
        else:
            self.x = np.zeros(self.state_dim)
        
        p_pos_init = 1e0 * dt**3 / 3
        p_vel_init = 1e0 * dt
        # Initialize state covariance matrix
        self.P = np.diag([
            p_pos_init, p_pos_init, p_pos_init,
            p_vel_init, p_vel_init, p_vel_init
        ])
        
        # Earth's gravitational constant (m³/s²)
        self.mu = 3.986004418e14
        
        # J2 perturbation constant and Earth radius
        self.J2 = 1.08262668e-3
        self.R_earth = 6378137.0  # Earth's equatorial radius in meters
        
        # Process noise parameters
        self.q_pos = dt**3 / 3 * 1e-1
        self.q_vel = dt * 1e-1

        mea_pos_noise = 1e-1
        mea_vel_noise = 1e-1
        
        # Measurement noise
        self.R = np.diag([
            mea_pos_noise, mea_pos_noise, mea_pos_noise,
            mea_vel_noise, mea_vel_noise, mea_vel_noise
        ])

    
    def state_transition(self, state, dt):
        """
        Nonlinear state transition using orbital dynamics with J2 perturbation
        """
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        r3 = r**3
        r5 = r**5
        
        # Basic gravitational acceleration
        ax = -self.mu * x / r3
        ay = -self.mu * y / r3
        az = -self.mu * z / r3
        
        # J2 perturbation acceleration
        factor = 1.5 * self.J2 * self.mu * self.R_earth**2 / r5
        
        # J2 acceleration components
        ax_j2 = factor * x * (5 * z**2 / r**2 - 1)
        ay_j2 = factor * y * (5 * z**2 / r**2 - 1)
        az_j2 = factor * z * (5 * z**2 / r**2 - 3)
        
        # Total acceleration
        ax_total = ax + ax_j2
        ay_total = ay + ay_j2
        az_total = az + az_j2
        
        # New state
        new_state = np.zeros_like(state)
        new_state[0] = x + vx*dt + 0.5*ax_total*dt**2
        new_state[1] = y + vy*dt + 0.5*ay_total*dt**2
        new_state[2] = z + vz*dt + 0.5*az_total*dt**2
        new_state[3] = vx + ax_total*dt
        new_state[4] = vy + ay_total*dt
        new_state[5] = vz + az_total*dt
        
        return new_state
    
    def calculate_jacobian(self, state, dt):
        """
        Calculate the Jacobian matrix F for the state transition including J2 effects
        """
        x, y, z = state[0:3]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        self.prev_r = r
        if r < 1e4:
            r = self.prev_r
        r3 = r**3
        r5 = r**5
        r7 = r**7
        
        # Basic gravitational terms
        dax_dx = -self.mu * (1/r3 - 3*x**2/r5)
        dax_dy = self.mu * 3*x*y/r5
        dax_dz = self.mu * 3*x*z/r5
        
        day_dx = self.mu * 3*x*y/r5
        day_dy = -self.mu * (1/r3 - 3*y**2/r5)
        day_dz = self.mu * 3*y*z/r5
        
        daz_dx = self.mu * 3*x*z/r5
        daz_dy = self.mu * 3*y*z/r5
        daz_dz = -self.mu * (1/r3 - 3*z**2/r5)
        
        # J2 terms
        J2_factor = 1.5 * self.J2 * self.mu * self.R_earth**2
        
        # Partial derivatives of J2 acceleration
        # These are complex expressions - adding only the most significant terms
        dax_dx_j2 = J2_factor * ((5*z**2/r**2 - 1)/r5 - 10*x**2*z**2/r7)
        day_dy_j2 = J2_factor * ((5*z**2/r**2 - 1)/r5 - 10*y**2*z**2/r7)
        daz_dz_j2 = J2_factor * ((15*z**2/r**2 - 3)/r5 - 10*z**3/r7)
        
        # Combine basic gravity and J2 terms
        F = np.zeros((6, 6))
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Position derivatives
        F[3:6, 0:3] = np.array([
            [dax_dx + dax_dx_j2, dax_dy, dax_dz],
            [day_dx, day_dy + day_dy_j2, day_dz],
            [daz_dx, daz_dy, daz_dz + daz_dz_j2]
        ]) * dt
        
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
        
        self.P = F @ self.P @ F.T + Q
    
    def update(self, measurement):
        """
        Update step
        """
        # Measurement matrix
        # H = np.eye((self.measurement_dim, self.state_dim))
        H = np.eye(self.measurement_dim) 
        
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

def filter_velocity_data(df):
    """
    Filter velocity data points beyond 100 m/s for each direction
    """
    filtered_df = df.copy()
    vel_threshold = 100000  # 100 m/s for realistic satellite velocities
    
    # Filter by direction and velocity
    filtered_data = []
    
    for direction in ['x', 'y', 'z']:
        direction_data = filtered_df[filtered_df['ECEF'] == direction]
        # Filter velocities
        valid_velocities = direction_data[abs(direction_data['velocity']) < vel_threshold]
        filtered_data.append(valid_velocities)
    
    # Combine filtered data
    filtered_df = pd.concat(filtered_data)
    filtered_df = filtered_df.sort_values('time')
    
    return filtered_df

def filter_extreme_data(df):
    """
    Combined filtering for plotting:
    1. Filter position data points beyond ±1e9
    2. Filter velocity data points beyond ±100 m/s
    """
    filtered_df = df.copy()
    
    # Position filtering
    pos_threshold = 1e9
    filtered_df = filtered_df[abs(filtered_df['position']) < pos_threshold]
    
    # Velocity filtering
    vel_threshold = 1e9
    filtered_df = filtered_df[abs(filtered_df['velocity']) < vel_threshold]
    
    return filtered_df

def process_and_filter_data(df):
    """
    Process and filter GPS data
    """
    # Filter extreme values
    filtered_df = filter_extreme_data(df)
    
    # Initialize EKF with first measurement
    first_group = filtered_df[filtered_df['time'] == filtered_df['time'].iloc[0]]
    initial_state = np.zeros(6)
    
    for _, row in first_group.iterrows():
        if row['ECEF'] == 'x':
            # convert km to m
            initial_state[0] = row['position'] * 1e3
            # convert dm/s to m/s   
            initial_state[3] = row['velocity'] * 1e-1
        elif row['ECEF'] == 'y':
            initial_state[1] = row['position'] * 1e3
            initial_state[4] = row['velocity'] * 1e-1
        elif row['ECEF'] == 'z':
            initial_state[2] = row['position'] * 1e3
            initial_state[5] = row['velocity'] * 1e-1
    
    # Initialize EKF
    ekf = ExtendedKalmanFilter(initial_state)
    
    # Process measurements
    results = []
    prev_time = None
    
    for time, group in filtered_df.groupby('time'):
        current_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        
        if prev_time is None:
            dt = 1800
        else:
            dt = (current_time - prev_time).total_seconds()
        
        # Predict
        ekf.predict(dt)
        
        # Prepare measurement
        measurement = np.zeros(6)
        for _, row in group.iterrows():
            if row['ECEF'] == 'x':
                measurement[0] = row['position']
                measurement[3] = row['velocity']
            elif row['ECEF'] == 'y':
                measurement[1] = row['position']
                measurement[4] = row['velocity']
            elif row['ECEF'] == 'z':
                measurement[2] = row['position']
                measurement[5] = row['velocity']
        
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
    
    # Plot filtered raw data
    filtered_df = filter_extreme_data(raw_df)
    grouped = filtered_df.groupby('time')
    
    # Create time series
    times_list = []
    x_raw, y_raw, z_raw = [], [], []
    
    for time, group in grouped:
        x = group[group['ECEF'] == 'x']['position'].values
        y = group[group['ECEF'] == 'y']['position'].values
        z = group[group['ECEF'] == 'z']['position'].values
        
        if len(x) > 0 and len(y) > 0 and len(z) > 0:
            times_list.append(pd.to_datetime(time))
            x_raw.append(x[0])
            y_raw.append(y[0])
            z_raw.append(z[0])
    
    # Convert lists to arrays
    raw_times = np.array(times_list)
    x_raw = np.array(x_raw)
    y_raw = np.array(y_raw)
    z_raw = np.array(z_raw)
    
    # Plot 3D trajectory
    ax1.scatter(x_raw, y_raw, z_raw, c='gray', s=10, alpha=0.3, label='Raw Data')
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
    ax2.scatter(raw_times, x_raw, c='gray', s=10, alpha=0.3, label='Raw X')
    ax2.scatter(times, filtered_results['x_est'], c='b', s=20, alpha=0.6, 
                label='X Position')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X Position (m)')
    ax2.set_title('X Position vs Time')
    ax2.legend()
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
    
    # Print velocity statistics
    print("\nEKF Velocity Statistics:")
    print(f"VX range: [{filtered_results['vx_est'].min():.2f}, {filtered_results['vx_est'].max():.2f}] m/s")
    print(f"VY range: [{filtered_results['vy_est'].min():.2f}, {filtered_results['vy_est'].max():.2f}] m/s")
    print(f"VZ range: [{filtered_results['vz_est'].min():.2f}, {filtered_results['vz_est'].max():.2f}] m/s")
    
    # Position error over time
    ax4 = fig.add_subplot(224)
    
    # Ensure the arrays have matching timestamps before calculating error
    filtered_times = pd.to_datetime(filtered_results['time'])
    raw_times_df = pd.DataFrame({'time': raw_times, 'x': x_raw, 'y': y_raw, 'z': z_raw})
    filtered_df = pd.DataFrame({
        'time': filtered_times,
        'x': filtered_results['x_est'],
        'y': filtered_results['y_est'],
        'z': filtered_results['z_est']
    })
    
    # Merge on matching timestamps
    merged_df = pd.merge(raw_times_df, filtered_df, on='time', suffixes=('_raw', '_filtered'))
    
    # Calculate error only for matching timestamps
    position_error = np.sqrt(
        (merged_df['x_filtered'] - merged_df['x_raw'])**2 +
        (merged_df['y_filtered'] - merged_df['y_raw'])**2 +
        (merged_df['z_filtered'] - merged_df['z_raw'])**2
    )
    
    ax4.scatter(merged_df['time'], position_error, c='r', s=20, alpha=0.6, label='Position Error')
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('Position Error vs Time')
    ax4.legend()
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('ekf_filtered_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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