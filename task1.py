import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# 1. Data Preprocessing
# ---------------------------

def load_and_preprocess_data(file_path):
    """
    Load the data from a CSV file, remove outliers, and convert units.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - filtered_data (pd.DataFrame): Preprocessed data with one row per timestamp.
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Display initial data information
    print("Initial Data:")
    print(data.head())
    print(f"Total data points: {len(data)}\n")

    # Convert 'time' column to datetime if it's not already
    if not np.issubdtype(data['time'].dtype, np.datetime64):
        data['time'] = pd.to_datetime(data['time'])

    # Pivot the data to have one row per timestamp with x, y, z positions and velocities
    pivot_position = data.pivot_table(index='time', columns='ECEF', values='position')
    pivot_velocity = data.pivot_table(index='time', columns='ECEF', values='velocity')
    pivot_clock = data.pivot_table(index='time', columns='ECEF', values='clock')

    # Flatten the column MultiIndex if present
    pivot_position.columns = [f'position_{col}' for col in pivot_position.columns]
    pivot_velocity.columns = [f'velocity_{col}' for col in pivot_velocity.columns]
    pivot_clock.columns = [f'clock_{col}' for col in pivot_clock.columns]

    # Combine all pivoted data into a single DataFrame
    combined_data = pd.concat([pivot_position, pivot_velocity, pivot_clock], axis=1).reset_index()

    # Remove any rows with missing data
    combined_data.dropna(inplace=True)

    # Define threshold for position components (in meters after conversion)
    position_threshold = 1e9  # meters

    # Convert units
    # Position: km to m
    for axis in ['x', 'y', 'z']:
        combined_data[f'position_{axis}'] = combined_data[f'position_{axis}'] * 1e3

    # Clock: μs to s
    for axis in ['x', 'y', 'z']:
        combined_data[f'clock_{axis}'] = combined_data[f'clock_{axis}'] * 1e-6

    # Velocity: dm/s to m/s
    for axis in ['x', 'y', 'z']:
        combined_data[f'velocity_{axis}'] = combined_data[f'velocity_{axis}'] * 1e-1

    # Remove rows where any position component exceeds the threshold
    condition = (
        (combined_data['position_x'] <= position_threshold) & (combined_data['position_x'] >= -position_threshold) &
        (combined_data['position_y'] <= position_threshold) & (combined_data['position_y'] >= -position_threshold) &
        (combined_data['position_z'] <= position_threshold) & (combined_data['position_z'] >= -position_threshold)
    )

    filtered_data = combined_data[condition].reset_index(drop=True)

    print("After Filtering and Unit Conversion:")
    print(filtered_data.head())
    print(f"Filtered data points: {len(filtered_data)}\n")

    return filtered_data

# ---------------------------
# 2. Extended Kalman Filter (EKF) Implementation
# ---------------------------

class EKF:
    def __init__(self, dt, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov):
        """
        Initialize the Extended Kalman Filter.

        Parameters:
        - dt (float): Time step in seconds.
        - initial_state (np.ndarray): Initial state vector [x, y, z, vx, vy, vz].
        - initial_covariance (np.ndarray): Initial covariance matrix (6x6).
        - process_noise_cov (np.ndarray): Process noise covariance matrix (6x6).
        - measurement_noise_cov (np.ndarray): Measurement noise covariance matrix (3x3).
        """
        self.dt = dt
        self.x = initial_state.reshape((6, 1))  # State vector
        self.P = initial_covariance            # Covariance matrix
        self.Q = process_noise_cov              # Process noise covariance
        self.R = measurement_noise_cov          # Measurement noise covariance

    def predict(self):
        """
        Predict the next state and covariance using the motion model with gravity.
        """
        # Unpack current state
        x, y, z, vx, vy, vz = self.x.flatten()

        # Compute distance from Earth's center
        r = np.sqrt(x**2 + y**2 + z**2)

        # Earth's gravitational parameter
        MU = 3.986004418e14  # m^3/s^2

        # Compute gravitational acceleration
        ax = -MU * x / r**3
        ay = -MU * y / r**3
        az = -MU * z / r**3

        # Predict the next state
        x_pred = x + vx * self.dt + 0.5 * ax * self.dt**2
        y_pred = y + vy * self.dt + 0.5 * ay * self.dt**2
        z_pred = z + vz * self.dt + 0.5 * az * self.dt**2
        vx_pred = vx + ax * self.dt
        vy_pred = vy + ay * self.dt
        vz_pred = vz + az * self.dt

        self.x = np.array([[x_pred],
                           [y_pred],
                           [z_pred],
                           [vx_pred],
                           [vy_pred],
                           [vz_pred]])

        # Compute the Jacobian matrix (F)
        F = np.eye(6)

        # Partial derivatives for position update
        F[0, 3] = self.dt
        F[1, 4] = self.dt
        F[2, 5] = self.dt

        # Partial derivatives for velocity update with respect to position
        d_ax_dx = (3 * MU * x**2) / r**5 - MU / r**3
        d_ax_dy = (3 * MU * x * y) / r**5
        d_ax_dz = (3 * MU * x * z) / r**5

        d_ay_dx = (3 * MU * y * x) / r**5
        d_ay_dy = (3 * MU * y**2) / r**5 - MU / r**3
        d_ay_dz = (3 * MU * y * z) / r**5

        d_az_dx = (3 * MU * z * x) / r**5
        d_az_dy = (3 * MU * z * y) / r**5
        d_az_dz = (3 * MU * z**2) / r**5 - MU / r**3

        # Populate the Jacobian matrix
        F[3, 0] = d_ax_dx * self.dt
        F[3, 1] = d_ax_dy * self.dt
        F[3, 2] = d_ax_dz * self.dt

        F[4, 0] = d_ay_dx * self.dt
        F[4, 1] = d_ay_dy * self.dt
        F[4, 2] = d_ay_dz * self.dt

        F[5, 0] = d_az_dx * self.dt
        F[5, 1] = d_az_dy * self.dt
        F[5, 2] = d_az_dz * self.dt

        # Update the covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Update the state with a new measurement.

        Parameters:
        - z (np.ndarray): Measurement vector [x, y, z].
        """
        # Measurement Matrix (H)
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        # Measurement Prediction
        z_pred = H @ self.x

        # Measurement Residual
        y = z.reshape((3, 1)) - z_pred

        # Residual Covariance
        S = H @ self.P @ H.T + self.R

        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update the state
        self.x = self.x + K @ y

        # Update the covariance
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

# ---------------------------
# 3. Main Execution
# ---------------------------

def main():
    # File path to your CSV data
    file_path = 'GPS meas.csv'  # Replace with your actual file path

    # Load and preprocess the data
    data = load_and_preprocess_data(file_path)

    # Define time step (30 minutes in seconds)
    dt = 30 * 60  # 1800 seconds

    # Initialize the EKF
    # Initial state: first measurement's position and velocity
    initial_position = data.iloc[0][['position_x', 'position_y', 'position_z']].values
    initial_velocity = data.iloc[0][['velocity_x', 'velocity_y', 'velocity_z']].values
    initial_state = np.hstack((initial_position, initial_velocity))  # [x, y, z, vx, vy, vz]

    # Initial covariance matrix (6x6)
    # Assuming high uncertainty in initial position and velocity
    initial_covariance = np.diag([1e3, 1e3, 1e3, 1e2, 1e2, 1e2])

    # Process noise covariance matrix (6x6)
    # Tune these values based on your system's expected process noise
    process_noise_std = np.array([10, 10, 10, 1, 1, 1])  # Example standard deviations
    Q = np.diag(process_noise_std**2)

    # Measurement noise covariance matrix (3x3)
    # Tune these values based on your measurement noise
    measurement_noise_std = np.array([1e1, 1e1, 1e1])  # Example standard deviations
    R = np.diag(measurement_noise_std**2)

    # Instantiate the EKF
    ekf = EKF(dt, initial_state, initial_covariance, Q, R)

    # Lists to store estimated states
    estimated_states = []
    estimated_states.append(ekf.x.flatten())

    # Iterate over each measurement
    for index, row in data.iterrows():
        if index == 0:
            continue  # Skip the first data point (already initialized)

        # Prediction step
        ekf.predict()

        # Measurement vector [x, y, z]
        z = row[['position_x', 'position_y', 'position_z']].values

        # Update step
        ekf.update(z)

        # Store the estimated state
        estimated_states.append(ekf.x.flatten())

    # Convert estimated states to NumPy array for easier handling
    estimated_states = np.array(estimated_states)

    # Extract measured positions and velocities
    measured_positions = data[['position_x', 'position_y', 'position_z']].values
    measured_velocities = data[['velocity_x', 'velocity_y', 'velocity_z']].values

    # Extract estimated positions and velocities
    estimated_positions = estimated_states[:, 0:3]
    estimated_velocities = estimated_states[:, 3:6]

    # Extract time for plotting
    time = data['time'].values  # Ensure 'time' is in a plottable format

    # ---------------------------
    # 4. Plotting
    # ---------------------------

    # 4.1. 3D Trajectory Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(measured_positions[:,0], measured_positions[:,1], measured_positions[:,2], 
            label='Measured', color='blue', marker='o')
    ax.plot(estimated_positions[:,0], estimated_positions[:,1], estimated_positions[:,2], 
            label='EKF Estimated', color='red', linestyle='--', marker='x')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Satellite Trajectory: Measured vs EKF Estimated')
    ax.legend()
    plt.show()

    # 4.2. Position Components Over Time
    plt.figure(figsize=(15, 10))

    # X Position
    plt.subplot(3,1,1)
    plt.plot(time, measured_positions[:,0], label='Measured X', color='blue', marker='o')
    plt.plot(time, estimated_positions[:,0], label='Estimated X', color='red', linestyle='--', marker='x')
    plt.xlabel('Time')
    plt.ylabel('X Position (m)')
    plt.legend()
    plt.grid()

    # Y Position
    plt.subplot(3,1,2)
    plt.plot(time, measured_positions[:,1], label='Measured Y', color='blue', marker='o')
    plt.plot(time, estimated_positions[:,1], label='Estimated Y', color='red', linestyle='--', marker='x')
    plt.xlabel('Time')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid()

    # Z Position
    plt.subplot(3,1,3)
    plt.plot(time, measured_positions[:,2], label='Measured Z', color='blue', marker='o')
    plt.plot(time, estimated_positions[:,2], label='Estimated Z', color='red', linestyle='--', marker='x')
    plt.xlabel('Time')
    plt.ylabel('Z Position (m)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # 4.3. Velocity Components Over Time
    plt.figure(figsize=(15, 10))

    # Vx Velocity
    plt.subplot(3,1,1)
    plt.plot(time, measured_velocities[:,0], label='Measured Vx', color='blue', marker='o')
    plt.plot(time, estimated_velocities[:,0], label='Estimated Vx', color='red', linestyle='--', marker='x')
    plt.xlabel('Time')
    plt.ylabel('Vx (m/s)')
    plt.legend()
    plt.grid()

    # Vy Velocity
    plt.subplot(3,1,2)
    plt.plot(time, measured_velocities[:,1], label='Measured Vy', color='blue', marker='o')
    plt.plot(time, estimated_velocities[:,1], label='Estimated Vy', color='red', linestyle='--', marker='x')
    plt.xlabel('Time')
    plt.ylabel('Vy (m/s)')
    plt.legend()
    plt.grid()

    # Vz Velocity
    plt.subplot(3,1,3)
    plt.plot(time, measured_velocities[:,2], label='Measured Vz', color='blue', marker='o')
    plt.plot(time, estimated_velocities[:,2], label='Estimated Vz', color='red', linestyle='--', marker='x')
    plt.xlabel('Time')
    plt.ylabel('Vz (m/s)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # 4.4. Clock Over Time (Optional)
    plt.figure(figsize=(12, 6))
    # Assuming clock is the same for x, y, z components; plot one of them
    plt.plot(time, data['clock_x'].values, label='Clock X (s)', color='green', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Clock (s)')
    plt.title('Clock Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------------
    # 5. Save Estimated States (Optional)
    # ---------------------------
    # You can save the estimated states to a CSV file for further analysis
    estimated_df = pd.DataFrame(estimated_states, columns=['est_x', 'est_y', 'est_z', 'est_vx', 'est_vy', 'est_vz'])
    combined_df = pd.concat([data, estimated_df], axis=1)
    combined_df.to_csv('estimated_states.csv', index=False)
    print("Estimated states saved to 'estimated_states.csv'.")

if __name__ == "__main__":
    main()
