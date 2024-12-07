import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from task1 import ExtendedKalmanFilter, process_and_filter_data

def filter_extreme_data(df):
    """
    Filter out data points beyond ±1e8
    """
    filtered_df = df.copy()
    threshold = 1e4
    filtered_df = filtered_df[abs(filtered_df['position']) < threshold]
    return filtered_df

class ManeuverDetector:
    def __init__(self):
        # Thresholds for maneuver detection
        self.velocity_threshold = 0.001  # m/s (mm/s threshold for detecting velocity changes)
        self.position_threshold = 1.0    # m (position change threshold)
        self.window_size = 5  # Number of measurements for moving average
        self.min_duration = 30  # Minimum duration between maneuvers (seconds)
    
    def detect_maneuvers(self, ekf_results):
        """
        Detect potential maneuvers using EKF filtered results
        """
        maneuvers = []
        
        # Calculate velocity and position magnitudes
        velocities = np.sqrt(
            ekf_results['vx_est']**2 + 
            ekf_results['vy_est']**2 + 
            ekf_results['vz_est']**2
        )
        
        positions = np.sqrt(
            ekf_results['x_est']**2 + 
            ekf_results['y_est']**2 + 
            ekf_results['z_est']**2
        )
        
        # Calculate changes
        velocity_changes = np.abs(np.diff(velocities))
        position_changes = np.abs(np.diff(positions))
        
        # Use moving average to smooth out noise
        window = np.ones(self.window_size) / self.window_size
        smoothed_vel_changes = np.convolve(velocity_changes, window, mode='valid')
        smoothed_pos_changes = np.convolve(position_changes, window, mode='valid')
        
        # Skip first hour of data (initialization period)
        times = pd.to_datetime(ekf_results['time'])
        start_time = times.iloc[0]
        valid_indices = times > (start_time + pd.Timedelta(hours=1))
        
        # Ensure arrays have same length for comparison
        min_length = min(len(smoothed_vel_changes), len(smoothed_pos_changes))
        smoothed_vel_changes = smoothed_vel_changes[:min_length]
        smoothed_pos_changes = smoothed_pos_changes[:min_length]
        
        # Detect maneuvers based on both velocity and position changes
        maneuver_indices = np.where(
            (smoothed_vel_changes > self.velocity_threshold) & 
            (smoothed_pos_changes > self.position_threshold)
        )[0]
        
        # Adjust indices to account for the convolution window
        maneuver_indices = maneuver_indices + self.window_size - 1
        
        maneuver_indices = maneuver_indices[maneuver_indices < len(valid_indices)]
        maneuver_indices = maneuver_indices[valid_indices.iloc[maneuver_indices]]
        
        # Group consecutive indices into maneuver events
        if len(maneuver_indices) > 0:
            maneuver_events = []
            current_event = [maneuver_indices[0]]
            
            for i in range(1, len(maneuver_indices)):
                time_diff = (times.iloc[maneuver_indices[i]] - 
                           times.iloc[maneuver_indices[i-1]]).total_seconds()
                
                if time_diff <= self.min_duration:
                    current_event.append(maneuver_indices[i])
                else:
                    maneuver_events.append(current_event)
                    current_event = [maneuver_indices[i]]
            
            maneuver_events.append(current_event)
            
            # Record maneuver events
            for event in maneuver_events:
                start_idx = event[0]
                end_idx = event[-1]
                
                # Calculate total changes during maneuver
                total_velocity_change = np.sum(velocity_changes[start_idx:end_idx+1])
                total_position_change = np.sum(position_changes[start_idx:end_idx+1])
                
                maneuver = {
                    'start_time': times[start_idx],
                    'end_time': times[end_idx],
                    'velocity_change': total_velocity_change,
                    'position_change': total_position_change,
                    'duration': (times[end_idx] - times[start_idx]).total_seconds(),
                    'start_position': (
                        ekf_results.iloc[start_idx][['x_est', 'y_est', 'z_est']].values
                    ),
                    'start_velocity': (
                        ekf_results.iloc[start_idx][['vx_est', 'vy_est', 'vz_est']].values
                    )
                }
                maneuvers.append(maneuver)
        
        return maneuvers

def plot_maneuvers(ekf_results, maneuvers):
    """
    Plot satellite trajectory and detected maneuvers
    """
    fig = plt.figure(figsize=(15, 12))
    
    # 3D trajectory
    ax1 = fig.add_subplot(211, projection='3d')
    
    # Plot EKF trajectory
    ax1.scatter(ekf_results['x_est'], ekf_results['y_est'], ekf_results['z_est'],
                c='blue', s=10, alpha=0.6, label='EKF Trajectory')
    
    # Highlight maneuvers
    for maneuver in maneuvers:
        pos = maneuver['start_position']
        ax1.scatter(pos[0], pos[1], pos[2], 
                   c='red', s=100, marker='*', label='Maneuver')
        ax1.text(pos[0], pos[1], pos[2],
                f"Maneuver\n{maneuver['start_time'].strftime('%Y-%m-%d %H:%M')}\n"
                f"ΔV: {maneuver['velocity_change']*1000:.1f} mm/s\n"
                f"ΔP: {maneuver['position_change']:.1f} m",
                fontsize=8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Satellite Trajectory with Detected Maneuvers')
    
    # Velocity magnitude plot
    ax2 = fig.add_subplot(212)
    times = pd.to_datetime(ekf_results['time'])
    velocities = np.sqrt(
        ekf_results['vx_est']**2 + 
        ekf_results['vy_est']**2 + 
        ekf_results['vz_est']**2
    )
    
    ax2.plot(times, velocities, 'b-', alpha=0.6, label='Velocity Magnitude')
    
    # Add maneuvers to velocity plot
    for maneuver in maneuvers:
        ax2.axvline(x=maneuver['start_time'], color='r', linestyle='--', alpha=0.5)
        ax2.scatter(maneuver['start_time'], 
                   np.sqrt(np.sum(maneuver['start_velocity']**2)),
                   c='red', s=100, marker='*')
        ax2.annotate(f'ΔV: {maneuver["velocity_change"]*1000:.1f} mm/s\n'
                    f'ΔP: {maneuver["position_change"]:.1f} m',
                    xy=(maneuver['start_time'], 
                        np.sqrt(np.sum(maneuver['start_velocity']**2))),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->'))
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Magnitude with Detected Maneuvers')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Remove duplicate labels
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys())
    
    plt.tight_layout()
    return fig

def main():
    # Read the GPS measurements
    df = pd.read_csv('GPS meas.csv')
    
    # Filter outliers first
    filtered_df = filter_extreme_data(df)
    
    # Process data using EKF from task1
    ekf_results = process_and_filter_data(filtered_df)
    
    # Initialize maneuver detector
    detector = ManeuverDetector()
    
    # Detect maneuvers using EKF results
    maneuvers = detector.detect_maneuvers(ekf_results)
    
    # Print detected maneuvers
    print(f"\nDetected {len(maneuvers)} potential maneuvers:")
    for i, maneuver in enumerate(maneuvers, 1):
        print(f"\nManeuver {i}:")
        print(f"Start time: {maneuver['start_time']}")
        print(f"End time: {maneuver['end_time']}")
        print(f"Velocity change: {maneuver['velocity_change']*1000:.1f} mm/s")
        print(f"Position change: {maneuver['position_change']:.1f} m")
        print(f"Duration: {maneuver['duration']:.1f} seconds")
    
    # Plot results
    fig = plot_maneuvers(ekf_results, maneuvers)
    plt.savefig('maneuver_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 