import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class ManeuverDetector:
    def __init__(self):
        # Thresholds for maneuver detection
        self.accel_threshold = 0.1  # m/s² (threshold for detecting significant acceleration changes)
        self.velocity_change_threshold = 5.0  # m/s (threshold for sudden velocity changes)
        self.window_size = 3  # Number of measurements to use for acceleration calculation
    
    def calculate_acceleration(self, positions, times):
        """
        Calculate acceleration using central difference method
        """
        # Convert time differences to seconds
        dt = np.diff(times)
        
        # Calculate velocities
        velocities = np.diff(positions) / dt
        
        # Calculate acceleration
        dt_accel = (dt[:-1] + dt[1:]) / 2
        accelerations = np.diff(velocities) / dt_accel
        
        return accelerations
    
    def filter_extreme_data(self, df):
        """
        Filter out data points beyond ±1e10
        """
        # Create a copy of the dataframe
        filtered_df = df.copy()
        
        # Filter out positions beyond threshold
        threshold = 1e9
        filtered_df = filtered_df[abs(filtered_df['position']) < threshold]
        
        return filtered_df
    
    def detect_maneuvers(self, df):
        """
        Detect potential maneuvers in satellite trajectory
        """
        # First filter out extreme data points
        filtered_df = self.filter_extreme_data(df)
        
        # Convert time strings to datetime objects
        times = pd.to_datetime(filtered_df['time'])
        time_diffs = np.array([(t - times[0]).total_seconds() for t in times])
        
        # Group by timestamp to get complete state at each time
        grouped = filtered_df.groupby('time')
        
        # Initialize arrays for positions and detected maneuvers
        positions_x = []
        positions_y = []
        positions_z = []
        timestamps = []
        maneuvers = []
        
        # Process each timestamp
        for time, group in grouped:
            x = group[group['ECEF'] == 'x']['position'].values
            y = group[group['ECEF'] == 'y']['position'].values
            z = group[group['ECEF'] == 'z']['position'].values
            
            if len(x) > 0 and len(y) > 0 and len(z) > 0:
                positions_x.append(x[0])
                positions_y.append(y[0])
                positions_z.append(z[0])
                timestamps.append(time)
        
        # Convert to numpy arrays
        positions_x = np.array(positions_x)
        positions_y = np.array(positions_y)
        positions_z = np.array(positions_z)
        timestamps = np.array(timestamps)
        
        # Calculate accelerations for each axis
        times_sec = np.array([(pd.to_datetime(t) - pd.to_datetime(timestamps[0])).total_seconds() 
                            for t in timestamps])
        
        # Need at least 3 points to calculate acceleration
        if len(times_sec) >= 3:
            ax = self.calculate_acceleration(positions_x, times_sec)
            ay = self.calculate_acceleration(positions_y, times_sec)
            az = self.calculate_acceleration(positions_z, times_sec)
            
            # Calculate total acceleration magnitude
            total_accel = np.sqrt(ax[:-1]**2 + ay[:-1]**2 + az[:-1]**2)
            
            # Detect potential maneuvers
            maneuver_indices = np.where(total_accel > self.accel_threshold)[0]
            
            # Group consecutive indices into maneuver events
            if len(maneuver_indices) > 0:
                maneuver_events = []
                current_event = [maneuver_indices[0]]
                
                for i in range(1, len(maneuver_indices)):
                    if maneuver_indices[i] - maneuver_indices[i-1] <= 2:  # Consider consecutive or near-consecutive points as same event
                        current_event.append(maneuver_indices[i])
                    else:
                        maneuver_events.append(current_event)
                        current_event = [maneuver_indices[i]]
                
                maneuver_events.append(current_event)
                
                # Record maneuver events
                for event in maneuver_events:
                    start_idx = event[0] + 1  # Adjust index due to acceleration calculation
                    end_idx = event[-1] + 1
                    
                    maneuver = {
                        'start_time': timestamps[start_idx],
                        'end_time': timestamps[end_idx],
                        'max_acceleration': np.max(total_accel[event]),
                        'duration': (pd.to_datetime(timestamps[end_idx]) - 
                                   pd.to_datetime(timestamps[start_idx])).total_seconds()
                    }
                    maneuvers.append(maneuver)
        
        return maneuvers

def plot_maneuvers(df, maneuvers):
    """
    Plot satellite trajectory and detected maneuvers using scatter plots with timestamps
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create two subplots: 3D trajectory and timeline
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)
    
    # Group data by timestamp
    grouped = df.groupby('time')
    x, y, z = [], [], []
    times = []
    
    for time, group in grouped:
        x_val = group[group['ECEF'] == 'x']['position'].values
        y_val = group[group['ECEF'] == 'y']['position'].values
        z_val = group[group['ECEF'] == 'z']['position'].values
        
        if len(x_val) > 0 and len(y_val) > 0 and len(z_val) > 0:
            x.append(x_val[0])
            y.append(y_val[0])
            z.append(z_val[0])
            times.append(pd.to_datetime(time))
    
    # Plot trajectory points in 3D
    scatter = ax1.scatter(x, y, z, c=range(len(times)), cmap='viridis', 
                         s=20, alpha=0.6, label='Satellite Position')
    
    # Highlight maneuvers in 3D plot
    for maneuver in maneuvers:
        start_time = maneuver['start_time']
        end_time = maneuver['end_time']
        
        # Find positions during maneuver
        maneuver_data = grouped.get_group(start_time)
        x_man = maneuver_data[maneuver_data['ECEF'] == 'x']['position'].values[0]
        y_man = maneuver_data[maneuver_data['ECEF'] == 'y']['position'].values[0]
        z_man = maneuver_data[maneuver_data['ECEF'] == 'z']['position'].values[0]
        
        # Plot maneuver point with marker
        ax1.scatter(x_man, y_man, z_man, c='red', s=100, marker='*', 
                   label='Maneuver')
        
        # Add text annotation
        ax1.text(x_man, y_man, z_man, 
                f'Maneuver\n{pd.to_datetime(start_time).strftime("%Y-%m-%d %H:%M")}',
                fontsize=8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Satellite Trajectory with Detected Maneuvers')
    
    # Plot timeline in bottom subplot
    times = np.array(times)
    ax2.scatter(times, np.zeros_like(times), c='blue', s=20, alpha=0.6, 
               label='Measurements')
    
    # Add maneuvers to timeline
    for maneuver in maneuvers:
        maneuver_time = pd.to_datetime(maneuver['start_time'])
        ax2.scatter(maneuver_time, 0, c='red', s=100, marker='*', 
                   label='Maneuver')
        ax2.annotate(f'Maneuver\n{maneuver["max_acceleration"]:.2f} m/s²',
                    xy=(maneuver_time, 0), xytext=(0, 20),
                    textcoords='offset points', ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->'))
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Events')
    ax2.set_title('Maneuver Timeline')
    
    # Format timeline x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Remove duplicate labels
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys())
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys())
    
    plt.tight_layout()
    return fig

def main():
    # Read the GPS measurements
    df = pd.read_csv('GPS meas.csv')
    
    # Initialize maneuver detector
    detector = ManeuverDetector()
    
    # Filter extreme data and detect maneuvers
    filtered_df = detector.filter_extreme_data(df)
    maneuvers = detector.detect_maneuvers(filtered_df)
    
    # Print filtering results
    total_measurements = len(df)
    filtered_measurements = len(filtered_df)
    filtered_out = total_measurements - filtered_measurements
    
    print(f"\nData Filtering Results:")
    print(f"Total measurements: {total_measurements}")
    print(f"Measurements after filtering: {filtered_measurements}")
    print(f"Points filtered out: {filtered_out}")
    
    # Print detected maneuvers
    print(f"\nDetected {len(maneuvers)} potential maneuvers:")
    for i, maneuver in enumerate(maneuvers, 1):
        print(f"\nManeuver {i}:")
        print(f"Start time: {maneuver['start_time']}")
        print(f"End time: {maneuver['end_time']}")
        print(f"Maximum acceleration: {maneuver['max_acceleration']:.2f} m/s²")
        print(f"Duration: {maneuver['duration']:.1f} seconds")
    
    # Plot results
    fig = plot_maneuvers(filtered_df, maneuvers)
    plt.savefig('maneuver_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 