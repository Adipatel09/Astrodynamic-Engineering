import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def filter_extreme_data(df):
    """
    Filter out data points beyond ±1e9
    """
    # Create a copy of the dataframe
    filtered_df = df.copy()
    
    # Filter out positions beyond threshold
    threshold = 1e9
    filtered_df = filtered_df[abs(filtered_df['position']) < threshold]
    
    return filtered_df

def plot_gps_data(df):
    """
    Plot GPS measurements using scatter plots
    """
    # Filter extreme values
    filtered_df = filter_extreme_data(df)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Group data by timestamp
    grouped = filtered_df.groupby('time')
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
    
    # Plot 3D trajectory
    scatter = ax1.scatter(x, y, z, c=range(len(times)), cmap='viridis', 
                         s=20, alpha=0.6, label='GPS Measurements')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D GPS Trajectory (Filtered)')
    
    # Position components over time
    times = np.array(times)
    
    # X position
    ax2 = fig.add_subplot(222)
    ax2.scatter(times, x, c='r', s=20, alpha=0.6, label='X Position')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X Position (m)')
    ax2.set_title('X Position vs Time')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.legend()
    
    # Y position
    ax3 = fig.add_subplot(223)
    ax3.scatter(times, y, c='g', s=20, alpha=0.6, label='Y Position')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_title('Y Position vs Time')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    ax3.legend()
    
    # Z position
    ax4 = fig.add_subplot(224)
    ax4.scatter(times, z, c='b', s=20, alpha=0.6, label='Z Position')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Z Position (m)')
    ax4.set_title('Z Position vs Time')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    return fig, filtered_df

def main():
    # Read the GPS measurements
    df = pd.read_csv('GPS meas.csv')
    
    # Plot results and get filtered data
    fig, filtered_df = plot_gps_data(df)
    
    # Print filtering results
    total_measurements = len(df)
    filtered_measurements = len(filtered_df)
    filtered_out = total_measurements - filtered_measurements
    
    print("\nData Filtering Results:")
    print(f"Total measurements: {total_measurements}")
    print(f"Measurements after filtering: {filtered_measurements}")
    print(f"Points filtered out: {filtered_out}")
    
    # Print basic statistics of filtered data
    print("\nFiltered GPS Data Statistics:")
    grouped = filtered_df.groupby('ECEF')
    for ecef, group in grouped:
        print(f"\n{ecef} Position Statistics:")
        print(f"Mean: {group['position'].mean():.2f} m")
        print(f"Std Dev: {group['position'].std():.2f} m")
        print(f"Min: {group['position'].min():.2f} m")
        print(f"Max: {group['position'].max():.2f} m")
    
    # Save and show plot
    plt.savefig('gps_measurements_filtered.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 