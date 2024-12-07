import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def estimate_measurement_noise(df):
    """
    Estimate measurement noise from GPS data using statistical methods
    """
    # Filter extreme outliers first
    pos_threshold = 1e9
    vel_threshold = 1e9
    
    filtered_df = df[
        (abs(df['position']) < pos_threshold) &
        (abs(df['velocity']) < vel_threshold)
    ]
    
    # Separate by ECEF direction
    noise_stats = {}
    
    for direction in ['x', 'y', 'z']:
        direction_data = filtered_df[filtered_df['ECEF'] == direction]
        
        if not direction_data.empty:
            # Calculate position noise
            pos_std = direction_data['position'].std()
            pos_var = direction_data['position'].var()
            
            # Calculate velocity noise
            vel_std = direction_data['velocity'].std()
            vel_var = direction_data['velocity'].var()
            
            # Store statistics
            noise_stats[direction] = {
                'position_std': pos_std,
                'position_var': pos_var,
                'velocity_std': vel_std,
                'velocity_var': vel_var
            }
    
    # Calculate clock noise
    clock_std = filtered_df['clock'].std()
    clock_var = filtered_df['clock'].var()
    dclock_std = filtered_df['dclock'].std()
    dclock_var = filtered_df['dclock'].var()
    
    noise_stats['clock'] = {
        'bias_std': clock_std,
        'bias_var': clock_var,
        'drift_std': dclock_std,
        'drift_var': dclock_var
    }
    
    return noise_stats

def plot_noise_distribution(df):
    """
    Plot noise distributions for position and velocity measurements
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Position distributions by direction
    pos_ax = axes[0, 0]
    for direction in ['x', 'y', 'z']:
        direction_data = df[df['ECEF'] == direction]
        if not direction_data.empty:
            pos_ax.hist(direction_data['position'], bins=50, alpha=0.5, label=f'{direction.upper()} Position')
    pos_ax.set_title('Position Measurement Distribution')
    pos_ax.set_xlabel('Position (m)')
    pos_ax.set_ylabel('Count')
    pos_ax.legend()
    
    # Velocity distributions by direction
    vel_ax = axes[0, 1]
    for direction in ['x', 'y', 'z']:
        direction_data = df[df['ECEF'] == direction]
        if not direction_data.empty:
            vel_ax.hist(direction_data['velocity'], bins=50, alpha=0.5, label=f'{direction.upper()} Velocity')
    vel_ax.set_title('Velocity Measurement Distribution')
    vel_ax.set_xlabel('Velocity (m/s)')
    vel_ax.set_ylabel('Count')
    vel_ax.legend()
    
    # Clock bias distribution
    clock_ax = axes[1, 0]
    clock_ax.hist(df['clock'], bins=50, alpha=0.5, color='purple')
    clock_ax.set_title('Clock Bias Distribution')
    clock_ax.set_xlabel('Clock Bias')
    clock_ax.set_ylabel('Count')
    
    # Clock drift distribution
    dclock_ax = axes[1, 1]
    dclock_ax.hist(df['dclock'], bins=50, alpha=0.5, color='orange')
    dclock_ax.set_title('Clock Drift Distribution')
    dclock_ax.set_xlabel('Clock Drift')
    dclock_ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('measurement_noise_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Read data
    df = pd.read_csv('GPS meas.csv')
    
    # Estimate noise
    noise_stats = estimate_measurement_noise(df)
    
    # Print noise statistics
    print("\nMeasurement Noise Statistics:")
    print("-" * 30)
    
    for direction in ['x', 'y', 'z']:
        print(f"\n{direction.upper()} Direction:")
        print(f"Position Std Dev: {noise_stats[direction]['position_std']:.2f} m")
        print(f"Position Variance: {noise_stats[direction]['position_var']:.2f} m²")
        print(f"Velocity Std Dev: {noise_stats[direction]['velocity_std']:.2f} m/s")
        print(f"Velocity Variance: {noise_stats[direction]['velocity_var']:.2f} (m/s)²")
    
    print("\nClock:")
    print(f"Bias Std Dev: {noise_stats['clock']['bias_std']:.2e}")
    print(f"Bias Variance: {noise_stats['clock']['bias_var']:.2e}")
    print(f"Drift Std Dev: {noise_stats['clock']['drift_std']:.2e}")
    print(f"Drift Variance: {noise_stats['clock']['drift_var']:.2e}")
    
    # Plot noise distributions
    plot_noise_distribution(df)
    
    # Save noise statistics to file
    with open('noise_statistics.txt', 'w') as f:
        f.write("Measurement Noise Statistics:\n")
        f.write("-" * 30 + "\n")
        
        for direction in ['x', 'y', 'z']:
            f.write(f"\n{direction.upper()} Direction:\n")
            f.write(f"Position Std Dev: {noise_stats[direction]['position_std']:.2f} m\n")
            f.write(f"Position Variance: {noise_stats[direction]['position_var']:.2f} m²\n")
            f.write(f"Velocity Std Dev: {noise_stats[direction]['velocity_std']:.2f} m/s\n")
            f.write(f"Velocity Variance: {noise_stats[direction]['velocity_var']:.2f} (m/s)²\n")
        
        f.write("\nClock:\n")
        f.write(f"Bias Std Dev: {noise_stats['clock']['bias_std']:.2e}\n")
        f.write(f"Bias Variance: {noise_stats['clock']['bias_var']:.2e}\n")
        f.write(f"Drift Std Dev: {noise_stats['clock']['drift_std']:.2e}\n")
        f.write(f"Drift Variance: {noise_stats['clock']['drift_var']:.2e}\n")

if __name__ == "__main__":
    main() 