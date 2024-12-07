from sgp4.api import Satrec
from sgp4.api import jday
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def parse_tle_file(filename):
    """Parse TLE file and return the latest TLE data"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Get the latest TLE (last two lines)
    line1 = lines[-2].strip()
    line2 = lines[-1].strip()
    
    return line1, line2

def propagate_orbit(line1, line2, start_time, end_time, step_seconds=1800):
    """
    Propagate orbit using SGP4 and return ECEF positions and velocities
    
    Args:
        line1, line2: TLE lines
        start_time: datetime object for start of propagation
        end_time: datetime object for end of propagation
        step_seconds: time step in seconds (default 30 minutes)
    """
    # Create satellite object
    satellite = Satrec.twoline2rv(line1, line2)
    
    # Create time array
    times = []
    current_time = start_time
    while current_time <= end_time:
        times.append(current_time)
        current_time += timedelta(seconds=step_seconds)
    
    # Lists to store results
    positions = []
    velocities = []
    timestamps = []
    
    for t in times:
        # Convert time to Julian date
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        
        # Get position and velocity
        error, position, velocity = satellite.sgp4(jd, fr)
        
        if error == 0:  # Only append if no error
            positions.append(position)
            velocities.append(velocity)
            timestamps.append(t)
    
    return np.array(positions), np.array(velocities), timestamps

def create_measurement_df(positions, velocities, timestamps):
    """Create a DataFrame in the same format as GPS meas.csv"""
    data = []
    
    for i in range(len(timestamps)):
        # Convert km to m for positions and km/s to dm/s for velocities
        pos = positions[i] * 1000  # km to m
        vel = velocities[i] * 10000  # km/s to dm/s
        
        # Add x, y, z coordinates
        for j, coord in enumerate(['x', 'y', 'z']):
            data.append({
                'time': timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
                'sv': 'L47',
                'ECEF': coord,
                'position': pos[j],
                'clock': 0.0,
                'velocity': vel[j],
                'dclock': 999999.999999
            })
    
    return pd.DataFrame(data)

def main():
    # Parse TLE file
    line1, line2 = parse_tle_file('sat000039452.txt')
    
    # Set time range (June 1-3, 2024)
    start_time = datetime(2024, 6, 1, 0, 0, 0)
    end_time = datetime(2024, 6, 3, 23, 59, 59)
    
    # Propagate orbit
    positions, velocities, timestamps = propagate_orbit(line1, line2, start_time, end_time)
    
    # Create DataFrame
    df = create_measurement_df(positions, velocities, timestamps)
    
    # Save to CSV
    df.to_csv('tle_propagated_data.csv', index=False)
    
    print(f"Generated {len(timestamps)} points from {start_time} to {end_time}")
    print("Data saved to tle_propagated_data.csv")

if __name__ == "__main__":
    main() 