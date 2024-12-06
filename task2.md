# Task 2: Maneuver Detection Algorithm Explanation

## Overview
The maneuver detection algorithm implemented in task2.py identifies potential satellite maneuvers by analyzing acceleration patterns in the trajectory data. The algorithm uses a combination of filtering and acceleration-based detection methods.

## Detection Logic

### 1. Data Preprocessing
- Filter out extreme position values (beyond ±1e9 meters) to remove unrealistic measurements
- Group measurements by timestamp to get complete state (x, y, z positions) at each time
- Convert timestamps to time differences in seconds for acceleration calculations

### 2. Acceleration Calculation
- Use central difference method to calculate accelerations:
  1. First derivative (velocity): v = Δposition/Δt
  2. Second derivative (acceleration): a = Δvelocity/Δt
- Calculate total acceleration magnitude from all three axes:
  ```python
  total_accel = √(ax² + ay² + az²)
  ```

### 3. Maneuver Detection Criteria
- Primary threshold: acceleration > 0.1 m/s² (self.accel_threshold)
  - This threshold was chosen as it's significantly above typical orbital perturbations
  - Natural orbital motion typically has smaller accelerations
- Group consecutive detections:
  - Maneuvers often span multiple measurements
  - Points within 2 time steps are considered part of the same maneuver
  - This helps avoid counting a single maneuver multiple times

### 4. Maneuver Characterization
For each detected maneuver, we record:
- Start time: When acceleration exceeds threshold
- End time: When acceleration returns below threshold
- Maximum acceleration: Peak acceleration during the maneuver
- Duration: Time between start and end

## Visualization
The algorithm provides two visualization perspectives:
1. 3D Trajectory Plot:
   - Shows complete satellite path
   - Marks maneuver points with red stars
   - Includes timestamp labels for maneuvers

2. Timeline Plot:
   - Shows temporal distribution of measurements
   - Marks maneuvers with timing and acceleration magnitude
   - Provides clear view of maneuver sequence

## Detection Parameters
Key parameters that affect detection:
- Acceleration threshold (0.1 m/s²)
- Position value filter (±1e9 meters)
- Time window for grouping (2 steps)
- Velocity change threshold (5.0 m/s)

## Limitations and Considerations
1. The algorithm assumes:
   - Maneuvers cause detectable acceleration changes
   - Valid measurements have reasonable position values
   - Maneuvers have finite duration

2. Potential false positives can come from:
   - Measurement noise
   - Orbital perturbations
   - Data gaps

3. Potential false negatives can occur with:
   - Very gradual maneuvers
   - Maneuvers during data gaps
   - Maneuvers below threshold

## Future Improvements
The algorithm could be enhanced by:
- Adding adaptive thresholds based on measurement noise
- Incorporating velocity change criteria
- Including orbital dynamics models
- Adding statistical confidence measures 