# Satellite State Estimation and Maneuver Detection

This project implements a solution for satellite state estimation, maneuver detection, and state propagation using GNSS data.

## 1. GNSS Data Assimilation

### Filter Selection
We chose the Extended Kalman Filter (EKF) for this task because:
- It handles nonlinear orbital dynamics
- Provides optimal state estimation for near-Gaussian noise
- Efficiently processes sequential measurements
- Well-suited for real-time satellite tracking

### State Space Model
State vector (8 components):
- Position (x, y, z) in ECEF coordinates
- Velocity (vx, vy, vz)
- Clock bias and drift

### Filter Parameters
1. Initial State Covariance (P):
   - Position uncertainty: 1000 m²
   - Velocity uncertainty: 1000 (m/s)²
   - Clock states uncertainty: 1000 units²
   - Reasoning: Large initial uncertainty to avoid filter overconfidence

2. Process Noise (Q):
   - Position: 1.0 m²/s³
   - Velocity: 1.0 m²/s
   - Clock: 0.1 units²/s
   - Reasoning: Based on expected satellite dynamics and clock stability

3. Measurement Noise (R):
   - Position: 10.0 m²
   - Clock: 10.0 units²
   - Reasoning: Based on typical GNSS measurement accuracy

## 2. Maneuver Detection

### Detection Algorithm
We implement maneuver detection by:
1. Monitoring innovation sequence
2. Calculating velocity changes
3. Using statistical thresholds for detection

### Detection Criteria
- Innovation magnitude exceeds 3σ threshold
- Sudden velocity changes above nominal orbital dynamics
- Multiple consecutive detections required for confirmation

## 3. State Propagation and TLE Comparison

### Propagation Method
- Uses full orbital dynamics model
- Includes J2 perturbation effects
- Propagates both state and covariance

### TLE Comparison
- Compares propagated states with TLE-derived positions
- Analyzes position and velocity differences
- Quantifies prediction accuracy degradation

## Implementation Details

[Technical implementation details remain the same...]