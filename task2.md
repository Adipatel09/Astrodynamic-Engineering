# Task 2: Maneuver Detection Visualization Explanation

The visualization in task2.py uses colors and markers to help identify satellite positions and maneuvers. Here's what each color and marker represents:

## 3D Trajectory Plot (Top)

### Points
- **Blue dots**: Regular satellite position measurements
  - Size: 20 points
  - Alpha (transparency): 0.6
  - These represent the normal trajectory points

### Maneuver Indicators
- **Red stars (*)**: Detected maneuvers
  - Size: 100 points
  - These larger red markers highlight where maneuvers were detected
  - Each maneuver point includes a text label showing the timestamp

## Timeline Plot (Bottom)

### Points
- **Blue dots**: Regular measurements
  - Size: 20 points
  - Alpha (transparency): 0.6
  - Shows the temporal distribution of measurements

### Maneuver Indicators
- **Red stars (*)**: Maneuver events
  - Size: 100 points
  - Each maneuver has:
    - Yellow annotation box showing acceleration magnitude
    - Arrow pointing to the exact time of maneuver
    - Timestamp in format "YYYY-MM-DD HH:MM"

## Color Significance

1. **Blue**: Used for regular, nominal satellite positions
   - Indicates normal orbital motion
   - Lower alpha value helps reduce visual clutter

2. **Red**: Used to highlight maneuvers
   - High contrast against blue points
   - Star shape makes them easily distinguishable
   - Larger size ensures visibility

3. **Yellow**: Used for annotation boxes
   - High visibility against both light and dark backgrounds
   - Semi-transparent (alpha: 0.5) to not obscure other elements
   - Contains important maneuver information

This color scheme was chosen to:
- Maximize contrast between normal operations and maneuvers
- Ensure visibility of important events
- Maintain readability of annotations
- Reduce visual clutter while preserving information 