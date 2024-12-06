import numpy as np

class ManeuverDetector:
    def __init__(self, innovation_threshold=3.0, velocity_threshold=100.0):
        self.innovation_threshold = innovation_threshold
        self.velocity_threshold = velocity_threshold
        self.previous_velocity = None
        
    def detect_maneuver(self, state, innovation, innovation_covariance):
        """
        Detect potential maneuvers using multiple criteria
        """
        # Current velocity vector
        velocity = state[3:6]
        
        # Check innovation magnitude
        normalized_innovation = np.sqrt(innovation.T @ np.linalg.inv(innovation_covariance) @ innovation)
        innovation_check = normalized_innovation > self.innovation_threshold
        
        # Check velocity change
        velocity_check = False
        if self.previous_velocity is not None:
            velocity_change = np.linalg.norm(velocity - self.previous_velocity)
            velocity_check = velocity_change > self.velocity_threshold
        
        self.previous_velocity = velocity.copy()
        
        return innovation_check or velocity_check 