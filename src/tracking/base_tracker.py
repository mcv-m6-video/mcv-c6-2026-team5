from abc import ABC, abstractmethod

class BaseTracker(ABC):
    """
    Abstract class for Task 2.1 (Overlap) and Task 2.2 (Kalman).
    """
    
    def __init__(self):
        self.tracks = [] # List of active track objects
        self.frame_count = 0

    @abstractmethod
    def update(self, detections):
        """
        Core logic:
        1. Receive detections from the current frame.
        2. Match them to existing tracks (Association).
        3. Create new tracks or kill old ones.
        
        Args:
            detections: List of boxes [[x1, y1, x2, y2, score], ...]
        Returns:
            active_tracks: List of objects with assigned IDs.
        """
        pass
    