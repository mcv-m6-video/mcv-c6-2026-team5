from abc import ABC, abstractmethod

class BaseTracker(ABC):
    def __init__(self):
        self.next_track_id = 1
        self.active_tracks = []

    @abstractmethod
    def update(self, detections):
        """
        Updates the tracker with the detections from the current frame.
        
        Args:
            detections (list or np.ndarray): A list of bounding boxes for the current frame
                                             in format [x1, y1, x2, y2].
                                             
        Returns:
            list of dicts: The updated active tracks containing 'id' and 'bbox'.
        """
        pass