import numpy as np
from scipy.optimize import linear_sum_assignment
from .base_tracker import BaseTracker

class MaxIoUTracker(BaseTracker): # class with max age implementation
    def __init__(self, iou_threshold=0.4, max_age=3):
        """
        Args:
            iou_threshold (float): Minimum IoU to match a detection.
            max_age (int): Maximum frames to keep a track alive without detection.
        """
        super().__init__()
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        
        # We need a more complex track structure now
        # self.tracks is a list of dicts: 
        # {'id': int, 'bbox': [x1,y1,x2,y2], 'age': int, 'hits': int}
        self.tracks = [] 
        self.next_track_id = 1

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections):
        """
        detections: List of boxes [x1, y1, x2, y2]
        """
        # 1. PREDICT: Increment age of all tracks (assume they are lost for now)
        for t in self.tracks:
            t['age'] += 1

        # 2. MATCHING
        matched_indices = []
        unmatched_dets = []
        
        if len(self.tracks) > 0 and len(detections) > 0:
            # Build IoU Matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for t, track in enumerate(self.tracks):
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = self._compute_iou(track['bbox'], det)
            
            # Hungarian Algorithm (Maximize IoU -> Minimize 1-IoU)
            # row_indices correspond to tracks, col_indices correspond to detections
            cost_matrix = 1.0 - iou_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_set = set()
            for r, c in zip(row_indices, col_indices):
                if iou_matrix[r, c] >= self.iou_threshold:
                    # Valid Match
                    self.tracks[r]['bbox'] = detections[c] # Update position
                    self.tracks[r]['age'] = 0              # Reset age (found!)
                    self.tracks[r]['hits'] += 1
                    matched_set.add(c)
                else:
                    # Match found but IoU too low -> Treat as unmatched
                    pass
            
            # Identify unmatched detections
            for d in range(len(detections)):
                if d not in matched_set:
                    unmatched_dets.append(d)
        else:
            unmatched_dets = list(range(len(detections)))

        # 3. CREATE NEW TRACKS
        for d_idx in unmatched_dets:
            self.tracks.append({
                'id': self.next_track_id,
                'bbox': detections[d_idx],
                'hits': 1,
                'age': 0
            })
            self.next_track_id += 1

        # 4. DELETE DEAD TRACKS
        # Keep track if it was just found (age=0) OR it's still within max_age
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

        # Return format: List of objects with 'id' and 'bbox'
        return [{'id': t['id'], 'bbox': t['bbox']} for t in self.tracks]