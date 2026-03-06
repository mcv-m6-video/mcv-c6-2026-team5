import numpy as np
from scipy.optimize import linear_sum_assignment
from .base_tracker import BaseTracker

class KalmanFilter:
    """
    A simple Kalman Filter for tracking bounding boxes in image space.
    State: [u, v, s, r, u_dot, v_dot, s_dot]
    - (u, v): Center position
    - s: Scale (area)
    - r: Aspect ratio (w/h)
    - *_dot: Velocities
    """
    def __init__(self, bbox, process_noise_scale=1.0, measurement_noise_scale=1.0, initial_cov_scale=10.0):
        # Initialize state mean and covariance
        self.kf_dim = 7
        
        # ---  Initialize state as 7x1 (4 pos + 3 vel) ---
        z = self.convert_bbox_to_z(bbox)
        self.x = np.zeros((self.kf_dim, 1))
        self.x[:4] = z # Set position
        # Velocities (indices 4, 5, 6) remain 0 initially
        
        self.F = np.eye(self.kf_dim) # State transition matrix
        self.H = np.eye(4, self.kf_dim) # Measurement matrix (we measure only first 4)
        
        # P: Covariance matrix (Uncertainty of the current state)
        self.P = np.eye(self.kf_dim) * initial_cov_scale
        self.P[4:, 4:] *= 1000.0 # High uncertainty for initial velocity
        
        # R: Measurement noise (Detector uncertainty)
        self.R = np.eye(4) * measurement_noise_scale
        self.R[2:, 2:] *= 10.0 # Trust position more than size
        
        # Q: Process noise (Motion uncertainty)
        self.Q = np.eye(self.kf_dim) * process_noise_scale
        self.Q[4:, 4:] *= 0.01 # Velocity changes are usually smaller

        # Set motion model (Constant Velocity)
        # x += dx, y += dy, s += ds
        for i in range(4):
            if i != 3: # Aspect ratio is assumed constant
                self.F[i, i+4] = 1.0

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box [x1, y1, x2, y2] and returns state vector [u, v, s, r]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x=None):
        """
        Takes a state vector x [u, v, s, r...] and returns a bounding box [x1, y1, x2, y2]
        """
        if x is None: x = self.x
        x_c, y_c, s, r = x[:4]
        
        # If the filter predicts negative size, we clamp it to a small positive number
        if s <= 0: s = 1e-6
        if r <= 0: r = 1e-6
        
        w = np.sqrt(s * r)
        h = s / w
        return np.array([x_c - w/2., y_c - h/2., x_c + w/2., y_c + h/2.]).reshape((4,))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.convert_x_to_bbox(self.x)

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        z = self.convert_bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x) # Residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman Gain
        
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)


class KalmanTracker(BaseTracker):
    def __init__(self, iou_threshold=0.3, max_age=3, 
                 process_noise_scale=1.0, measurement_noise_scale=1.0):
        """
        Args:
            iou_threshold (float): Minimum IoU to associate a detection with a track.
            max_age (int): Maximum number of frames to keep a track alive without hits.
            process_noise_scale (float): Multiplier for Q (System uncertainty). 
                                         Low = Strict constant velocity. High = Erratic motion allowed.
            measurement_noise_scale (float): Multiplier for R (Measurement uncertainty).
                                             Low = Trust detector. High = Smooth out detector jitter.
        """
        super().__init__()
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        
        # We store these to pass them to new KalmanFilter instances
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        
        self.tracks = []
        self.track_id_count = 1

    def _iou_batch(self, bboxes1, bboxes2):
        """
        Computes IoU matrix between N predicted boxes and M detected boxes.
        """
        iou_matrix = np.zeros((len(bboxes1), len(bboxes2)))
        for i, box1 in enumerate(bboxes1):
            for j, box2 in enumerate(bboxes2):
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                
                iou = inter_area / (box1_area + box2_area - inter_area + 1e-10)
                iou_matrix[i, j] = iou
        return iou_matrix

    def update(self, detections):
        """
        detections: List/Array of [x1, y1, x2, y2]
        """
        # 1. PREDICT: Move all tracks forward using Kalman Filter
        predicted_boxes = []
        for t in self.tracks:
            pred_box = t['kf'].predict()
            t['bbox'] = pred_box 
            t['age'] += 1        
            predicted_boxes.append(pred_box)

        # 2. MATCHING
        matched_indices = []
        unmatched_dets = []
        
        if len(predicted_boxes) > 0 and len(detections) > 0:
            iou_mat = self._iou_batch(predicted_boxes, detections)
            
            # Minimize cost (1 - IoU)
            row_idx, col_idx = linear_sum_assignment(1 - iou_mat)
            
            matched_set = set()
            for r, c in zip(row_idx, col_idx):
                if iou_mat[r, c] >= self.iou_threshold:
                    t = self.tracks[r]
                    t['kf'].update(detections[c]) 
                    
                    # CHANGE: Use the Kalman State (Smoothed) instead of raw detection
                    # This visualizes the benefit of the filter
                    # t['bbox'] = t['kf'].convert_x_to_bbox(t['kf'].x)
                    
                    # Update internal Kalman state (keep this!)
                    t['kf'].update(detections[c]) 

                    # Output the raw, perfectly tight detection box to the evaluator
                    t['bbox'] = detections[c]
                    
                    t['hits'] += 1
                    t['age'] = 0
                    matched_set.add(c)
            
            for d in range(len(detections)):
                if d not in matched_set:
                    unmatched_dets.append(d)
        else:
            unmatched_dets = list(range(len(detections)))

        # 3. NEW TRACKS
        for d_idx in unmatched_dets:
            # Create new KF with the custom noise parameters
            new_kf = KalmanFilter(
                detections[d_idx],
                process_noise_scale=self.process_noise_scale,
                measurement_noise_scale=self.measurement_noise_scale
            )
            self.tracks.append({
                'kf': new_kf,
                'id': self.track_id_count,
                'bbox': detections[d_idx],
                'hits': 1,
                'age': 0
            })
            self.track_id_count += 1

        # 4. REMOVE DEAD TRACKS
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

        # Return results
        ret = []
        for t in self.tracks:
            if t['age'] == 0: 
                ret.append({'id': t['id'], 'bbox': t['bbox']})
            
        return ret