import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from src.tracking.base_tracker import BaseTracker
from src.optical_flow.state_of_art_estimators import compute_neuflow

class NeuFlowTracker(BaseTracker):
    def __init__(self, model, device="cuda", half=False, iou_threshold=0.4, max_age=3, alpha=0.1):
        super().__init__()
        self.model = model
        self.device = device
        self.half = half
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.alpha = alpha
        self.infer_w, self.infer_h = 768, 432
        
        self.tracks = []
        self.next_track_id = 1
        
        # We now cache the ALREADY INTERPOLATED tensor to save 50% of the resizing overhead
        self.prev_img_t = None 

    def _iou_batch(self, bboxes1, bboxes2):
        """Vectorized IoU calculation using NumPy broadcasting. Much faster than nested loops."""
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)))
            
        b1 = np.array(bboxes1)
        b2 = np.array(bboxes2)
        
        lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
        rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        
        return inter / (union + 1e-6)

    def update(self, curr_frame, detections):
        # 1. STANDARDIZE DETECTIONS TO NUMPY ONCE
        if torch.is_tensor(detections):
            detections = detections.cpu().numpy()
        elif isinstance(detections, list):
            detections = np.array(detections)
            
        orig_h, orig_w = curr_frame.shape[1:]
        scale_x = self.infer_w / orig_w
        scale_y = self.infer_h / orig_h

        # 2. INTERPOLATE CURRENT FRAME ONCE
        img2_t = F.interpolate(
            curr_frame.unsqueeze(0).to(self.device), 
            size=(self.infer_h, self.infer_w), 
            mode='bilinear', 
            align_corners=False
        )
        if self.half:
            img2_t = img2_t.half()

        if self.prev_img_t is not None and len(self.tracks) > 0:
            
            # 3. FAST PURE PYTORCH INFERENCE
            with torch.inference_mode():
                flow_predictions = self.model(self.prev_img_t, img2_t)
            
            # 4. MOVE FLOW TO CPU FOR FAST SLICING
            # Extracting patches and calculating medians in NumPy is 10x faster 
            # than doing it sequentially on the GPU with .item() calls
            flow_np = flow_predictions[-1][0].permute(1, 2, 0).cpu().numpy()

            for t in self.tracks:
                x1, y1, x2, y2 = t['bbox']
                
                fx1 = max(0, int(x1 * scale_x))
                fy1 = max(0, int(y1 * scale_y))
                fx2 = min(self.infer_w, int(x2 * scale_x))
                fy2 = min(self.infer_h, int(y2 * scale_y))
                
                h_crop = fy2 - fy1
                w_crop = fx2 - fx1
                
                cy1 = fy1 + int(h_crop * 0.25)
                cy2 = fy2 - int(h_crop * 0.25)
                cx1 = fx1 + int(w_crop * 0.25)
                cx2 = fx2 - int(w_crop * 0.25)
                
                shift_x, shift_y = 0.0, 0.0
                
                if cx2 > cx1 and cy2 > cy1:
                    flow_crop = flow_np[cy1:cy2, cx1:cx2]
                    shift_x = float(np.median(flow_crop[:, :, 0])) / scale_x
                    shift_y = float(np.median(flow_crop[:, :, 1])) / scale_y
                
                t['bbox'] = np.array([x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y])
                t['age'] += 1

        else:
            for t in self.tracks:
                t['age'] += 1
        
        # 5. CACHE THE INTERPOLATED TENSOR
        self.prev_img_t = img2_t 

        matched_set = set()
        unmatched_dets = []

        if len(self.tracks) > 0 and len(detections) > 0:
            # 6. VECTORIZED IoU CALCULATION
            track_boxes = np.array([t['bbox'] for t in self.tracks])
            iou_matrix = self._iou_batch(track_boxes, detections)
            
            cost_matrix = 1.0 - iou_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_indices, col_indices):
                if iou_matrix[r, c] >= self.iou_threshold:
                    pred_box = self.tracks[r]['bbox']
                    det_box = detections[c]
                    
                    self.tracks[r]['bbox'] = self.alpha * pred_box + (1.0 - self.alpha) * det_box
                    self.tracks[r]['age'] = 0
                    self.tracks[r]['hits'] += 1
                    matched_set.add(c)
            
            for d in range(len(detections)):
                if d not in matched_set:
                    unmatched_dets.append(d)
                    
        else:
            unmatched_dets = list(range(len(detections)))

        for d_idx in unmatched_dets:
            self.tracks.append({
                'id': self.next_track_id,
                'bbox': detections[d_idx].copy(),
                'hits': 1,
                'age': 0
            })
            self.next_track_id += 1

        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

        # Convert back to standard lists just before returning
        return [{'id': t['id'], 'bbox': t['bbox'].tolist()} for t in self.tracks if t['age'] == 0]