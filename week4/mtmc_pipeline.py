import os
import pickle
import torch
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image

from ultralytics import YOLO

from tqdm import tqdm

class MTMCPipeline:
    def __init__(self, detector, tracker, vit_extractor, vit_transforms, conf_thresh=0.5, device='cuda'):
        self.detector = detector
        self.tracker = tracker
        self.vit_extractor = vit_extractor
        self.vit_transforms = vit_transforms
        self.conf_thresh = conf_thresh
        self.device = device
        self.roi_masks = {}

    def load_roi(self, cam_id, roi_path):
        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        _, roi_binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
        self.roi_masks[cam_id] = roi_binary

    def is_in_roi(self, bbox, cam_id):
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2) // 2, y2 
        roi = self.roi_masks[cam_id]
        cy, cx = min(cy, roi.shape[0] - 1), min(cx, roi.shape[1] - 1)
        return roi[cy, cx] == 255

    def process_single_camera(self, cam_id, dataset, cached_bboxes=None):
        tracklet_data = {}
        all_gt = []    # NEW: Store Ground Truth
        all_preds = [] # NEW: Store Predictions
        
        # Iterate directly through the dataset
        with torch.no_grad():
            for frame_idx in tqdm(range(len(dataset))):
                # if frame_idx==400:break
                # Unpack target to get Ground Truth
                img_tensor, target, _ = dataset[frame_idx]
                img_tensor = img_tensor.to(self.device)
                
                # --- NEW: Process Ground Truth for Metrics ---
                gt_boxes = target.get('boxes', [])
                if 'track_id' in target and len(target['track_id']) > 0:
                     gt_ids = target['track_id'].cpu().numpy()
                else:
                     gt_ids = np.arange(len(gt_boxes))

                frame_gt = []
                for i, box in enumerate(gt_boxes):
                    frame_gt.append({
                        'bbox': box.cpu().numpy(),
                        'id': int(gt_ids[i])       
                    })
                all_gt.append(frame_gt)
                # ---------------------------------------------
                
                # Convert tensor back to numpy/OpenCV format for cropping
                img_rgb = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                if cached_bboxes is not None:
                    bboxes = cached_bboxes[frame_idx]
                else:
                    # 1. Detector inference
                    if type(self.detector) == YOLO:
                        outputs = self.detector(img_bgr, conf=self.conf_thresh, iou=0.45,               
                                                imgsz=1280, device=self.device, verbose=False)[0]
                        raw_bboxes = outputs.boxes.xyxy.detach().cpu().numpy()
                    else:
                        outputs = self.detector(img_tensor.unsqueeze(0))[0]
                        mask = outputs['scores'] >= self.conf_thresh
                        raw_bboxes = outputs['boxes'][mask].detach().cpu().numpy()
                    
                    # Filter by ROI
                    bboxes = [b for b in raw_bboxes if self.is_in_roi(b, cam_id)]
                    bboxes = np.array(bboxes) if len(bboxes) > 0 else np.empty((0, 4))
                
                # 2. Tracker update (expects tensor and boxes)
                tracked_objects = self.tracker.update(img_tensor, bboxes)
                
                # --- NEW: Process Predictions for Metrics ---
                frame_preds = []
                for obj in tracked_objects:
                    frame_preds.append({
                        'bbox': obj['bbox'], 
                        'id': int(obj['id']) 
                    })
                all_preds.append(frame_preds)
                # --------------------------------------------
                
                for obj in tracked_objects:
                    t_id, bbox = obj['id'], obj['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Ensure valid crop boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
                    
                    crop = img_rgb[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    # Apply ViT transforms
                    crop_pil = Image.fromarray(crop).convert("RGB")
                    crop_tensor = self.vit_transforms(crop_pil).unsqueeze(0).to(self.device)
                    
                    emb = self.vit_extractor(crop_tensor) 
                    norm = emb["embeddings_norm"][0].detach().cpu()
                    # print(norm)
                    
                    # Store embeddings and timestamps
                    if t_id not in tracklet_data:
                        tracklet_data[t_id] = {
                            'embeddings': [], 'start_frame': frame_idx, 'end_frame': frame_idx, 'bboxes': {}
                        }
                    
                    tracklet_data[t_id]['embeddings'].append(norm)
                    tracklet_data[t_id]['end_frame'] = frame_idx
                    tracklet_data[t_id]['bboxes'][frame_idx] = [x1, y1, x2, y2]
                
        # 3. Aggregate tracklet embeddings
        aggregated_tracklets = {}
        for t_id, data in tracklet_data.items():
            stacked_embs = torch.stack(data['embeddings'])
            mean_emb = torch.mean(stacked_embs, dim=0)
            aggregated_tracklets[t_id] = {
                'embedding': torch.nn.functional.normalize(mean_emb, p=2, dim=0).numpy(),
                'start_frame': data['start_frame'],
                'end_frame': data['end_frame'],
                'bboxes': data['bboxes']
            }
            
        # Return the metrics data alongside the tracklets
        return aggregated_tracklets, all_gt, all_preds
    
    
    
    


    
    
    def cross_camera_matching(self, cam1, cam2, cam1_tracklets, cam2_tracklets, start_times, threshold=0.3):
        ids_1 = list(cam1_tracklets.keys())
        ids_2 = list(cam2_tracklets.keys())
        
        if not ids_1 or not ids_2:
            return []

        embs_1 = np.array([cam1_tracklets[i]['embedding'] for i in ids_1])
        embs_2 = np.array([cam2_tracklets[i]['embedding'] for i in ids_2])
        
        dist_matrix = cdist(embs_1, embs_2, metric='cosine')
        
        # --- ABSOLUTE TIME FILTER ---
        # Allow a generous time gap (e.g., 300 seconds / 5 minutes) 
        # for a car to traverse the intersection or wait at a red light.
        # MAX_TIME_GAP_SECONDS = 300.0 
        PENALTY_NUM = 1e6
        
        PENALTY_NUM = 1e6
        min_gap, max_gap = get_time_bounds(cam1, cam2)
        
        for i, id1 in enumerate(ids_1):
            for j, id2 in enumerate(ids_2):
                # 1. Get middle frames
                t1_mid_frame = (cam1_tracklets[id1]['start_frame'] + cam1_tracklets[id1]['end_frame']) / 2
                t2_mid_frame = (cam2_tracklets[id2]['start_frame'] + cam2_tracklets[id2]['end_frame']) / 2
                
                # 2. Convert to absolute global time in seconds
                t1_sec = get_absolute_time(cam1, t1_mid_frame, start_times)
                t2_sec = get_absolute_time(cam2, t2_mid_frame, start_times)
                # 3. Calculate time gap
                time_gap = abs(t1_sec - t2_sec)
                
                # 4. Enforce Physics
                if time_gap < min_gap or time_gap > max_gap:
                    dist_matrix[i, j] = PENALTY_NUM
                    
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        global_matches = []
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < threshold:
                global_matches.append((ids_1[r], ids_2[c]))
                
        return global_matches

    def save_features(self, data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

def load_camera_timestamps(timestamp_file):
    """
    Reads the cam_timestamp/<subset>.txt file.
    Expects format: 'c001 0'
    """
    start_times = {}
    with open(timestamp_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                cam_id = parts[0]
                timestamp = float(parts[1])
                start_times[cam_id] = timestamp
                
    return start_times

def get_absolute_time(cam_id, frame_idx, start_times):
    """
    Converts a local frame index to absolute global seconds.
    Accounts for c015 running at 8 FPS instead of 10 FPS.
    """
    # The frame rate is 10 FPS for all videos except c015, which is 8 FPS.
    fps = 8.0 if cam_id == "c015" else 10.0 
    
    start_time = start_times[cam_id]
    return start_time + (frame_idx / fps)


def get_time_bounds(cam1, cam2):
    """
    Defines the allowed absolute time gap (in seconds) between two cameras.
    Format: (min_gap, max_gap)
    """
    # S03 (Cameras 10-15) point OUTWARD from the intersection.
    # It takes at least ~2-3 seconds to cross the blind spot in the center.
    # It takes a maximum of ~120 seconds if they get caught at a red light.
    
    # We strip the 'c0' and just sort the integers to make mapping easy
    c1, c2 = min(cam1, cam2), max(cam1, cam2)
    
    s03_pairs = [
        ('c010', 'c011'), ('c010', 'c012'), ('c010', 'c013'), ('c010', 'c014'), ('c010', 'c015'),
        ('c011', 'c012'), ('c011', 'c013'), ('c011', 'c014'), ('c011', 'c015'),
        ('c012', 'c013'), ('c012', 'c014'), ('c012', 'c015'),
        ('c013', 'c014'), ('c013', 'c015'),
        ('c014', 'c015')
    ]
    
    if (c1, c2) in s03_pairs:
        return (2.0, 120.0) # MUST be at least 2 seconds apart, max 2 minutes
        
    return (0, 90) # Fallback