import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

# --- Your Project Imports ---
from src.data.loader import AICityDataset
from src.detection.fine_tuned import FineTunedDetector
from src.tracking.iou_tracker import MaxIoUTracker
from src.tracking.kalman_tracker import KalmanTracker

# --- Import the Metric Calculator ---
# (Assumes calc_metrics.py is in the same folder)
from src.evaluation import calculate_tracking_metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 1. LOAD MODEL & DATASET
    print("Loading model and dataset...")
    
    # Initialize Detector
    detector = FineTunedDetector().get_model()
    detector.load_state_dict(torch.load(args.model_path, map_location=device))
    detector.to(device).eval()
    
    # Initialize Tracker
    if args.tracker == 'iou':
        tracker = MaxIoUTracker(iou_threshold=args.iou_thresh, max_age=args.max_age)
    else:
        tracker = KalmanTracker(iou_threshold=args.iou_thresh, max_age=args.max_age, process_noise_scale=args.proc_noise, measurement_noise_scale=args.meas_noise)
        
    dataset = AICityDataset(video_path=args.video_path, xml_path=args.xml_path)
    
    # 2. INFERENCE LOOP
    # We will store results in these lists instead of writing to files
    all_gt = []    # List of lists of dicts
    all_preds = [] # List of lists of dicts
    
    print(f"Running inference on {len(dataset)} frames...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            if idx % 3 != 0:
                continue
            # --- A. Process Ground Truth ---
            img_tensor, target = dataset[idx]
            gt_boxes = target.get('boxes', [])
            
            # NOTE: For IDF1 to be valid, GT must have consistent IDs across frames.
            # If your dataset loader doesn't provide IDs (e.g. target['labels']), 
            # we default to enumeration. 
            # WARNING: Enumeration (0,1,2..) breaks ID metrics if order changes!
            if 'track_id' in target and len(target['track_id']) > 0:
                 gt_ids = target['track_id'].cpu().numpy()
            else:
                 # Should not happen with corrected dataset, but safe fallback
                 print(f"Warning: No GT IDs in frame {idx}")
                 gt_ids = np.arange(len(gt_boxes))

            frame_gt = []
            for i, box in enumerate(gt_boxes):
                frame_gt.append({
                    'bbox': box.cpu().numpy(), # Ensure [x1, y1, x2, y2]
                    'id': int(gt_ids[i])       # Ensure Integer
                })
            all_gt.append(frame_gt)

            # --- B. Process Predictions ---
            # 1. Detect
            outputs = detector(img_tensor.unsqueeze(0).to(device))[0]
            
            # 2. Filter by Confidence
            mask = outputs['scores'] >= args.conf_thresh
            det_boxes = outputs['boxes'][mask].cpu().numpy()
            
            # 3. Update Tracker
            tracked_objects = tracker.update(det_boxes)
            
            # 4. Format Predictions
            frame_preds = []
            for obj in tracked_objects:
                frame_preds.append({
                    'bbox': obj['bbox'], # Tracker outputs [x1, y1, x2, y2]
                    'id': int(obj['id']) # Tracker ID
                })
            all_preds.append(frame_preds)

    # 3. CALCULATE METRICS
    print("\nCalculating Tracking Metrics (HOTA, IDF1)...")
    
    try:
        scores = calculate_tracking_metrics(all_gt, all_preds)
        
        print("\n" + "="*30)
        print(f" TRACKER: {args.tracker.upper()}")
        print("="*30)
        print(f" HOTA Score : {scores['HOTA']:.4f}")
        print(f" IDF1 Score : {scores['IDF1']:.4f}")
        print("="*30 + "\n")
        
    except Exception as e:
        print(f"\nError calculating metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=['iou', 'kalman'], default='kalman')
    parser.add_argument("--video_path", default="data/AICity_data/AICity_data/train/S03/c010/vdo.avi")
    parser.add_argument("--xml_path", default="data/gt/ai_challenge_s03_c010-full_annotation.xml")
    parser.add_argument("--model_path", default="models/fine_tuned_rcnn.pth")
    # Tuning parameters
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--max_age", type=int, default=3)
    # only kalman
    parser.add_argument("--proc_noise", type=float, default=1.0)
    parser.add_argument("--meas_noise", type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)