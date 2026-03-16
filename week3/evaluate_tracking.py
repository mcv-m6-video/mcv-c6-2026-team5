import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment # Added for FP matching

# --- Your Project Imports ---
from src.data.loader import AICityDataset
from src.detection.fine_tuned import FineTunedDetector
from src.tracking.iou_tracker import MaxIoUTracker
from src.tracking.kalman_tracker import KalmanTracker
from src.tracking.optical_tracker import NeuFlowTracker
from src.optical_flow.state_of_art_estimators import initialize_neuflow

# --- Import the Metric Calculator ---
from src.evaluation import calculate_tracking_metrics

def compute_fp_metrics(all_gt, all_preds, iou_thresh=0.5):
    """
    Computes exact False Positives, False Negatives, Precision, and Recall.
    """
    def bb_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    total_fp = 0
    total_tp = 0
    total_gt = 0
    total_preds = 0

    for gt_f, pred_f in zip(all_gt, all_preds):
        gts = [g['bbox'] for g in gt_f]
        preds = [p['bbox'] for p in pred_f]
        
        total_gt += len(gts)
        total_preds += len(preds)

        if len(gts) == 0:
            total_fp += len(preds) # All predictions are FPs if no GT exists
            continue
        if len(preds) == 0:
            continue

        # Build IoU matrix
        iou_matrix = np.zeros((len(gts), len(preds)))
        for i, gt in enumerate(gts):
            for j, p in enumerate(preds):
                iou_matrix[i, j] = bb_iou(gt, p)

        # Solve assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matches = 0
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_thresh:
                matches += 1

        total_tp += matches
        total_fp += (len(preds) - matches) # Leftover predictions are FPs

    total_fn = total_gt - total_tp
    precision = total_tp / total_preds if total_preds > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0

    return total_fp, total_fn, total_tp, precision, recall

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
    elif args.tracker == 'kalman':
        tracker = KalmanTracker(iou_threshold=args.iou_thresh, max_age=args.max_age, process_noise_scale=args.proc_noise, measurement_noise_scale=args.meas_noise)
    else:
        half = False
        model = initialize_neuflow(device=device, half=half)
        tracker = NeuFlowTracker(model=model, device=device, half=half, iou_threshold=args.iou_thresh, max_age=args.max_age)
        
    if args.easy_task.lower() in ['true', '1', 'yes']:
        easy_task = True
        print("Easy task enabled: Occluded and parked cars will be excluded from tracking.")
    else:
        easy_task = False
    dataset = AICityDataset(video_path=args.video_path, xml_path=args.xml_path, easy_task=easy_task)
    
    # 2. INFERENCE LOOP
    all_gt = []    
    all_preds = [] 
    
    print(f"Running inference on {len(dataset)} frames...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            # --- A. Process Ground Truth ---
            img_tensor, target = dataset[idx]
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

            # --- B. Process Predictions ---
            outputs = detector(img_tensor.unsqueeze(0).to(device))[0]
            
            mask = outputs['scores'] >= args.conf_thresh
            det_boxes = outputs['boxes'][mask].cpu().numpy()
            
            if type(tracker) == NeuFlowTracker:
                tracked_objects = tracker.update(img_tensor, det_boxes)
            else:
                tracked_objects = tracker.update(det_boxes)
            
            frame_preds = []
            for obj in tracked_objects:
                frame_preds.append({
                    'bbox': obj['bbox'], 
                    'id': int(obj['id']) 
                })
            all_preds.append(frame_preds)

    # 3. CALCULATE METRICS
    print("\nCalculating Tracking Metrics (HOTA, IDF1)...")
    
    try:
        # 1. Existing HOTA/IDF1 Metrics
        scores = calculate_tracking_metrics(all_gt, all_preds)
        
        # 2. New Exact FP Metrics
        if args.evaluate_FP and args.evaluate_FP.lower() in ['true', '1', 'yes']:
            print("\nCalculating Exact FP/FN Metrics...")
            total_fp, total_fn, total_tp, precision, recall = compute_fp_metrics(all_gt, all_preds)
        else:
            total_fp = total_fn = total_tp = precision = recall = None
            
        
        print("\n" + "="*40)
        print(f" TRACKER: {args.tracker.upper()}")
        print("="*40)
        print(f" HOTA Score      : {scores.get('HOTA', 0.0):.4f}")
        print(f" IDF1 Score      : {scores.get('IDF1', 0.0):.4f}")
        print("-" * 40)
        print(f" Total False Positives (FP) : {total_fp}")
        print(f" Total False Negatives (FN) : {total_fn}")
        print(f" Total True Positives  (TP) : {total_tp}")
        print("-" * 40)
        print(f" Precision (Detector Trust) : {precision:.4f}")
        print(f" Recall    (GT Captured)    : {recall:.4f}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"\nError calculating metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=['iou', 'kalman', 'neuflow'], default='kalman')
    parser.add_argument("--video_path", default="data/AICity_data/AICity_data/train/S03/c010/vdo.avi")
    parser.add_argument("--xml_path", default="data/gt/ai_challenge_s03_c010-full_annotation.xml")
    parser.add_argument("--model_path", default="models/fine_tuned_rcnn.pth")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--proc_noise", type=float, default=1.0)
    parser.add_argument("--meas_noise", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--easy_task", type=str, default="false", help='Exclude occluded/parked cars')
    parser.add_argument("--evaluate_FP", type=str, help='Compute exact FP/FN metrics')
    args = parser.parse_args()
    main(args)