import optuna
import json
import numpy as np
import argparse
import torch
from tqdm import tqdm

# --- Project Imports ---
from src.tracking.optical_tracker import NeuFlowTracker
from src.evaluation import calculate_tracking_metrics
from src.optical_flow.state_of_art_estimators import initialize_neuflow
from src.data.loader import AICityDataset

# --- 1. GLOBAL SETUP ---
# Load the data ONCE globally
print("Loading raw detections from JSON...")
with open("data/raw_detections.json", 'r') as f:
    RAW_DATA = json.load(f)

# Convert lists back to numpy for speed inside the loop
ALL_GT = RAW_DATA['ground_truth']
ALL_DETS = RAW_DATA['detections']
print(f"Loaded {len(ALL_GT)} frames.")

# Initialize model and dataset ONCE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading NeuFlow model on {device}...")
NEUFLOW_MODEL = initialize_neuflow(device=device, half=False)

print("Loading Dataset to fetch frames for Optical Flow...")
DATASET = AICityDataset(video_path="data/AICity_data/AICity_data/train/S03/c010/vdo.avi", 
                        xml_path="data/gt/ai_challenge_s03_c010-full_annotation.xml")


def objective(trial):
    # --- 2. SEARCH SPACE ---
    # Standard Params
    conf_thresh = trial.suggest_float("conf_thresh", 0.3, 0.9, step=0.05)
    iou_thresh = trial.suggest_float("iou_thresh", 0.1, 0.7, step=0.05)
    max_age = trial.suggest_int("max_age", 1, 30)
    
    # NeuFlow Smoothing Param (0.0 = pure detection, 1.0 = pure flow prediction)
    alpha = trial.suggest_float("alpha", 0.0, 1.0, step=0.05) 
    
    # --- 3. INITIALIZE TRACKER ---
    tracker = NeuFlowTracker(model=NEUFLOW_MODEL,
                             device=device,
                             half=False,
                             iou_threshold=iou_thresh, 
                             max_age=max_age, 
                             alpha=alpha)
    # print("neuflow initialized")
    
    preds_for_eval = []

    # --- 4. TRACKING LOOP ---
    for frame_idx, raw_dets in enumerate(ALL_DETS):
        # Fetch actual image tensor for optical flow
        img_tensor, _ = DATASET[frame_idx]
        
        # A. Filter Detections by current conf_thresh
        filtered_boxes = []
        for d in raw_dets:
            if d[4] >= conf_thresh:
                filtered_boxes.append(d[:4]) 
        
        if len(filtered_boxes) > 0:
            det_tensor = np.array(filtered_boxes)
        else:
            det_tensor = np.empty((0, 4))
            
        # B. Update Tracker (Passing both frame and detections)
        # print("updating tracker")
        tracked_objects = tracker.update(img_tensor, det_tensor)
        # print("tracker updated")
        
        # C. Format for Evaluation
        frame_preds = [{
            'bbox': t['bbox'], 
            'id': int(t['id'])
        } for t in tracked_objects]
        
        preds_for_eval.append(frame_preds)
    # print("tracking loop ended")

    # --- 5. CALCULATE SCORE ---
    try:
        metrics = calculate_tracking_metrics(ALL_GT, preds_for_eval)
        # Using a weighted combination of IDF1 (Identity) and HOTA (Overall)
        return metrics['IDF1']*0.5 + metrics['HOTA']*0.5
    except Exception as e:
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    args = parser.parse_args()

    print(f"Starting optimization for {args.trials} trials...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n" + "="*30)
    print(" BEST PARAMETERS ")
    print("="*30)
    print(f"Best Score (0.5*IDF1 + 0.5*HOTA): {study.best_value:.4f}")
    print("Params:", study.best_params)
    
    with open("best_neuflow_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)