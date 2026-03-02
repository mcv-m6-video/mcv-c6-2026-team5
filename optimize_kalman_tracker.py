import optuna
import json
import numpy as np
import argparse
from tqdm import tqdm

# --- Your Project Imports ---
from src.tracking.kalman_tracker import KalmanTracker
from src.evaluation import calculate_tracking_metrics

# Load the data ONCE globally
print("Loading raw detections from JSON...")
with open("data/raw_detections.json", 'r') as f:
    RAW_DATA = json.load(f)

# Convert lists back to numpy for speed inside the loop
ALL_GT = RAW_DATA['ground_truth']
ALL_DETS = RAW_DATA['detections']
print(f"Loaded {len(ALL_GT)} frames.")

def objective(trial):
    # --- 1. SEARCH SPACE ---
    # Standard Params
    conf_thresh = trial.suggest_float("conf_thresh", 0.3, 0.9, step=0.05)
    iou_thresh = trial.suggest_float("iou_thresh", 0.1, 0.7, step=0.05)
    max_age = trial.suggest_int("max_age", 1, 30)
    
    # Kalman Params (Use Log Scale!)
    # We explore values like 0.1, 1.0, 10.0, 100.0 evenly
    process_noise_scale = trial.suggest_float("proc_noise", 0.01, 100.0, log=True)
    measurement_noise_scale = trial.suggest_float("meas_noise", 0.01, 100.0, log=True)
    

    # 2. Initialize Tracker
    tracker = KalmanTracker(iou_threshold=iou_thresh, 
                            max_age=max_age, 
                            process_noise_scale=process_noise_scale, 
                            measurement_noise_scale=measurement_noise_scale)
    
    preds_for_eval = []

    # 3. Fast Loop (No Neural Net!)
    for frame_idx, raw_dets in enumerate(ALL_DETS):
        # A. Filter Detections by current conf_thresh
        # raw_det is [x1, y1, x2, y2, score]
        # We perform list comprehension filtering
        filtered_boxes = []
        for d in raw_dets:
            if d[4] >= conf_thresh:
                filtered_boxes.append(d[:4]) # Keep only [x1, y1, x2, y2]
        
        # Convert to numpy for tracker
        if len(filtered_boxes) > 0:
            det_tensor = np.array(filtered_boxes)
        else:
            det_tensor = np.empty((0, 4))
            
        # B. Update Tracker
        tracked_objects = tracker.update(det_tensor)
        
        # C. Format for Evaluation
        frame_preds = [{
            'bbox': t['bbox'], 
            'id': int(t['id'])
        } for t in tracked_objects]
        
        preds_for_eval.append(frame_preds)

    # 4. Calculate Score
    try:
        metrics = calculate_tracking_metrics(ALL_GT, preds_for_eval)
        # You can maximize HOTA, IDF1, or a combination (e.g. 0.5*HOTA + 0.5*IDF1)
        # return metrics['IDF1']*0.5 + metrics['HOTA']*0.5
        return metrics['HOTA']
    except Exception as e:
        # If tracking fails (e.g. no tracks), return 0
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    args = parser.parse_args()

    print(f"Starting optimization for {args.trials} trials...")
    
    # Create Study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Results
    print("\n" + "="*30)
    print(" BEST PARAMETERS ")
    print("="*30)
    print(f"Best HOTA: {study.best_value:.4f}")
    print("Params:", study.best_params)
    
    # Save best params to file for easy reading
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f)