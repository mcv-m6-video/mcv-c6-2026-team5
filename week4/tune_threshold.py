import os
import json
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path

# Import only what is needed for matching and evaluation
from mtmc_pipeline import MTMCPipeline, load_camera_timestamps
from global_id_graph_matching import generate_global_ids
from src.evaluation import calculate_tracking_metrics

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def evaluate_mtmc_for_tuning(mot_output_dir, dataset_root, cameras):
    """A modified evaluator that RETURNS the scores instead of just printing them."""
    global_gt = defaultdict(list)
    global_preds = defaultdict(list)
    
    for cam_id in cameras:
        # Load GT
        gt_path = os.path.join(dataset_root, cam_id, "gt/gt.txt")
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame, obj_id, x, y, w, h = map(float, parts[:6])
                    virtual_frame = f"{cam_id}_{int(frame)}"
                    global_gt[virtual_frame].append({'bbox': [x, y, x + w, y + h], 'id': int(obj_id)})
                    
        # Load Preds
        pred_path = os.path.join(mot_output_dir, f"{cam_id}.txt")
        if os.path.exists(pred_path):
            with open(pred_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame, obj_id, x, y, w, h = map(float, parts[:6])
                    virtual_frame = f"{cam_id}_{int(frame)}"
                    global_preds[virtual_frame].append({'bbox': [x, y, x + w, y + h], 'id': int(obj_id)})

    all_virtual_frames = sorted(list(set(global_gt.keys()).union(set(global_preds.keys()))))
    list_gt = [global_gt[vf] for vf in all_virtual_frames]
    list_preds = [global_preds[vf] for vf in all_virtual_frames]
    
    # Return the metrics dictionary
    return calculate_tracking_metrics(list_gt, list_preds)

def main():
    config = load_config("config.json")
    dataset_root = Path(config['dataset_root'])
    
    # 1. Automatically detect sequence and cameras
    sequence_name = dataset_root.name  # Extracts "S03"
    cameras = sorted([d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('c')])
    print(f"Loaded Sequence {sequence_name}. Detected {len(cameras)} cameras: {cameras}")

    # 2. Dynamically locate the timestamp file
    # If dataset_root is "data/AIC22/train/S03", parents[1] goes up to "data/AIC22"
    timestamp_file = dataset_root.parents[1] / "cam_timestamp" / f"{sequence_name}.txt"
    
    if timestamp_file.exists():
        print(f"Loading timestamps from {timestamp_file}...")
        start_times = load_camera_timestamps(timestamp_file)
    else:
        print(f"Warning: Timestamp file {timestamp_file} not found! Defaulting to 0 seconds.")
        start_times = {cam: 0.0 for cam in cameras}
    
    output_dir = config['output_dir']
    
    print("Loading cached tracklets...")
    all_tracklets = {}
    for cam_id in cameras:
        cache_path = os.path.join(output_dir, f"{cam_id}_full_cache.pkl")
        if not os.path.exists(cache_path):
            print(f"Error: Cache for {cam_id} not found. Run main script first.")
            return
        with open(cache_path, 'rb') as f:
            cam_tracklets, _, _ = pickle.load(f)
            all_tracklets[cam_id] = cam_tracklets

    # Initialize a dummy pipeline just for the cross_camera_matching method
    pipeline = MTMCPipeline(None, None, None, None)

    # Define the range of thresholds to test (e.g., 0.02 to 0.20 in steps of 0.02)
    thresholds_to_test = np.arange(0.02, 0.22, 0.02)
    
    results = {}
    best_idf1 = 0.0
    best_thresh = 0.0


    print("\nStarting Grid Search for Optimal Threshold...")
    print("-" * 50)
    
    for thresh in thresholds_to_test:
        thresh = round(thresh, 3)
        print(f"Testing Threshold: {thresh}...")
        
        # 1. Match
        all_camera_matches = {}
        for i in range(len(cameras)):
            for j in range(i + 1, len(cameras)):
                cam1, cam2 = cameras[i], cameras[j]
                matches = pipeline.cross_camera_matching(cam1, cam2,
                    all_tracklets[cam1], all_tracklets[cam2], start_times, threshold=thresh
                )
                all_camera_matches[(cam1, cam2)] = matches
                
        # 2. Global IDs
        global_ids_mapping = generate_global_ids(all_camera_matches, all_tracklets, start_times)
        
        # 3. Export MOT
        mot_output_dir = os.path.join(output_dir, "mot_results_tuning")
        os.makedirs(mot_output_dir, exist_ok=True)
        
        for cam_id, tracklets in all_tracklets.items():
            out_file = os.path.join(mot_output_dir, f"{cam_id}.txt")
            with open(out_file, 'w') as f:
                for local_id, data in tracklets.items():
                    if cam_id in global_ids_mapping and local_id in global_ids_mapping[cam_id]:
                        global_id = global_ids_mapping[cam_id][local_id]
                        for frame_idx, bbox in data['bboxes'].items():
                            x1, y1, x2, y2 = bbox
                            f.write(f"{frame_idx + 1},{global_id},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1\n")
                            
        # 4. Evaluate
        scores = evaluate_mtmc_for_tuning(mot_output_dir, dataset_root, cameras)
        idf1 = scores.get('IDF1', 0.0)
        hota = scores.get('HOTA', 0.0)
        
        results[thresh] = {'IDF1': idf1, 'HOTA': hota}
        print(f"  -> Result: IDF1 = {idf1:.4f} | HOTA = {hota:.4f}")
        
        if idf1 > best_idf1:
            best_idf1 = idf1
            best_thresh = thresh

    # 5. Print Summary
    print("\n" + "="*50)
    print(" TUNING SUMMARY")
    print("="*50)
    print(f"{'Threshold':<15} | {'IDF1':<15} | {'HOTA':<15}")
    print("-" * 50)
    for t, metric in results.items():
        print(f"{t:<15} | {metric['IDF1']:<15.4f} | {metric['HOTA']:<15.4f}")
    print("="*50)
    print(f"🏆 BEST THRESHOLD (by IDF1): {best_thresh} (IDF1: {best_idf1:.4f})")
    print("="*50)

if __name__ == "__main__":
    main()