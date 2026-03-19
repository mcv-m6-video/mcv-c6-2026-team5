import os
import json
import pickle
import optuna
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm

from mtmc_pipeline import MTMCPipeline, load_camera_timestamps
from global_id_graph_matching import generate_global_ids
from src.evaluation import calculate_tracking_metrics
from src.data.loader import AICityDataset
from reid.embedding_extract import ViTEmbeddingModel
from src.tracking.optical_tracker import NeuFlowTracker
from src.optical_flow.state_of_art_estimators import initialize_neuflow

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_reid(embed_cfg, model_state, device):
    model = ViTEmbeddingModel(
        model_name=embed_cfg["model_name"], embedding_dim=embed_cfg["embedding_dim"],
        pretrained=False, dropout=embed_cfg["dropout"]
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model

def evaluate_mtmc_for_tuning(mot_output_dir, dataset_root, cameras):
    global_gt = defaultdict(list)
    global_preds = defaultdict(list)
    
    for cam_id in cameras:
        gt_path = os.path.join(dataset_root, cam_id, "gt/gt.txt")
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame, obj_id, x, y, w, h = map(float, parts[:6])
                    global_gt[f"{cam_id}_{int(frame)}"].append({'bbox': [x, y, x + w, y + h], 'id': int(obj_id)})
                    
        pred_path = os.path.join(mot_output_dir, f"{cam_id}.txt")
        if os.path.exists(pred_path):
            with open(pred_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame, obj_id, x, y, w, h = map(float, parts[:6])
                    global_preds[f"{cam_id}_{int(frame)}"].append({'bbox': [x, y, x + w, y + h], 'id': int(obj_id)})

    all_virtual_frames = sorted(list(set(global_gt.keys()).union(set(global_preds.keys()))))
    list_gt = [global_gt[vf] for vf in all_virtual_frames]
    list_preds = [global_preds[vf] for vf in all_virtual_frames]
    return calculate_tracking_metrics(list_gt, list_preds)

def main():
    config = load_config("config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_root = Path(config['dataset_root'])
    sequence_name = dataset_root.name
    cameras = sorted([d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('c')])
    
    timestamp_file = dataset_root.parents[1] / "cam_timestamp" / f"{sequence_name}.txt"
    start_times = load_camera_timestamps(timestamp_file) if timestamp_file.exists() else {cam: 0.0 for cam in cameras}
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # STAGE 1: CACHE DETECTIONS (Run YOLO once and kill it)
    # ---------------------------------------------------------
    yolo_cache_path = os.path.join(output_dir, "yolo_bboxes_cache.pkl")
    if not os.path.exists(yolo_cache_path):
        print("\n--- STAGE 1: Generating YOLO Bounding Box Cache ---")
        detector = YOLO(config['detector']['model'])
        dummy_pipeline = MTMCPipeline(detector, None, None, None, conf_thresh=config['conf_thresh'], device=device)
        
        all_bboxes = {}
        
        for cam_id in cameras:
            print(f"Running YOLO on {cam_id}...")
            all_bboxes[cam_id] = {}
            cam_dir = dataset_root / cam_id
            dataset = AICityDataset(video_path=str(cam_dir / "vdo.avi"), xml_path=str(cam_dir / "gt/gt.txt"))
            dummy_pipeline.load_roi(cam_id, str(cam_dir / "roi.jpg"))
            
            for frame_idx in tqdm(range(len(dataset))):
                img_tensor, _, img_rgb = dataset[frame_idx]
                import cv2
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                outputs = detector(img_bgr, conf=config['conf_thresh'], iou=0.45, imgsz=1280, device=device, verbose=False)[0]
                raw_bboxes = outputs.boxes.xyxy.detach().cpu().numpy()
                bboxes = [b for b in raw_bboxes if dummy_pipeline.is_in_roi(b, cam_id)]
                all_bboxes[cam_id][frame_idx] = np.array(bboxes) if len(bboxes) > 0 else np.empty((0, 4))
                
        with open(yolo_cache_path, 'wb') as f:
            pickle.dump(all_bboxes, f)
        
        # Free GPU Memory!
        del detector
        torch.cuda.empty_cache()
        print("YOLO Cache generated and model removed from GPU.")
    else:
        print("\nLoading YOLO Bounding Box Cache...")
        with open(yolo_cache_path, 'rb') as f:
            all_bboxes = pickle.load(f)

    # ---------------------------------------------------------
    # STAGE 2: PREPARE RE-ID & NEUFLOW FOR OPTUNA
    # ---------------------------------------------------------
    ckpt = torch.load(config['embeding']['model'], map_location=device)
    vit_extractor = load_reid(ckpt["config"], ckpt["model_state"], device)
    tfms_embeds = transforms.Compose([
        transforms.Resize((ckpt["config"]["image_size"], ckpt["config"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # NeuFlow model stays loaded, we just spawn new Trackers in the loop
    neuflow_model = initialize_neuflow(device=device, half=False)

    # ---------------------------------------------------------
    # STAGE 3: OPTUNA OBJECTIVE
    # ---------------------------------------------------------
    def objective(trial):
        print(f"\n--- Starting Trial {trial.number} ---")
        
        # 1. Suggest HEAVY Parameters
        opt_iou_thresh = trial.suggest_float("iou_thresh", 0.1, 0.5)
        opt_max_age = trial.suggest_int("max_age", 1, 10)
        opt_alpha = trial.suggest_float("alpha", 0.1, 0.9)
        
        # 2. Suggest FAST Parameters
        opt_cosine = trial.suggest_float("cosine_threshold", 0.05, 0.25)
        opt_min_gap = trial.suggest_float("min_gap", 0.5, 4.0)
        opt_max_gap = trial.suggest_float("max_gap", 60.0, 150.0)

        # Monkey-patch physics
        def optuna_time_bounds(cam1, cam2):
            return (opt_min_gap, opt_max_gap)
        import mtmc_pipeline, global_id_graph_matching
        mtmc_pipeline.get_time_bounds = optuna_time_bounds
        global_id_graph_matching.get_time_bounds = optuna_time_bounds

        all_tracklets = {}
        
        # Process cameras with new tracker params, USING CACHED BBOXES
        for cam_id in cameras:
            # Create a fresh tracker for this camera with Optuna's suggested params
            tracker = NeuFlowTracker(model=neuflow_model, device=device, half=False, 
                                     iou_threshold=opt_iou_thresh, max_age=opt_max_age, alpha=opt_alpha)
            
            pipeline = MTMCPipeline(None, tracker, vit_extractor, tfms_embeds, conf_thresh=config['conf_thresh'], device=device)
            pipeline.load_roi(cam_id, str(dataset_root / cam_id / "roi.jpg"))
            
            dataset = AICityDataset(video_path=str(dataset_root / cam_id / "vdo.avi"), xml_path=str(dataset_root / cam_id / "gt/gt.txt"))
            
            # Pass the cached bounding boxes!
            cam_tracklets, _, _ = pipeline.process_single_camera(cam_id, dataset, cached_bboxes=all_bboxes[cam_id])
            all_tracklets[cam_id] = cam_tracklets
            
        # Cross-Camera Matching
        pipeline = MTMCPipeline(None, None, None, None) # Dummy for matching func
        all_camera_matches = {}
        for i in range(len(cameras)):
            for j in range(i + 1, len(cameras)):
                cam1, cam2 = cameras[i], cameras[j]
                matches = pipeline.cross_camera_matching(cam1, cam2, all_tracklets[cam1], all_tracklets[cam2], start_times, threshold=opt_cosine)
                all_camera_matches[(cam1, cam2)] = matches
                
        # Global IDs & Export
        global_ids_mapping = generate_global_ids(all_camera_matches, all_tracklets, start_times)
        mot_output_dir = os.path.join(output_dir, f"mot_optuna_trial_{trial.number}")
        os.makedirs(mot_output_dir, exist_ok=True)
        
        for cam_id, tracklets in all_tracklets.items():
            out_file = os.path.join(mot_output_dir, f"{cam_id}.txt")
            with open(out_file, 'w') as f:
                for local_id, data in tracklets.items():
                    if cam_id in global_ids_mapping and local_id in global_ids_mapping[cam_id]:
                        global_id = global_ids_mapping[cam_id][local_id]
                        for frame_idx, bbox in data['bboxes'].items():
                            x1, y1, x2, y2 = bbox
                            f.write(f"{int(frame_idx) + 1},{global_id},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1\n")
                            
        # Evaluate
        scores = evaluate_mtmc_for_tuning(mot_output_dir, dataset_root, cameras)
        import shutil
        shutil.rmtree(mot_output_dir) # Cleanup
        
        return scores.get('IDF1', 0.0)

    # ---------------------------------------------------------
    # RUN STUDY
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STARTING HEAVY OPTUNA BAYESIAN SEARCH")
    print("="*50)
    
    study = optuna.create_study(direction="maximize")
    # Run 25 trials. With caching, this will take a few hours instead of days.
    study.optimize(objective, n_trials=25)

    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE 🏆")
    print("="*50)
    print(f"Best IDF1 Score: {study.best_value:.4f}")
    print("Best Parameters to put in your config.json:")
    for key, value in study.best_params.items():
        if key == "max_age":
            print(f"  {key}: {int(value)}")
        else:
            print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()