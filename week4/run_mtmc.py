import json
import os
from pathlib import Path

from ultralytics import YOLO
# from src.detection.off_the_shelf import OffTheShelfDetector
from src.tracking.optical_tracker import NeuFlowTracker
from src.detection.fine_tuned import FineTunedDetector
from src.tracking.optical_tracker import NeuFlowTracker
from src.optical_flow.state_of_art_estimators import initialize_neuflow
from src.data.loader import AICityDataset
from src.evaluation import calculate_tracking_metrics, compute_fp_metrics, evaluate_mtmc_globally

from mtmc_pipeline import MTMCPipeline, load_camera_timestamps
from reid.embedding_extract import ViTEmbeddingModel

from global_id_graph_matching import generate_global_ids
import pickle
from torchvision import transforms


import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
    
def load_reid(embed_cfg, model_state, device):
    model = ViTEmbeddingModel(
        model_name=embed_cfg["model_name"],
        embedding_dim=embed_cfg["embedding_dim"],
        pretrained=False,
        dropout=embed_cfg["dropout"],
    ).to(device)
    
    # Check if model_state is the dictionary of weights
    if isinstance(model_state, dict):
        model.load_state_dict(model_state)
    else:
        raise TypeError("The loaded model_state is not a dictionary. Check your checkpoint structure.")
        
    model.eval()
    return model

def main(config_path="config.json"):
    #condig json
    config = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Runing this code with device: {device}")

    # initialize REID embedings
    reid_path = config['embeding']['model']
    ckpt = torch.load(reid_path, map_location=device)
    print(f"Keys available in checkpoint: {ckpt.keys()}")
    model_state = ckpt["model_state"]
    embed_cfg = ckpt["config"]
    vit_extractor = load_reid(embed_cfg, model_state, device)
    
    tfms_embeds = transforms.Compose([
        transforms.Resize((embed_cfg["image_size"], embed_cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    
    # initialize detector
    print(f"Loading Fine-Tuned Detector from {config['detector']['model']}...")
    if 'rcnn' in config['detector']['model']:
        detector = FineTunedDetector() 
        detector = detector.get_model()
        state_dict = torch.load(config['detector']['model'], map_location=device)
        detector.load_state_dict(state_dict)

        detector.to(device).eval()
    else:
        detector = YOLO(config['detector']['model'])


    # initialize tracker
    neuflow_model = initialize_neuflow(device=device, half=False)
    tracker = NeuFlowTracker(model=neuflow_model, 
                                device=device, 
                                half=False, 
                                iou_threshold=config['tracker']['iou_thresh'], 
                                max_age=config['tracker']['max_age'],
                                alpha=config['tracker']['alpha'])
    
    
    # Initialize your specific classes 
    # detector = OffTheShelfDetector(config['detector']['model'], config['detector']['conf_thresh'])
    # vit_extractor = ViTEmbeddingModel(config['vit']['model_name'], config['vit']['device'])
    
    pipeline = MTMCPipeline(detector, tracker, vit_extractor, tfms_embeds, conf_thresh=config['conf_thresh'], device=config['vit']['device'])
    
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
    
    os.makedirs(config['output_dir'], exist_ok=True)
    all_tracklets = {}
    
    # Run 1: Intra-camera processing
    use_cache = config.get('use_cache', False)
    
    for cam_id in cameras:
        cache_path = os.path.join(config['output_dir'], f"{cam_id}_full_cache.pkl")
        
        if use_cache and os.path.exists(cache_path):
            print(f"\n[CACHE HIT] Loading precalculated data for camera {cam_id}...")
            with open(cache_path, 'rb') as f:
                cam_tracklets, all_gt, all_preds = pickle.load(f)
            all_tracklets[cam_id] = cam_tracklets
            
        else:
            print(f"\n[INFERENCE] Running intra camera processing on cam {cam_id}...")
            cam_dir = dataset_root / cam_id
            video_path = str(cam_dir / "vdo.avi")
            roi_path = str(cam_dir / "roi.jpg")
            gt_path = str(cam_dir / "gt/gt.txt")
            
            pipeline.load_roi(cam_id, roi_path)
            
            dataset = AICityDataset(video_path=video_path, xml_path=gt_path)
            cam_tracklets, all_gt, all_preds = pipeline.process_single_camera(cam_id, dataset)
            all_tracklets[cam_id] = cam_tracklets
            
            # Save everything to cache for future runs
            print(f"Saving cache to {cache_path}...")
            with open(cache_path, 'wb') as f:
                pickle.dump((cam_tracklets, all_gt, all_preds), f)
            
            # (Optional) Keep your old specific feature save if you need it elsewhere
            out_path = os.path.join(config['output_dir'], f"{cam_id}_features.pkl")
            pipeline.save_features(cam_tracklets, out_path)
        
        # --- Calculate and Print Metrics ---
        print(f"\n" + "="*40)
        print(f" SINGLE CAMERA METRICS: {cam_id}")
        print("="*40)
        try:
            scores = calculate_tracking_metrics(all_gt, all_preds)
            total_fp, total_fn, total_tp, precision, recall = compute_fp_metrics(all_gt, all_preds)
            
            print(f" HOTA Score      : {scores.get('HOTA', 0.0):.4f}")
            print(f" IDF1 Score      : {scores.get('IDF1', 0.0):.4f}")
            print("-" * 40)
            print(f" Total False Positives (FP) : {total_fp}")
            print(f" Total False Negatives (FN) : {total_fn}")
            print(f" Total True Positives  (TP) : {total_tp}")
            print("-" * 40)
            print(f" Precision : {precision:.4f}")
            print(f" Recall    : {recall:.4f}")
        except Exception as e:
            print(f" Error calculating metrics for {cam_id}: {e}")
        print("="*40 + "\n")
        
        torch.cuda.empty_cache()
        
    

    # Run 2: Inter-camera matching
    all_camera_matches = {}
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            cam1, cam2 = cameras[i], cameras[j]
            # print(f"\nDistances {cam1} to {cam2}:")
            matches = pipeline.cross_camera_matching(
                cam1,
                cam2,
                all_tracklets[cam1], 
                all_tracklets[cam2], 
                start_times,
                threshold=config['matching']['cosine_threshold']
            )
            all_camera_matches[(cam1, cam2)] = matches
            
            print(f"Global matches between {cam1} and {cam2}: {matches}")
    
    global_ids_maping = generate_global_ids(all_camera_matches, all_tracklets, start_times)
    print(global_ids_maping)
    
    # Run 3: Export MOT format for MTMC Evaluation
    mot_output_dir = os.path.join(config['output_dir'], "mot_results")
    os.makedirs(mot_output_dir, exist_ok=True)
    
    for cam_id, tracklets in all_tracklets.items():
        if cam_id not in global_ids_maping:
            continue
            
        out_file = os.path.join(mot_output_dir, f"{cam_id}.txt")
        
        with open(out_file, 'w') as f:
            for local_id, data in tracklets.items():
                if local_id in global_ids_maping[cam_id]:
                    global_id = global_ids_maping[cam_id][local_id]
                    
                    for frame_idx, bbox in data['bboxes'].items():
                        # AI City MOT format: [frame, ID, left, top, width, height, 1, -1, -1, -1]
                        x1, y1, x2, y2 = bbox
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Frame index is usually 1-based in MOT format
                        f.write(f"{frame_idx + 1},{global_id},{x1},{y1},{width},{height},1,-1,-1,-1\n")
                        
    print(f"Exported MTMC results to {mot_output_dir}. Ready for official MTMC evaluation.")
    
    evaluate_mtmc_globally(mot_output_dir, config['dataset_root'], cameras)
    
if __name__ == "__main__":
    main()