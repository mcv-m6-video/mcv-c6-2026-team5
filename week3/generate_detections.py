import torch
import json
import numpy as np
from tqdm import tqdm
import os

# --- Your Imports ---
from src.data.loader import AICityDataset
from src.detection.fine_tuned import FineTunedDetector

# CONFIG
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
XML_PATH = "data/gt/ai_challenge_s03_c010-full_annotation.xml"
MODEL_PATH = "models/fine_tuned_rcnn_fold_0_strat_A_w_ocluded.pth"
OUTPUT_FILE = "data/raw_detections.json"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating detections on {device}...")

    # 1. Load Model & Data
    detector = FineTunedDetector().get_model()
    detector.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    detector.to(device).eval()
    
    dataset = AICityDataset(video_path=VIDEO_PATH, xml_path=XML_PATH)
    
    # Data storage
    cached_data = {
        "detections": [], # List of list of [x1, y1, x2, y2, score]
        "ground_truth": [] # List of list of {'bbox': [], 'id': int}
    }

    print("Running inference (ONCE)...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            img_tensor, target = dataset[idx]

            # --- A. Save Ground Truth ---
            # Extract IDs (fallback to index if 'labels' missing)
            # Check for the new track_id field
            if 'track_id' in target and len(target['track_id']) > 0:
                 gt_ids = target['track_id'].cpu().numpy()
            else:
                 # Should not happen with corrected dataset, but safe fallback
                 print(f"Warning: No GT IDs in frame {idx}")
                 gt_ids = np.arange(len(target['boxes']))

            frame_gt = []
            boxes = target['boxes'].cpu().numpy()
            for i, box in enumerate(boxes):
                frame_gt.append({
                    'bbox': box.tolist(), # Convert to list for JSON serialization
                    'id': int(gt_ids[i])
                })
            cached_data["ground_truth"].append(frame_gt)

            # --- B. Save Detections ---
            outputs = detector(img_tensor.unsqueeze(0).to(device))[0]
            
            # Save EVERYTHING with score > 0.01 (Very low threshold)
            # We will filter strictly later during optimization
            mask = outputs['scores'] > 0.01 
            
            det_boxes = outputs['boxes'][mask].cpu().numpy()
            det_scores = outputs['scores'][mask].cpu().numpy()
            
            frame_dets = []
            for box, score in zip(det_boxes, det_scores):
                # Format: [x1, y1, x2, y2, score]
                entry = box.tolist()
                entry.append(float(score)) 
                frame_dets.append(entry)
                
            cached_data["detections"].append(frame_dets)

    # Save to disk
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(cached_data, f)
    print("Done! Now run the optimizer.")

if __name__ == "__main__":
    main()