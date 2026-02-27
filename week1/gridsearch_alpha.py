import torch
import numpy as np
import os
import csv
import random
from tqdm import tqdm
from torchcodec.decoders import VideoDecoder
from src.background.gaussian import SingleGaussian
from src.utils.post_processing import apply_morphology, get_bboxes_from_mask, merge_bboxes_by_distance
from src.data.parser import load_gt_xml
from src.evaluation.coco_eval import evaluate_coco
import cv2

# --- CONFIGURATION ---
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
GT_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
ROI_PATH = "data/AICity_data/AICity_data/train/S03/c010/roi.jpg"
OUTPUT_CSV = "results/grid_search_alpha.csv"

# Grid Search Range
ALPHAS = np.arange(0.5, 6.5, 0.5)
# ALPHAS = [3.0]

# Fixed Parameters for Task 1.1 Baseline
PARAMS = {
    'detection_mode': 'gray',
    'shadow_method': 'none',
    'shadow_params': {},
    'kernel_opening_size': 5,
    'kernel_closing_size': 5,
    'morph_shape': 'ellipse',
    'morph_op': 'open_close',
    'min_area': 3000,
    'merge_dist': 50
}

def run_grid_search():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Setup Data
    decoder = VideoDecoder(VIDEO_PATH, device="cpu")
    total_frames = decoder.metadata.num_frames
    train_len = int(total_frames * 0.25)
    
    gt_boxes_all = load_gt_xml(GT_PATH)
    # Filter GT to only include validation frames (after training)
    test_gt = {k: v for k, v in gt_boxes_all.items() if k >= train_len}
    
    roi = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
    roi_tensor = torch.from_numpy(roi).to(device).float() / 255.0
    
    results = []
    
    print(f"Starting Grid Search on Alphas: {ALPHAS}")
    print(f"Training set: 0-{train_len}, Test set: {train_len}-{total_frames}")
    
    # Optimization: Instantiate model once and fit once
    # We cheat slightly: fit() computes mean/std. This is independent of Alpha.
    # So we compute stats once and reuse them.
    
    print("Pre-computing Background Stats (One-time)...")
    base_model = SingleGaussian(alpha=1.0, device=device)
    base_model.fit(decoder, num_train_frames=train_len)
    mean_gray = base_model.mean_gray
    std = base_model.std
    
    # --- GRID SEARCH LOOP ---
    for alpha in ALPHAS:
        print(f"\nEvaluating Alpha = {alpha}...")
        
        # Init Model with computed stats
        model = SingleGaussian(alpha=float(alpha), device=device)
        model.mean_gray = mean_gray
        model.std = std
        
        predictions = {}
        
        # Inference Loop (Test Frames)
        # Using tqdm to show progress
        pbar = tqdm(range(train_len, total_frames), desc=f"Inf Alpha {alpha}")
        
        for i in pbar:
            frame = decoder[i] # (C, H, W) Tensor
            frame = frame.to(device).float()
            
            # 1. Apply Model
            fg_mask = model.apply(
                frame, 
                detection_mode=PARAMS['detection_mode'], 
                shadow_method=PARAMS['shadow_method'],
                shadow_params=PARAMS['shadow_params']
            )
            
            # 2. Apply ROI
            fg_mask = fg_mask & (roi_tensor > 0)
            
            # 3. Morphology
            # Ensure mask is suitable for morphology_gpu (expects tensor)
            # In your gaussian.py apply() returns a boolean tensor.
            cleaned_mask = apply_morphology(
                fg_mask, 
                operation=PARAMS['morph_op'],
                kernel_opening_size=PARAMS['kernel_opening_size'],
                kernel_closing_size=PARAMS['kernel_closing_size'],
                morph_shape=PARAMS['morph_shape']
            )
            
            # 4. Box Extraction
            cleaned_mask_np = cleaned_mask.cpu().numpy().astype('uint8') * 255
            boxes = get_bboxes_from_mask(cleaned_mask_np, min_area=PARAMS['min_area'])
            
            h, w = frame.shape[1], frame.shape[2]
            boxes = merge_bboxes_by_distance(boxes, min_distance=PARAMS['merge_dist'], frame_height=h)
            
            if len(boxes) > 0:
                predictions[i] = boxes
        
        # Evaluate
        # mAP = evaluate_map(predictions, test_gt)
        mAP = evaluate_coco(test_gt, predictions, h, w)
        print(f"Alpha {alpha} => mAP: {mAP:.4f}")
        results.append([alpha, mAP])
        
    # Save Results
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'map'])
        writer.writerows(results)
    
    print(f"Grid search complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_grid_search()