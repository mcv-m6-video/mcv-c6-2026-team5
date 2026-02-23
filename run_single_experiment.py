import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
from torchcodec.decoders import VideoDecoder

# --- Project Imports ---
# Assumes you have the refactored files we discussed
from src.background.gaussian import RecursiveGaussian
from src.utils.post_processing import apply_morphology, get_bboxes_from_mask, merge_bboxes_by_distance
from src.data.parser import load_gt_xml
from src.evaluation.coco_eval import evaluate_coco

# --- Configuration ---
# Update these paths if necessary
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
GT_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
ROI_PATH = "data/AICity_data/AICity_data/train/S03/c010/roi.jpg"
OUTPUT_VIDEO = "results/best_experiment.mp4"
SPLIT_RATIO = 0.25

# --- HYPERPARAMETERS ---
# Replace these with the best values found by Optuna
# alpha,rho,shadow_method,tau_s,tau_h,shadow_alpha,shadow_beta,morph_kernel,morph_op,min_area,merge_dist,mAP
# 2.8191757877076453,0.013645347808113136,hsv,27,85,0.6998176904639454,0.9957806359317366,3,open,400,31,0.6539704129995161

PARAMS = {
    # Background Model
    'alpha': 2.8191757877076453,            # Threshold multiplier
    'rho': 0.013645347808113136,             # Learning rate
    
    # Shadow Removal
    'shadow_method': "hsv",
    'tau_s': 27,
    'tau_h': 85,
    'shadow_alpha': 0.6998176904639454,
    'shadow_beta': 0.9957806359317366,
    
    # Post-Processing
    'morph_kernel': 3,       # Kernel size (3, 5, 7...)
    'morph_op': "open_close",# "open", "close", "open_close"
    'min_area': 400,         # Min pixel area for a car
    'merge_dist': 31         # Merge boxes closer than this
}

def draw_boxes(frame, boxes, color=(0, 255, 0), label="Car", thickness=2):
    """Helper to draw bounding boxes on an image."""
    img = frame.copy()
    for box in boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        # Add label background for readability
        # (Optional: uncomment to add text labels)
        # cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def run_experiment(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    print(f"Loading video from: {VIDEO_PATH}")
    # 1. Load Video & Metadata
    decoder = VideoDecoder(VIDEO_PATH, device="cpu")
    total_frames = decoder.metadata.num_frames
    train_len = int(total_frames * SPLIT_RATIO)
    width = decoder.metadata.width
    height = decoder.metadata.height
    fps = decoder.metadata.average_fps

    # 2. Load Ground Truth & ROI
    print("Loading Ground Truth and ROI...")
    gt_boxes = load_gt_xml(GT_PATH)
    # Filter GT for the testing phase only
    gt_boxes_test = {k: v for k, v in gt_boxes.items() if k >= train_len}
    
    roi_mask = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        raise FileNotFoundError(f"ROI mask not found at {ROI_PATH}")
    roi_mask_tensor = torch.from_numpy(roi_mask).to(device).float() / 255.0

    # 3. Initialize & Train Model
    print(f"Training Recursive Gaussian on first {train_len} frames...")
    model = RecursiveGaussian(alpha=params['alpha'], rho=params['rho'], device=device)
    model.fit(decoder, num_train_frames=train_len)

    # 4. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    print(f"Inference started. Saving output to: {OUTPUT_VIDEO}")

    # 5. Inference Loop
    pred_boxes_test = {}
    
    shadow_params = {
        "alpha": params['shadow_alpha'],
        "beta": params['shadow_beta'],
        "tau_s": params['tau_s'],
        "tau_h": params['tau_h']
    }

    # Iterate through testing frames
    for i in tqdm(range(train_len, total_frames), desc="Processing Frames"):
        # Load frame (C, H, W)
        frame_tensor = decoder[i].to(device).float()
        
        # A. Apply Background Subtraction
        fg_mask = model.apply(
            frame_tensor, 
            shadow_method=params['shadow_method'],
            shadow_params=shadow_params
        )
        
        # B. Apply ROI Mask
        fg_mask = (fg_mask > 0) & (roi_mask_tensor > 0)
        mask_np = fg_mask.cpu().numpy().astype('uint8') * 255
        
        # C. Post-Processing (Morphology)
        mask_clean = apply_morphology(
            mask_np, 
            kernel_opening_size=params['morph_kernel'],
            kernel_closing_size=params['morph_kernel'], 
            operation=params['morph_op']
        )
        
        # D. Extract Bounding Boxes
        boxes = get_bboxes_from_mask(mask_clean, min_area=params['min_area'])
        
        # E. Merge Overlapping/Nearby Boxes
        boxes = merge_bboxes_by_distance(boxes, min_distance=params['merge_dist'], frame_height=height)
        
        if boxes:
            pred_boxes_test[i] = boxes

        # F. Visualization
        # Convert Torch Tensor (RGB) -> Numpy (BGR) for OpenCV
        frame_vis = frame_tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)

        # Draw Ground Truth (Green)
        gt_on_frame = gt_boxes_test.get(i, [])
        frame_vis = draw_boxes(frame_vis, gt_on_frame, color=(0, 255, 0), label="GT")
        
        # Draw Predictions (Red)
        frame_vis = draw_boxes(frame_vis, boxes, color=(0, 0, 255), label="Pred")
        
        # Add Frame Counter
        cv2.putText(frame_vis, f"Frame: {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        writer.write(frame_vis)

    writer.release()
    print("Video processing complete.")

    # 6. Evaluation
    print("\nCalculating Metrics...")
    map50 = evaluate_coco(gt_boxes_test, pred_boxes_test, height, width)
    
    print("="*30)
    print(f"FINAL RESULT (mAP@0.5): {map50:.4f}")
    print("="*30)

if __name__ == "__main__":
    run_experiment(PARAMS)