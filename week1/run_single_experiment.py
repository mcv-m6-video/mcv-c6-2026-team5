import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
from torchcodec.decoders import VideoDecoder

# --- Project Imports ---
# Assumes you have the refactored files we discussed
from src.background.gaussian import RecursiveGaussian, SingleGaussian
from src.utils.post_processing import apply_morphology, get_bboxes_from_mask, merge_bboxes_by_distance
from src.data.parser import load_gt_xml
from src.evaluation.coco_eval import evaluate_coco
from visualize_frame_comparison import USE_RECURSIVE

# --- Configuration ---
# Update these paths if necessary
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
GT_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
ROI_PATH = "data/AICity_data/AICity_data/train/S03/c010/roi.jpg"
OUTPUT_VIDEO = "results/best_experiment.mp4"
SPLIT_RATIO = 0.25

USE_RECURSIVE = True # Set to False to use SingleGaussian instead of RecursiveGaussian
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
# alpha,rho,detection_mode,update_buffer,shadow_method,tau_s,tau_h,shadow_alpha,shadow_beta,kernel_opening_size,kernel_closing_size,morph_shape,morph_op,min_area,merge_dist,mAP
# 5.517641094945545,0.036568795190752415,gray,2,hsv,50,93,0.5000457179903269,0.9342603776131376,3,11,rect,close,583,12,0.6634981080246408
PARAMS = { # BEST MODEL FROM BAYESIAN OPTIMIZATION (0.6635 mAP) 0.6883 w/o ROI
    # Background Model
    'alpha': 5.517641094945545,            # Threshold multiplier
    'rho': 0.036568795190752415,             # Learning rate
    'detection_mode': "gray",
    'update_buffer': 2,
    
    # Shadow Removal
    'shadow_method': "hsv",
    'tau_s': 50,
    'tau_h': 93,
    'shadow_alpha': 0.5000457179903269,
    'shadow_beta': 0.9342603776131376,
    
    # Post-Processing
    # Kernel size (3, 5, 7...)
    'kernel_opening_size': 3,
    'kernel_closing_size': 11,
    'morph_op': "close",# "open", "close", "open_close"
    'morph_shape': "rect",
    'min_area': 583,         # Min pixel area for a car
    'merge_dist': 12         # Merge boxes closer than this
}

PARAMS_SINGLE ={
    'alpha': 3,            # Threshold multiplier
    'shadow_method': "none",
    'detection_mode': "gray",
    'tau_s': 60,
    'tau_h': 40,
    'shadow_alpha': 0.54,
    'shadow_beta': 0.99,
    'kernel_opening_size': 5,
    'kernel_closing_size': 5,
    'morph_shape': 'ellipse',
    'morph_op': 'open_close',
    'min_area': 1500,
    'merge_dist': 50
}

def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2):
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
    if USE_RECURSIVE:
        model = RecursiveGaussian(alpha=params['alpha'], rho=params['rho'], device=device)
    else:
        model = SingleGaussian(alpha=params['alpha'], device=device)
    model.fit(decoder, num_train_frames=train_len)

    # 4. Setup Video Writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    # bg_writer = cv2.VideoWriter(OUTPUT_VIDEO.replace(".mp4", "_background.mp4"), fourcc, fps, (width, height))
    # std_writer = cv2.VideoWriter(OUTPUT_VIDEO.replace(".mp4", "_stddev.mp4"), fourcc, fps, (width, height))
    # foreground_before_morph_writer = cv2.VideoWriter(OUTPUT_VIDEO.replace(".mp4", "_foreground_before_morph.mp4"), fourcc, fps, (width, height))
    # foreground_after_morph_writer = cv2.VideoWriter(OUTPUT_VIDEO.replace(".mp4", "_foreground_after_morph.mp4"), fourcc, fps, (width, height))
    # foreground_dilated_writer = cv2.VideoWriter(OUTPUT_VIDEO.replace(".mp4", "_foreground_dilated.mp4"), fourcc, fps, (width, height))
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
        if USE_RECURSIVE:
            fg_mask_tensor = model.apply(
                frame_tensor, 
                shadow_method=params['shadow_method'],
                shadow_params=shadow_params,
                detection_mode=params['detection_mode'],
                update_buffer=params['update_buffer']
            )
        else:
            fg_mask_tensor = model.apply(
                frame_tensor, 
                shadow_method=params['shadow_method'],
                shadow_params=shadow_params,
                detection_mode=params['detection_mode']
            )
        
        # B. Apply ROI Mask
        # fg_mask_tensor = (fg_mask_tensor > 0) & (roi_mask_tensor > 0)
        # fg_mask_tensor = (fg_mask_tensor.float()*255).to(torch.uint8)

        # C. Post-Processing (Morphology)
        cleaned_mask = apply_morphology(
            fg_mask_tensor, 
            kernel_opening_size=params['kernel_opening_size'],
            kernel_closing_size=params['kernel_closing_size'], 
            operation=params['morph_op'],
            morph_shape=params['morph_shape'],
        )
        cleaned_mask = cleaned_mask.cpu().numpy().astype('uint8') * 255
        # D. Extract Bounding Boxes
        boxes = get_bboxes_from_mask(cleaned_mask, min_area=params['min_area'])
        
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
        frame_vis = draw_boxes(frame_vis, gt_on_frame, color=(0, 255, 0))
        
        # Draw Predictions (Red)
        frame_vis = draw_boxes(frame_vis, boxes, color=(0, 0, 255))
        
        # Add Frame Counter
        cv2.putText(frame_vis, f"Frame: {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # writer.write(frame_vis)
        
        # # Optional: Save background model visualization1
        # frame_before_morph = (fg_mask_uint_tensor.cpu().numpy().astype('uint8'))
        # frame_before_morph_vis = cv2.cvtColor(frame_before_morph, cv2.COLOR_GRAY2BGR)
        # foreground_before_morph_writer.write(frame_before_morph_vis)
        
        # frame_after_morph_vis = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        # foreground_after_morph_writer.write(frame_after_morph_vis)
        
        # bg_frame = model.mean_rgb.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        # bg_frame_vis = cv2.cvtColor(bg_frame, cv2.COLOR_RGB2BGR)
        # bg_writer.write(bg_frame_vis)
        
        
        # std_frame = model.std.cpu().numpy()
        # max_val = std_frame.max() if std_frame.max() > 0 else 1
        # std_norm = (std_frame / max_val * 255).astype(np.uint8)
        # heatmap_frame = cv2.applyColorMap(std_norm, cv2.COLORMAP_JET)
        # std_writer.write(heatmap_frame)
        
        # fg_dilated_vis = cv2.cvtColor(fg_dilated.cpu().numpy().astype('uint8') * 255, cv2.COLOR_GRAY2BGR)
        # foreground_dilated_writer.write(fg_dilated_vis)
        
        
        

    # writer.release()
    # foreground_before_morph_writer.release()
    # foreground_after_morph_writer.release()
    # bg_writer.release()
    # foreground_dilated_writer.release()
    print("Video processing complete.")

    # 6. Evaluation
    print("\nCalculating Metrics...")
    map50 = evaluate_coco(gt_boxes_test, pred_boxes_test, height, width)
    
    print("="*30)
    print(f"FINAL RESULT (mAP@0.5): {map50:.4f}")
    print("="*30)

if __name__ == "__main__":
    run_experiment(PARAMS)