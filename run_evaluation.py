import torch
from torchcodec.decoders import VideoDecoder
from src.background.gaussian import SingleGaussian
from src.utils.post_processing import post_process_mask
from src.data.parser import load_gt_xml
from src.evaluation.map import compute_map_randomized
from src.visualization.debugger import visualize_comparison

import numpy as np
import cv2
from tqdm import tqdm
import os

# --- Config ---
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
GT_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
ALPHA = 3.0
SPLIT_RATIO = 0.25
SAVE_DEBUG = True
DEBUG_SAVE_PATH = "results/task1_1/"

# 1. Load Data
decoder = VideoDecoder(VIDEO_PATH, device="cpu")
total_frames = decoder.metadata.num_frames
train_len = int(total_frames * SPLIT_RATIO)

print("Loading Ground Truth...")
gt_boxes = load_gt_xml(GT_PATH)

# Filter GT to only include the testing frames (75%)
gt_boxes_test = {k: v for k, v in gt_boxes.items() if k >= train_len}
print(f"Loaded GT for {len(gt_boxes_test)} frames.")

# Extract metadata for the video writer
fps = decoder.metadata.average_fps  # Get FPS from torchcodec
width = decoder.metadata.width
height = decoder.metadata.height

# 2. Train Model
print("Training Gaussian Model...")
model = SingleGaussian(alpha=ALPHA, device="cuda")
model.fit(decoder, num_train_frames=train_len)

# 3. Inference & Collection
pred_boxes_test = {} # {frame_id: [[x,y,w,h], ...]}

debug_writer = None
if SAVE_DEBUG:
    debug_path = os.path.join(DEBUG_SAVE_PATH, "debug_comparison.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (width, height)) #, isColor=True)
    print(f"Saving debug video to: {debug_path}")


print("Running Inference...")
for i in tqdm(range(train_len, total_frames)):
    
    frame_tensor = decoder[i].to("cuda").float()
    
    # Prediction
    fg_mask = model.apply(frame_tensor)
    mask_np = fg_mask.cpu().numpy().astype('uint8') * 255
    
    # Post-processing (Essential for getting boxes!)
    _, boxes = post_process_mask(mask_np, min_area=150)
    
    # Store predictions
    if len(boxes) > 0:
        pred_boxes_test[i] = boxes
    if debug_writer:
        current_gt = gt_boxes_test.get(i, [])
        original_img = frame_tensor.cpu().numpy().astype(np.uint8).transpose(1,2,0)
        
        debug_frame = visualize_comparison(
            frame=frame_tensor.cpu().numpy().astype(np.uint8).transpose(1,2,0), # Adjust for your tensor shape
            gt_boxes=current_gt,
            pred_boxes=boxes
        )
        debug_writer.write(debug_frame)
if debug_writer:
    debug_writer.release()

        

# 4. Evaluate (Task 1.2)
print("Evaluating mAP (Randomized Ranking)...")
mAP = compute_map_randomized(gt_boxes_test, pred_boxes_test, n_runs=10, iou_thresh=0.3)

print(f"Final mAP (Alpha={ALPHA}): {mAP:.4f}")