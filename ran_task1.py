import torch
from torchcodec.decoders import VideoDecoder
from src.background.gaussian import SingleGaussian
from src.utils.post_processing import post_process_mask
import cv2
import numpy as np
import os
from tqdm import tqdm

# --- Configuration ---
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi" # Update this path!
OUTPUT_DIR = "results/task1_1"
ALPHA = 7.0 # Maybe we can do some tests with this
SPLIT_RATIO = 0.25
SAVE_VISUALIZATION = True  # Set True to save the video

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Video
decoder = VideoDecoder(VIDEO_PATH) #, device=device)
total_frames = decoder.metadata.num_frames
train_len = int(total_frames * SPLIT_RATIO)
test_len = total_frames - train_len

# Extract metadata for the video writer
fps = decoder.metadata.average_fps  # Get FPS from torchcodec
width = decoder.metadata.width
height = decoder.metadata.height

print(f"Video Info: {width}x{height} @ {fps} FPS")
print(f"Training on first {train_len} frames. Inference on {test_len} frames.")

# 2. Train Model
model = SingleGaussian(alpha=ALPHA, device=device)
model.fit(decoder, num_train_frames=train_len)

# save mean as image
mean_img = model.mean.cpu().numpy().astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "mean_image.png"), mean_img)

# 3. Setup Video Writer
video_writer = None
if SAVE_VISUALIZATION:
    output_path = os.path.join(OUTPUT_DIR, "foreground_masks.mp4")
    # 'mp4v' is a reliable codec for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # isColor=False because we are saving a grayscale mask. 
    # Change to True if you want to save the original colored video with overlay.
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    print(f"Saving video to: {output_path}")

# 4. Inference Loop
print("Starting Inference...")
for i in tqdm(range(train_len, total_frames)):
    
    # Get frame (C, H, W) float tensor
    frame_tensor = decoder[i].float()
    
    # Predict (Returns Boolean Tensor on GPU)
    fg_mask = model.apply(frame_tensor)
    
    # Convert to CPU Numpy for saving
    # shape: (H, W), values: 0 or 255, type: uint8
    mask_np = fg_mask.cpu().numpy().astype(np.uint8) * 255
    
    # clean_mask: The mask without noise
    # bboxes: List of [x, y, w, h] for valid cars
    clean_mask, bboxes = post_process_mask(mask_np, min_area=150)
    
    # Write to video
    if video_writer:
        # Convert original frame to BGR for drawing
        original_img = frame_tensor.cpu().numpy().astype(np.uint8)
        if original_img.shape[0] == 3: # C, H, W
             original_img = original_img.transpose(1, 2, 0)
             original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        else: # H, W (Grayscale)
             original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

        # Draw Bounding Boxes
        for (x, y, w, h) in bboxes:
            # Green Box for predictions
            cv2.rectangle(clean_mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # Draw Raw Mask overlay (Red) to compare
        # original_img[mask_np > 0] = [0, 0, 255] # Optional: See raw noise vs clean boxes
            
        video_writer.write(clean_mask)

# Cleanup
if video_writer:
    video_writer.release()

print("Done.")