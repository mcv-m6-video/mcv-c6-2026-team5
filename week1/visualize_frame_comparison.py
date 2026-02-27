# import cv2
# import torch
# import numpy as np
# import os
# from torchcodec.decoders import VideoDecoder
# from src.background.gaussian import SingleGaussian, RecursiveGaussian
# from src.utils.post_processing import apply_morphology, get_bboxes_from_mask, merge_bboxes_by_distance
# from src.data.parser import load_gt_xml

# # --- CONFIGURATION ---
# VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
# GT_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
# ROI_PATH = "data/AICity_data/AICity_data/train/S03/c010/roi.jpg"
# FRAME_TO_PLOT = 560  # Choose a frame with cars (e.g., 600, 850, 1100)

# # --- MODEL SELECTION ---
# # Set to True for Task 2.1 (Adaptive), False for Task 1.1 (Static)
# USE_RECURSIVE = False

# # --- PARAMS (Adjust per model) ---
# if USE_RECURSIVE:
#     OUTPUT_IMG = f"results/frames_viz/viz_recursive_frame_{FRAME_TO_PLOT}.jpg"
#     print("--- Mode: RECURSIVE GAUSSIAN (Task 2.1) ---")
#     PARAMS = {
#         'alpha': 5.711, 
#         'rho': 0.055,
#         'detection_mode': 'gray',
#         'update_buffer': 2,
#         'shadow_method': 'hsv', 
#         'shadow_params': {
#             'alpha': 0.54,
#             'beta': 0.99,
#             'tau_s': 65,
#             'tau_h': 95
#         },
#         'kernel_opening_size': 3,
#         'kernel_closing_size': 13,
#         'morph_shape': 'rect',
#         'morph_op': 'close',
#         'min_area': 566,
#         'merge_dist': 15
#     }
# else:
#     OUTPUT_IMG = f"results/frames_viz/viz_single_frame_{FRAME_TO_PLOT}.jpg"
#     print("--- Mode: SINGLE GAUSSIAN (Task 1.1) ---")
#     PARAMS = {
#         'alpha': 3.0, 
#         'rho': 0,     # Not used
#         'detection_mode': 'gray',
#         'update_buffer': 0, # Not used
#         'shadow_method': 'none', # Task 1.1 had no shadow removal
#         'shadow_params': {},
#         'kernel_opening_size': 5, 
#         'kernel_closing_size': 5,
#         'morph_shape': 'ellipse',
#         'morph_op': 'open_close',
#         'min_area': 150,
#         'merge_dist': 30
#     }

# def compute_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = boxA[2] * boxA[3]
#     boxBArea = boxB[2] * boxB[3]
#     return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# def get_max_iou(pred_box, gt_boxes):
#     if not gt_boxes: return 0.0
#     return max([compute_iou(pred_box, gt) for gt in gt_boxes])

# def run_visualization():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     # 1. Load Video using TorchCodec
#     decoder = VideoDecoder(VIDEO_PATH, device="cpu")
#     total_frames = decoder.metadata.num_frames
#     train_len = int(total_frames * 0.25)
    
#     gt_boxes_all = load_gt_xml(GT_PATH)
#     roi = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
#     roi_tensor = torch.from_numpy(roi).to(device).float() / 255.0

#     # 2. Initialize & Fit Model
#     if USE_RECURSIVE:
#         model = RecursiveGaussian(alpha=PARAMS['alpha'], rho=PARAMS['rho'], device=device)
#     else:
#         model = SingleGaussian(alpha=PARAMS['alpha'], device=device)

#     # --- TRAINING PHASE (First 25%) ---
#     # The .fit() method inside your gaussian.py handles the streaming internally
#     # using video_decoder[i]
#     model.fit(decoder, num_train_frames=train_len)

#     # --- INFERENCE PHASE ---
#     print(f"Seeking to frame {FRAME_TO_PLOT}...")
    
#     if USE_RECURSIVE:
#         # Recursive models MUST process every frame sequentially to update the background
#         # We start from where training left off
#         print("Recursive mode: Processing frames sequentially to update background...")
        
#         # We only need to "burn in" frames up to the target
#         # We can optimize slightly by not doing full morphology/box extraction for the skip frames
#         # just the model.apply() to update the background.
        
#         for i in range(train_len, FRAME_TO_PLOT + 1):
#             frame = decoder[i]
            
#             # Run Model & Update Background
#             fg_mask = model.apply(
#                 frame, 
#                 detection_mode=PARAMS['detection_mode'], 
#                 update_buffer=PARAMS['update_buffer'],
#                 shadow_method=PARAMS['shadow_method'],
#                 shadow_params=PARAMS['shadow_params']
#             )
            
#             # If this is the target frame, capture data and break
#             if i == FRAME_TO_PLOT:
#                 target_frame = frame
#                 target_fg_mask = fg_mask
#                 break
                
#     else:
#         # Single Gaussian (Static) can jump directly to the target frame!
#         # The background model (mean/std) was frozen after .fit()
#         print("Single Gaussian mode: Jumping directly to target frame...")
#         target_frame = decoder[FRAME_TO_PLOT]
        
#         target_fg_mask = model.apply(
#             target_frame, 
#             detection_mode=PARAMS['detection_mode'],
#             shadow_method=PARAMS['shadow_method'],
#             shadow_params=PARAMS['shadow_params']
#         )

#     # --- POST-PROCESSING & VISUALIZATION (Target Frame Only) ---
#     print(f"Visualizing Frame {FRAME_TO_PLOT}...")
    
#     # Apply ROI
#     target_fg_mask = target_fg_mask & (roi_tensor > 0)
#     target_fg_mask = (target_fg_mask.float()*255).to(torch.uint8)
#     # GPU Morphology
#     cleaned_mask = apply_morphology(
#         target_fg_mask, 
#         operation=PARAMS['morph_op'],
#         kernel_opening_size=PARAMS['kernel_opening_size'],
#         kernel_closing_size=PARAMS['kernel_closing_size'],
#         morph_shape=PARAMS['morph_shape']
#     )
    
#     # Box Extraction (CPU)
#     frame_h, frame_w = target_frame.shape[1], target_frame.shape[2]
#     cleaned_mask = cleaned_mask.cpu().numpy().astype('uint8') * 255
#     pred_boxes = get_bboxes_from_mask(cleaned_mask, min_area=PARAMS['min_area'])
#     pred_boxes = merge_bboxes_by_distance(pred_boxes, min_distance=PARAMS['merge_dist'], frame_height=frame_h)

#     # --- DRAWING ---
#     # Convert Torch -> Numpy (H, W, C) BGR for OpenCV
#     vis_img = target_frame.cpu().numpy()
#     if vis_img.shape[0] == 3: vis_img = vis_img.transpose(1, 2, 0) # CHW -> HWC
#     vis_img = np.ascontiguousarray(vis_img, dtype=np.uint8)
#     vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR) # TorchCodec gives RGB

#     mask_np = cleaned_mask.astype('uint8')
#     # 1. Draw Segmentation Mask (Cyan tint)
#     colored_mask = np.zeros_like(vis_img)
#     colored_mask[:, :, 0] = mask_np # Blue
#     colored_mask[:, :, 1] = mask_np # Green
    
#     mask_indices = mask_np > 0
#     vis_img[mask_indices] = cv2.addWeighted(vis_img[mask_indices], 0.6, colored_mask[mask_indices], 0.4, 0)

#     # 2. Draw Ground Truth (Green)
#     gt_boxes = gt_boxes_all.get(FRAME_TO_PLOT, [])
#     for box in gt_boxes:
#         x, y, w, h = map(int, box)
#         cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(vis_img, "GT", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # 3. Draw Predictions (Color Coded by IoU)
#     for box in pred_boxes:
#         x, y, w, h = map(int, box)
#         max_iou = get_max_iou([x, y, w, h], gt_boxes)
        
#         if max_iou > 0.5:
#             color = (255, 0, 0) # Blue (Good)
#             label = f"IoU={max_iou:.2f}"
#         elif max_iou > 0.3:
#             color = (0, 255, 255) # Yellow (Okay)
#             label = f"IoU={max_iou:.2f}"
#         else:
#             color = (0, 0, 255) # Red (False Positive)
#             label = "FP"
        
#         cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(vis_img, label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # Save
#     os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
#     cv2.imwrite(OUTPUT_IMG, vis_img)
#     print(f"Saved visualization to {OUTPUT_IMG}")

# if __name__ == "__main__":
#     run_visualization()

import cv2
import torch
import numpy as np
import os
from torchcodec.decoders import VideoDecoder
from src.background.gaussian import SingleGaussian, RecursiveGaussian
from src.utils.post_processing import apply_morphology, get_bboxes_from_mask, merge_bboxes_by_distance
from src.data.parser import load_gt_xml

# --- CONFIGURATION ---
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
GT_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
ROI_PATH = "data/AICity_data/AICity_data/train/S03/c010/roi.jpg"
FRAME_TO_PLOT = 575  # Change this to the frame you want to visualize

# --- MODEL SELECTION ---
# Set to True for Task 2.1 (Adaptive), False for Task 1.1 (Static)
USE_RECURSIVE = False

# --- PARAMS ---
if USE_RECURSIVE:
    OUTPUT_PREFIX = f"results/frames_viz/recursive_frame_{FRAME_TO_PLOT}"
    print("--- Mode: RECURSIVE GAUSSIAN (Task 2.1) ---")
    PARAMS = {
        'alpha': 5.711, 
        'rho': 0.055,
        'detection_mode': 'gray',
        'update_buffer': 2,
        'shadow_method': 'hsv', 
        'shadow_params': {'alpha': 0.54, 'beta': 0.99, 'tau_s': 65, 'tau_h': 95},
        'kernel_opening_size': 3,
        'kernel_closing_size': 13,
        'morph_shape': 'rect',
        'morph_op': 'close',
        'min_area': 566,
        'merge_dist': 15
    }
else:
    OUTPUT_PREFIX = f"results/frames_viz/single_frame_{FRAME_TO_PLOT}"
    print("--- Mode: SINGLE GAUSSIAN (Task 1.1) ---")
    PARAMS = {
        'alpha': 3,
        'rho': 0,
        'detection_mode': 'gray',
        'update_buffer': 0,
        'shadow_method': 'none',
        'shadow_params': {},
        'kernel_opening_size': 5, 
        'kernel_closing_size': 5,
        'morph_shape': 'ellipse',
        'morph_op': 'open_close',
        'min_area': 300,
        'merge_dist': 30
    }

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_max_iou(pred_box, gt_boxes):
    if not gt_boxes: return 0.0
    return max([compute_iou(pred_box, gt) for gt in gt_boxes])

def run_visualization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Data
    decoder = VideoDecoder(VIDEO_PATH, device="cpu")
    total_frames = decoder.metadata.num_frames
    train_len = int(total_frames * 0.25)
    
    gt_boxes_all = load_gt_xml(GT_PATH)
    roi = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
    roi_tensor = torch.from_numpy(roi).to(device).float() / 255.0

    # 2. Init Model
    if USE_RECURSIVE:
        model = RecursiveGaussian(alpha=PARAMS['alpha'], rho=PARAMS['rho'], device=device)
    else:
        model = SingleGaussian(alpha=PARAMS['alpha'], device=device)

    # 3. Train
    print("Training model...")
    model.fit(decoder, num_train_frames=train_len)

    # 4. Inference
    print(f"Seeking to frame {FRAME_TO_PLOT}...")
    
    if USE_RECURSIVE:
        # Recursive must burn-in
        for i in range(train_len, FRAME_TO_PLOT + 1):
            frame = decoder[i]
            fg_mask = model.apply(
                frame, 
                detection_mode=PARAMS['detection_mode'], 
                update_buffer=PARAMS['update_buffer'],
                shadow_method=PARAMS['shadow_method'],
                shadow_params=PARAMS['shadow_params']
            )
            if i == FRAME_TO_PLOT:
                target_frame = frame
                target_fg_mask = fg_mask
                break
    else:
        # Single Gaussian jumps directly
        target_frame = decoder[FRAME_TO_PLOT]
        target_fg_mask = model.apply(
            target_frame, 
            detection_mode=PARAMS['detection_mode'],
            shadow_method=PARAMS['shadow_method'],
            shadow_params=PARAMS['shadow_params']
        )

    # 5. Post-Processing
    print(f"Processing Frame {FRAME_TO_PLOT}...")
    target_fg_mask = target_fg_mask & (roi_tensor > 0)
    target_fg_mask = (target_fg_mask.float()*255).to(torch.uint8)
    
    cleaned_mask = apply_morphology(
        target_fg_mask, 
        operation=PARAMS['morph_op'],
        kernel_opening_size=PARAMS['kernel_opening_size'],
        kernel_closing_size=PARAMS['kernel_closing_size'],
        morph_shape=PARAMS['morph_shape']
    )
    
    # Extract Boxes
    cleaned_mask_np = cleaned_mask.cpu().numpy().astype('uint8') * 255
    pred_boxes = get_bboxes_from_mask(cleaned_mask_np, min_area=PARAMS['min_area'])
    pred_boxes = merge_bboxes_by_distance(pred_boxes, min_distance=PARAMS['merge_dist'], frame_height=target_frame.shape[1])

    # --- SAVE 4 SEPARATE IMAGES ---
    os.makedirs(os.path.dirname(OUTPUT_PREFIX), exist_ok=True)

    # Convert Torch Frame to BGR Numpy
    vis_img = target_frame.cpu().numpy()
    if vis_img.shape[0] == 3: vis_img = vis_img.transpose(1, 2, 0)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # IMAGE 1: Original
    cv2.imwrite(f"{OUTPUT_PREFIX}_1_original.jpg", vis_img)
    print(f"Saved: {OUTPUT_PREFIX}_1_original.jpg")

    # IMAGE 2: Ground Truth Only
    vis_gt = vis_img.copy()
    gt_boxes = gt_boxes_all.get(FRAME_TO_PLOT, [])
    for box in gt_boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(vis_gt, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis_gt, "GT", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(f"{OUTPUT_PREFIX}_2_gt.jpg", vis_gt)
    print(f"Saved: {OUTPUT_PREFIX}_2_gt.jpg")

    # IMAGE 3: Mask Calculation
    cv2.imwrite(f"{OUTPUT_PREFIX}_3_mask.jpg", cleaned_mask_np)
    print(f"Saved: {OUTPUT_PREFIX}_3_mask.jpg")

    # IMAGE 4: Final Prediction (GT + Predictions)
    vis_final = vis_gt.copy() # Start with GT drawn
    for box in pred_boxes:
        x, y, w, h = map(int, box)
        max_iou = get_max_iou([x, y, w, h], gt_boxes)
        
        if max_iou > 0.5:
            color = (255, 0, 0) # Blue
            label = f"{max_iou:.2f}"
        elif max_iou > 0.3:
            color = (0, 255, 255) # Yellow
            label = f"{max_iou:.2f}"
        else:
            color = (0, 0, 255) # Red
            label = "FP"
        
        cv2.rectangle(vis_final, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis_final, label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite(f"{OUTPUT_PREFIX}_4_final.jpg", vis_final)
    print(f"Saved: {OUTPUT_PREFIX}_4_final.jpg")

if __name__ == "__main__":
    run_visualization()