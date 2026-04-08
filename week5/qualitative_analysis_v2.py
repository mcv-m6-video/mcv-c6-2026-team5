import pickle
import os
import cv2
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision

from model.model_classification import Model
from util.io import load_json
from util.dataset import load_classes

class DummyArgs:
    def __contains__(self, key):
        return hasattr(self, key)

def load_model_from_config(model_name):
    """Helper function to load a model and its specific config arguments."""
    config_path = f'config/{model_name}.json'
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return None, None
        
    config = load_json(config_path)
    
    m_args = DummyArgs()
    m_args.feature_arch = config.get('feature_arch', 'rny002')
    m_args.num_classes = config.get('num_classes', 12)
    m_args.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    m_args.loss_type = config.get('loss_type', 'bce')
    m_args.class_aware_sampling = config.get('class_aware_sampling', False)
    m_args.temporal_head = config.get('temporal_head', 'max_pool')
    m_args.tcn_num_layers = config.get('tcn_num_layers', 3)
    m_args.tcn_kernel_size = config.get('tcn_kernel_size', 3)
    m_args.tcn_hidden_dim = config.get('tcn_hidden_dim', None)
    m_args.tcn_dropout = config.get('tcn_dropout', 0.2)

    print(f"Initializing {model_name} on {m_args.device}...")
    model = Model(args=m_args)
    
    save_dir = config['save_dir'] + '/' + model_name
    ckpt_path = os.path.join(save_dir, 'checkpoints', 'checkpoint_best.pt')
    
    if os.path.exists(ckpt_path):
        model.load(torch.load(ckpt_path, map_location=m_args.device))
        print(f" -> {model_name} weights loaded successfully.")
    else:
        print(f"Error: Could not find checkpoint at {ckpt_path}")
        return None, None
        
    return model, config

def get_top_preds(probs, idx_to_class, top_k=2, threshold=0.15):
    """Formats the top predictions into a string like 'PASS (0.73), DRIVE (0.69)'"""
    top_indices = np.argsort(probs)[::-1][:top_k]
    preds = []
    for i in top_indices:
        if probs[i] > threshold or len(preds) == 0:
            preds.append(f"{idx_to_class[i + 1]} ({probs[i]:.2f})")
    return ", ".join(preds)

def main():
    parser = argparse.ArgumentParser(description="Compare Baseline and Best Model qualitative results.")
    parser.add_argument('--model_base', type=str, default='baseline', help='Config name for Baseline')
    parser.add_argument('--model_best', type=str, default='best_model', help='Config name for Best Model')
    parser.add_argument('--store_dir', type=str, default='/ghome/group05/datasets/SN-BAS-2025_savedata_stride2/splits')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--clip_len', type=int, default=50)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--num_clips', type=int, default=5)
    parser.add_argument('--target_class', type=int, default=None, help='Filter by specific class ID')
    parser.add_argument('--indices', type=int, nargs='+', default=None)
    parser.add_argument('--find_improvement', action='store_true', help='Search clips where Best is correct and Base is wrong')
    
    args = parser.parse_args()

    # --- 1. LOAD BOTH MODELS ---
    model_base, config_base = load_model_from_config(args.model_base)
    model_best, config_best = load_model_from_config(args.model_best)
    
    if model_base is None or model_best is None:
        return

    # --- 2. LOAD CLASS DICTIONARY ---
    classes_dict = load_classes(os.path.join('data', config_base.get('dataset', 'soccernetball'), 'class.txt'))
    idx_to_class = {v: k for k, v in classes_dict.items()}
    ordered_class_names = [idx_to_class[i] for i in range(1, 13)]

    # --- 3. LOAD DATA SPLITS ---
    store_path = os.path.join(args.store_dir, f'LEN{args.clip_len}SPLIT{args.split}')
    with open(os.path.join(store_path, 'frame_paths.pkl'), 'rb') as f:
        frame_paths = pickle.load(f)
    with open(os.path.join(store_path, 'labels.pkl'), 'rb') as f:
        labels_store = pickle.load(f)

    # Filtering logic (MODIFIED: Strict limit of max 3 labels per clip)
    if args.indices is not None:
        action_clip_indices = args.indices
        folder_suffix = "_CUSTOM_INDICES"
    else:
        if args.target_class is not None:
            # Must contain target class AND have between 1 and 3 labels total
            action_clip_indices = [i for i, labels in enumerate(labels_store) 
                                   if any(lbl['label'] == args.target_class for lbl in labels) 
                                   and 0 < len(labels) <= 3]
            folder_suffix = f"_CLASS{args.target_class}_MAX3LABELS"
        else:
            action_clip_indices = [i for i, labels in enumerate(labels_store) if 0 < len(labels) <= 3]
            folder_suffix = "_RANDOM_ACTIONS_MAX3LABELS"
        
        if len(action_clip_indices) == 0:
            print("No matching clips found with those constraints!")
            return
            
    if args.find_improvement:
        folder_suffix += "_IMPROVEMENTS"

    random.shuffle(action_clip_indices)

    output_dir = os.path.join("qualitative_results", f"COMPARE_{args.model_base}_vs_{args.model_best}{folder_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    if args.find_improvement:
        print(f"\n[SEARCH MODE] Scanning dataset for perfect comparisons (Best Correct / Base Wrong)...")
    else:
        print(f"\nGenerating clips into '{output_dir}'...")

    # --- 4. GENERATION LOOP ---
    playback_fps = args.fps / args.stride
    generated_count = 0

    for idx in action_clip_indices:
        if generated_count >= args.num_clips:
            break 

        paths_metadata = frame_paths[idx]
        clip_labels = labels_store[idx]
        if not paths_metadata or paths_metadata[1] == -1: 
            continue 
            
        base_path, start, pad_start, pad_end, ndigits, length = paths_metadata
        
        actual_frame_paths = [None] * pad_start
        for j in range(length - pad_start - pad_end):
            f_name = f"frame{start + j * args.stride}.jpg" if ndigits == -1 else f"{str(start + j * args.stride).zfill(ndigits)}.jpg"
            actual_frame_paths.append(os.path.join(base_path, f_name))
        actual_frame_paths += [None] * pad_end

        # --- PREPARE TENSOR FOR INFERENCE ---
        tensor_frames = []
        valid_shape = None
        
        for p in actual_frame_paths:
            if p is not None and os.path.exists(p):
                img = torchvision.io.read_image(p)
                valid_shape = img.shape
                break
                
        if valid_shape is None:
            continue

        for p in actual_frame_paths:
            if p is None or not os.path.exists(p):
                tensor_frames.append(torch.zeros(valid_shape, dtype=torch.uint8))
            else:
                tensor_frames.append(torchvision.io.read_image(p))
                
        seq = torch.stack(tensor_frames, dim=0)

        # --- RUN INFERENCE ON BOTH MODELS ---
        with torch.no_grad():
            probs_base = model_base.predict(seq)[0]
            probs_best = model_best.predict(seq)[0]
            
        top1_idx_base = np.argmax(probs_base)
        top1_idx_best = np.argmax(probs_best)
        
        gt_classes = [l['label'] for l in clip_labels]
        
        # --- SEARCH LOGIC ---
        if args.find_improvement:
            best_is_correct = (top1_idx_best + 1) in gt_classes
            base_is_wrong = (top1_idx_base + 1) not in gt_classes
            
            if not best_is_correct or not base_is_wrong:
                continue 

        # Format strings using the new multi-label helper function
        pred_base_str = get_top_preds(probs_base, idx_to_class)
        pred_best_str = get_top_preds(probs_best, idx_to_class)

        # Clean GT String (MODIFIED: Shows exact instances with frame numbers)
        gt_strings = [f"{idx_to_class[l['label']]} (fr {l['label_idx']})" for l in clip_labels]
        gt_string = ", ".join(gt_strings) if gt_strings else "Background"

        # --- PLOT CONFIDENCE DISTRIBUTION COMPARISON ---
        plt.figure(figsize=(12, 6))
        x = np.arange(len(ordered_class_names))
        width = 0.35
        
        plt.bar(x - width/2, probs_base, width, label='Baseline', color='#A1C9F4', edgecolor='black')
        plt.bar(x + width/2, probs_best, width, label='Best Model', color='#FFB482', edgecolor='black')
        
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        plt.xticks(x, ordered_class_names, rotation=45, ha='right', fontweight='bold')
        plt.ylim(0, 1.0)
        plt.ylabel('Probability Score', fontweight='bold')
        plt.title(f'Clip {idx} Predictions Comparison', fontweight='bold')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.gca().set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'clip_{idx}_distribution.png'))
        plt.close()

        # --- GENERATE VIDEO WITH NEW HUD OVERLAY ---
        first_frame = cv2.imread(next(p for p in actual_frame_paths if p is not None))
        orig_height, width, _ = first_frame.shape
        
        # HUD settings for smaller text and non-overlapping canvas
        hud_height = 100
        new_height = orig_height + hud_height
        
        out_path = os.path.join(output_dir, f'clip_{idx}.mp4')
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), playback_fps, (width, new_height))
        
        print(f"\n[FOUND MATCH] Generating Clip {idx}")
        print(f" -> Saving {out_path}")
        
        # BGR Colors matching your image
        C_WHITE = (255, 255, 255)
        C_GREEN = (0, 255, 0)
        C_YELLOW = (0, 220, 255) 

        for frame_idx, p in enumerate(actual_frame_paths):
            if p is None or not os.path.exists(p):
                frame = np.zeros((orig_height, width, 3), dtype=np.uint8)
            else:
                frame = cv2.imread(p)

            # Create a new blank canvas tall enough for the HUD + Video
            canvas = np.zeros((new_height, width, 3), dtype=np.uint8)
            
            # Place the video frame AT THE BOTTOM of the canvas
            canvas[hud_height:new_height, 0:width] = frame

            # Draw Text in the black HUD area at the top
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65  # Reduced size
            thickness = 2      # Kept at 2 so it's readable when compressed
            
            cv2.putText(canvas, f"Frame {frame_idx}/{args.clip_len}", (15, 25), font, font_scale, C_WHITE, thickness, cv2.LINE_AA)
            cv2.putText(canvas, f"GT:   {gt_string}", (15, 50), font, font_scale, C_GREEN, thickness, cv2.LINE_AA)
            cv2.putText(canvas, f"Base: {pred_base_str}", (15, 75), font, font_scale, C_YELLOW, thickness, cv2.LINE_AA)
            cv2.putText(canvas, f"Best: {pred_best_str}", (15, 95), font, font_scale, C_YELLOW, thickness, cv2.LINE_AA)

            # Flashing Action Marker (+/- 2 frames of the actual event)
            # Offset the Y coordinate by the hud_height so it draws over the actual video
            for lbl in clip_labels:
                if abs(lbl['label_idx'] - frame_idx) <= 2:
                    cv2.putText(canvas, f"--> {idx_to_class[lbl['label']]} <--", (width//2 - 120, hud_height + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            out.write(canvas)
        out.release()
        
        generated_count += 1
        
    print("\nGeneration complete!")

if __name__ == '__main__':
    main()