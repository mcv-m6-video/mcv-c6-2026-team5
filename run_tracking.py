import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

# Import Detectors and Trackers
from src.detection.fine_tuned import FineTunedDetector
from src.tracking.iou_tracker import MaxIoUTracker
from src.tracking.kalman_tracker import KalmanTracker

from src.data.loader import AICityDataset 

def get_track_color(track_id):
    """Generates a consistent, distinct BGR color for a given track ID."""
    np.random.seed(track_id)
    color = np.random.randint(0, 255, size=3).tolist()
    return tuple(color)

def preprocess_frame(frame, device):
    """Converts an OpenCV BGR frame to a PyTorch RGB tensor."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_frame = torch.from_numpy(rgb_frame).float() / 255.0
    tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor_frame

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize Object Detector
    print(f"Loading Fine-Tuned Detector from {args.model_path}...")
    detector = FineTunedDetector() 
    model = detector.get_model()
    
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model weights not found at {args.model_path}!")
    
    model.to(device)
    model.eval()
    

    # 2. Initialize the chosen Tracker
    if args.tracker == 'iou':
        print(f"Initializing Max IoU Tracker (Task 2.1)...")
        tracker = MaxIoUTracker(iou_threshold=args.iou_thresh, max_age=args.max_age)
        output_filename = "results/tracking_iou.mp4"
    elif args.tracker == 'kalman':
        print(f"Initializing Kalman Filter Tracker (Task 2.2)...")
        tracker = KalmanTracker(iou_threshold=args.iou_thresh, 
                                max_age=args.max_age,
                                process_noise_scale=args.proc_noise,
                                measurement_noise_scale=args.meas_noise)
        output_filename = "results/tracking_kalman.mp4"
    else:
        raise ValueError("Invalid tracker type specified.")

    # Override output filename if user specifically provided one
    if args.output_video:
        
        output_filename = args.output_video

    # 3. Initialize AICity Dataset
    # This will handle frame extraction and caching automatically.
    print("Initializing dataset. This may take a moment on the first run...")
    dataset = AICityDataset(video_path=args.video_path, xml_path=args.xml_path)
    print("Dataset initialized successfully.")
    
    # 4. Setup Video Writer using properties from cached frames
    first_frame_path = os.path.join(dataset.cache_dir, dataset.imgs[0])
    sample_frame = cv2.imread(first_frame_path)
    height, width, _ = sample_frame.shape
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 10, (width, height))
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    # Note: Using a standard FPS of 25-30 might look more natural than the detector's FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 25, (width, height))

    print(f"Starting tracking on {len(dataset)} frames...")
    # 4. Processing Loop
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Processing Video ({args.tracker})"):
            img_tensor, _ = dataset[idx]

            
            # Add batch dimension and move to device
            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Run Detection
            outputs = model(input_tensor)[0]
            
            # Filter Detections
            mask = outputs['scores'] >= args.conf_thresh
            boxes = outputs['boxes'][mask].cpu().numpy()

            # Update Tracker
            tracked_objects = tracker.update(boxes)

            # --- Visualization ---
            # For drawing, we need the original BGR frame. We load it directly
            # from the cache, which is very fast.
            frame_path = os.path.join(dataset.cache_dir, dataset.imgs[idx])
            frame = cv2.imread(frame_path)

            for obj in tracked_objects:
                track_id = obj['id']
                bbox = obj['bbox']
                color = get_track_color(track_id)
                
                x1, y1, x2, y2 = map(lambda v: max(0, int(v)), bbox)
                
                # Draw Bounding Box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out.write(frame)

    # 6. Release Resources
    out.release()
    print(f"Tracking complete! Video saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 2.1 and 2.2 Tracking")
    parser.add_argument("--xml_path", type=str, default="data/gt/ai_challenge_s03_c010-full_annotation.xml",
                        help="Path to the ground truth XML file")
    parser.add_argument("--tracker", type=str, choices=['iou', 'kalman'], default='iou',
                        help="Choose which tracker to run: 'iou' (Task 2.1) or 'kalman' (Task 2.2)")
    parser.add_argument("--video_path", type=str, default="data/AICity_data/AICity_data/train/S03/c010/vdo.avi", 
                        help="Path to the input video")
    parser.add_argument("--model_path", type=str, default="models/fine_tuned_rcnn.pth", 
                        help="Path to the detection model")
    parser.add_argument("--output_video", type=str, default="", 
                        help="Custom path to save the output video (optional)")
    parser.add_argument("--conf_thresh", type=float, default=0.5, 
                        help="Confidence threshold for object detection")
    parser.add_argument("--iou_thresh", type=float, default=0.3, 
                        help="Minimum IoU threshold to assign a track ID")
    parser.add_argument("--max_age", type=int, default=1, 
                        help="Max frames to keep a track alive without detections (Kalman only)")
    parser.add_argument("--proc_noise", type=float, default=1.0)
    parser.add_argument("--meas_noise", type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)