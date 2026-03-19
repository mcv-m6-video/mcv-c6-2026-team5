import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Import Detectors and Trackers
from src.detection.fine_tuned import FineTunedDetector
from src.tracking.iou_tracker import MaxIoUTracker
from src.tracking.kalman_tracker import KalmanTracker
from src.tracking.optical_tracker import NeuFlowTracker
from src.optical_flow.state_of_art_estimators import initialize_neuflow
from src.data.loader import AICityDataset 

def get_track_color(track_id):
    np.random.seed(track_id)
    color = np.random.randint(0, 255, size=3).tolist()
    return tuple(color)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading Fine-Tuned Detector from {args.model_path}...")
    detector = FineTunedDetector() 
    detector_model = detector.get_model()
    
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        detector_model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model weights not found at {args.model_path}!")
    
    detector_model.to(device)
    detector_model.eval()
    
    if args.tracker == 'iou':
        print(f"Initializing Max IoU Tracker...")
        tracker = MaxIoUTracker(iou_threshold=args.iou_thresh, max_age=args.max_age)
        output_filename = "results/tracking_iou.mp4"
    elif args.tracker == 'kalman':
        print(f"Initializing Kalman Filter Tracker...")
        tracker = KalmanTracker(iou_threshold=args.iou_thresh, 
                                max_age=args.max_age,
                                process_noise_scale=args.proc_noise,
                                measurement_noise_scale=args.meas_noise)
        output_filename = "results/tracking_kalman.mp4"
    elif args.tracker == 'neuflow':
        print(f"Initializing NeuFlow Tracker...")
        half = False
        neuflow_model = initialize_neuflow(device=device, half=half)
        tracker = NeuFlowTracker(model=neuflow_model, 
                                 device=device, 
                                 half=half, 
                                 iou_threshold=args.iou_thresh, 
                                 max_age=args.max_age,
                                 alpha=args.alpha)
        output_filename = "results/tracking_neuflow.mp4"
    else:
        raise ValueError("Invalid tracker type specified.")

    if args.output_video:
        output_filename = args.output_video

    print("Initializing dataset. This may take a moment on the first run...")
    if args.easy_task.lower() in ['true', '1', 'yes']:
        easy_task = True
        print("Easy task enabled: Occluded and parked cars will be excluded from tracking.")
    else:
        easy_task = False
    dataset = AICityDataset(video_path=args.video_path, xml_path=args.xml_path, easy_task=easy_task)
    print("Dataset initialized successfully.")
    
    first_frame_path = os.path.join(dataset.cache_dir, dataset.imgs[0])
    sample_frame = cv2.imread(first_frame_path)
    height, width, _ = sample_frame.shape
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 20, (width, height))

    track_history = defaultdict(list)
    max_history_len = 30

    print(f"Starting tracking on {len(dataset)} frames...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Processing Video ({args.tracker})"):
            img_tensor, target = dataset[idx]

            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            outputs = detector_model(input_tensor)[0]
            
            mask = outputs['scores'] >= args.conf_thresh
            boxes = outputs['boxes'][mask].cpu().numpy()

            if args.tracker == 'neuflow':
                tracked_objects = tracker.update(img_tensor, boxes)
            else:
                tracked_objects = tracker.update(boxes)

            frame_path = os.path.join(dataset.cache_dir, dataset.imgs[idx])
            frame = cv2.imread(frame_path)
            
            active_ids = []

            for obj in tracked_objects:
                track_id = obj['id']
                bbox = obj['bbox']
                active_ids.append(track_id)
                color = get_track_color(track_id)
                
                x1, y1, x2, y2 = map(lambda v: max(0, int(v)), bbox)
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                track_history[track_id].append((center_x, center_y))
                
                if len(track_history[track_id]) > max_history_len:
                    track_history[track_id].pop(0)

                points = track_history[track_id]
                for i in range(1, len(points)):
                    thickness = int(np.sqrt(64 / float(len(points) - i + 1)) * 1.5)
                    cv2.line(frame, points[i - 1], points[i], color, thickness)
                
                cv2.circle(frame, (center_x, center_y), 4, color, -1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if 'boxes' in target and len(target['boxes']) > 0:
                gt_boxes = target['boxes'].cpu().numpy()
                gt_ids = target.get('track_id', torch.zeros(len(gt_boxes))).cpu().numpy()
                
                for gt_box, gt_id in zip(gt_boxes, gt_ids):
                    gx1, gy1, gx2, gy2 = map(lambda v: max(0, int(v)), gt_box)
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    gt_label = f"GT: {int(gt_id)}"
                    (w, h), _ = cv2.getTextSize(gt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (gx1, gy1 - h - 10), (gx1 + w, gy1), (0, 255, 0), -1)
                    cv2.putText(frame, gt_label, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            for tid in list(track_history.keys()):
                if tid not in active_ids:
                    track_history[tid].pop(0)
                    if len(track_history[tid]) == 0:
                        del track_history[tid]

            out.write(frame)

    out.release()
    print(f"Tracking complete! Video saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 2.1 and 2.2 Tracking")
    parser.add_argument("--xml_path", type=str, default="data/gt/ai_challenge_s03_c010-full_annotation.xml")
    
    parser.add_argument("--tracker", type=str, choices=['iou', 'kalman', 'neuflow'], default='iou')
    
    parser.add_argument("--video_path", type=str, default="data/AICity_data/AICity_data/train/S03/c010/vdo.avi")
    parser.add_argument("--model_path", type=str, default="models/fine_tuned_rcnn.pth")
    parser.add_argument("--output_video", type=str, default="")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--max_age", type=int, default=1)
    
    parser.add_argument("--proc_noise", type=float, default=1.0)
    parser.add_argument("--meas_noise", type=float, default=1.0)
    
    parser.add_argument("--alpha", type=float, default=0.5)
    
    parser.add_argument("--easy_task", type=str, help='If set, will exclude occluded and parked cars from tracking')
    
    args = parser.parse_args()
    main(args)