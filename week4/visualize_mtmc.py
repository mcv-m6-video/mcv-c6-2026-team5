import cv2
import os
import argparse
from collections import defaultdict, deque
import numpy as np

def load_mot_data(file_path):
    """
    Loads MOT format data.
    Returns a dict: data_by_frame[frame_num] = [(id, x1, y1, x2, y2)]
    """
    data_by_frame = defaultdict(list)
    if not os.path.exists(file_path):
        return data_by_frame
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            data_by_frame[frame].append((obj_id, int(x), int(y), int(x + w), int(y + h)))
    return data_by_frame

def main(args):
    print(f"Loading Video: {args.video_path}")
    cap = cv2.VideoCapture(args.video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare Video Writer
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    # Load Data
    print("Loading Ground Truth and Predictions...")
    gt_data = load_mot_data(args.gt_path)
    pred_data = load_mot_data(args.pred_path)
    
    # Dictionary to store the trajectory history: {global_id: deque([(cx, cy), ...])}
    trajectories = defaultdict(lambda: deque(maxlen=args.track_length))

    frame_idx = 1 # MOT format is 1-based indexing
    
    print("Rendering Video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        active_ids = []

        # 1. Draw Ground Truth (Green, Thick)
        if args.show_gt and frame_idx in gt_data:
            for _, gx1, gy1, gx2, gy2 in gt_data[frame_idx]:
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
                gt_label = "GT"
                (w, h), _ = cv2.getTextSize(gt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)
                cv2.rectangle(frame, (gx1, gy1 - h - 10), (gx1 + w, gy1), (0, 255, 0), -1)
                cv2.putText(frame, gt_label, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

        # 2. Draw Predictions and Trajectories
        if frame_idx in pred_data:
            for global_id, x1, y1, x2, y2 in pred_data[frame_idx]:
                active_ids.append(global_id)
                color = (0, 0, 255) # BGR Plain Red
                
                # Calculate Center Point for Trajectory
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trajectories[global_id].append((cx, cy))
                
                # Draw Trajectory Tail dynamically
                points = list(trajectories[global_id])
                for i in range(1, len(points)):
                    thickness = int(np.sqrt(64 / float(len(points) - i + 1)) * 2.0)
                    cv2.line(frame, points[i - 1], points[i], color, thickness)
                cv2.circle(frame, (cx, cy), 5, color, -1)

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw Label with solid background
                label = f"ID: {global_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
                
        # 3. Cleanup logic: Fade out missing tracks
        for tid in list(trajectories.keys()):
            if tid not in active_ids:
                # Car is no longer on screen, pop oldest point to create a fading effect
                if len(trajectories[tid]) > 0:
                    trajectories[tid].popleft() 
                    
                    # Keep rendering the fading tail until it's completely gone
                    points = list(trajectories[tid])
                    color = (0, 0, 255)
                    for i in range(1, len(points)):
                        thickness = int(np.sqrt(64 / float(len(points) - i + 1)) * 2.0)
                        cv2.line(frame, points[i - 1], points[i], color, thickness)
                        
                if len(trajectories[tid]) == 0:
                    del trajectories[tid]

        # Write frame to video
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Video saved successfully to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MTMC Tracking Results")
    parser.add_argument("--video_path", required=True, help="Path to the original vdo.avi")
    parser.add_argument("--gt_path", required=True, help="Path to the gt.txt file")
    parser.add_argument("--pred_path", required=True, help="Path to your exported mot_results/cam_id.txt")
    parser.add_argument("--output_path", default="results/mtmc/visualizations/output.mp4", help="Path to save the output video")
    parser.add_argument("--track_length", type=int, default=50, help="How many frames of history to show for the trajectory line")
    parser.add_argument("--show_gt", action="store_true", help="Flag to enable rendering of Ground Truth boxes")
    
    args = parser.parse_args()
    main(args)