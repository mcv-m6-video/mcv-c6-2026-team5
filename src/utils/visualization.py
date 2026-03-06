import cv2
import numpy as np
from pathlib import Path
# from src.optical_flow.state_of_art_estimators import compute_neuflow

def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def create_flow_video(image_dir: str, output_path: str, fps: int = 10):
    image_paths = sorted(list(Path(image_dir).glob("*.png"))) # Adjust extension if needed
    
    if len(image_paths) < 2:
        print("Not enough images to compute flow.")
        return

    first_frame = cv2.imread(str(image_paths[0]))
    h, w = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # model = initialize_neuflow(device="cuda")
    
    for i in range(len(image_paths) - 1):
        img1 = cv2.imread(str(image_paths[i]))
        img2 = cv2.imread(str(image_paths[i+1]))
        
        # flow = compute_neuflow(model, img1, img2, device="cuda")
        
        # flow_bgr = flow_to_hsv(flow)
        # out.write(flow_bgr)
        
    out.release()