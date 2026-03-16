import time
import cv2
import os
from pathlib import Path
import torch
import numpy as np

from src.optical_flow.flow_metrics import read_kitti_flow, compute_msen, compute_pepn
from src.utils.visualization import flow_to_hsv
from src.optical_flow.pyflow_estimator import compute_pyflow
from src.optical_flow.state_of_art_estimators import initialize_neuflow, compute_neuflow

def evaluate_optical_flow(img1_path: str, img2_path: str, gt_path: str, save_hsv: bool = False, output_dir: str = "results/optical_flow/"):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    image_height, image_width = img1.shape[:2]
    flow_gt, valid_mask = read_kitti_flow(gt_path)

    if save_hsv:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "gt_flow.png"), flow_to_hsv(flow_gt))

    results = {}

    # 1. PyFlow Evaluation

    modes = ["default", "fast"]
    for mode in modes:
        start_time = time.time()
        flow_est = compute_pyflow(img1, img2, mode=mode)
        runtime = time.time() - start_time
        
        method_name = f"PyFlow ({mode})"
        if save_hsv:
            original_plus_flow = np.vstack([img1, flow_to_hsv(flow_est)])
            cv2.imwrite(os.path.join(output_dir, f"pyflow_{mode}_hsv.png"), original_plus_flow)
            
        results[method_name] = {
            'MSEN': compute_msen(flow_gt, flow_est, valid_mask),
            'PEPN': compute_pepn(flow_gt, flow_est, valid_mask),
            'Runtime': runtime
        }

    # 2. NeuFlow v2 Evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_height, image_width = img1.shape[:2]
    half = False

    neuflow_model = initialize_neuflow(device=device, half=half)
    # Warmup for accurate GPU timing
    _ = compute_neuflow(neuflow_model, img1, img2, device=device, half=half)
    
    torch.cuda.synchronize()
    start_time = time.time()
    flow_neuflow = compute_neuflow(neuflow_model, img1, img2, device=device, half=half)
    torch.cuda.synchronize()
    neuflow_runtime = time.time() - start_time
    
    print("NaNs:", np.isnan(flow_neuflow).sum())
    print("Infs:", np.isinf(flow_neuflow).sum())
    print("min:", np.nanmin(flow_neuflow), "max:", np.nanmax(flow_neuflow))
    # flow_neuflow = np.nan_to_num(flow_neuflow, nan=0.0, posinf=0.0, neginf=0.0)
    print("flow min:", flow_neuflow.min())
    print("flow max:", flow_neuflow.max())
    print("flow mean:", np.abs(flow_neuflow).mean())
    
    gt_h, gt_w = flow_gt.shape[:2]
    
    pred_h, pred_w = flow_neuflow.shape[:2]

    flow_neuflow = cv2.resize(flow_neuflow, (gt_w, gt_h))

    flow_neuflow[:, :, 0] *= (gt_w / pred_w)
    flow_neuflow[:, :, 1] *= (gt_h / pred_h)

    if save_hsv:
        original_plus_flow = np.vstack([img1, flow_to_hsv(flow_neuflow)])
        cv2.imwrite(os.path.join(output_dir, "neuflow_hsv.png"), original_plus_flow)

    results['NeuFlow v2'] = {
        'MSEN': compute_msen(flow_gt, flow_neuflow, valid_mask),
        'PEPN': compute_pepn(flow_gt, flow_neuflow, valid_mask),
        'Runtime': neuflow_runtime
    }

    return results
if __name__ == "__main__":
    kitti_dir = Path("data/data_stereo_flow/training/")
    img1_path = str(kitti_dir / "colored_0/000045_10.png")
    img2_path = str(kitti_dir / "colored_0/000045_11.png")
    gt_path = str(kitti_dir / "flow_noc/000045_10.png")
    print(f"image1: {img1_path}\nimage2: {img2_path}\ngt_image: {gt_path}")

    metrics = evaluate_optical_flow(img1_path, img2_path, gt_path, save_hsv=True)
    
    for method, data in metrics.items():
        print(f"--- {method} ---")
        print(f"MSEN: {data['MSEN']:.4f}")
        print(f"PEPN: {data['PEPN']:.2f}%")
        print(f"Runtime: {data['Runtime']:.4f}s")