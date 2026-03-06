import time
import cv2
import os
from pathlib import Path

from src.optical_flow.flow_metrics import read_kitti_flow, compute_msen, compute_pepn
from src.utils.visualization import flow_to_hsv
from src.optical_flow.pyflow_estimator import compute_pyflow

def evaluate_optical_flow(img1_path: str, img2_path: str, gt_path: str, save_hsv: bool = False, output_dir: str = "results/optical_flow/"):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    flow_gt, valid_mask = read_kitti_flow(gt_path)

    if save_hsv:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "gt_flow.png"), flow_to_hsv(flow_gt))

    results = {}

    modes = ["default", "fast"]
    for mode in modes:
        start_time = time.time()
        flow_est = compute_pyflow(img1, img2, mode=mode)
        runtime = time.time() - start_time
        
        method_name = f"PyFlow ({mode})"
        if save_hsv:
            cv2.imwrite(os.path.join(output_dir, f"pyflow_{mode}_hsv.png"), flow_to_hsv(flow_est))
            
        results[method_name] = {
            'MSEN': compute_msen(flow_gt, flow_est, valid_mask),
            'PEPN': compute_pepn(flow_gt, flow_est, valid_mask),
            'Runtime': runtime
        }

    return results

if __name__ == "__main__":
    kitti_dir = Path("data/kitti")
    img1_path = str(kitti_dir / "000045_10.png")
    img2_path = str(kitti_dir / "000045_11.png")
    gt_path = str(kitti_dir / "gt_000045_10.png")

    metrics = evaluate_optical_flow(img1_path, img2_path, gt_path, save_hsv=True)
    
    for method, data in metrics.items():
        print(f"--- {method} ---")
        print(f"MSEN: {data['MSEN']:.4f}")
        print(f"PEPN: {data['PEPN']:.2f}%")
        print(f"Runtime: {data['Runtime']:.4f}s")