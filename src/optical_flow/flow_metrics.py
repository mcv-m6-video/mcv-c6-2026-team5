import numpy as np
import cv2

def read_kitti_flow(flow_path: str) -> tuple[np.ndarray, np.ndarray]:
    flow_image = cv2.imread(flow_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    print(flow_image.shape)
    flow_image = flow_image[:, :, ::-1].astype(np.float32)
    
    flow_u = (flow_image[:, :, 0] - 2**15) / 64.0
    flow_v = (flow_image[:, :, 1] - 2**15) / 64.0
    valid_mask = flow_image[:, :, 2] > 0
    
    flow_gt = np.dstack((flow_u, flow_v))
    return flow_gt, valid_mask

def compute_msen(flow_gt: np.ndarray, flow_est: np.ndarray, valid_mask: np.ndarray) -> float:
    error = np.linalg.norm(flow_gt - flow_est, axis=2)
    msen = np.mean(error[valid_mask])
    return float(msen)

def compute_pepn(flow_gt: np.ndarray, flow_est: np.ndarray, valid_mask: np.ndarray, threshold: float = 3.0) -> float:
    error = np.linalg.norm(flow_gt - flow_est, axis=2)
    erroneous_pixels = np.sum((error > threshold) & valid_mask)
    total_valid_pixels = np.sum(valid_mask)
    
    pepn = (erroneous_pixels / total_valid_pixels) * 100
    return float(pepn)