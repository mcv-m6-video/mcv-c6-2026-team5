import numpy as np
import pyflow

def compute_pyflow(img1: np.ndarray, img2: np.ndarray, mode: str = "default") -> np.ndarray:
    if len(img1.shape) == 2:
        img1 = np.stack([img1]*3, axis=-1)
        img2 = np.stack([img2]*3, axis=-1)
    img1 = img1.astype(np.float64) + np.random.normal(0, 1e-6, img1.shape)
    img2 = img2.astype(np.float64) + np.random.normal(0, 1e-6, img2.shape)
    
    img1_float = np.ascontiguousarray(img1.astype(float) / 255.0)
    img2_float = np.ascontiguousarray(img2.astype(float) / 255.0)
    # print(f"unique img1 values: {np.unique(img1_float)}")

    
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    colType = 0  # 0: RGB, 1: Gray, 2: Gradient
    
    if mode == "default":
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
    elif mode == "fast":
        nOuterFPIterations = 3
        nInnerFPIterations = 1
        nSORIterations = 10
    else:
        raise ValueError("Mode must be 'default' or 'fast'.")

    u, v, _ = pyflow.coarse2fine_flow(
        img1_float, img2_float, alpha, ratio, minWidth, 
        nOuterFPIterations, nInnerFPIterations, nSORIterations, colType
    )
    
    flow = np.dstack((u, v))
    return flow