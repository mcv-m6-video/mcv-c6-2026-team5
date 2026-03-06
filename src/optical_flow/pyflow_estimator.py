import numpy as np
import pyflow

def compute_pyflow(img1: np.ndarray, img2: np.ndarray, mode: str = "default") -> np.ndarray:
    img1_float = np.ascontiguousarray(img1.astype(float) / 255.0)
    img2_float = np.ascontiguousarray(img2.astype(float) / 255.0)
    
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    colType = 1 
    
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

    u, v, _ = pyflow.calcOpticalFlow(
        img1_float, img2_float, alpha, ratio, minWidth, 
        nOuterFPIterations, nInnerFPIterations, nSORIterations, colType
    )
    
    flow = np.dstack((u, v))
    return flow