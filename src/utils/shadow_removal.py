import cv2
import numpy as np
import torch

def remove_shadows(frame_tensor, bg_mean_tensor, fg_mask_tensor, method="hsv", device='cuda', **kwargs):
    """
    Generic Shadow Removal Dispatcher.
    
    Args:
        method (str): "hsv" (Background Comparison) or "lab" (Luminance/Saturation Thresholding)
        **kwargs: Arguments specific to the chosen method.
    """
    # Common: Prepare data on CPU/Numpy for OpenCV
    def to_cv_img(tensor):
        if tensor is None: return None
        img = tensor.detach().cpu().numpy().astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3: 
            img = img.transpose(1, 2, 0) # CHW -> HWC
        return img 

    fg_mask_np = fg_mask_tensor.detach().cpu().numpy().astype(np.uint8) * 255
    frame_np = to_cv_img(frame_tensor)
    
    # Check if we have color data
    if frame_np.ndim < 3 or frame_np.shape[2] != 3:
        return fg_mask_tensor

    # --- DISPATCH METHOD ---
    if method == "hsv":
        bg_np = to_cv_img(bg_mean_tensor)
        is_shadow = _detect_shadows_hsv(frame_np, bg_np, **kwargs)
    elif method == "lab":
        is_shadow = _detect_shadows_lab(frame_np, **kwargs)
    else:
        raise ValueError(f"Unknown shadow removal method: {method}")

    # Clean Mask (Set shadow pixels to 0)
    cleaned_np = fg_mask_np.copy()
    cleaned_np[is_shadow] = 0 
    
    return torch.from_numpy(cleaned_np).to(device) > 128


# based on https://ieeexplore.ieee.org/document/1233909
def _detect_shadows_hsv(frame_bgr, bg_bgr, alpha=0.4, beta=0.99, tau_s=60, tau_h=40):
    # Convert BGR (OpenCV) to HSV
    hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2HSV) # Assuming TorchCodec gave RGB
    hsv_bg = cv2.cvtColor(bg_bgr, cv2.COLOR_RGB2HSV)

    # Split Channels
    h_i, s_i, v_i = cv2.split(hsv_frame)
    h_b, s_b, v_b = cv2.split(hsv_bg)

    # Ratio of Brightness
    v_ratio = v_i.astype(float) / (v_b.astype(float) + 1.0)
    
    mask_v = (v_ratio >= alpha) & (v_ratio <= beta)
    mask_s = np.abs(s_i.astype(float) - s_b.astype(float)) <= tau_s
    
    diff_h = np.abs(h_i.astype(float) - h_b.astype(float))
    diff_h = np.minimum(diff_h, 180 - diff_h)
    mask_h = diff_h <= tau_h
    
    return mask_v & mask_s & mask_h

# based on https://opencv.org/blog/shadow-correction-using-opencv/
def _detect_shadows_lab(frame_rgb, sensitivity=1.0): # Sensitivity ~0/2.0
    """
    Detects shadows based on Low Lightness (L) and Low Saturation (S).
    Does NOT require a background model.
    """
    # 1. Convert RGB to LAB to get Lightness
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    
    # 2. Convert RGB to HSV to get Saturation
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[:, :, 1] / 255.0 # Normalize to 0-1
    
    # 3. Logic from snippet: (L < 0.5 * sensitivity) & (S < 0.5)
    # Note: OpenCV L channel is 0-255.
    # We normalize L to 0-1 for the formula
    L_norm = L / 255.0
    
    shadow_cond = (L_norm < (0.5 * sensitivity)) & (S < 0.5)
    
    # Optional: Morphological cleanup (from snippet mask_blur logic)
    mask = shadow_cond.astype(np.uint8) * 255
    # Use a small Gaussian blur to match the snippet's "soft mask" feel
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask > 0