# import cv2
# import numpy as np
# import torch

# def remove_shadows(frame_tensor, bg_mean_tensor, fg_mask_tensor, method="hsv", device='cuda', **kwargs):
#     """
#     Generic Shadow Removal Dispatcher.
    
#     Args:
#         method (str): "hsv" (Background Comparison) or "lab" (Luminance/Saturation Thresholding)
#         **kwargs: Arguments specific to the chosen method.
#     """
#     # Common: Prepare data on CPU/Numpy for OpenCV
#     def to_cv_img(tensor):
#         if tensor is None: return None
#         img = tensor.detach().cpu().numpy().astype(np.uint8)
#         if img.ndim == 3 and img.shape[0] == 3: 
#             img = img.transpose(1, 2, 0) # CHW -> HWC
#         return img 

#     fg_mask_np = fg_mask_tensor.detach().cpu().numpy().astype(np.uint8) * 255
#     frame_np = to_cv_img(frame_tensor)
    
#     # Check if we have color data
#     if frame_np.ndim < 3 or frame_np.shape[2] != 3:
#         return fg_mask_tensor

#     # --- DISPATCH METHOD ---
#     if method == "hsv":
#         bg_np = to_cv_img(bg_mean_tensor)
#         is_shadow = _detect_shadows_hsv(frame_np, bg_np, **kwargs)
#     elif method == "lab":
#         is_shadow = _detect_shadows_lab(frame_np, **kwargs)
#     else:
#         raise ValueError(f"Unknown shadow removal method: {method}")

#     # Clean Mask (Set shadow pixels to 0)
#     cleaned_np = fg_mask_np.copy()
#     cleaned_np[is_shadow] = 0 
    
#     return torch.from_numpy(cleaned_np).to(device) > 128


# # based on https://ieeexplore.ieee.org/document/1233909
# def _detect_shadows_hsv(frame_bgr, bg_bgr, alpha=0.1, beta=3, tau_s=60, tau_h=40):
#     # Convert BGR (OpenCV) to HSV
#     hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2HSV) # Assuming TorchCodec gave RGB
#     hsv_bg = cv2.cvtColor(bg_bgr, cv2.COLOR_RGB2HSV)

#     # Split Channels
#     h_i, s_i, v_i = cv2.split(hsv_frame)
#     h_b, s_b, v_b = cv2.split(hsv_bg)

#     # Ratio of Brightness
#     v_ratio = v_i.astype(float) / (v_b.astype(float) + 1.0)
    
#     mask_v = (v_ratio >= alpha) & (v_ratio <= beta)
#     mask_s = np.abs(s_i.astype(float) - s_b.astype(float)) <= tau_s
    
#     diff_h = np.abs(h_i.astype(float) - h_b.astype(float))
#     diff_h = np.minimum(diff_h, 180 - diff_h)
#     mask_h = diff_h <= tau_h
    
#     return mask_v & mask_s & mask_h

# # based on https://opencv.org/blog/shadow-correction-using-opencv/
# def _detect_shadows_lab(frame_rgb, sensitivity=1.0): # Sensitivity ~0/2.0
#     """
#     Detects shadows based on Low Lightness (L) and Low Saturation (S).
#     Does NOT require a background model.
#     """
#     # 1. Convert RGB to LAB to get Lightness
#     lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
#     L, A, B = cv2.split(lab)
    
#     # 2. Convert RGB to HSV to get Saturation
#     hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
#     S = hsv[:, :, 1] / 255.0 # Normalize to 0-1
    
#     # 3. Logic from snippet: (L < 0.5 * sensitivity) & (S < 0.5)
#     # Note: OpenCV L channel is 0-255.
#     # We normalize L to 0-1 for the formula
#     L_norm = L / 255.0
    
#     shadow_cond = (L_norm < (0.5 * sensitivity)) & (S < 0.5)
    
#     # Optional: Morphological cleanup (from snippet mask_blur logic)
#     mask = shadow_cond.astype(np.uint8) * 255
#     # Use a small Gaussian blur to match the snippet's "soft mask" feel
#     mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
#     return mask > 0

import torch
import math

def rgb_to_hsv_torch(rgb_tensor):
    """
    Convert RGB Tensor (B, C, H, W) or (C, H, W) to HSV.
    Range: RGB [0, 1] -> HSV [0, 1] (H is 0-1, not 0-360)
    """
    if rgb_tensor.ndim == 3:
        rgb_tensor = rgb_tensor.unsqueeze(0) # Add batch dim for stability
        
    r, g, b = rgb_tensor[:, 0, ...], rgb_tensor[:, 1, ...], rgb_tensor[:, 2, ...]

    max_val, _ = torch.max(rgb_tensor, dim=1)
    min_val, _ = torch.min(rgb_tensor, dim=1)
    diff = max_val - min_val

    # EPS to avoid division by zero
    eps = 1e-6
    
    # 1. Value
    v = max_val

    # 2. Saturation
    s = torch.where(max_val > eps, diff / (max_val + eps), torch.zeros_like(max_val))

    # 3. Hue
    # H calculation depends on which channel is max
    mask_r = (max_val == r)
    mask_g = (max_val == g)
    mask_b = (max_val == b)

    h = torch.zeros_like(max_val)
    
    # If R is max: (G - B) / diff
    h[mask_r] = (g[mask_r] - b[mask_r]) / (diff[mask_r] + eps)
    # If G is max: 2.0 + (B - R) / diff
    h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / (diff[mask_g] + eps)
    # If B is max: 4.0 + (R - G) / diff
    h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / (diff[mask_b] + eps)

    # Normalize H to [0, 1] range (OpenCV uses 0-180, we use 0-1)
    h = (h / 6.0) % 1.0

    return torch.stack([h, s, v], dim=1).squeeze(0)

def remove_shadows(frame_tensor, bg_mean_tensor, fg_mask_tensor, 
                   alpha=0.4, beta=0.99, tau_s=60, tau_h=40, 
                   method="hsv", device='cuda', **kwargs):
    """
    GPU-Accelerated Shadow Removal.
    All inputs must be Tensors on the same device.
    """
    # Ensure inputs are normalized [0, 1] for the math
    # Frame and BG are likely 0-255 or 0-1. Let's force 0-1.
    if frame_tensor.max() > 1.0: frame_tensor = frame_tensor / 255.0
    if bg_mean_tensor.max() > 1.0: bg_mean_tensor = bg_mean_tensor / 255.0
    
    # 1. Convert to HSV (Stay on GPU!)
    hsv_frame = rgb_to_hsv_torch(frame_tensor)
    hsv_bg = rgb_to_hsv_torch(bg_mean_tensor)

    # Extract channels (H, S, V are all 0-1 now)
    h_i, s_i, v_i = hsv_frame[0], hsv_frame[1], hsv_frame[2]
    h_b, s_b, v_b = hsv_bg[0], hsv_bg[1], hsv_bg[2]

    # 2. Calculate Masks (Vectorized GPU Ops)
    
    # Value Ratio: V_curr / V_bg
    # alpha <= ratio <= beta
    v_ratio = v_i / (v_b + 1e-6)
    mask_v = (v_ratio >= alpha) & (v_ratio <= beta)

    # Saturation Difference
    # We need to scale tau_s to [0, 1] range because OpenCV uses 0-255
    # tau_s = 60 (OpenCV) -> 60/255.0 approx 0.23
    tau_s_norm = tau_s /255.0
    diff_s = torch.abs(s_i - s_b)
    mask_s = diff_s <= tau_s_norm

    # Hue Difference
    # Hue is circular [0, 1]. OpenCV tau_h=40 (0-180) -> 40/180 approx 0.22
    tau_h_norm = tau_h/180.0
    diff_h = torch.abs(h_i - h_b)
    diff_h = torch.min(diff_h, 1.0 - diff_h) # Wrap around
    mask_h = diff_h <= tau_h_norm

    # 3. Combine
    is_shadow = mask_v & mask_s & mask_h

    # 4. Clean Mask
    # fg_mask_tensor is Boolean or Float.
    # Set pixels to False/0 where is_shadow is True
    cleaned_mask = fg_mask_tensor.clone()
    
    # If Boolean
    if cleaned_mask.dtype == torch.bool:
        cleaned_mask[is_shadow] = False
    else:
        cleaned_mask[is_shadow] = 0

    return cleaned_mask