import torch
from .base import BackgroundModel # Assuming you created the abstract base class
from src.utils.shadow_removal import remove_shadows
import numpy

class SingleGaussian(BackgroundModel):
    def __init__(self, alpha: float, device: str = 'cuda'):
        self.alpha = alpha
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mean_gray = None
        self.mean_rgb = None
        self.std = None

    def fit(self, video_decoder, num_train_frames: int):
        """
        Compute Mean and Variance (Std) from the first N frames.
        Uses torchcodec decoder to stream frames efficiently.
        """
        print(f"Training Single Gaussian on {num_train_frames} frames...")
        
        # Method 1: Iterative (Low Memory Usage)
        # We calculate running sum and sum_squares to avoid loading all frames into RAM
        sum_x = None
        sum_x2 = None
        sum_rgb = None
        
        for i in range(num_train_frames):
            # Load frame, convert to float, normalize to [0, 255] range logic (keep as float 0-255 for formula)
            frame = video_decoder[i].to(self.device).float()
            
            # Squeeze batch dim if present (C, H, W)
            if frame.ndim == 4: frame = frame.squeeze(0)
            
            if sum_rgb is None:
                sum_rgb = torch.zeros_like(frame)
            sum_rgb += frame
            
            # Grayscale conversion (standard for this specific assignment)
            # 0.299R + 0.587G + 0.114B
            if frame.shape[0] == 3:
                frame = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            
            if sum_x is None:
                sum_x = torch.zeros_like(frame)
                sum_x2 = torch.zeros_like(frame)

            sum_x += frame
            sum_x2 += frame ** 2

        # Calculate Mean and Std
        self.mean_gray = sum_x / num_train_frames
        self.mean_rgb = sum_rgb / num_train_frames
        
        variance = (sum_x2 / num_train_frames) - (self.mean_gray ** 2)
        
        # Clamp variance to avoid negative values due to float precision
        variance = torch.clamp(variance, min=0)
        self.std = torch.sqrt(variance)
        
        print("Training complete.")

    def apply(self, frame: torch.Tensor, shadow_method: str = "hsv", 
              shadow_params: dict = None, 
              detection_mode: str = "gray",   # "gray" or "rgb"
             )-> torch.Tensor:        
        """
        Apply foreground detection: |I - u| >= alpha * (std + 2)
        Input: Tensor (C, H, W) or (H, W) in range [0, 255]
        Output: Binary Mask (H, W) boolean
        Shadow method hsv or lab
        """
        if self.mean_gray is None:
            raise RuntimeError("Model not trained. Run .fit() first.")

        # Ensure frame is on the correct device and is float
        frame = frame.to(self.device).float()
        
        if frame.ndim == 4: frame = frame.squeeze(0)
        if detection_mode == "rgb":
            # RGB Detection: Foreground if ANY channel deviates enough
            # We use mean_rgb (3, H, W) instead of mean_gray
            diff = torch.abs(frame - self.mean_rgb) 
            # Threshold needs to adapt to 3 channels. 
            # We approximate std for RGB simply or reuse the gray std broadcasted 
            # (A proper RGB Mahalanobis distance is better but expensive; this is a fast approximation)
            threshold = self.alpha * (self.std + 2) 
            # If any channel (R, G, or B) exceeds threshold, it's FG
            fg_mask = torch.any(diff >= threshold, dim=0)
            
        else:
            # Convert to grayscale if needed
            if frame.shape[0] == 3:
                gray_frame = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]

            # The Formula from Slides: |I_i - mu_i| >= alpha * (sigma_i + 2)
            diff = torch.abs(gray_frame - self.mean_gray)
            threshold = self.alpha * (self.std + 2)
            
            # Create binary mask (True = Foreground)
            fg_mask = diff >= threshold

        if shadow_method != "none":
            if shadow_method == "hsv":
                method_kwargs = {
                    "alpha": 0.5,
                    "beta": 0.5,
                    "tau_s": 60,
                    "tau_h": 40
                }
            elif shadow_method == "lab":
                method_kwargs = {
                    "sensitivity": 0.95
                }
            else:
                raise ValueError(f"Unknown shadow removal method: {shadow_method} (none to disable)")
            
            fg_mask = remove_shadows(
                frame_tensor=frame,
                bg_mean_tensor=self.mean_rgb,
                fg_mask_tensor=fg_mask,
                method=shadow_method,
                device=self.device,
                **method_kwargs
            )

        return fg_mask


class RecursiveGaussian(BackgroundModel):
    def __init__(self, alpha: float, rho: float, device: str = 'cuda'):
        """
        Args:
            alpha: Threshold multiplier for detection.
            rho: Learning rate (0 < rho < 1). Controls how fast bg adapts.
        """
        self.alpha = alpha
        self.rho = rho
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mean_gray = None
        self.mean_rgb = None
        self.std = None        

    def fit(self, video_decoder, num_train_frames: int):
        """
        Initialization: Compute initial Mean and Std from the first N frames.
        Same as Task 1.1 (Non-Adaptive initialization).
        """
        print(f"Initializing Recursive Gaussian on {num_train_frames} frames...")
        
        sum_x = None
        sum_x2 = None
        sum_rgb = None
        
        for i in range(num_train_frames):
            frame = video_decoder[i].to(self.device).float()
            
            
            if sum_rgb is None:
                sum_rgb = torch.zeros_like(frame)
            sum_rgb += frame
            
            # Grayscale conversion (if needed)
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            elif frame.ndim == 4: # Handle batch dimension
                frame = frame.squeeze(0)
                if frame.shape[0] == 3:
                    frame = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            
            if sum_x is None:
                sum_x = torch.zeros_like(frame)
                sum_x2 = torch.zeros_like(frame)
                

            sum_x += frame
            sum_x2 += frame ** 2

        self.mean_gray = sum_x / num_train_frames
        self.mean_rgb = sum_rgb / num_train_frames
        variance = (sum_x2 / num_train_frames) - (self.mean_gray ** 2)
        variance = torch.clamp(variance, min=0)
        self.std = torch.sqrt(variance)
        
        print("Initialization complete.")

    def apply(self, frame: torch.Tensor, 
              shadow_method: str = "hsv", 
              shadow_params: dict = None, 
              detection_mode: str = "gray",   # "gray" or "rgb"
              update_buffer: int = 0 )-> torch.Tensor:         # 0, 1, 2... dilation iterations) 
        """
        1. Predict Foreground.
        2. Update Background Model (Mean & Variance) for BG pixels only.
        Shadow method hsv or lab
        """
        if self.mean_gray is None:
            raise RuntimeError("Model not initialized. Run .fit() first.")
        
        if shadow_params is None:
            shadow_params = {}  

        frame = frame.to(self.device).float()
        if frame.ndim == 4: frame = frame.squeeze(0)
        # Prepare Frame
        if detection_mode == "rgb":
            # RGB Detection: Foreground if ANY channel deviates enough
            # We use mean_rgb (3, H, W) instead of mean_gray
            diff = torch.abs(frame - self.mean_rgb) 
            # Threshold needs to adapt to 3 channels. 
            # We approximate std for RGB simply or reuse the gray std broadcasted 
            # (A proper RGB Mahalanobis distance is better but expensive; this is a fast approximation)
            threshold = self.alpha * (self.std + 2) 
            # If any channel (R, G, or B) exceeds threshold, it's FG
            fg_mask = torch.any(diff >= threshold, dim=0)
            
        else:
            #detect grayscale if needed
            if frame.ndim == 3 and frame.shape[0] == 3:
                gray_frame = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            else:
                gray_frame = frame

            # --- STEP 1: PREDICT (Foreground Detection) ---
            diff = torch.abs(gray_frame - self.mean_gray)
            # Formula: |I - mu| >= alpha * (sigma + 2) [cite: 273]
            threshold = self.alpha * (self.std + 2)
            fg_mask = diff >= threshold
        
        # --- STEP 2: UPDATE (Recursive Adaptation) ---
        # "if pixel in Background then update" 
        # bg_mask is True where the pixel is Background
        
        if shadow_method != "none":
            # if shadow_method == "hsv":
            #     method_kwargs = {
            #         "alpha": 0.4,
            #         "beta": 0.9,
            #         "tau_s": 25,
            #         "tau_h": 90
            #     }
            # elif shadow_method == "lab":
            #     method_kwargs = {
            #         "sensitivity": 0.95
            #     }
            # else:
            #     raise ValueError(f"Unknown shadow removal method: {shadow_method} (none to disable)")
            updated_fg_mask = remove_shadows(
                frame_tensor=frame,
                bg_mean_tensor=self.mean_rgb,
                fg_mask_tensor=fg_mask,
                method=shadow_method,
                device=self.device,
                **shadow_params
            )
            fg_mask = updated_fg_mask
        
        bg_mask = ~fg_mask
        
        if update_buffer > 0:
            # We use MaxPool as a fast GPU dilation
            # Kernel size 3=1 pixel dilation, 5=2 pixels, etc.
            k_size = 2 * update_buffer + 1
            fg_dilated = torch.nn.functional.max_pool2d(
                fg_mask.float().unsqueeze(0).unsqueeze(0), 
                kernel_size=k_size, stride=1, padding=update_buffer
            ).squeeze() > 0
            update_mask = ~fg_dilated
            bg_mask = update_mask

        if detection_mode == "rgb" and (frame.ndim == 3 and frame.shape[0] == 3):
             gray_frame = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        
        # We only update pixels where bg_mask is True.
        # Formula: mu_t = rho * I_t + (1 - rho) * mu_{t-1} 
        self.mean_gray[bg_mask] = (self.rho * gray_frame[bg_mask]) + \
                             ((1 - self.rho) * self.mean_gray[bg_mask])

        self.mean_rgb[:, bg_mask] = (self.rho * frame[:, bg_mask]) + \
                             ((1 - self.rho) * self.mean_rgb[:, bg_mask])

        # Formula: sigma^2_t = rho * (I_t - mu_t)^2 + (1 - rho) * sigma^2_{t-1} 
        # Note: We use the *new* mean for the variance calculation variance update
        current_variance = self.std[bg_mask] ** 2
        diff_sq = (gray_frame[bg_mask] - self.mean_gray[bg_mask]) ** 2
        
        new_variance = (self.rho * diff_sq) + \
                       ((1 - self.rho) * current_variance)
        
        self.std[bg_mask] = torch.sqrt(new_variance + 1e-6)  # Add small epsilon to avoid sqrt of zero

        return fg_mask