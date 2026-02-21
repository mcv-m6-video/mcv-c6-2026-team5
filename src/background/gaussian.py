import torch
from .base import BackgroundModel # Assuming you created the abstract base class
import numpy

class SingleGaussian(BackgroundModel):
    def __init__(self, alpha: float, device: str = 'cuda'):
        self.alpha = alpha
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mean = None
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
        
        for i in range(num_train_frames):
            # Load frame, convert to float, normalize to [0, 255] range logic (keep as float 0-255 for formula)
            frame = video_decoder[i].to(self.device).float()
            
            # Squeeze batch dim if present (C, H, W)
            if frame.ndim == 4: frame = frame.squeeze(0)
            
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
        self.mean = sum_x / num_train_frames
        variance = (sum_x2 / num_train_frames) - (self.mean ** 2)
        
        # Clamp variance to avoid negative values due to float precision
        variance = torch.clamp(variance, min=0)
        self.std = torch.sqrt(variance)
        
        print("Training complete.")

    def apply(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Apply foreground detection: |I - u| >= alpha * (std + 2)
        Input: Tensor (C, H, W) or (H, W) in range [0, 255]
        Output: Binary Mask (H, W) boolean
        """
        if self.mean is None:
            raise RuntimeError("Model not trained. Run .fit() first.")

        # Ensure frame is on the correct device and is float
        frame = frame.to(self.device).float()
        
        if frame.ndim == 4: frame = frame.squeeze(0)
        
        # Convert to grayscale if needed
        if frame.shape[0] == 3:
             frame = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]

        # The Formula from Slides: |I_i - mu_i| >= alpha * (sigma_i + 2)
        diff = torch.abs(frame - self.mean)
        threshold = self.alpha * (self.std + 2)
        
        # Create binary mask (True = Foreground)
        mask = diff >= threshold
        return mask