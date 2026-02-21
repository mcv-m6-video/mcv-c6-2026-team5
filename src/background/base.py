from abc import ABC, abstractmethod
import numpy as np

class BackgroundModel(ABC):
    @abstractmethod
    def fit(self, images: np.ndarray, gt_masks: np.ndarray = None):
        """
        Train the background model.
        images: List of frames or 3D numpy array.
        gt_masks: Optional ground truth (for supervised methods later).
        """
        pass

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Predict foreground mask for a single image.
        Returns: Binary mask (0=bg, 255=fg)
        """
        pass