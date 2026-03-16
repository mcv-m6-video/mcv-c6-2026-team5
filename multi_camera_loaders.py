import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import torchvision.transforms.functional as TF
import numpy as np
from src.data.loader import AICityDataset
class MultiCameraDataset(Dataset):
    """
    Wraps multiple single-camera datasets and truncates to the shortest one.
    Returns a dict with all camera frames and homographies for a given timestep.
    """
    def __init__(self, datasets: list[AICityDataset], homographies: list[torch.Tensor]):
        assert len(datasets) == len(homographies)
        self.datasets     = datasets
        self.homographies = homographies
        # Truncate to shortest video
        self.length = min(len(d) for d in datasets)
        print(f"Video lengths: {[len(d) for d in datasets]}")
        print(f"Using {self.length} frames (truncated to shortest)")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        frames = torch.stack([d[idx] for d in self.datasets], dim=0)  # (S, 3, H, W)
        H_mats = torch.stack(self.homographies, dim=0)                 # (S, 3, 3)
        return frames, H_mats