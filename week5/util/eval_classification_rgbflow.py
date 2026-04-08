#!/usr/bin/env python3
"""
Evaluation utilities for RGB + Optical Flow classification.
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def evaluate_rgbflow(model, dataset):
    """
    Evaluate RGB+Flow multi-label classification.

    Args:
        model: RGB+Flow model with model.predict(rgb, flow)
        dataset: dataset returning dict with:
            - 'frame': [T, 3, H, W]
            - 'flow':  [T, 2, H, W]
            - 'label': [C]

    Returns:
        ap_score: np.ndarray of shape [C]
    """
    all_labels = []
    all_scores = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating RGB+Flow"):
        sample = dataset[idx]

        rgb = sample['frame']
        flow = sample['flow']
        label = sample['label']

        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        pred = model.predict(rgb, flow)   # expected shape [1, C] or [C]

        pred = np.asarray(pred)
        if pred.ndim == 2:
            pred = pred[0]

        all_scores.append(pred)
        all_labels.append(label)

    all_scores = np.asarray(all_scores)
    all_labels = np.asarray(all_labels)

    ap_score = []
    for c in range(all_labels.shape[1]):
        y_true = all_labels[:, c]
        y_score = all_scores[:, c]

        # avoid crash if a class has no positives in this split
        if np.sum(y_true) == 0:
            ap = 0.0
        else:
            ap = average_precision_score(y_true, y_score)

        ap_score.append(ap)

    return np.asarray(ap_score, dtype=np.float32)