import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import cv2
import numpy as np
class PredictionHead(nn.Module):
    """
    3x3 conv -> activation -> 1x1 conv
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class BEVDetectionHeads(nn.Module):
    """
    Heads for:
      - center heatmap: (B, 1, Hg, Wg)
      - offset map:     (B, 2, Hg, Wg)
    """
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()

        self.center_head = PredictionHead(
            in_channels=in_channels,
            out_channels=1,
            hidden_channels=hidden_channels,
        )

        self.offset_head = PredictionHead(
            in_channels=in_channels,
            out_channels=2,   # dx, dy
            hidden_channels=hidden_channels,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        center_logits = self.center_head(x)     # (B,1,Hg,Wg)
        offsets = self.offset_head(x)           # (B,2,Hg,Wg)

        return {
            "center_logits": center_logits,
            "center_prob": torch.sigmoid(center_logits),
            "offsets": offsets,
        }
    
class ReIDHead(nn.Module):
    """
    Dense ReID embedding map.
    """
    def __init__(self, in_channels: int, embedding_dim: int = 64, hidden_channels: int = 128):
        super().__init__()
        self.embed_head = PredictionHead(
            in_channels=in_channels,
            out_channels=embedding_dim,
            hidden_channels=hidden_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed_head(x)  # (B, Cid, H, W)
        emb = F.normalize(emb, dim=1)
        return emb

class IdentityClassifier(nn.Module):
    """
    Classification over sampled embeddings.
    """
    def __init__(self, embedding_dim: int, num_ids: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, embedding_dim)
        return self.fc(x)
    
def sample_embeddings_at_indices(
    emb_map: torch.Tensor,   # (B, C, H, W)
    indices: torch.Tensor,   # (N, 3): [batch_idx, y, x]
) -> torch.Tensor:
    """
    Sample embedding vectors from an embedding map.

    Returns:
        sampled: (N, C)
    """
    b = indices[:, 0]
    y = indices[:, 1]
    x = indices[:, 2]

    sampled = emb_map[b, :, y, x]   # (N, C)
    return sampled

class BEVPredictionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reid_dim: int = 64,
        hidden_channels: int = 128,
    ):
        super().__init__()

        self.det_heads = BEVDetectionHeads(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        )

        self.reid_head = ReIDHead(
            in_channels=in_channels,
            embedding_dim=reid_dim,
            hidden_channels=hidden_channels,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.det_heads(x)
        out["reid_embedding"] = self.reid_head(x)
        return out


import torch
import torch.nn.functional as F


def decode_bev_centers(center_prob, offsets, score_thresh=0.7, max_det=100):
    """
    center_prob: (B, 1, H, W)
    offsets:     (B, 2, H, W)

    Returns:
        detections: list of tensors, one per batch item
                    each tensor is (N, 3) with [x, y, score]
                    in BEV grid coordinates (float, after offset)
    """
    B, _, H, W = center_prob.shape

    # local maxima via max pooling
    pooled = F.max_pool2d(center_prob, kernel_size=3, stride=1, padding=1)
    keep = (center_prob == pooled) & (center_prob >= score_thresh)

    detections = []

    for b in range(B):
        ys, xs = torch.where(keep[b, 0])
        scores = center_prob[b, 0, ys, xs]

        if len(scores) == 0:
            detections.append(torch.empty((0, 3), device=center_prob.device))
            continue

        # sort by score
        scores, order = torch.sort(scores, descending=True)
        order = order[:max_det]

        ys = ys[order]
        xs = xs[order]
        scores = scores[:max_det]

        dx = offsets[b, 0, ys, xs]
        dy = offsets[b, 1, ys, xs]

        x = xs.float() + dx
        y = ys.float() + dy

        det = torch.stack([x, y, scores], dim=1)  # (N, 3)
        detections.append(det)

    return detections


def visualize_bev_detections(center_prob, detections, title="BEV detections"):
    """
    center_prob: (1,1,H,W)
    detections: output of decode_bev_centers(...)[0]
    """
    heatmap = center_prob[0, 0].detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar()

    if detections.shape[0] > 0:
        x = detections[:, 0].detach().cpu().numpy()
        y = detections[:, 1].detach().cpu().numpy()
        plt.scatter(x, y, s=30, c="cyan", marker="o")

    plt.title(title)
    plt.show()

def bev_grid_to_local_xy(detections, x_min, x_max, y_min, y_max, Wg, Hg):
    """
    detections: (N, 3) [x_grid, y_grid, score]
    returns:    (N, 3) [x_local_m, y_local_m, score]
    """
    if detections.shape[0] == 0:
        return detections

    x_grid = detections[:, 0]
    y_grid = detections[:, 1]
    score = detections[:, 2]

    x_local = x_min + x_grid * ((x_max - x_min) / Wg)
    y_local = y_min + y_grid * ((y_max - y_min) / Hg)

    return torch.stack([x_local, y_local, score], dim=1)


def draw_points_on_image(image_tensor, image_dets, radius=6):
    """
    image_tensor: (3,H,W), RGB, [0,1]
    image_dets: (N,3) [u, v, score]
    """
    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8).copy()

    H, W = img.shape[:2]

    for det in image_dets.detach().cpu().numpy():
        u, v, score = det
        u = int(round(u))
        v = int(round(v))

        if 0 <= u < W and 0 <= v < H:
            cv2.circle(img, (u, v), radius, (0, 255, 255), 2)

    plt.figure(figsize=(12, 7))
    plt.imshow(img)
    plt.title("Projected detections on image")
    plt.axis("off")
    plt.show()


