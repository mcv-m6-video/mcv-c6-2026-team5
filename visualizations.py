import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def visualize_features(feat: torch.Tensor, title: str = "Feature map", n_channels: int = 8):
    """
    Visualize a feature map tensor with shape (1, C, H, W) or (C, H, W).
    Shows: mean activation, max activation, and first n_channels individual maps.
    """
    if feat.dim() == 4:
        feat = feat[0]           # (C, H, W)
    feat = feat.detach().cpu()

    mean_map = feat.mean(dim=0).numpy()
    max_map  = feat.max(dim=0).values.numpy()
    channels = feat[:n_channels].numpy()   # (n, H, W)

    total = 2 + n_channels
    fig, axes = plt.subplots(2, (total + 1) // 2, figsize=(14, 5))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="medium")

    for ax in axes:
        ax.axis("off")

    def show(ax, data, t, cmap="inferno"):
        ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(t, fontsize=9)
        ax.axis("off")

    show(axes[0], mean_map, "Mean over channels")
    show(axes[1], max_map,  "Max over channels")

    for i, ch in enumerate(channels):
        show(axes[2 + i], ch, f"Channel {i}")

    plt.tight_layout()
    plt.show()


def visualize_bev(bev: torch.Tensor, title: str = "BEV projection", n_channels: int = 8):
    """Same as visualize_features but labelled for BEV space."""
    visualize_features(bev, title=title, n_channels=n_channels)


def visualize_side_by_side(feat: torch.Tensor, bev: torch.Tensor):
    """
    One figure: left = camera feature mean, right = BEV feature mean.
    Quick sanity check that projection worked.
    """
    feat = feat[0].detach().cpu() if feat.dim() == 4 else feat.detach().cpu()
    bev  = bev[0].detach().cpu()  if bev.dim()  == 4 else bev.detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Camera features  →  BEV projection", fontsize=12)

    axes[0].imshow(feat.mean(0).numpy(), cmap="inferno", aspect="auto")
    axes[0].set_title(f"Camera feature mean  {tuple(feat.shape)}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(bev.mean(0).numpy(), cmap="inferno", aspect="auto")
    axes[1].set_title(f"BEV feature mean  {tuple(bev.shape)}", fontsize=9)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()