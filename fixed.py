import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ConvNeXt import ConvNeXtFirstStage
from Decoder import BEVResNet18Decoder
from src.data.loader import AICityDataset
from Heads import BEVPredictionModule, decode_bev_centers, bev_grid_to_local_xy

from multi_camera_bev import (
    load_homography,
    get_image_size_from_video,
    estimate_local_bev_bounds_from_roi,
    project_to_bev_local,
    local_xy_to_image_xy,
    local_xy_to_latlon,
    latlon_to_local_xy,
    apply_homography,
)


# -------------------------
# ID counting & remapping
# -------------------------

def count_unique_ids(camera_configs: list[dict]) -> tuple[int, dict[int, int]]:
    """
    Scan all GT files and build a compact 0-based remapping of track IDs.

    AI City GT format (MOT):
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

    Raw IDs can be large and non-contiguous (e.g. 1, 34, 207, 534 ...).
    We sort them and map to 0, 1, 2, ... so the classifier weight matrix
    is exactly n_identities × reid_dim with no wasted rows.

    Returns:
        n_identities : total number of unique track IDs across all cameras
        id_remap     : dict mapping raw_id → compact_id
    """
    raw_ids: set[int] = set()

    for cfg in camera_configs:
        gt_path = Path(cfg["xml_path"])
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")

        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    raw_ids.add(int(parts[1]))
                except ValueError:
                    continue

    sorted_ids = sorted(raw_ids)
    id_remap   = {raw: compact for compact, raw in enumerate(sorted_ids)}
    n          = len(sorted_ids)
    print(f"Found {n} unique track IDs across {len(camera_configs)} camera(s). "
          f"Raw ID range: [{sorted_ids[0]}, {sorted_ids[-1]}]")
    return n, id_remap


# -------------------------
# Multi-camera dataset wrapper
# -------------------------

class MultiCameraDataset(Dataset):
    """
    Wraps multiple per-camera AICityDatasets.
    Each __getitem__ returns one frame from every camera simultaneously,
    truncated to the shortest camera's length.

    id_remap: dict[raw_id -> compact_id] produced by count_unique_ids().
              If None, raw IDs are passed through unchanged (not recommended
              for training — the classifier size would be wrong).
    """

    def __init__(self, camera_configs: list[dict], id_remap: dict[int, int] | None = None):
        self.datasets = [
            AICityDataset(video_path=cfg["video_path"], xml_path=cfg["xml_path"])
            for cfg in camera_configs
        ]
        self.id_remap = id_remap
        self.length = min(len(ds) for ds in self.datasets)
        print(f"MultiCameraDataset: {len(self.datasets)} cameras, "
              f"{self.length} frames (truncated to shortest).")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        images, targets = [], []
        for ds in self.datasets:
            img, tgt = ds[idx]
            # Remap raw track IDs → compact 0-based IDs
            if self.id_remap is not None and tgt is not None and "ids" in tgt:
                tgt = dict(tgt)   # shallow copy — don't mutate the cache
                tgt["ids"] = torch.tensor(
                    [self.id_remap.get(int(i), -1) for i in tgt["ids"]],
                    dtype=torch.long,
                )
            images.append(img)
            targets.append(tgt)
        return images, targets


# -------------------------
# Encoder  (paper §3.1)
# -------------------------
# The paper uses three blocks of ResNet/Swin, each downsampling by 2, then
# upsamples and concatenates each layer's output into a feature pyramid that
# gives a final spatial resolution of H/4 × W/4 at Cf=128 channels.
# We wrap ConvNeXt the same way: freeze it, attach a lightweight FPN adapter
# that guarantees the 128-channel H/4×W/4 contract regardless of what the
# backbone actually outputs.

class FPNAdapter(nn.Module):
    """
    Lightweight 1×1 projection + 3×3 mixer to enforce the
    Cf=128, H/4×W/4 output contract described in EarlyBird §3.1.

    ConvNeXt first-stage outputs C_in channels at H/4×W/4 already,
    so this is just a channel projection with a small spatial mix.
    """

    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# -------------------------
# Aggregation  (paper §3.3)
# -------------------------
# "We concatenate all feature maps along the channel dimension
#  S×Cf×Hg×Wg → (S·Cf)×Hg×Wg, then apply two 2D convolutions
#  to reduce to Cg=128."

class EarlyBirdAggregation(nn.Module):
    """
    Camera-count-agnostic aggregation matching EarlyBird §3.3.

    The original paper concatenates S×Cf channels then reduces with two Conv2D.
    That bakes S into the first conv weight — so a model trained on 3 cameras
    breaks at val time if val has a different number of cameras.

    Fix: apply a shared per-camera Conv2D (weight shared across cameras) to
    project each camera's Cf features to out_channels, then sum across cameras.
    This is equivalent to the paper's approach when S is fixed, but works for
    any S at both train and val time with a single set of weights.

    Shape: (B, S, C, Hg, Wg) → (B, out_channels, Hg, Wg)
    """

    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        # Applied identically to every camera's feature map
        self.per_cam = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Second conv operates on the summed feature — same size
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        # bev: (B, S, C, Hg, Wg)
        B, S, C, Hg, Wg = bev.shape
        # Apply shared conv to every camera, then sum (mean would also work)
        cam_feats = bev.reshape(B * S, C, Hg, Wg)
        cam_feats = self.per_cam(cam_feats)                  # (B*S, out, Hg, Wg)
        cam_feats = cam_feats.reshape(B, S, -1, Hg, Wg)
        fused     = cam_feats.sum(dim=1)                     # (B, out, Hg, Wg)
        return self.fuse(fused)                              # (B, out, Hg, Wg)


# -------------------------
# Image-view auxiliary heads  (paper §3.4)
# -------------------------
# The paper adds two auxiliary heads on the *image-feature* maps:
#   • A center head → 1×Hf×Wf  (2D box center heatmap)
#   • A foot head   → 1×Hf×Wf  (bottom-center / foot location)
# Both trained with Focal Loss; they push the encoder to develop
# spatially precise activations before BEV projection.

class ImageViewAuxHeads(nn.Module):
    """
    Center + foot-location detection heads on image-view features.
    Each: Conv2d(Cf,Cf,3,pad=1) → ReLU → Conv2d(Cf,1,1)
    """

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.center_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        self.foot_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor) -> dict:
        return {
            "img_center_logits": self.center_head(feats),   # (B, 1, Hf, Wf)
            "img_foot_logits":   self.foot_head(feats),     # (B, 1, Hf, Wf)
        }


# -------------------------
# Per-camera 2D box regression head
# -------------------------
# Predicts (w, h) in *normalised* image coordinates at every feature-map
# cell, conditioned on the centre heatmap.  We follow CenterNet's convention:
#   • width  = w_pixels / img_w   (in [0, 1])
#   • height = h_pixels / img_h   (in [0, 1])
# At inference we read out the (w, h) at each detected centre and recover
# the full xyxy box:
#   cx_px = (u + offset_x) * stride      stride = 4  (H/4 feature map)
#   cy_px = (v + offset_y) * stride
#   w_px  = w_norm * img_w
#   h_px  = h_norm * img_h
#   box   = [cx_px - w_px/2, cy_px - h_px/2, cx_px + w_px/2, cy_px + h_px/2]
#
# Trained with IoU loss (robust to scale variation across vehicle sizes)
# plus L1 on the raw normalised values as a secondary regulariser.
# A separate uncertainty log_var balances this against the other losses.

class ImageViewBoxHead(nn.Module):
    """
    Per-camera 2D bounding-box size head.
    Input:  (B, Cf, Hf, Wf)  — image-view features from FPN adapter
    Output: (B, 2, Hf, Wf)   — [w_norm, h_norm] predicted at every cell
    """

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Sigmoid(),   # clamp output to (0, 1) — normalised w/h
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)   # (B, 2, Hf, Wf)


# -------------------------
# Re-ID head  (paper §3.4)
# -------------------------
# Applied to BOTH image-view features and BEV features.
# Produces Cid=64 dimensional embeddings.
# Supervised by Cross-Entropy (identity classification) + SupCon loss.

class ReIDHead(nn.Module):
    """Re-ID embedding head. Conv2d(C, Cid, 1) + BN."""

    def __init__(self, in_channels: int, reid_dim: int = 64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, reid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reid_dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.embed(feats)   # (B, reid_dim, H, W)


# -------------------------
# Full model  (EarlyBird §3)
# -------------------------

class EarlyBirdModel(nn.Module):
    """
    Multi-camera BEV detection + tracking model following EarlyBird §3.

    Pipeline (per forward):
      1. Encode each camera's image  →  Cf=128, H/4×W/4  [frozen backbone + FPN adapter]
      2. Auxiliary image-view heads  →  2D center & foot heatmaps
      3. Re-ID on image features     →  64-dim embedding map
      4. Project each camera to shared BEV grid  [your coord logic, unchanged]
      5. Concat + 2×Conv aggregate   →  128×Hg×Wg
      6. ResNet-18 FPN decoder       →  128×Hg×Wg  (large receptive field)
      7. BEV prediction heads        →  center logits, offsets, re-ID embeddings
    """

    def __init__(
        self,
        n_cameras:     int,
        bev_grid:      dict,
        image_sizes:   list[tuple],
        homographies:  list[torch.Tensor],
        freeze_backbone: bool = True,
        base_channels: int   = 128,
        reid_dim:      int   = 64,
        n_identities:  int   = 100,
    ):
        super().__init__()

        self.n_cameras    = n_cameras
        self.bev_grid     = bev_grid
        self.image_sizes  = image_sizes
        self.reid_dim     = reid_dim
        self.n_identities = n_identities

        for i, H in enumerate(homographies):
            self.register_buffer(f"H_{i}", H)

        # ── Backbone (frozen ConvNeXt) ─────────────────────────────────
        self.backbone = ConvNeXtFirstStage(pretrained=True)
        if freeze_backbone:
            self._freeze(self.backbone)
        backbone_ch = self._probe_backbone_channels()

        # ── FPN adapter: backbone_ch → 128, enforces H/4×W/4 contract ─
        self.fpn_adapter = FPNAdapter(in_channels=backbone_ch, out_channels=base_channels)

        # ── Auxiliary image-view heads (paper §3.4) ────────────────────
        self.img_aux_heads = ImageViewAuxHeads(in_channels=base_channels)

        # ── Per-camera 2D box regression head ─────────────────────────
        self.img_box_head = ImageViewBoxHead(in_channels=base_channels)

        # ── Re-ID head on image features (paper §3.4) ─────────────────
        self.img_reid_head = ReIDHead(in_channels=base_channels, reid_dim=reid_dim)

        # ── BEV aggregation: per-cam conv + sum + fuse (paper §3.3) ───
        self.aggregator = EarlyBirdAggregation(
            in_channels=base_channels,
            out_channels=base_channels,
        )

        # ── ResNet-18 FPN decoder (paper §3.3) ────────────────────────
        self.decoder = BEVResNet18Decoder(
            in_channels=base_channels,
            base_channels=base_channels,
            out_channels=base_channels,
        )

        # ── BEV prediction heads (paper §3.4) ─────────────────────────
        self.bev_center_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )
        self.bev_offset_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2, kernel_size=1),
        )

        # ── Re-ID head on BEV features (paper §3.4) ───────────────────
        self.bev_reid_head = ReIDHead(in_channels=base_channels, reid_dim=reid_dim)

        # ── Identity classifier shared by both re-ID heads ────────────
        # Paper: linear layer trained with Cross-Entropy to ground-truth ID
        self.id_classifier = nn.Linear(reid_dim, n_identities)

        # ── Learnable log-variance for uncertainty loss balancing ──────
        # Paper follows FairMOT (Kendall et al.) — one scalar per loss term.
        # Terms: bev_center, bev_offset, img_center, img_foot,
        #        img_reid_ce, img_reid_supcon, bev_reid_ce, bev_reid_supcon,
        #        img_box  (added: per-camera 2D box regression)
        self.log_vars = nn.Parameter(torch.zeros(9))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze(module: nn.Module):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def _probe_backbone_channels(self) -> int:
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(dummy)
        return out.shape[1]

    def _get_H(self, cam_idx: int) -> torch.Tensor:
        return getattr(self, f"H_{cam_idx}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: list[torch.Tensor]) -> dict:
        """
        Args:
            images: list of (B, 3, Hi, Wi) per camera (may differ in size)

        Returns dict with keys:
            bev_center_logits  (B, 1, Hg, Wg)
            bev_center_prob    (B, 1, Hg, Wg)
            bev_offsets        (B, 2, Hg, Wg)
            bev_reid           (B, reid_dim, Hg, Wg)
            img_center_logits  list[(B, 1, Hf, Wf)]  — one per camera
            img_foot_logits    list[(B, 1, Hf, Wf)]  — one per camera
            img_reid           list[(B, reid_dim, Hf, Wf)]  — one per camera
            img_box_wh         list[(B, 2, Hf, Wf)]  — normalised (w, h) per camera
        """
        g = self.bev_grid

        bev_per_cam     = []
        img_center_list = []
        img_foot_list   = []
        img_reid_list   = []
        img_box_list    = []

        for cam_idx, img in enumerate(images):
            B            = img.shape[0]
            img_h, img_w = self.image_sizes[cam_idx]
            H            = self._get_H(cam_idx)   # (1, 3, 3)

            # 1. Backbone (frozen — no grad through here)
            with torch.no_grad():
                raw_feats = self.backbone(img)       # (B, backbone_ch, Hf, Wf)

            # 2. FPN adapter → (B, 128, Hf, Wf)
            feats = self.fpn_adapter(raw_feats)

            # 3. Auxiliary image-view heads
            aux = self.img_aux_heads(feats)
            img_center_list.append(aux["img_center_logits"])
            img_foot_list.append(aux["img_foot_logits"])

            # 4. Re-ID on image features
            img_reid_list.append(self.img_reid_head(feats))  # (B, reid_dim, Hf, Wf)

            # 5. Per-camera 2D box size prediction  →  (B, 2, Hf, Wf)
            img_box_list.append(self.img_box_head(feats))

            # 5. Project each sample to shared BEV grid
            #    project_to_bev_local expects (S, C, Hf, Wf), loop over batch
            bevs = []
            for b in range(B):
                bev = project_to_bev_local(
                    features=feats[b].unsqueeze(0),   # (1, 128, Hf, Wf)
                    H_ground_to_image=H,
                    Hg=g["Hg"], Wg=g["Wg"],
                    x_min=g["x_min"], x_max=g["x_max"],
                    y_min=g["y_min"], y_max=g["y_max"],
                    img_h=img_h, img_w=img_w,
                    origin=g["origin"],
                )
                bevs.append(bev.squeeze(0))           # (128, Hg, Wg)
            bev_per_cam.append(torch.stack(bevs, dim=0))   # (B, 128, Hg, Wg)

        # 6. Stack → (B, S, 128, Hg, Wg)
        bev_stack = torch.stack(bev_per_cam, dim=1)

        # 7. Concat aggregation → (B, 128, Hg, Wg)
        bev_agg = self.aggregator(bev_stack)

        # 8. ResNet-18 FPN decoder → (B, 128, Hg, Wg)
        bev_decoded = self.decoder(bev_agg)

        # 9. BEV prediction heads
        bev_center_logits = self.bev_center_head(bev_decoded)   # (B, 1, Hg, Wg)
        bev_offsets       = self.bev_offset_head(bev_decoded)   # (B, 2, Hg, Wg)
        bev_reid          = self.bev_reid_head(bev_decoded)     # (B, reid_dim, Hg, Wg)

        return {
            "bev_center_logits": bev_center_logits,
            "bev_center_prob":   torch.sigmoid(bev_center_logits),
            "bev_offsets":       bev_offsets,
            "bev_reid":          bev_reid,
            "img_center_logits": img_center_list,
            "img_foot_logits":   img_foot_list,
            "img_reid":          img_reid_list,
            "img_box_wh":        img_box_list,   # list[(B, 2, Hf, Wf)]  norm (w, h)
        }


# -------------------------
# Losses  (paper §3.4)
# -------------------------

def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CornerNet / CenterNet focal loss.
    pred_logits : (B, 1, H, W)
    target      : (B, 1, H, W)  Gaussian-smoothed heatmap in [0,1], peak=1
    """
    pred     = torch.sigmoid(pred_logits)
    pos_mask = (target == 1).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = -torch.log(pred + 1e-8) * (1 - pred) ** 2 * pos_mask
    neg_loss = -torch.log(1 - pred + 1e-8) * pred ** 2 * (1 - target) ** 4 * neg_mask

    n_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def offset_l1_loss(
    pred:   torch.Tensor,   # (B, 2, H, W)
    target: torch.Tensor,   # (B, 2, H, W)
    mask:   torch.Tensor,   # (B, 1, H, W)  1 at object centres
) -> torch.Tensor:
    return (F.l1_loss(pred, target, reduction="none") * mask).sum() / mask.sum().clamp(min=1)


def box_iou_loss(
    pred_wh:  torch.Tensor,   # (B, 2, H, W)  normalised (w, h) in [0,1]
    target_wh: torch.Tensor,  # (B, 2, H, W)  normalised (w, h) in [0,1]
    mask:     torch.Tensor,   # (B, 1, H, W)  1 at object centres
) -> torch.Tensor:
    """
    IoU loss on predicted vs target box sizes, evaluated only at object centres.

    Both pred and target are (w_norm, h_norm) — width and height normalised
    by the image dimensions.  We compute IoU treating each prediction as a
    box centred at its grid cell (position cancels out in the IoU ratio).

    IoU loss = 1 - IoU, averaged over positive cells.
    Combined with a secondary L1 term for faster early-training convergence.
    """
    mask_flat   = mask[:, 0].bool()                     # (B, H, W)
    pred_flat   = pred_wh.permute(0, 2, 3, 1)[mask_flat]    # (N, 2)
    target_flat = target_wh.permute(0, 2, 3, 1)[mask_flat]  # (N, 2)

    if pred_flat.shape[0] == 0:
        return pred_wh.sum() * 0.0

    pred_w,   pred_h   = pred_flat[:, 0],   pred_flat[:, 1]
    target_w, target_h = target_flat[:, 0], target_flat[:, 1]

    inter_w = torch.min(pred_w, target_w)
    inter_h = torch.min(pred_h, target_h)
    inter   = inter_w * inter_h
    union   = pred_w * pred_h + target_w * target_h - inter + 1e-7
    iou     = inter / union

    iou_loss = (1 - iou).mean()
    l1_loss  = F.l1_loss(pred_flat, target_flat)
    return iou_loss + 0.1 * l1_loss



def supcon_loss(
    embeddings:  torch.Tensor,   # (N, reid_dim)  raw (will be L2-normalised inside)
    labels:      torch.Tensor,   # (N,)  integer identity IDs
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al. 2020).
    Pulls same-identity embeddings together, pushes different-identity ones apart.
    """
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0

    embeddings = F.normalize(embeddings, dim=1)
    sim        = torch.matmul(embeddings, embeddings.T) / temperature   # (N, N)

    labels   = labels.unsqueeze(1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)

    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim            = sim - logits_max.detach()
    exp_sim        = torch.exp(sim)
    self_mask      = torch.eye(embeddings.shape[0], device=embeddings.device).bool()
    exp_sim        = exp_sim.masked_fill(self_mask, 0)

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    n_pos    = pos_mask.sum(dim=1).clamp(min=1)
    loss     = -(pos_mask * log_prob).sum(dim=1) / n_pos

    return loss.mean()


class EarlyBirdLoss(nn.Module):
    """
    Full EarlyBird loss (paper §3.4) + per-camera 2D box regression:

        L = Σ_i  exp(-s_i) * L_i  +  s_i

    where s_i = log_var_i are learned scalars (uncertainty weighting,
    following FairMOT / Kendall et al.).

    Nine terms (indices match model.log_vars):
        0  bev_center   (focal)
        1  bev_offset   (L1)
        2  img_center   (focal, averaged over cameras)
        3  img_foot     (focal, averaged over cameras)
        4  img_reid_ce  (cross-entropy over identities, image view)
        5  img_reid_sc  (SupCon, image view)
        6  bev_reid_ce  (cross-entropy over identities, BEV view)
        7  bev_reid_sc  (SupCon, BEV view)
        8  img_box      (IoU + L1 on normalised w/h, averaged over cameras)
    """

    def __init__(self, id_classifier: nn.Linear):
        super().__init__()
        self.id_classifier = id_classifier

    @staticmethod
    def _uncertainty_weight(log_var: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        return torch.exp(-log_var) * loss + log_var

    def _reid_losses(
        self,
        reid_map:     torch.Tensor,   # (B, reid_dim, H, W)
        center_mask:  torch.Tensor,   # (B, 1, H, W)   1 at object centres
        id_labels:    torch.Tensor,   # (B, 1, H, W)   identity index, -1 = bg
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings at object centres, compute CE + SupCon."""
        ce_total = reid_map.new_zeros(1)
        sc_total = reid_map.new_zeros(1)
        n_valid  = 0

        for b in range(reid_map.shape[0]):
            ys, xs = (center_mask[b, 0] > 0).nonzero(as_tuple=True)
            if ys.numel() == 0:
                continue
            embs  = reid_map[b, :, ys, xs].T        # (n, D)
            ids   = id_labels[b, 0, ys, xs].long()  # (n,)
            valid = ids >= 0
            if valid.sum() == 0:
                continue
            embs, ids = embs[valid], ids[valid]

            logits    = self.id_classifier(F.normalize(embs, dim=1))
            ce_total  = ce_total + F.cross_entropy(logits, ids)
            sc_total  = sc_total + supcon_loss(embs, ids)
            n_valid  += 1

        denom = max(n_valid, 1)
        return ce_total / denom, sc_total / denom

    def forward(
        self,
        preds: dict,
        # BEV targets
        bev_heatmap:      torch.Tensor,
        bev_offsets:      torch.Tensor,
        bev_offset_mask:  torch.Tensor,
        bev_id_map:       torch.Tensor,
        # Image-view targets (list, one per camera)
        img_center_maps:  list[torch.Tensor],
        img_foot_maps:    list[torch.Tensor],
        img_center_masks: list[torch.Tensor],
        img_id_maps:      list[torch.Tensor],
        img_box_targets:  list[torch.Tensor],  # each (B, 2, Hf, Wf) norm (w,h)
        log_vars:         torch.Tensor,        # (9,) learnable
    ) -> dict:

        # ── BEV losses ─────────────────────────────────────────────────
        l_bev_center = focal_loss(preds["bev_center_logits"], bev_heatmap)
        l_bev_offset = offset_l1_loss(preds["bev_offsets"], bev_offsets, bev_offset_mask)
        l_bev_reid_ce, l_bev_reid_sc = self._reid_losses(
            preds["bev_reid"], bev_offset_mask, bev_id_map
        )

        # ── Image-view losses (averaged over cameras) ──────────────────
        l_img_center = sum(
            focal_loss(logits, gt)
            for logits, gt in zip(preds["img_center_logits"], img_center_maps)
        ) / len(img_center_maps)

        l_img_foot = sum(
            focal_loss(logits, gt)
            for logits, gt in zip(preds["img_foot_logits"], img_foot_maps)
        ) / len(img_foot_maps)

        l_img_reid_ce = preds["bev_reid"].new_zeros(1)
        l_img_reid_sc = preds["bev_reid"].new_zeros(1)
        for reid_map, mask, id_map in zip(
            preds["img_reid"], img_center_masks, img_id_maps
        ):
            ce, sc = self._reid_losses(reid_map, mask, id_map)
            l_img_reid_ce = l_img_reid_ce + ce
            l_img_reid_sc = l_img_reid_sc + sc
        l_img_reid_ce = l_img_reid_ce / len(preds["img_reid"])
        l_img_reid_sc = l_img_reid_sc / len(preds["img_reid"])

        # ── Per-camera 2D box regression loss (IoU + L1) ───────────────
        l_img_box = sum(
            box_iou_loss(pred_wh, tgt_wh, mask)
            for pred_wh, tgt_wh, mask in zip(
                preds["img_box_wh"], img_box_targets, img_center_masks
            )
        ) / len(preds["img_box_wh"])

        # ── Uncertainty-weighted total (paper / FairMOT) ───────────────
        raw = [
            l_bev_center, l_bev_offset,
            l_img_center, l_img_foot,
            l_img_reid_ce, l_img_reid_sc,
            l_bev_reid_ce, l_bev_reid_sc,
            l_img_box,
        ]
        total = sum(self._uncertainty_weight(log_vars[i], raw[i]) for i in range(9))

        return {
            "total":       total,
            "bev_center":  l_bev_center,
            "bev_offset":  l_bev_offset,
            "img_center":  l_img_center,
            "img_foot":    l_img_foot,
            "img_reid_ce": l_img_reid_ce,
            "img_reid_sc": l_img_reid_sc,
            "bev_reid_ce": l_bev_reid_ce,
            "bev_reid_sc": l_bev_reid_sc,
            "img_box":     l_img_box,
        }


# -------------------------
# Target builder
# -------------------------

def collate_multicam(batch):
    n_cams         = len(batch[0][0])
    images_by_cam  = [torch.stack([b[0][c] for b in batch]) for c in range(n_cams)]
    targets_by_cam = [[b[1][c] for b in batch] for c in range(n_cams)]
    return images_by_cam, targets_by_cam


def build_targets(
    targets_by_cam:  list,
    Hg: int, Wg: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    homographies:    list[torch.Tensor],
    image_sizes:     list[tuple],
    origin:          tuple,
    device:          torch.device,
    feat_sizes:      list[tuple] | None = None,  # actual (Hf, Wf) per camera from encoder
    sigma_bev:       float = 2.0,
    sigma_img:       float = 4.0,
) -> dict:
    """
    Build all supervision tensors needed by EarlyBirdLoss.

    feat_sizes: list of (Hf, Wf) — the *actual* spatial size of the encoder
                feature map for each camera.  Always pass this from the model's
                forward output so targets match predictions exactly.
                Falls back to img_h//4, img_w//4 if not provided.

    Returns a dict with keys:
        bev_heatmap       (B, 1, Hg, Wg)
        bev_offsets       (B, 2, Hg, Wg)
        bev_offset_mask   (B, 1, Hg, Wg)
        bev_id_map        (B, 1, Hg, Wg)
        img_center_maps   list[(B, 1, Hf, Wf)]  per camera
        img_foot_maps     list[(B, 1, Hf, Wf)]  per camera
        img_center_masks  list[(B, 1, Hf, Wf)]  per camera
        img_id_maps       list[(B, 1, Hf, Wf)]  per camera
        img_box_targets   list[(B, 2, Hf, Wf)]  normalised (w, h) per camera
    """
    B     = len(targets_by_cam[0])
    n_cam = len(targets_by_cam)

    bev_heatmap     = torch.zeros(B, 1, Hg, Wg, device=device)
    bev_offsets     = torch.zeros(B, 2, Hg, Wg, device=device)
    bev_offset_mask = torch.zeros(B, 1, Hg, Wg, device=device)
    bev_id_map      = torch.full((B, 1, Hg, Wg), -1, dtype=torch.long, device=device)

    img_center_maps  = [None] * n_cam
    img_foot_maps    = [None] * n_cam
    img_center_masks = [None] * n_cam
    img_id_maps      = [None] * n_cam
    img_box_targets  = [None] * n_cam

    def gaussian_splat(heatmap, ix, iy, sigma):
        r = int(3 * sigma)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < heatmap.shape[0] and 0 <= nx < heatmap.shape[1]:
                    val = math.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
                    if val > heatmap[ny, nx]:
                        heatmap[ny, nx] = val

    for cam_idx in range(n_cam):
        img_h, img_w = image_sizes[cam_idx]

        # Use actual encoder output size if provided, else fall back to //4
        if feat_sizes is not None:
            Hf, Wf = feat_sizes[cam_idx]
        else:
            Hf, Wf = img_h // 4, img_w // 4

        H_cam = homographies[cam_idx]   # (1, 3, 3)

        img_center_maps[cam_idx]  = torch.zeros(B, 1, Hf, Wf, device=device)
        img_foot_maps[cam_idx]    = torch.zeros(B, 1, Hf, Wf, device=device)
        img_center_masks[cam_idx] = torch.zeros(B, 1, Hf, Wf, device=device)
        img_id_maps[cam_idx]      = torch.full(
            (B, 1, Hf, Wf), -1, dtype=torch.long, device=device
        )
        # 2 channels: [w_norm, h_norm] — zero where no object
        img_box_targets[cam_idx]  = torch.zeros(B, 2, Hf, Wf, device=device)

        for b in range(B):
            tgt = targets_by_cam[cam_idx][b]
            if tgt is None or len(tgt) == 0:
                continue

            boxes = tgt["boxes"]   # (N, 4) xyxy image pixels
            ids   = tgt.get("ids", torch.zeros(len(boxes), dtype=torch.long))

            # ── Image-view targets ──────────────────────────────────────
            for n in range(len(boxes)):
                x1, y1, x2, y2 = boxes[n].tolist()

                # Box center → feature-map coords
                cx_f = ((x1 + x2) / 2) / img_w * Wf
                cy_f = ((y1 + y2) / 2) / img_h * Hf
                # Foot (bottom-center) → feature-map coords
                fu_f = ((x1 + x2) / 2) / img_w * Wf
                fv_f = y2              / img_h * Hf

                gaussian_splat(img_center_maps[cam_idx][b, 0],
                               int(round(cx_f)), int(round(cy_f)), sigma_img)
                gaussian_splat(img_foot_maps[cam_idx][b, 0],
                               int(round(fu_f)), int(round(fv_f)), sigma_img)

                # Centre mask & ID (for re-ID sampling) + box size target
                ix_f, iy_f = int(round(cx_f)), int(round(cy_f))
                if 0 <= ix_f < Wf and 0 <= iy_f < Hf:
                    img_center_masks[cam_idx][b, 0, iy_f, ix_f] = 1.0
                    img_id_maps[cam_idx][b, 0, iy_f, ix_f]      = ids[n]
                    # Normalised box dimensions — GT is (left, top, w, h) in
                    # pixel coords; x1/x2/y1/y2 were derived from xyxy above.
                    w_norm = (x2 - x1) / img_w
                    h_norm = (y2 - y1) / img_h
                    img_box_targets[cam_idx][b, 0, iy_f, ix_f] = w_norm
                    img_box_targets[cam_idx][b, 1, iy_f, ix_f] = h_norm

            # ── BEV targets (your coord logic, unchanged) ───────────────
            u   = 0.5 * (boxes[:, 0] + boxes[:, 2])
            v   = boxes[:, 3]
            pts_img = torch.stack([u, v], dim=1).to(torch.float64)

            H_inv      = torch.inverse(H_cam[0].to(torch.float64))
            pts_ground = apply_homography(H_inv, pts_img)       # (N, 2) lat/lon
            pts_local, _ = latlon_to_local_xy(pts_ground)       # (N, 2) x/y metres

            gx = (pts_local[:, 0] - x_min) / (x_max - x_min) * (Wg - 1)
            gy = (pts_local[:, 1] - y_min) / (y_max - y_min) * (Hg - 1)

            for n, (cx, cy) in enumerate(zip(gx.tolist(), gy.tolist())):
                ix, iy = int(round(cx)), int(round(cy))
                if not (0 <= ix < Wg and 0 <= iy < Hg):
                    continue
                gaussian_splat(bev_heatmap[b, 0], ix, iy, sigma_bev)
                bev_offsets[b, 0, iy, ix]     = cx - ix
                bev_offsets[b, 1, iy, ix]     = cy - iy
                bev_offset_mask[b, 0, iy, ix] = 1.0
                bev_id_map[b, 0, iy, ix]      = ids[n]

    return {
        "bev_heatmap":      bev_heatmap,
        "bev_offsets":      bev_offsets,
        "bev_offset_mask":  bev_offset_mask,
        "bev_id_map":       bev_id_map,
        "img_center_maps":  img_center_maps,
        "img_foot_maps":    img_foot_maps,
        "img_center_masks": img_center_masks,
        "img_id_maps":      img_id_maps,
        "img_box_targets":  img_box_targets,
    }


# -------------------------
# Shared epoch runner  (train + val)
# -------------------------

LOG_KEYS = [
    "total", "bev_center", "bev_offset",
    "img_center", "img_foot",
    "img_reid_ce", "img_reid_sc",
    "bev_reid_ce", "bev_reid_sc",
    "img_box",
]
LV_LABELS = ["bev_c", "bev_off", "img_c", "img_f",
             "img_rce", "img_rsc", "bev_rce", "bev_rsc", "box"]


def run_epoch(
    model:        EarlyBirdModel,
    loader:       DataLoader,
    criterion:    EarlyBirdLoss,
    Hg: int, Wg: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    homographies: list,
    image_sizes:  list,
    shared_origin: tuple,
    device:       torch.device,
    optimizer:    torch.optim.Optimizer | None = None,
    scheduler=None,
    grad_accum:   int = 1,
    epoch_label:  str = "",
) -> dict[str, float]:
    """
    One full pass over `loader`.  Pass optimizer=None for validation.
    Returns averaged loss dict.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    model.backbone.eval()   # always keep frozen BN in eval mode

    totals   = {k: 0.0 for k in LOG_KEYS}
    n_batches = 0

    if is_train:
        optimizer.zero_grad()

    pbar = tqdm(
        loader,
        desc=epoch_label,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )

    with torch.set_grad_enabled(is_train):
        for step, (images_by_cam, targets_by_cam) in enumerate(pbar):
            images_by_cam = [imgs.to(device) for imgs in images_by_cam]

            preds = model(images_by_cam)

            # Read actual encoder output sizes from predictions — this is the
            # ground truth for target tensor sizes, not img_h//4 which can
            # differ depending on ConvNeXt's internal padding behaviour.
            feat_sizes = [
                (logits.shape[2], logits.shape[3])
                for logits in preds["img_center_logits"]
            ]

            t = build_targets(
                targets_by_cam=targets_by_cam,
                Hg=Hg, Wg=Wg,
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                homographies=homographies,
                image_sizes=image_sizes,
                origin=shared_origin,
                device=device,
                feat_sizes=feat_sizes,
            )

            losses = criterion(
                preds=preds,
                bev_heatmap=t["bev_heatmap"],
                bev_offsets=t["bev_offsets"],
                bev_offset_mask=t["bev_offset_mask"],
                bev_id_map=t["bev_id_map"],
                img_center_maps=t["img_center_maps"],
                img_foot_maps=t["img_foot_maps"],
                img_center_masks=t["img_center_masks"],
                img_id_maps=t["img_id_maps"],
                img_box_targets=t["img_box_targets"],
                log_vars=model.log_vars,
            )

            if is_train:
                (losses["total"] / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

            for k in LOG_KEYS:
                totals[k] += losses[k].item()
            n_batches += 1

            # Live tqdm postfix — show the three most diagnostic values
            pbar.set_postfix(
                loss=f"{losses['total'].item():.3f}",
                bev_c=f"{losses['bev_center'].item():.3f}",
                box=f"{losses['img_box'].item():.3f}",
            )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# -------------------------
# Validation metrics
# -------------------------

def compute_val_metrics(
    model:         EarlyBirdModel,
    loader:        DataLoader,
    Hg: int, Wg: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    homographies:  list,
    image_sizes:   list,
    shared_origin: tuple,
    device:        torch.device,
    det_thresh:    float = 0.4,    # NMS score threshold (paper §3.5)
    dist_thresh_m: float = 0.5,    # true-positive radius in metres (paper §4.1)
    iou_thresh:    float = 0.5,    # box IoU threshold for 2D box TP
) -> dict[str, float]:
    """
    Compute detection metrics on the validation set.

    BEV metrics  — evaluated on ground-plane detections:
        precision, recall, F1   (Euclidean distance ≤ dist_thresh_m)

    2D box metrics  — evaluated per-camera on image-space boxes:
        box_precision, box_recall, box_F1   (IoU ≥ iou_thresh)

    Both use the same detect-then-match logic: for each predicted centre
    that clears det_thresh we find the nearest GT centre; if it is within
    the threshold and not yet matched it counts as a TP.
    """
    model.eval()
    model.backbone.eval()

    # Accumulate over the whole val set
    bev_tp = bev_fp = bev_fn = 0
    box_tp = box_fp = box_fn = 0

    # metres-per-cell for converting grid coords → metres
    x_scale = (x_max - x_min) / (Wg - 1)
    y_scale = (y_max - y_min) / (Hg - 1)

    with torch.no_grad():
        for images_by_cam, targets_by_cam in tqdm(
            loader, desc="  val metrics", leave=False, dynamic_ncols=True
        ):
            images_by_cam = [imgs.to(device) for imgs in images_by_cam]
            preds         = model(images_by_cam)
            B             = images_by_cam[0].shape[0]

            feat_sizes = [
                (logits.shape[2], logits.shape[3])
                for logits in preds["img_center_logits"]
            ]

            # ── BEV detection metrics ──────────────────────────────────
            center_prob = preds["bev_center_prob"]   # (B, 1, Hg, Wg)
            offsets     = preds["bev_offsets"]       # (B, 2, Hg, Wg)

            # Simple 3×3 max-pool NMS  (paper §3.5)
            hmax = F.max_pool2d(center_prob, kernel_size=3, stride=1, padding=1)
            keep = (center_prob == hmax) & (center_prob > det_thresh)

            t = build_targets(
                targets_by_cam=targets_by_cam,
                Hg=Hg, Wg=Wg,
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                homographies=homographies,
                image_sizes=image_sizes,
                origin=shared_origin,
                device=device,
                feat_sizes=feat_sizes,
            )

            for b in range(B):
                # --- Predicted BEV centres in metres ---
                ys_pred, xs_pred = keep[b, 0].nonzero(as_tuple=True)
                pred_x_m = (xs_pred.float() + offsets[b, 0, ys_pred, xs_pred]) * x_scale + x_min
                pred_y_m = (ys_pred.float() + offsets[b, 1, ys_pred, xs_pred]) * y_scale + y_min
                pred_pts  = torch.stack([pred_x_m, pred_y_m], dim=1)  # (P, 2)

                # --- GT BEV centres in metres ---
                gt_ys, gt_xs = (t["bev_offset_mask"][b, 0] > 0).nonzero(as_tuple=True)
                gt_x_m = gt_xs.float() * x_scale + x_min
                gt_y_m = gt_ys.float() * y_scale + y_min
                gt_pts  = torch.stack([gt_x_m, gt_y_m], dim=1)  # (G, 2)

                n_pred, n_gt = pred_pts.shape[0], gt_pts.shape[0]
                bev_fn += n_gt

                if n_pred == 0 or n_gt == 0:
                    bev_fp += n_pred
                    continue

                # Greedy nearest-neighbour matching
                dists    = torch.cdist(pred_pts.float(), gt_pts.float())  # (P, G)
                matched_gt = set()
                for p in range(n_pred):
                    min_d, min_g = dists[p].min(dim=0)
                    g = int(min_g)
                    if min_d.item() <= dist_thresh_m and g not in matched_gt:
                        bev_tp   += 1
                        bev_fn   -= 1
                        matched_gt.add(g)
                    else:
                        bev_fp += 1

            # ── 2D box metrics  (per camera) ──────────────────────────
            for cam_idx, (img_box_pred, img_mask, img_box_tgt) in enumerate(zip(
                preds["img_box_wh"],
                t["img_center_masks"],
                t["img_box_targets"],
            )):
                img_h, img_w = image_sizes[cam_idx]
                Hf, Wf       = img_h // 4, img_w // 4
                stride       = 4

                # Use BEV centre detections projected to image space as
                # the proxy for "which cells fired" in image view.
                # Simpler: use img_center_logits threshold directly.
                img_logits   = preds["img_center_logits"][cam_idx]  # (B,1,Hf,Wf)
                img_prob     = torch.sigmoid(img_logits)
                img_hmax     = F.max_pool2d(img_prob, 3, 1, 1)
                img_keep     = (img_prob == img_hmax) & (img_prob > det_thresh)

                for b in range(B):
                    ys_p, xs_p = img_keep[b, 0].nonzero(as_tuple=True)
                    # GT centres
                    ys_g, xs_g = (img_mask[b, 0] > 0).nonzero(as_tuple=True)

                    n_p, n_g = ys_p.shape[0], ys_g.shape[0]
                    box_fn += n_g

                    if n_p == 0 or n_g == 0:
                        box_fp += n_p
                        continue

                    # Reconstruct predicted boxes in pixel space
                    # (centre from cell index × stride, size from head output)
                    pred_cx = (xs_p.float() + 0.5) * stride   # (P,)
                    pred_cy = (ys_p.float() + 0.5) * stride
                    pred_w  = img_box_pred[b, 0, ys_p, xs_p] * img_w
                    pred_h  = img_box_pred[b, 1, ys_p, xs_p] * img_h
                    pred_boxes = torch.stack([
                        pred_cx - pred_w / 2, pred_cy - pred_h / 2,
                        pred_cx + pred_w / 2, pred_cy + pred_h / 2,
                    ], dim=1)   # (P, 4)

                    # Reconstruct GT boxes in pixel space
                    gt_cx = (xs_g.float() + 0.5) * stride
                    gt_cy = (ys_g.float() + 0.5) * stride
                    gt_w  = img_box_tgt[b, 0, ys_g, xs_g] * img_w
                    gt_h  = img_box_tgt[b, 1, ys_g, xs_g] * img_h
                    gt_boxes = torch.stack([
                        gt_cx - gt_w / 2, gt_cy - gt_h / 2,
                        gt_cx + gt_w / 2, gt_cy + gt_h / 2,
                    ], dim=1)   # (G, 4)

                    # Greedy IoU matching
                    matched_gt = set()
                    for p in range(n_p):
                        pb = pred_boxes[p]
                        ious = _box_iou_1d(pb.unsqueeze(0), gt_boxes)   # (G,)
                        best_iou, best_g = ious.max(dim=0)
                        g = int(best_g)
                        if best_iou.item() >= iou_thresh and g not in matched_gt:
                            box_tp   += 1
                            box_fn   -= 1
                            matched_gt.add(g)
                        else:
                            box_fp += 1

    def _prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        return p, r, f

    bev_p, bev_r, bev_f1 = _prf(bev_tp, bev_fp, bev_fn)
    box_p, box_r, box_f1 = _prf(box_tp, box_fp, box_fn)

    return {
        "bev_precision": bev_p, "bev_recall": bev_r, "bev_F1": bev_f1,
        "box_precision": box_p, "box_recall": box_r, "box_F1": box_f1,
        "bev_tp": bev_tp, "bev_fp": bev_fp, "bev_fn": bev_fn,
        "box_tp": box_tp, "box_fp": box_fp, "box_fn": box_fn,
    }


def _box_iou_1d(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """IoU between one box (1,4) and many boxes (N,4), all xyxy."""
    ix1 = torch.max(box[:, 0], boxes[:, 0])
    iy1 = torch.max(box[:, 1], boxes[:, 1])
    ix2 = torch.min(box[:, 2], boxes[:, 2])
    iy2 = torch.min(box[:, 3], boxes[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    a1    = (box[:, 2]   - box[:, 0])   * (box[:, 3]   - box[:, 1])
    a2    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-7)


# -------------------------
# Training entry point
# -------------------------

def train(
    train_camera_configs: list[dict],
    val_camera_configs:   list[dict] | None = None,
    n_identities:         int | None = None,
    epochs:               int   = 50,
    max_steps:            int | None = None,  # overrides epochs if set; total optimizer steps
    batch_size:           int   = 4,          # 4 fits comfortably on 8GB VRAM
    grad_accum:           int   = 4,          # effective batch = batch_size × grad_accum = 16
    lr:                   float = 1e-3,
    cell_size:            float = 0.4,
    roi_margin_m:         float = 3.0,
    val_every:            int   = 5,
    device_str:           str   = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    print(f"Training on {device}")

    # ------------------------------------------------------------------
    # 0. Count unique identities from TRAIN set only
    # ------------------------------------------------------------------
    n_identities, id_remap = count_unique_ids(train_camera_configs)

    # ------------------------------------------------------------------
    # 0b. Resolve epochs vs max_steps
    # ------------------------------------------------------------------
    # We need to know the dataset size before we can convert max_steps → epochs,
    # so we do a lightweight scan here (no model yet).
    # If max_steps is given it takes priority over epochs.
    _tmp_ds   = MultiCameraDataset(train_camera_configs, id_remap=id_remap)
    steps_per_epoch = max(len(_tmp_ds) // batch_size, 1) // grad_accum
    if max_steps is not None:
        epochs = max(math.ceil(max_steps / max(steps_per_epoch, 1)), 1)
        print(f"max_steps={max_steps} → {epochs} epochs "
              f"({steps_per_epoch} optimizer steps/epoch)")
    else:
        print(f"epochs={epochs}  ({steps_per_epoch} optimizer steps/epoch, "
              f"~{epochs * steps_per_epoch} total steps)")

    # ------------------------------------------------------------------
    # 1. Per-camera metadata  (train)
    # ------------------------------------------------------------------
    homographies, image_sizes, bev_bounds_list, origins = [], [], [], []

    for cfg in train_camera_configs:
        H = load_homography(cfg["calibration_path"])
        homographies.append(H)
        image_sizes.append(get_image_size_from_video(cfg["video_path"]))
        bounds, origin = estimate_local_bev_bounds_from_roi(
            roi_path=cfg["roi_path"],
            H_ground_to_image=H,
            margin_m=roi_margin_m,
            max_points=3000,
        )
        bev_bounds_list.append(bounds)
        origins.append(origin)

    x_min = min(b[0] for b in bev_bounds_list)
    x_max = max(b[1] for b in bev_bounds_list)
    y_min = min(b[2] for b in bev_bounds_list)
    y_max = max(b[3] for b in bev_bounds_list)
    Wg    = int((x_max - x_min) / cell_size)
    Hg    = int((y_max - y_min) / cell_size)
    shared_origin = origins[0]

    bev_grid = dict(
        Hg=Hg, Wg=Wg,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        origin=shared_origin,
    )
    print(f"Shared BEV grid: Hg={Hg}, Wg={Wg}")

    # ------------------------------------------------------------------
    # 2. Datasets & dataloaders
    # ------------------------------------------------------------------
    train_dataset = MultiCameraDataset(train_camera_configs, id_remap=id_remap)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_multicam,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = None
    if val_camera_configs:
        # Val set uses the same id_remap so classifier indices are consistent.
        # IDs that appear only in val (never in train) will map to -1 and be
        # excluded from the re-ID loss — that's the correct behaviour.
        val_dataset = MultiCameraDataset(val_camera_configs, id_remap=id_remap)
        val_loader  = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_multicam,
            drop_last=False,
            pin_memory=True,
        )
        print(f"Val set: {len(val_dataset)} frames across "
              f"{len(val_camera_configs)} camera(s).")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = EarlyBirdModel(
        n_cameras=len(train_camera_configs),
        bev_grid=bev_grid,
        image_sizes=image_sizes,
        homographies=homographies,
        freeze_backbone=True,
        base_channels=128,
        reid_dim=64,
        n_identities=n_identities,
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params  = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f"Params — total: {total_params:,}  trainable: {trainable_params:,}  "
          f"backbone (frozen): {backbone_params:,}")
    assert backbone_params == 0, "Backbone must be fully frozen!"

    # ------------------------------------------------------------------
    # 4. Optimizer + one-cycle LR scheduler  (paper §4.2)
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    total_steps = epochs * (len(train_loader) // grad_accum)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max(total_steps, 1),
        pct_start=0.3,
        anneal_strategy="cos",
    )
    criterion = EarlyBirdLoss(id_classifier=model.id_classifier)

    # ------------------------------------------------------------------
    # 5. History  (for overfitting detection)
    # ------------------------------------------------------------------
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "bev_F1": [], "box_F1": [],
    }
    best_val_loss  = float("inf")
    best_ckpt_path = Path("checkpoints") / "earlybird_best.pth"
    best_ckpt_path.parent.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 6. Epoch loop
    # ------------------------------------------------------------------
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch", dynamic_ncols=True)

    for epoch in epoch_bar:
        ep1 = epoch + 1

        # ── Train ──────────────────────────────────────────────────────
        train_avg = run_epoch(
            model=model, loader=train_loader, criterion=criterion,
            Hg=Hg, Wg=Wg,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            homographies=homographies, image_sizes=image_sizes,
            shared_origin=shared_origin, device=device,
            optimizer=optimizer, scheduler=scheduler,
            grad_accum=grad_accum,
            epoch_label=f"Train {ep1:03d}/{epochs}",
        )
        history["train_loss"].append(train_avg["total"])

        # ── Val loss (every epoch) ─────────────────────────────────────
        val_avg     = {}
        val_metrics = {}
        if val_loader is not None:
            val_avg = run_epoch(
                model=model, loader=val_loader, criterion=criterion,
                Hg=Hg, Wg=Wg,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                homographies=homographies, image_sizes=image_sizes,
                shared_origin=shared_origin, device=device,
                optimizer=None,   # no gradient updates
                epoch_label=f"  Val {ep1:03d}/{epochs}",
            )
            history["val_loss"].append(val_avg["total"])

        # ── Per-epoch tqdm outer bar update ───────────────────────────
        postfix = {
            "tr_loss": f"{train_avg['total']:.3f}",
            "lr":      f"{scheduler.get_last_lr()[0]:.2e}",
        }
        if val_avg:
            postfix["val_loss"] = f"{val_avg['total']:.3f}"
            gap = val_avg["total"] - train_avg["total"]
            postfix["gap"] = f"{gap:+.3f}"   # positive = overfitting signal
        epoch_bar.set_postfix(postfix)

        # ── Loss-weight line (printed once per epoch to stdout) ────────
        lv          = model.log_vars.detach().cpu().tolist()
        weights_str = "  ".join(f"{l}={math.exp(-v):.2f}" for l, v in zip(LV_LABELS, lv))
        tqdm.write(
            f"Epoch {ep1:03d}/{epochs} | "
            f"train={train_avg['total']:.3f}"
            + (f"  val={val_avg['total']:.3f}" if val_avg else "")
            + f"  bev_c={train_avg['bev_center']:.3f}"
            f"  bev_off={train_avg['bev_offset']:.3f}"
            f"  box={train_avg['img_box']:.3f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
        )
        tqdm.write(f"           loss weights → {weights_str}")

        # ── Full val metric summary every `val_every` epochs ──────────
        if val_loader is not None and ep1 % val_every == 0:
            val_metrics = compute_val_metrics(
                model=model, loader=val_loader,
                Hg=Hg, Wg=Wg,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                homographies=homographies, image_sizes=image_sizes,
                shared_origin=shared_origin, device=device,
            )
            history["bev_F1"].append(val_metrics["bev_F1"])
            history["box_F1"].append(val_metrics["box_F1"])

            sep = "─" * 62
            tqdm.write(f"\n{sep}")
            tqdm.write(f"  Validation summary — epoch {ep1:03d}/{epochs}")
            tqdm.write(sep)
            tqdm.write(
                f"  BEV  │ P={val_metrics['bev_precision']:.3f}  "
                f"R={val_metrics['bev_recall']:.3f}  "
                f"F1={val_metrics['bev_F1']:.3f}  "
                f"(TP={val_metrics['bev_tp']}  FP={val_metrics['bev_fp']}  "
                f"FN={val_metrics['bev_fn']})"
            )
            tqdm.write(
                f"  Box  │ P={val_metrics['box_precision']:.3f}  "
                f"R={val_metrics['box_recall']:.3f}  "
                f"F1={val_metrics['box_F1']:.3f}  "
                f"(TP={val_metrics['box_tp']}  FP={val_metrics['box_fp']}  "
                f"FN={val_metrics['box_fn']})"
            )
            # Overfitting indicator
            if len(history["train_loss"]) >= val_every:
                recent_train = history["train_loss"][-val_every:]
                recent_val   = history["val_loss"][-val_every:]
                avg_gap      = sum(v - t for v, t in zip(recent_val, recent_train)) / val_every
                trend        = "↑ widening" if avg_gap > 0.05 else "↓ stable"
                tqdm.write(
                    f"  Gap  │ avg(val-train) over last {val_every} epochs = "
                    f"{avg_gap:+.3f}  {trend}"
                )
            tqdm.write(sep + "\n")

        # ── Save best checkpoint (by val loss) ────────────────────────
        if val_avg and val_avg["total"] < best_val_loss:
            best_val_loss = val_avg["total"]
            torch.save({
                "epoch":           ep1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "bev_grid":        bev_grid,
                "image_sizes":     image_sizes,
                "n_identities":    n_identities,
                "val_loss":        best_val_loss,
            }, best_ckpt_path)
            tqdm.write(f"  ★  New best val loss {best_val_loss:.4f} → {best_ckpt_path}")

    # ------------------------------------------------------------------
    # 7. Final checkpoint
    # ------------------------------------------------------------------
    final_ckpt = Path("checkpoints") / "earlybird_final.pth"
    torch.save({
        "epoch":           epochs,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "bev_grid":        bev_grid,
        "image_sizes":     image_sizes,
        "n_identities":    n_identities,
        "history":         history,
    }, final_ckpt)
    print(f"\nFinal checkpoint saved → {final_ckpt}")
    if val_loader is not None:
        print(f"Best checkpoint (val)  → {best_ckpt_path}  (loss={best_val_loss:.4f})")
    return model, history


# -------------------------
# Entry point
# -------------------------

# Train cameras
# Train cameras
TRAIN_CAMERA_CONFIGS = [
    {
        "video_path":       r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c001\vdo.avi",
        "xml_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c001\gt\gt.txt",
        "calibration_path": r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c001\calibration.txt",
        "roi_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c001\roi.jpg",
    },
    {
        "video_path":       r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c002\vdo.avi",
        "xml_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c002\gt\gt.txt",
        "calibration_path": r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c002\calibration.txt",
        "roi_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c002\roi.jpg",
    },
    {
        "video_path":       r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c003\vdo.avi",
        "xml_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c003\gt\gt.txt",
        "calibration_path": r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c003\calibration.txt",
        "roi_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c003\roi.jpg",
    },
    # 
    # Add more train cameras here
]

# Validation cameras — can be different scenes or a held-out subset.
# Set to None to skip validation entirely.
VAL_CAMERA_CONFIGS = [
    {
        "video_path":       r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c010\vdo.avi",
        "xml_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c010\gt\gt.txt",
        "calibration_path": r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c010\calibration.txt",
        "roi_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c010\roi.jpg",
    },
    {
        "video_path":       r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c011\vdo.avi",
        "xml_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c011\gt\gt.txt",
        "calibration_path": r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c011\calibration.txt",
        "roi_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c011\roi.jpg",
    },
    {
        "video_path":       r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c012\vdo.avi",
        "xml_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c012\gt\gt.txt",
        "calibration_path": r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c012\calibration.txt",
        "roi_path":         r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S03\c012\roi.jpg",
    },
    # {
    #     "video_path":       r"...\c002\vdo.avi",
    #     "xml_path":         r"...\c002\gt\gt.txt",
    #     "calibration_path": r"...\c002\calibration.txt",
    #     "roi_path":         r"...\c002\roi.jpg",
    # },
]   # empty list → treated as None (no val)

if __name__ == "__main__":
    train(
        train_camera_configs=TRAIN_CAMERA_CONFIGS,
        val_camera_configs=VAL_CAMERA_CONFIGS or None,
        epochs=1,          # use this OR max_steps, not both
        #max_steps=5000,        # stop after 5000 optimizer steps regardless of epoch count
        batch_size=4,          # 4 fits on 8GB VRAM; drop to 2 if OOM
        grad_accum=4,          # effective batch = 16
        lr=1e-3,
        val_every=5,
    )
