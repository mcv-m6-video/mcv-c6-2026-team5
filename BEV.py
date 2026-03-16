import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ConvNeXt import ConvNeXtFirstStage
from Decoder import BEVResNet18Decoder
from src.data.loader import AICityDataset
from visualizations import (visualize_bev, visualize_features,
                            visualize_side_by_side)
from Heads import BEVPredictionModule, decode_bev_centers, visualize_bev_detections, bev_grid_to_local_xy, draw_points_on_image

# -------------------------
# Coordinate conversions
# -------------------------
def apply_homography(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    Apply a 3x3 homography to 2D points.

    Args:
        H: (3, 3)
        pts: (N, 2)

    Returns:
        projected_pts: (N, 2)
    """
    H = H.to(torch.float64)
    pts = pts.to(torch.float64)

    pts_h = torch.cat(
        [pts, torch.ones((pts.shape[0], 1), dtype=torch.float64, device=pts.device)],
        dim=1,
    ).T  # (3, N)

    proj = H @ pts_h
    proj = proj / (proj[2:3] + 1e-12)

    return proj[:2].T


def latlon_to_local_xy(latlon_pts: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
    """
    Convert latitude/longitude points to a local XY frame in meters.

    Args:
        latlon_pts: (N, 2), [lat, lon]

    Returns:
        xy_m: (N, 2), local coordinates in meters
        origin: (lat0, lon0)
    """
    latlon_pts = latlon_pts.to(torch.float64)

    lat0 = latlon_pts[:, 0].mean()
    lon0 = latlon_pts[:, 1].mean()

    earth_radius = 6378137.0

    lat = latlon_pts[:, 0] * math.pi / 180.0
    lon = latlon_pts[:, 1] * math.pi / 180.0
    lat0_rad = lat0 * math.pi / 180.0
    lon0_rad = lon0 * math.pi / 180.0

    x = (lon - lon0_rad) * math.cos(lat0_rad) * earth_radius
    y = (lat - lat0_rad) * earth_radius

    xy_m = torch.stack([x, y], dim=1)
    return xy_m, (float(lat0), float(lon0))


def local_xy_to_latlon(xy_m: torch.Tensor, origin: tuple[float, float]) -> torch.Tensor:
    """
    Convert local XY coordinates in meters back to lat/lon.

    Args:
        xy_m: (..., 2), [x, y] in meters
        origin: (lat0, lon0)

    Returns:
        latlon: (..., 2), [lat, lon]
    """
    xy_m = xy_m.to(torch.float64)
    lat0, lon0 = origin

    earth_radius = 6378137.0
    lat0_rad = lat0 * math.pi / 180.0
    lon0_rad = lon0 * math.pi / 180.0

    x = xy_m[..., 0]
    y = xy_m[..., 1]

    lat = y / earth_radius + lat0_rad
    lon = x / (earth_radius * math.cos(lat0_rad)) + lon0_rad

    lat = lat * 180.0 / math.pi
    lon = lon * 180.0 / math.pi

    return torch.stack([lat, lon], dim=-1)


# -------------------------
# Dataset / calibration utils
# -------------------------
def load_homography(calibration_path: str | Path) -> torch.Tensor:
    """
    Load CityFlow calibration.txt homography matrix.

    Returns:
        H_ground_to_image: (1, 3, 3)
    """
    calibration_path = Path(calibration_path)

    with open(calibration_path, "r") as f:
        first_line = f.readline().strip()

    parts = first_line.split(";")
    first_row = list(map(float, parts[0].split()[2:]))
    second_row = list(map(float, parts[1].split()))
    third_row = list(map(float, parts[2].split()))

    H = torch.tensor([first_row, second_row, third_row], dtype=torch.float64)
    return H.unsqueeze(0)  # (1, 3, 3)


def boxes_to_bottom_centers(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert xyxy boxes to bottom-center points.

    Args:
        boxes: (N, 4)

    Returns:
        pts: (N, 2)
    """
    u = 0.5 * (boxes[:, 0] + boxes[:, 2])
    v = boxes[:, 3]
    return torch.stack([u, v], dim=1)

def estimate_local_bev_bounds_from_roi(
    roi_path: str | Path,
    H_ground_to_image: torch.Tensor,
    margin_m: float = 5.0,
    max_points: int = 5000,
    y_start_ratio: float = 0.65,
) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
    """
    Estimate BEV bounds from the lower part of a binary ROI mask.

    Args:
        roi_path: path to binary ROI image
        H_ground_to_image: (1, 3, 3)
        margin_m: extra margin around bounds
        max_points: max sampled ROI pixels
        y_start_ratio: only use ROI pixels below this fraction of image height

    Returns:
        bounds: (x_min, x_max, y_min, y_max)
        origin: (lat0, lon0)
    """
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise FileNotFoundError(f"Could not read ROI mask: {roi_path}")

    h, w = roi.shape
    y_thresh = int(h * y_start_ratio)

    ys, xs = np.where((roi > 0) & (np.arange(h)[:, None] >= y_thresh))

    if len(xs) == 0:
        raise RuntimeError("No ROI pixels found in the selected lower image region.")

    pts_img = np.stack([xs, ys], axis=1)

    if len(pts_img) > max_points:
        idx = np.random.choice(len(pts_img), size=max_points, replace=False)
        pts_img = pts_img[idx]

    pts_img = torch.tensor(pts_img, dtype=torch.float64)

    H_img_to_ground = torch.inverse(H_ground_to_image[0].to(torch.float64))
    pts_ground = apply_homography(H_img_to_ground, pts_img)

    pts_local, origin = latlon_to_local_xy(pts_ground)

    x = pts_local[:, 0].cpu().numpy()
    y = pts_local[:, 1].cpu().numpy()

    # robust bounds
    x_min = np.percentile(x, 1) - margin_m
    x_max = np.percentile(x, 99) + margin_m
    y_min = np.percentile(y, 1) - margin_m
    y_max = np.percentile(y, 99) + margin_m

    return (float(x_min), float(x_max), float(y_min), float(y_max)), origin

# -------------------------
# BEV projection
# -------------------------
def project_to_bev_local(
    features: torch.Tensor,   # (S, C, Hf, Wf)
    H_ground_to_image: torch.Tensor,  # (S, 3, 3)
    Hg: int,
    Wg: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    img_h: int,
    img_w: int,
    origin: tuple[float, float],
) -> torch.Tensor:
    """
    Project camera feature maps to a BEV grid defined in local metric coordinates.

    Returns:
        bev: (S, C, Hg, Wg)
    """
    S, C, Hf, Wf = features.shape
    device = features.device

    xs = torch.linspace(x_min, x_max, Wg, device=device, dtype=torch.float64)
    ys = torch.linspace(y_min, y_max, Hg, device=device, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    xy_local = torch.stack([grid_x, grid_y], dim=-1)  # (Hg, Wg, 2)
    latlon = local_xy_to_latlon(xy_local, origin)     # (Hg, Wg, 2)

    ones = torch.ones((Hg, Wg, 1), device=device, dtype=torch.float64)
    ground_pts = torch.cat([latlon, ones], dim=-1).reshape(-1, 3).T  # (3, N)

    H_ground_to_image = H_ground_to_image.to(device=device, dtype=torch.float64)
    proj = torch.bmm(H_ground_to_image, ground_pts.unsqueeze(0).expand(S, -1, -1))

    u = proj[:, 0, :] / (proj[:, 2, :] + 1e-12)
    v = proj[:, 1, :] / (proj[:, 2, :] + 1e-12)

    u_feat = u * (Wf / img_w)
    v_feat = v * (Hf / img_h)

    u_norm = 2.0 * (u_feat / (Wf - 1)) - 1.0
    v_norm = 2.0 * (v_feat / (Hf - 1)) - 1.0

    sample_grid = torch.stack([u_norm, v_norm], dim=-1).reshape(S, Hg, Wg, 2)

    bev = F.grid_sample(
        features.to(torch.float32),
        sample_grid.to(torch.float32),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return bev


def project_valid_mask_local(
    H_ground_to_image: torch.Tensor,
    Hg: int,
    Wg: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    img_h: int,
    img_w: int,
    origin: tuple[float, float],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Compute which BEV cells project inside the image.

    Returns:
        valid_mask: (S, Hg, Wg)
    """
    xs = torch.linspace(x_min, x_max, Wg, device=device, dtype=torch.float64)
    ys = torch.linspace(y_min, y_max, Hg, device=device, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    xy_local = torch.stack([grid_x, grid_y], dim=-1)
    latlon = local_xy_to_latlon(xy_local, origin)

    ones = torch.ones((Hg, Wg, 1), device=device, dtype=torch.float64)
    ground_pts = torch.cat([latlon, ones], dim=-1).reshape(-1, 3).T

    H_ground_to_image = H_ground_to_image.to(device=device, dtype=torch.float64)
    proj = torch.bmm(
        H_ground_to_image,
        ground_pts.unsqueeze(0).expand(H_ground_to_image.shape[0], -1, -1),
    )

    u = proj[:, 0, :] / (proj[:, 2, :] + 1e-12)
    v = proj[:, 1, :] / (proj[:, 2, :] + 1e-12)

    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return valid.reshape(H_ground_to_image.shape[0], Hg, Wg).float()


# -------------------------
# BEV aggregation
# -------------------------
class CameraAttentionAggregation(nn.Module):
    """
    Camera-wise attention aggregation over projected BEV features.
    Input:  (B, S, C, Hg, Wg)
    Output: (B, C, Hg, Wg)
    """
    def __init__(self, in_channels: int, n_cameras: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * n_cameras, n_cameras, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        B, S, C, Hg, Wg = bev_features.shape

        cat = bev_features.reshape(B, S * C, Hg, Wg)
        attn = self.attention(cat).unsqueeze(2)   # (B, S, 1, Hg, Wg)
        fused = (bev_features * attn).sum(dim=1)  # (B, C, Hg, Wg)

        return self.proj(fused)


def local_xy_to_image_xy(
    detections_local,
    H_ground_to_image,
    origin,
):
    """
    detections_local: (N,3) [x_local, y_local, score]
    H_ground_to_image: (3,3)
    origin: (lat0, lon0)

    returns:
        image_dets: (N,3) [u, v, score]
    """
    if detections_local.shape[0] == 0:
        return detections_local

    xy_local = detections_local[:, :2]
    score = detections_local[:, 2]

    latlon = local_xy_to_latlon(xy_local, origin)   # (N,2)
    uv = apply_homography(H_ground_to_image, latlon)

    return torch.cat([uv, score.unsqueeze(1)], dim=1)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    video_path = r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c001\vdo.avi"
    xml_path = r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c001\gt\gt.txt"
    calibration_path = r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c001\calibration.txt"
    roi_path = r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01\c001\roi.jpg"


    img_h, img_w = 1080, 1920
    cell_size = 0.4

    dataset = AICityDataset(video_path=video_path, xml_path=xml_path)
    model = ConvNeXtFirstStage(pretrained=True).eval()

    H_ground_to_image = load_homography(calibration_path)

    (x_min, x_max, y_min, y_max), origin = estimate_local_bev_bounds_from_roi(
        roi_path=roi_path,
        H_ground_to_image=H_ground_to_image,
        margin_m=3.0,
        max_points=3000,
    )
    
    # (x_min, x_max, y_min, y_max), origin = estimate_local_bev_bounds_from_image(
    # H_ground_to_image= H_ground_to_image,
    #     img_h= img_h,
    #     img_w= img_w)
    # (x_min, x_max, y_min, y_max), origin = estimate_local_bev_bounds(
    #     dataset=dataset,
    #     H_ground_to_image=H_ground_to_image,
    #     max_frames=200,
    #     margin_m=5.0,
    # )
    Wg = int((x_max - x_min) / cell_size)
    Hg = int((y_max - y_min) / cell_size)

    print("ROI-based BEV bounds:")
    print(x_min, x_max, y_min, y_max)
    print("Origin:", origin)
    print("Grid size:", Hg, Wg)

    image_tensor, target = dataset[0]

    with torch.no_grad():
        features = model(image_tensor)  # expected: (1, C, Hf, Wf)

    valid_mask = project_valid_mask_local(
        H_ground_to_image=H_ground_to_image,
        Hg=Hg,
        Wg=Wg,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        img_h=img_h,
        img_w=img_w,
        origin=origin,
    )

    bev = project_to_bev_local(
        features=features,
        H_ground_to_image=H_ground_to_image,
        Hg=Hg,
        Wg=Wg,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        img_h=img_h,
        img_w=img_w,
        origin=origin,
    )

    # Optional visualization
    visualize_features(features, title="ConvNeXt features (camera view)")
    visualize_bev(bev, title="BEV projected features")
    visualize_side_by_side(features, bev)

    # Example single-camera aggregation path
    bev_batch = bev.unsqueeze(0)  # (B=1, S=1, C, Hg, Wg)
    aggregator = CameraAttentionAggregation(in_channels=bev.shape[1], n_cameras=bev.shape[0])
    bev_fused = aggregator(bev_batch)

    decoder = BEVResNet18Decoder(
    in_channels=bev_fused.shape[1],
    base_channels=128,
    out_channels=bev_fused.shape[1],
    )

    bev_decoded = decoder(bev_fused)

    print("Projected BEV:", bev.shape)
    print("Fused BEV:", bev_fused.shape)
    print("Decoded BEV:", bev_decoded.shape)

    pred_module = BEVPredictionModule(
        in_channels=128,
        reid_dim=64,
        hidden_channels=128,
    )

    preds = pred_module(bev_decoded)

    print(preds["center_logits"].shape)   # (B,1,Hg,Wg)
    print(preds["center_prob"].shape)     # (B,1,Hg,Wg)
    print(preds["offsets"].shape)         # (B,2,Hg,Wg)
    print(preds["reid_embedding"].shape)  # (B,64,Hg,Wg)

    dets = decode_bev_centers(
        preds["center_prob"],
        preds["offsets"],
        score_thresh=0.3,
        max_det=50,
    )

    visualize_bev_detections(preds["center_prob"], dets[0], title="Predicted BEV centers")

    dets = decode_bev_centers(preds["center_prob"], preds["offsets"], score_thresh=0.3)
    dets_local = bev_grid_to_local_xy(dets[0], x_min, x_max, y_min, y_max, Wg, Hg)
    dets_img = local_xy_to_image_xy(dets_local, H_ground_to_image[0].double(), origin)

    draw_points_on_image(image_tensor, dets_img)