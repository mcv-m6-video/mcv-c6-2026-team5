import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvNeXt import ConvNeXtFirstStage
from pathlib import Path
import os
from torch.utils.data import Dataset, DataLoader
from multi_camera_loaders import MultiCameraDataset, AICityDataset
import cv2
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
# ------------------------------------------------------------------ #
# 1. BEV Projection  (Eq. 2 from paper)
# ------------------------------------------------------------------ #
def project_to_bev(features, H_mats, Hg, Wg, x_min, y_min, cell_cm=40.0):
    S, C, H_img, W_img = features.shape
    device = features.device

    # Build grid in float64 for numerical precision
    xs_world = torch.arange(Wg, device=device, dtype=torch.float64) * cell_cm + x_min
    ys_world = torch.arange(Hg, device=device, dtype=torch.float64) * cell_cm + y_min
    grid_y, grid_x = torch.meshgrid(ys_world, xs_world, indexing="ij")
    ones       = torch.ones_like(grid_x)
    ground_pts = torch.stack([grid_x, grid_y, ones], dim=-1).reshape(-1, 3).T  # (3, N) float64

    # Invert in float64 for precision, then convert to float32 for bmm
    H_world_to_img = torch.linalg.inv(H_mats.double()).float()  # (S, 3, 3) float32
    ground_pts_f32 = ground_pts.float()                          # (3, N)    float32

    uvs = torch.bmm(H_world_to_img, ground_pts_f32.unsqueeze(0).expand(S, -1, -1))
    u = uvs[:, 0, :] / (uvs[:, 2, :] + 1e-8)
    v = uvs[:, 1, :] / (uvs[:, 2, :] + 1e-8)

    u_norm = (u / (W_img - 1)) * 2.0 - 1.0
    v_norm = (v / (H_img - 1)) * 2.0 - 1.0

    sample_grid = torch.stack([u_norm, v_norm], dim=-1).reshape(S, Hg, Wg, 2)

    return F.grid_sample(
        features, sample_grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )

def scale_homography(H: torch.Tensor, orig_size: tuple, target_size: tuple) -> torch.Tensor:
    """
    Adjust homography when the image is resized.
    orig_size / target_size: (H, W)
    """
    orig_h,   orig_w   = orig_size
    target_h, target_w = target_size

    scale_u = target_w / orig_w   # x-axis scale
    scale_v = target_h / orig_h   # y-axis scale

    # Scale matrix: maps original pixel coords → resized pixel coords
    S = torch.tensor([
        [scale_u, 0,       0],
        [0,       scale_v, 0],
        [0,       0,       1],
    ], dtype=torch.float32)

    return S @ H  # (3, 3)

# ------------------------------------------------------------------ #
# 2. Aggregation: channel-cat + two Conv2d  (Section 3.2)
# ------------------------------------------------------------------ #
class BEVAggregation(nn.Module):
    def __init__(self, in_channels: int, n_cameras: int, out_channels: int = 128):
        super().__init__()
        combined = in_channels * n_cameras
        mid      = out_channels + (combined - out_channels) // 2
        self.convs = nn.Sequential(
            nn.Conv2d(combined, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        # bev: (B, S, C, Hg, Wg)
        B, S, C, Hg, Wg = bev.shape
        x = bev.reshape(B, S * C, Hg, Wg)   # channel-cat across cameras
        return self.convs(x)                  # (B, 128, Hg, Wg)


# ------------------------------------------------------------------ #
# 3. ResNet-18 + FPN decoder  (Section 3.3)
# ------------------------------------------------------------------ #
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class BEVDecoder(nn.Module):
    def __init__(self, in_channels: int = 128):
        super().__init__()
        # Encoder (2 stages — safe for 50×37 input)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), ResBlock(256),
        )  # Hg/2, Wg/2
        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), ResBlock(512),
        )  # Hg/4, Wg/4

        # FPN laterals
        self.lat2 = nn.Conv2d(512, 128, 1)
        self.lat1 = nn.Conv2d(256, 128, 1)

        # FPN merges (after upsample + concat)
        self.merge1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.merge0 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 128, Hg, Wg)
        e1 = self.enc1(x)    # (B, 256, Hg/2, Wg/2)
        e2 = self.enc2(e1)   # (B, 512, Hg/4, Wg/4)

        p2 = self.lat2(e2)   # (B, 128, Hg/4, Wg/4)

        p1 = self.merge1(torch.cat([
            self.lat1(e1),
            F.interpolate(p2, size=e1.shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1))            # (B, 128, Hg/2, Wg/2)

        p0 = self.merge0(torch.cat([
            x,
            F.interpolate(p1, size=x.shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1))            # (B, 128, Hg, Wg)

        return p0


# ------------------------------------------------------------------ #
# 4. Heads: detection heatmap + ReID  (Section 3.4, FairMOT-style)
# ------------------------------------------------------------------ #
class DetectionHead(nn.Module):
    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),  # (dx, dy) sub-cell offset
        )

    def forward(self, x):
        return self.heatmap(x), self.offset(x)


class ReIDHead(nn.Module):
    def __init__(self, in_channels: int = 128, reid_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, reid_dim, 1),
        )

    def forward(self, x):
        return F.normalize(self.head(x), dim=1)  # L2-normalised


# ------------------------------------------------------------------ #
# 5. Full EarlyBird model
# ------------------------------------------------------------------ #

class EarlyBird(nn.Module):
    def __init__(
        self,
        n_cameras:     int   = 4,
        feat_channels: int   = 128,
        bev_channels:  int   = 128,
        reid_dim:      int   = 128,
        Hg:            int   = 100,
        Wg:            int   = 100,
        x_min:         float = 0.0,
        y_min:         float = 0.0,
        cell_cm:       float = 40.0,
    ):
        super().__init__()
        self.Hg      = Hg
        self.Wg      = Wg
        self.x_min   = x_min
        self.y_min   = y_min
        self.cell_cm = cell_cm

        self.aggregation = BEVAggregation(feat_channels, n_cameras, bev_channels)
        self.decoder     = BEVDecoder(bev_channels)
        self.det_head    = DetectionHead(bev_channels)
        self.reid_head   = ReIDHead(bev_channels, reid_dim)

    def forward(self, features: torch.Tensor, H_mats: torch.Tensor):
        B, S, C, H_img, W_img = features.shape

        bev_views = []
        for b in range(B):
            bev = project_to_bev(
                features[b], H_mats[b],
                self.Hg, self.Wg,
                self.x_min, self.y_min, self.cell_cm,
            )
            bev_views.append(bev)
        bev = torch.stack(bev_views, dim=0)  # (B, S, C, Hg, Wg)

        fused   = self.aggregation(bev)
        decoded = self.decoder(fused)

        heatmap, offset = self.det_head(decoded)
        reid_emb        = self.reid_head(decoded)

        return heatmap, offset, reid_emb


# ------------------------------------------------------------------ #
# 6. Loss  (MSE heatmap + L1 offset + triplet ReID)
# ------------------------------------------------------------------ #
class EarlyBirdLoss(nn.Module):
    def __init__(self, w_det=1.0, w_off=1.0, w_reid=1.0):
        super().__init__()
        self.w_det  = w_det
        self.w_off  = w_off
        self.w_reid = w_reid

    def forward(self, heatmap, offset, reid_emb, gt_heatmap, gt_offset, gt_mask):
        # Detection loss: MSE on Gaussian heatmap
        det_loss = F.mse_loss(heatmap, gt_heatmap)

        # Offset loss: L1 only at ground-truth locations
        off_loss = (F.l1_loss(offset, gt_offset, reduction="none") * gt_mask).sum() \
                   / (gt_mask.sum() + 1e-8)

        # ReID loss: triplet loss at detection locations (simplified)
        reid_loss = self._triplet_loss(reid_emb, gt_mask)

        total = self.w_det * det_loss + self.w_off * off_loss + self.w_reid * reid_loss
        return total, {"det": det_loss, "offset": off_loss, "reid": reid_loss}

    def _triplet_loss(self, reid_emb, gt_mask, margin=0.3):
        # Extract embeddings at positive (pedestrian) locations
        # reid_emb: (B, D, Hg, Wg),  gt_mask: (B, 1, Hg, Wg)
        B, D, Hg, Wg = reid_emb.shape
        emb_flat  = reid_emb.permute(0, 2, 3, 1).reshape(B, -1, D)  # (B, N, D)
        mask_flat = gt_mask.reshape(B, -1).bool()                     # (B, N)

        loss = torch.tensor(0.0, device=reid_emb.device)
        count = 0
        for b in range(B):
            pos_emb = emb_flat[b][mask_flat[b]]   # (P, D)
            if pos_emb.shape[0] < 2:
                continue
            # All pairwise distances among positives (anchor-positive pairs)
            dist = torch.cdist(pos_emb, pos_emb)  # (P, P)
            ap   = dist.mean()
            # Negatives: all non-pedestrian locations
            neg_emb = emb_flat[b][~mask_flat[b]]  # (N, D)
            if neg_emb.shape[0] == 0:
                continue
            an = torch.cdist(pos_emb, neg_emb).min(dim=1).values.mean()
            loss  += F.relu(ap - an + margin)
            count += 1

        return loss / max(count, 1)


# ------------------------------------------------------------------ #
# 1. Decode heatmap peaks → detections in BEV grid coordinates
# ------------------------------------------------------------------ #
def decode_heatmap(
    heatmap:    torch.Tensor,   # (B, 1, Hg, Wg)
    offset:     torch.Tensor,   # (B, 2, Hg, Wg)
    reid_emb:   torch.Tensor,   # (B, D, Hg, Wg)
    threshold:  float = 0.3,
    max_dets:   int   = 100,
) -> list[dict]:
    """
    Returns a list of detections per batch item:
    [
        {
            "bev_coords": (N, 2)  float  grid coords (x, y) with sub-cell offset,
            "scores":     (N,)    float  heatmap confidence,
            "embeddings": (N, D)  float  L2-normalised ReID embeddings,
        },
        ...  # one dict per batch item
    ]
    """
    B, _, Hg, Wg = heatmap.shape

    # NMS: suppress non-peak cells via max-pooling
    heatmap_nms = heatmap * (
        F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1) == heatmap
    ).float()

    results = []
    for b in range(B):
        hm   = heatmap_nms[b, 0]             # (Hg, Wg)
        off  = offset[b]                      # (2, Hg, Wg)
        emb  = reid_emb[b]                    # (D, Hg, Wg)

        # Find cells above threshold
        mask  = hm > threshold                # (Hg, Wg)
        ys, xs = mask.nonzero(as_tuple=True)  # peak grid positions

        if len(xs) == 0:
            results.append({"bev_coords": torch.zeros(0, 2),
                            "scores":     torch.zeros(0),
                            "embeddings": torch.zeros(0, emb.shape[0])})
            continue

        scores = hm[ys, xs]

        # Keep top-k if too many detections
        if len(scores) > max_dets:
            topk   = scores.topk(max_dets).indices
            xs, ys, scores = xs[topk], ys[topk], scores[topk]

        # Add sub-cell offset for finer localisation
        dx = off[0, ys, xs]   # (N,)
        dy = off[1, ys, xs]   # (N,)
        bev_x = xs.float() + dx
        bev_y = ys.float() + dy

        # Extract ReID embeddings at peak locations
        embeddings = emb[:, ys, xs].T   # (N, D)

        results.append({
            "bev_coords": torch.stack([bev_x, bev_y], dim=1),   # (N, 2)
            "scores":     scores,
            "embeddings": embeddings,
        })

    return results


# ------------------------------------------------------------------ #
# 2. Convert BEV grid coords → world metres
# ------------------------------------------------------------------ #
def bev_to_world(
    bev_coords: torch.Tensor,   # (N, 2) grid indices
    x_min:      float,          # cm
    y_min:      float,          # cm
    cell_cm:    float = 40.0,
) -> torch.Tensor:              # (N, 2) metres
    world_cm = bev_coords * cell_cm + torch.tensor([x_min, y_min])
    return world_cm / 100.0     # cm → metres


# ------------------------------------------------------------------ #
# 3. Simple tracker using Hungarian matching on position + ReID
# ------------------------------------------------------------------ #
class Track:
    def __init__(self, track_id: int, world_pos: np.ndarray, embedding: np.ndarray):
        self.id        = track_id
        self.pos       = world_pos       # (2,) metres
        self.embedding = embedding       # (D,)
        self.hits      = 1
        self.misses    = 0

    def update(self, world_pos: np.ndarray, embedding: np.ndarray):
        self.pos       = world_pos
        self.embedding = embedding
        self.hits     += 1
        self.misses    = 0


class BEVTracker:
    def __init__(
        self,
        max_misses:     int   = 5,
        pos_weight:     float = 0.5,   # balance position vs ReID in cost
        dist_threshold: float = 2.0,   # metres — max allowed match distance
    ):
        self.tracks        = []
        self.next_id       = 1
        self.max_misses    = max_misses
        self.pos_weight    = pos_weight
        self.dist_threshold = dist_threshold

    def update(self, world_positions: np.ndarray, embeddings: np.ndarray) -> list[int]:
        """
        Args:
            world_positions: (N, 2) metres
            embeddings:      (N, D) L2-normalised

        Returns:
            ids: list of N track IDs for each detection
        """
        if len(self.tracks) == 0 or len(world_positions) == 0:
            return self._init_new_tracks(world_positions, embeddings)

        # Build cost matrix: weighted sum of position distance + ReID distance
        track_pos  = np.array([t.pos       for t in self.tracks])  # (M, 2)
        track_emb  = np.array([t.embedding for t in self.tracks])  # (M, D)

        # Position cost: euclidean distance in metres
        pos_cost = np.linalg.norm(
            world_positions[:, None] - track_pos[None], axis=2
        )   # (N, M)

        # ReID cost: cosine distance (1 - cosine_similarity)
        sim      = embeddings @ track_emb.T                         # (N, M)
        reid_cost = 1.0 - sim                                       # (N, M)

        cost = self.pos_weight * pos_cost + (1 - self.pos_weight) * reid_cost  # (N, M)

        # Hungarian matching
        det_idx, trk_idx = linear_sum_assignment(cost)

        assigned_ids   = [-1] * len(world_positions)
        matched_tracks = set()

        for d, t in zip(det_idx, trk_idx):
            if pos_cost[d, t] > self.dist_threshold:
                continue   # too far — treat as new detection
            self.tracks[t].update(world_positions[d], embeddings[d])
            assigned_ids[d] = self.tracks[t].id
            matched_tracks.add(t)

        # Unmatched detections → new tracks
        for d in range(len(world_positions)):
            if assigned_ids[d] == -1:
                track = Track(self.next_id, world_positions[d], embeddings[d])
                self.tracks.append(track)
                assigned_ids[d] = self.next_id
                self.next_id += 1

        # Unmatched tracks → increment misses, prune dead tracks
        for t, track in enumerate(self.tracks):
            if t not in matched_tracks:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return assigned_ids

    def _init_new_tracks(self, positions, embeddings):
        ids = []
        for pos, emb in zip(positions, embeddings):
            track = Track(self.next_id, pos, emb)
            self.tracks.append(track)
            ids.append(self.next_id)
            self.next_id += 1
        return ids


# ------------------------------------------------------------------ #
# 4. Project BEV world position back to each camera image
# ------------------------------------------------------------------ #
def world_to_image(
    world_pos_m:  torch.Tensor,   # (N, 2) metres
    H_mat:        torch.Tensor,   # (3, 3) image → world CM (original resolution)
    img_h:        int,            # target height (480)
    img_w:        int,            # target width  (640)
    orig_h:       int,            # original height (1080 or 960)
    orig_w:       int,            # original width  (1920 or 1280)
    bbox_size_px: int = 50,
) -> torch.Tensor:
    N = world_pos_m.shape[0]
    if N == 0:
        return torch.zeros(0, 4)

    world_cm = world_pos_m * 100.0
    ones = torch.ones(N, 1, device=world_pos_m.device)
    pts  = torch.cat([world_cm, ones], dim=1).T   # (3, N)

    H_inv = torch.linalg.inv(H_mat)
    uvs   = H_inv @ pts                            # (3, N) in original resolution

    u = uvs[0] / (uvs[2] + 1e-8)
    v = uvs[1] / (uvs[2] + 1e-8)

    # Scale from original resolution → target resolution
    u = u * (img_w / orig_w)
    v = v * (img_h / orig_h)

    valid = uvs[2] > 0
    half  = bbox_size_px // 2

    x1 = (u - half).clamp(0, img_w);  x2 = (u + half).clamp(0, img_w)
    y1 = (v - half).clamp(0, img_h);  y2 = (v + half).clamp(0, img_h)

    x1[~valid] = 0;  x2[~valid] = 0
    y1[~valid] = 0;  y2[~valid] = 0

    return torch.stack([x1, y1, x2, y2], dim=1)


# ------------------------------------------------------------------ #
# 5. Visualise on each camera frame
# ------------------------------------------------------------------ #
import cv2

COLORS = [
    (255, 0,   0),   (0, 255,   0),   (0,   0, 255),
    (255, 255, 0),   (0, 255, 255),   (255, 0, 255),
    (128, 0,   255), (255, 128,  0),
]

def draw_detections(
    frames:      torch.Tensor,   # (S, 3, H, W)  float [0,1]
    track_ids:   list[int],
    bboxes_per_cam: list[torch.Tensor],  # S x (N, 4)
) -> list[np.ndarray]:
    images = []
    S = frames.shape[0]
    for s in range(S):
        img = (frames[s].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bboxes = bboxes_per_cam[s]
        for i, (bbox, tid) in enumerate(zip(bboxes, track_ids)):
            x1, y1, x2, y2 = bbox.int().tolist()
            color = COLORS[tid % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"ID {tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        images.append(img)
    return images


if __name__ == "__main__":

    TARGET_SIZE = (480, 640)
    model = ConvNeXtFirstStage(pretrained=True)
    ROOT_PATH = Path(r"c:\Users\maiol\Desktop\Master\C6\project\AI_CITY_CHALLENGE_2022_TRAIN\train\S01")
    sequences = sorted([x for x in ROOT_PATH.iterdir() if x.is_dir()])

    homographies = []
    datasets     = []

    # ── Pass 1: load homographies and datasets ──────────────────────────
    for sequence in sequences:
        with open(os.path.join(sequence, "calibration.txt")) as f:
            for line in f:
                line       = line.split(";")
                first_row  = list(map(float, line[0].split()[2:]))
                second_row = list(map(float, line[1].split()))
                third_row  = list(map(float, line[2].split()))
                break

        H_raw = torch.tensor([first_row, second_row, third_row], dtype=torch.float64)
        homographies.append(H_raw)

        cam_dataset = AICityDataset(
            video_path=os.path.join(sequence, "vdo.avi"),
            calibration_path=os.path.join(sequence, "calibration.txt"),
            target_size=TARGET_SIZE,
        )
        datasets.append(cam_dataset)

    # ── Pass 2: compute world extent ────────────────────────────────────
    orig_sizes = []
    cam_ranges = []

    for i, sequence in enumerate(sequences):
        cap = cv2.VideoCapture(os.path.join(sequence, "vdo.avi"))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        orig_sizes.append((orig_h, orig_w))

        H_raw = homographies[i]
        visible_wx, visible_wy = [], []
        for v_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for u_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                pt  = torch.tensor([orig_w * u_frac, orig_h * v_frac, 1.0], dtype=torch.float64)
                uvs = H_raw @ pt
                wx  = (uvs[0] / uvs[2]).item()
                wy  = (uvs[1] / uvs[2]).item()
                if abs(wx) < 10000 and abs(wy) < 5000:
                    visible_wx.append(wx)
                    visible_wy.append(wy)

        if visible_wx:
            cam_ranges.append((min(visible_wx), max(visible_wx),
                               min(visible_wy), max(visible_wy)))

    # Union of all cameras for X (road length)
    margin  = 200
    x_min   = min(r[0] for r in cam_ranges) - margin
    x_max   = max(r[1] for r in cam_ranges) + margin
    cell_cm = 40.0
    Wg      = int((x_max - x_min) / cell_cm)

    # Y axis is compressed — expand manually to 10m for tracking
    y_raw_center = (min(r[2] for r in cam_ranges) + max(r[3] for r in cam_ranges)) / 2
    y_min = y_raw_center - 500
    y_max = y_raw_center + 500
    Hg    = int((y_max - y_min) / cell_cm)   # 25

    print(f"BEV grid: Wg={Wg}, Hg={Hg}")
    print(f"  X: {x_min:.0f} to {x_max:.0f} cm  ({(x_max-x_min)/100:.1f} m)")
    print(f"  Y: {y_min:.0f} to {y_max:.0f} cm  ({(y_max-y_min)/100:.1f} m)")

    # ── Build dataset and model ──────────────────────────────────────────
    full_dataset = MultiCameraDataset(datasets, homographies)
    dataloader   = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=0)

    earlybird = EarlyBird(
        n_cameras=len(sequences),
        feat_channels=128,
        Hg=Hg,
        Wg=Wg,
        x_min=x_min,
        y_min=y_min,
        cell_cm=cell_cm,
    )
    tracker = BEVTracker(max_misses=5, pos_weight=0.5, dist_threshold=2.0)

    # ── Inference loop ───────────────────────────────────────────────────

    for frames, H_mats in dataloader:
        
        S      = frames.shape[1]
        frames = frames.squeeze(0)

        with torch.no_grad():
            features = torch.stack(
                [model(frames[s].unsqueeze(0)) for s in range(S)], dim=1
            )
            heatmap, offset, reid_emb = earlybird(
                features,
                torch.stack(homographies).unsqueeze(0)  # float64, not from dataloader
            )

        det = {
            "bev_coords": torch.tensor([[
                (3530.0 - x_min) / cell_cm,
                (204.0  - y_min) / cell_cm,
            ]]),
            "scores":     torch.tensor([1.0]),
            "embeddings": torch.zeros(1, 128),
        }

        world_pos    = bev_to_world(
            det["bev_coords"], x_min=x_min, y_min=y_min, cell_cm=cell_cm
        ).cpu().numpy()
        world_tensor = torch.from_numpy(world_pos).double()

        H0    = homographies[0]
        H_inv = torch.linalg.inv(H0)

        # Test point: camera 1 image center
        u_test, v_test = 960.0, 540.0

        # Convention A: H maps image→world  (what we assumed)
        pt_img = torch.tensor([u_test, v_test, 1.0], dtype=torch.float64)
        w_a    = H0 @ pt_img
        wx_a   = (w_a[0] / w_a[2]).item()
        wy_a   = (w_a[1] / w_a[2]).item()
        print(f"Convention A (H=img→world): ({u_test},{v_test}) → world ({wx_a:.1f}, {wy_a:.1f})")

        # Now go back: world→image using H_inv
        pt_w   = torch.tensor([wx_a, wy_a, 1.0], dtype=torch.float64)
        back_a = H_inv @ pt_w
        print(f"  back w={back_a[2].item():.4f}  pixels=({(back_a[0]/back_a[2]).item():.1f}, {(back_a[1]/back_a[2]).item():.1f})")

        # Convention B: H maps world→image  (alternative)
        # Then H_inv maps image→world
        pt_img2 = torch.tensor([u_test, v_test, 1.0], dtype=torch.float64)
        w_b     = H_inv @ pt_img2
        wx_b    = (w_b[0] / w_b[2]).item()
        wy_b    = (w_b[1] / w_b[2]).item()
        print(f"\nConvention B (H=world→image): ({u_test},{v_test}) → world ({wx_b:.1f}, {wy_b:.1f})")

        # Now go back: world→image using H directly
        pt_w2  = torch.tensor([wx_b, wy_b, 1.0], dtype=torch.float64)
        back_b = H0 @ pt_w2
        print(f"  back w={back_b[2].item():.4f}  pixels=({(back_b[0]/back_b[2]).item():.1f}, {(back_b[1]/back_b[2]).item():.1f})")

        embeddings = det["embeddings"].cpu().numpy()
        track_ids  = tracker.update(world_pos, embeddings)

        bboxes_per_cam = [
            world_to_image(
                world_tensor, homographies[s],
                img_h=TARGET_SIZE[0], img_w=TARGET_SIZE[1],
                orig_h=orig_sizes[s][0], orig_w=orig_sizes[s][1],
            )
            for s in range(S)
        ]

        annotated = draw_detections(frames, track_ids, bboxes_per_cam)
        for s, img in enumerate(annotated):
            cv2.imshow(f"Camera {s+1}", img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()