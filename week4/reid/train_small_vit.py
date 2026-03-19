import os
import re
import random
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import timm


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    csv_path: str = "reid_annotations.csv"
    save_dir: str = "./checkpoints_reid_cityflow"

    train_scenes: tuple = ("S01", "S04")
    val_scenes: tuple = ("S03",)

    image_size: int = 224
    num_workers: int = 4

    batch_size: int = 64
    num_instances: int = 4   # K in PK sampler
    epochs: int = 40

    model_name: str = "vit_tiny_patch16_224"
    embedding_dim: int = 256
    pretrained: bool = True
    dropout: float = 0.0

    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4

    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    supcon_weight: float = 0.5
    temperature: float = 0.07

    min_images_per_id: int = 2
    amp: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def extract_scene_from_path(path: str) -> str:
    """
    Example:
      /.../images/train/S01_c001_frame_0055.jpg -> S01
    """
    m = re.search(r"(S\d+)_c\d+_frame_", str(path), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def extract_camera_from_path(path: str) -> str:
    """
    Example:
      /.../S01_c001_frame_0055.jpg -> c001
    """
    m = re.search(r"S\d+_(c\d+)_frame_", str(path), flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return "unknown"


# ============================================================
# Data preparation
# ============================================================

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"image_path", "identity", "source_frame_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df["identity"] = df["identity"].astype(str)
    df["scene"] = df["source_frame_path"].apply(extract_scene_from_path)
    df["camera"] = df["source_frame_path"].apply(extract_camera_from_path)

    return df


def build_train_val_split(df: pd.DataFrame, train_scenes, val_scenes):
    train_df = df[df["scene"].isin(train_scenes)].copy()
    val_df = df[df["scene"].isin(val_scenes)].copy()

    if len(train_df) == 0:
        raise ValueError(f"No train rows found for scenes {train_scenes}")
    if len(val_df) == 0:
        raise ValueError(f"No val rows found for scenes {val_scenes}")

    return train_df, val_df


def filter_ids_with_min_samples(df: pd.DataFrame, min_images_per_id: int):
    counts = df["identity"].value_counts()
    keep_ids = counts[counts >= min_images_per_id].index
    return df[df["identity"].isin(keep_ids)].copy()


# ============================================================
# Dataset
# ============================================================

class ReIDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, relabel=False):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

        self.image_paths = self.df["image_path"].tolist()
        self.identity_strings = self.df["identity"].tolist()
        self.cameras = self.df["camera"].tolist()
        self.scenes = self.df["scene"].tolist()

        if relabel:
            unique_ids = sorted(set(self.identity_strings))
            self.id2label = {id_str: i for i, id_str in enumerate(unique_ids)}
            self.labels = [self.id2label[x] for x in self.identity_strings]
        else:
            self.id2label = None
            self.labels = self.identity_strings

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return (
            img,
            self.labels[idx],
            self.cameras[idx],
            self.scenes[idx],
            self.image_paths[idx],
        )


# ============================================================
# PK Sampler
# ============================================================

class PKSampler(Sampler):
    """
    Samples P identities x K images each.
    batch_size = P * K
    """

    def __init__(self, labels, batch_size, num_instances):
        self.labels = labels
        self.batch_size = batch_size
        self.num_instances = num_instances

        assert batch_size % num_instances == 0
        self.num_pids_per_batch = batch_size // num_instances

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        label_to_pool = {
            label: inds.copy() for label, inds in self.label_to_indices.items()
        }

        for label in label_to_pool:
            random.shuffle(label_to_pool[label])

        available_labels = self.unique_labels.copy()
        final_indices = []

        while len(available_labels) >= self.num_pids_per_batch:
            selected_labels = random.sample(available_labels, self.num_pids_per_batch)
            batch = []

            for label in selected_labels:
                inds = label_to_pool[label]

                if len(inds) >= self.num_instances:
                    chosen = inds[:self.num_instances]
                    label_to_pool[label] = inds[self.num_instances:]
                else:
                    chosen = inds.copy()
                    while len(chosen) < self.num_instances:
                        chosen.append(random.choice(self.label_to_indices[label]))
                    label_to_pool[label] = []

                batch.extend(chosen)

            final_indices.extend(batch)

            new_available = []
            for label in available_labels:
                if len(label_to_pool[label]) >= 1:
                    new_available.append(label)
            available_labels = new_available

        return iter(final_indices)

    def __len__(self):
        return len(self.labels)


# ============================================================
# Model
# ============================================================

class ViTEmbeddingModel(nn.Module):
    def __init__(self, model_name, embedding_dim, pretrained=True, dropout=0.0):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        feat_dim = self.backbone.num_features

        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.embedding = nn.Linear(feat_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)

        nn.init.normal_(self.embedding.weight, std=0.001)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.bn(feat)
        feat = self.dropout(feat)

        emb = self.embedding(feat)
        emb = self.embedding_bn(emb)
        emb_norm = l2_normalize(emb)

        return {
            "embeddings": emb,
            "embeddings_norm": emb_norm,
        }


# ============================================================
# Losses
# ============================================================

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        dist = torch.cdist(embeddings, embeddings, p=2)
        n = dist.size(0)

        labels = labels.view(n, 1)
        mask_pos = labels.eq(labels.t())
        mask_neg = ~mask_pos

        eye = torch.eye(n, device=embeddings.device, dtype=torch.bool)
        mask_pos = mask_pos & ~eye

        dist_ap = torch.where(mask_pos, dist, torch.full_like(dist, -1e9)).max(dim=1)[0]
        dist_an = torch.where(mask_neg, dist, torch.full_like(dist, 1e9)).min(dim=1)[0]

        valid = (dist_ap > -1e8) & (dist_an < 1e8)
        if valid.sum() == 0:
            return embeddings.sum() * 0.0

        return F.relu(dist_ap[valid] - dist_an[valid] + self.margin).mean()


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        device = embeddings.device
        n = embeddings.size(0)

        sim = torch.matmul(embeddings, embeddings.t()) / self.temperature

        eye = torch.eye(n, dtype=torch.bool, device=device)
        labels = labels.view(n, 1)
        mask_pos = labels.eq(labels.t()) & ~eye

        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        exp_sim = torch.exp(sim) * (~eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = mask_pos.sum(dim=1)
        valid = pos_counts > 0
        if valid.sum() == 0:
            return embeddings.sum() * 0.0

        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (pos_counts + 1e-12)
        return -mean_log_prob_pos[valid].mean()


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()

    all_embs = []
    all_ids = []
    all_cams = []

    for images, labels, cameras, scenes, paths in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        embs = out["embeddings_norm"].cpu()

        all_embs.append(embs)
        all_ids.extend(labels)
        all_cams.extend(cameras)

    all_embs = torch.cat(all_embs, dim=0)
    return all_embs, all_ids, all_cams


@torch.no_grad()
def compute_retrieval_metrics(model, loader, device, exclude_same_camera=False):
    embs, ids, cams = extract_embeddings(model, loader, device)
    dist = torch.cdist(embs, embs, p=2)
    n = dist.size(0)

    aps = []
    correct_rank1 = 0
    valid_queries = 0

    for i in range(n):
        order = torch.argsort(dist[i])

        matches = []
        for j in order.tolist():
            if j == i:
                continue
            if exclude_same_camera and cams[j] == cams[i]:
                continue
            matches.append(j)

        if len(matches) == 0:
            continue

        match_labels = torch.tensor([1.0 if ids[j] == ids[i] else 0.0 for j in matches])
        num_rel = int(match_labels.sum().item())

        if num_rel == 0:
            continue

        valid_queries += 1
        if match_labels[0].item() == 1.0:
            correct_rank1 += 1

        cumsum = torch.cumsum(match_labels, dim=0)
        ranks = torch.arange(1, len(matches) + 1, dtype=torch.float32)
        precision_at_k = cumsum / ranks
        ap = (precision_at_k * match_labels).sum() / num_rel
        aps.append(ap.item())

    rank1 = correct_rank1 / max(valid_queries, 1)
    mAP = sum(aps) / max(len(aps), 1)

    return {
        "rank1": rank1,
        "mAP": mAP,
        "num_queries": valid_queries,
    }


# ============================================================
# Train
# ============================================================

def train_one_epoch(model, loader, optimizer, tri_loss_fn, supcon_loss_fn, scaler, device):
    model.train()

    total_loss = 0.0
    total_tri = 0.0
    total_supcon = 0.0
    n_batches = 0

    for images, labels, cameras, scenes, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=CFG.amp and device.startswith("cuda")):
            out = model(images)
            emb = out["embeddings_norm"]

            tri = tri_loss_fn(emb, labels)
            supcon = supcon_loss_fn(emb, labels)
            loss = CFG.triplet_weight * tri + CFG.supcon_weight * supcon

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_tri += tri.item()
        total_supcon += supcon.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "triplet": total_tri / max(n_batches, 1),
        "supcon": total_supcon / max(n_batches, 1),
    }


# ============================================================
# Main
# ============================================================

def main():
    set_seed(CFG.seed)
    os.makedirs(CFG.save_dir, exist_ok=True)

    df = load_dataframe(CFG.csv_path)
    train_df, val_df = build_train_val_split(df, CFG.train_scenes, CFG.val_scenes)

    train_df = filter_ids_with_min_samples(train_df, CFG.min_images_per_id)

    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Train IDs:  {train_df['identity'].nunique()}")
    print(f"Val IDs:    {val_df['identity'].nunique()}")

    train_tfms = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((CFG.image_size, CFG.image_size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ReIDDataset(train_df, transform=train_tfms, relabel=True)
    val_ds = ReIDDataset(val_df, transform=val_tfms, relabel=False)

    train_sampler = PKSampler(
        labels=train_ds.labels,
        batch_size=CFG.batch_size,
        num_instances=CFG.num_instances,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        sampler=train_sampler,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    model = ViTEmbeddingModel(
        model_name=CFG.model_name,
        embedding_dim=CFG.embedding_dim,
        pretrained=CFG.pretrained,
        dropout=CFG.dropout,
    ).to(CFG.device)

    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.bn.parameters()) +
        list(model.embedding.parameters()) +
        list(model.embedding_bn.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CFG.lr_backbone},
            {"params": head_params, "lr": CFG.lr_head},
        ],
        weight_decay=CFG.weight_decay,
    )

    tri_loss_fn = BatchHardTripletLoss(margin=CFG.triplet_margin)
    supcon_loss_fn = SupervisedContrastiveLoss(temperature=CFG.temperature)

    scaler = torch.cuda.amp.GradScaler(enabled=CFG.amp and CFG.device.startswith("cuda"))

    best_map = -1.0

    for epoch in range(1, CFG.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            tri_loss_fn,
            supcon_loss_fn,
            scaler,
            CFG.device,
        )

        val_stats = compute_retrieval_metrics(
            model,
            val_loader,
            CFG.device,
            exclude_same_camera=False,
        )

        print(
            f"Epoch [{epoch}/{CFG.epochs}] | "
            f"train_loss={train_stats['loss']:.4f} "
            f"(tri={train_stats['triplet']:.4f}, supcon={train_stats['supcon']:.4f}) | "
            f"val_rank1={val_stats['rank1']:.4f} | "
            f"val_mAP={val_stats['mAP']:.4f} | "
            f"queries={val_stats['num_queries']}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": CFG.__dict__,
        }

        torch.save(ckpt, os.path.join(CFG.save_dir, "last.pt"))

        if val_stats["mAP"] > best_map:
            best_map = val_stats["mAP"]
            torch.save(ckpt, os.path.join(CFG.save_dir, "best.pt"))
            print(f"  -> saved new best checkpoint (mAP={best_map:.4f})")


if __name__ == "__main__":
    main()