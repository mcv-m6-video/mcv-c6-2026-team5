"""
Two-stream RGB + Optical Flow classification model.

Design:
- RGB stream: per-frame 2D backbone -> temporal head -> logits
- Flow stream: per-frame 2D backbone -> temporal head -> logits
- Late fusion at logit level
"""

# Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

# Local imports
from model.modules import BaseRGBModel, FCLayers, step


# ---------------------------------------------------------------------
# Temporal pooling / temporal blocks
# ---------------------------------------------------------------------

class AttentionPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        weights = self.attn(x)                  # [B, T, 1]
        weights = torch.softmax(weights, dim=1)
        pooled = (x * weights).sum(dim=1)       # [B, D]
        return pooled


class TemporalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.residual = None
        if in_dim != out_dim:
            self.residual = nn.Conv1d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, D, T]
        out = self.conv(x)
        out = self.bn(out)

        res = x if self.residual is None else self.residual(x)

        out = self.relu(out + res)
        out = self.dropout(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, kernel_size=3, dropout=0.2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            dilation = 2 ** i  # 1, 2, 4 for 3 layers
            layers.append(
                TemporalConvBlock(
                    in_dim=in_ch,
                    out_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)   # [B, D, T]
        x = self.net(x)         # [B, H, T]
        x = x.transpose(1, 2)   # [B, T, H]
        return x


class MultiScaleTCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()

        self.branch_short = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.branch_medium = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.branch_long = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.residual = None
        if in_dim != hidden_dim:
            self.residual = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        self.out_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)   # [B, D, T]

        out_short = self.branch_short(x)
        out_medium = self.branch_medium(x)
        out_long = self.branch_long(x)

        out = torch.cat([out_short, out_medium, out_long], dim=1)
        out = self.fuse(out)

        res = x if self.residual is None else self.residual(x)
        out = torch.relu(out + res)

        out = out.transpose(1, 2)  # [B, T, H]
        return out


# ---------------------------------------------------------------------
# Single stream
# ---------------------------------------------------------------------

class SingleStreamModel(nn.Module):
    """
    One stream for either RGB or Flow.

    Input:
        x: [B, T, C, H, W]
    Output:
        logits: [B, num_classes]
    """
    def __init__(self, args=None, in_chans=3, modality='rgb'):
        super().__init__()
        self.modality = modality
        self._feature_arch = args.feature_arch

        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            model_name = {
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]]

            features = timm.create_model(
                model_name,
                pretrained=True,
                in_chans=in_chans
            )
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(self._feature_arch)

        self._features = features
        self.temporal_head = getattr(args, 'temporal_head', 'tcn_maxpool')

        tcn_hidden_dim = args.tcn_hidden_dim if args.tcn_hidden_dim is not None else self._d

        self.temporal_net = None
        self.temporal_pool = None

        if self.temporal_head == 'max_pool':
            fc_in_dim = self._d

        elif self.temporal_head == 'attn_pool':
            self.temporal_pool = AttentionPool1D(self._d)
            fc_in_dim = self._d

        elif self.temporal_head == 'tcn_maxpool':
            self.temporal_net = TemporalConvNet(
                in_dim=self._d,
                hidden_dim=tcn_hidden_dim,
                num_layers=args.tcn_num_layers,
                kernel_size=args.tcn_kernel_size,
                dropout=args.tcn_dropout
            )
            fc_in_dim = tcn_hidden_dim

        elif self.temporal_head == 'tcn_attn':
            self.temporal_net = TemporalConvNet(
                in_dim=self._d,
                hidden_dim=tcn_hidden_dim,
                num_layers=args.tcn_num_layers,
                kernel_size=args.tcn_kernel_size,
                dropout=args.tcn_dropout
            )
            self.temporal_pool = AttentionPool1D(tcn_hidden_dim)
            fc_in_dim = tcn_hidden_dim

        elif self.temporal_head == 'ms_tcn_maxpool':
            self.temporal_net = MultiScaleTCN(
                in_dim=self._d,
                hidden_dim=tcn_hidden_dim,
                dropout=args.tcn_dropout
            )
            fc_in_dim = tcn_hidden_dim

        elif self.temporal_head == 'ms_tcn_attn':
            self.temporal_net = MultiScaleTCN(
                in_dim=self._d,
                hidden_dim=tcn_hidden_dim,
                dropout=args.tcn_dropout
            )
            self.temporal_pool = AttentionPool1D(tcn_hidden_dim)
            fc_in_dim = tcn_hidden_dim

        else:
            raise ValueError(f"Unknown temporal_head: {self.temporal_head}")

        self._fc = FCLayers(fc_in_dim, args.num_classes)

        # RGB-only augmentation. Disabled for flow to avoid modality mismatch.
        if modality == 'rgb':
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                # IMPORTANT: no horizontal flip here, because flow would also need sign correction
            ])
            self.standardization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
            ])
        else:
            self.augmentation = None
            self.flow_clip_value = getattr(args, 'flow_clip_value', 20.0)

    def forward(self, x):
        # x: [B, T, C, H, W]
        if self.modality == 'rgb':
            x = self.normalize_rgb(x)
            if self.training and self.augmentation is not None:
                x = self.augment_rgb(x)
            x = self.standardize_rgb(x)
        else:
            x = self.normalize_flow(x)

        batch_size, clip_len, channels, height, width = x.shape

        feat = self._features(
            x.view(-1, channels, height, width)
        ).reshape(batch_size, clip_len, self._d)  # [B, T, D]

        if self.temporal_head == 'max_pool':
            feat = torch.max(feat, dim=1)[0]

        elif self.temporal_head == 'attn_pool':
            feat = self.temporal_pool(feat)

        elif self.temporal_head in ['tcn_maxpool', 'ms_tcn_maxpool']:
            feat = self.temporal_net(feat)         # [B, T, H]
            feat = torch.max(feat, dim=1)[0]       # [B, H]

        elif self.temporal_head in ['tcn_attn', 'ms_tcn_attn']:
            feat = self.temporal_net(feat)         # [B, T, H]
            feat = self.temporal_pool(feat)        # [B, H]

        logits = self._fc(feat)
        return logits

    def normalize_rgb(self, x):
        return x / 255.0

    def augment_rgb(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standardize_rgb(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standardization(x[i])
        return x

    def normalize_flow(self, x):
        # Assumes flow already dequantized back to real values in roughly [-flow_clip_value, flow_clip_value]
        return x / self.flow_clip_value


# ---------------------------------------------------------------------
# Two-stream wrapper
# ---------------------------------------------------------------------

class Model(BaseRGBModel):

    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()

            self.fusion_alpha = getattr(args, 'fusion_alpha', 0.5)

            self.rgb_stream = SingleStreamModel(args=args, in_chans=3, modality='rgb')
            self.flow_stream = SingleStreamModel(args=args, in_chans=2, modality='flow')

        def forward(self, rgb, flow):
            rgb_logits = self.rgb_stream(rgb)      # [B, C]
            flow_logits = self.flow_stream(flow)   # [B, C]

            fused_logits = self.fusion_alpha * rgb_logits + (1.0 - self.fusion_alpha) * flow_logits
            return fused_logits, rgb_logits, flow_logits

        def print_stats(self):
            print('Model params:',
                  sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

        self._loss_type = getattr(args, 'loss_type', 'bce')

        pos_weight = getattr(args, 'pos_weight', None)
        if pos_weight is not None:
            self._pos_weight = torch.tensor(
                pos_weight, dtype=torch.float32, device=self.device
            )
        else:
            self._pos_weight = None

    def focal_bce_with_logits(self, logits, targets, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt).pow(gamma)

        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = alpha_t * focal_weight

        return (focal_weight * bce).mean()

    def compute_loss(self, pred, label):
        if self._loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(pred, label)

        elif self._loss_type == 'weighted_bce':
            if self._pos_weight is None:
                raise ValueError("weighted_bce selected but pos_weight is None")
            return F.binary_cross_entropy_with_logits(
                pred, label, pos_weight=self._pos_weight
            )

        elif self._loss_type == 'focal_bce':
            return self.focal_bce_with_logits(pred, label)

        else:
            raise ValueError(f"Unknown loss_type: {self._loss_type}")

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.0

        with torch.no_grad() if optimizer is None else nullcontext():
            for batch in tqdm(loader):
                rgb = batch['frame'].to(self.device).float()
                flow = batch['flow'].to(self.device).float()
                label = batch['label'].to(self.device).float()

                with torch.cuda.amp.autocast():
                    fused_logits, rgb_logits, flow_logits = self._model(rgb, flow)
                    loss = self.compute_loss(fused_logits, label)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict_logits(self, rgb, flow):
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.FloatTensor(rgb)
        if not isinstance(flow, torch.Tensor):
            flow = torch.FloatTensor(flow)

        if len(rgb.shape) == 4:   # [T, 3, H, W]
            rgb = rgb.unsqueeze(0)
        if len(flow.shape) == 4:  # [T, 2, H, W]
            flow = flow.unsqueeze(0)

        if rgb.device != self.device:
            rgb = rgb.to(self.device)
        if flow.device != self.device:
            flow = flow.to(self.device)

        rgb = rgb.float()
        flow = flow.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                fused_logits, rgb_logits, flow_logits = self._model(rgb, flow)

        return fused_logits.detach().cpu(), rgb_logits.detach().cpu(), flow_logits.detach().cpu()

    def predict(self, rgb, flow):
        fused_logits, _, _ = self.predict_logits(rgb, flow)
        pred = torch.sigmoid(fused_logits)
        return pred.numpy()