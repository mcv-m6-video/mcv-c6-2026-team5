"""
File containing the main spotting model.

This version extends the baseline with optional frame-preserving temporal
modules inspired by the previous week's classification experiments:
- Temporal Shift over frame embeddings
- Single-scale dilated TCN neck
- Multi-scale TCN neck

"""

import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from model.modules import BaseRGBModel, FCLayers, step

def multiclass_focal_loss(logits, targets, weight=None, gamma=2.0, alpha=None):
    """
    logits: [N, C]
    targets: [N]
    weight: optional class weights tensor [C]
    gamma: focal focusing parameter
    alpha: optional scalar multiplier
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=weight)
    pt = torch.exp(-ce_loss)  # pt = predicted prob of true class
    focal_loss = ((1 - pt) ** gamma) * ce_loss

    if alpha is not None:
        focal_loss = alpha * focal_loss

    return focal_loss.mean()

class TemporalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length.")

        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = None if in_dim == out_dim else nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B, D, T]
        out = self.conv(x)
        out = self.bn(out)
        res = x if self.residual is None else self.residual(x)
        out = self.relu(out + res)
        out = self.dropout(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, kernel_size=3, dropout=0.2, dilations=None):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        if dilations is None:
            dilations = [2 ** i for i in range(num_layers)]
        elif len(dilations) != num_layers:
            raise ValueError("Length of dilations must match num_layers")

        layers = []
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            layers.append(
                TemporalConvBlock(
                    in_dim=in_ch,
                    out_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    dropout=dropout,
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
    def __init__(self, in_dim, hidden_dim, dropout=0.2, branch_dilations=(1, 2, 4), branch_kernel_sizes=(3, 3, 3)):
        super().__init__()
        if len(branch_dilations) != len(branch_kernel_sizes):
            raise ValueError("branch_dilations and branch_kernel_sizes must have same length")

        branches = []
        for dilation, kernel_size in zip(branch_dilations, branch_kernel_sizes):
            if kernel_size % 2 == 0:
                raise ValueError("All multi-scale kernel sizes must be odd")
            padding = ((kernel_size - 1) // 2) * dilation
            branches.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * len(branches), hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = None if in_dim == hidden_dim else nn.Conv1d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.out_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        outs = [branch(x) for branch in self.branches]
        out = self.fuse(torch.cat(outs, dim=1))
        res = x if self.residual is None else self.residual(x)
        out = torch.relu(out + res)
        out = out.transpose(1, 2)
        return out


class TemporalShift1D(nn.Module):
    def __init__(self, fold_div=4):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x):
        # x: [B, T, D]
        bsz, timesteps, dim = x.shape
        fold = dim // self.fold_div
        if fold == 0 or timesteps <= 1:
            return x

        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                 # shift left
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold] # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]            # keep rest
        return out


class Model(BaseRGBModel):

    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim
            else:
                raise NotImplementedError(self._feature_arch)

            self._features = features

            # Temporal modeling options (all frame-preserving)
            self.temporal_head = getattr(args, 'temporal_head', 'identity')
            self.use_temporal_shift = getattr(args, 'use_temporal_shift', False)
            self.temporal_shift = (
                TemporalShift1D(fold_div=getattr(args, 'temporal_shift_fold_div', 4))
                if self.use_temporal_shift else None
            )

            tcn_hidden_dim = getattr(args, 'tcn_hidden_dim', None) or self._d
            self.temporal_net = None
            fc_in_dim = self._d

            if self.temporal_head == 'identity':
                pass
            elif self.temporal_head == 'tcn':
                dilation_list = getattr(args, 'tcn_dilations', None)
                self.temporal_net = TemporalConvNet(
                    in_dim=self._d,
                    hidden_dim=tcn_hidden_dim,
                    num_layers=getattr(args, 'tcn_num_layers', 3),
                    kernel_size=getattr(args, 'tcn_kernel_size', 3),
                    dropout=getattr(args, 'tcn_dropout', 0.2),
                    dilations=dilation_list,
                )
                fc_in_dim = self.temporal_net.out_dim
            elif self.temporal_head == 'ms_tcn':
                self.temporal_net = MultiScaleTCN(
                    in_dim=self._d,
                    hidden_dim=tcn_hidden_dim,
                    dropout=getattr(args, 'tcn_dropout', 0.2),
                    branch_dilations=tuple(getattr(args, 'ms_tcn_dilations', [1, 2, 4])),
                    branch_kernel_sizes=tuple(getattr(args, 'ms_tcn_kernel_sizes', [3, 3, 3])),
                )
                fc_in_dim = self.temporal_net.out_dim
            else:
                raise ValueError(f"Unknown temporal_head: {self.temporal_head}")

            # Per-frame classifier for spotting.
            # Optional tiny temporal attention
            self.use_temporal_attention = getattr(args, 'use_temporal_attention', False)

            if self.use_temporal_attention:
                self._temporal_attn = nn.Linear(fc_in_dim, 1)
            else:
                self._temporal_attn = None
                
            self._fc = FCLayers(fc_in_dim, args.num_classes + 1)
            self.use_actionness = getattr(args, 'use_actionness', False)

            if self.use_actionness:
                self._actionness_fc = FCLayers(fc_in_dim, 1)
            else:
                self._actionness_fc = None
                
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x):
            x = self.normalize(x)
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = self.augment(x)

            x = self.standarize(x)
            im_feat = self._features(x.view(-1, channels, height, width)).reshape(batch_size, clip_len, self._d)

            if self.temporal_shift is not None:
                im_feat = im_feat + self.temporal_shift(im_feat)

            if self.temporal_net is not None:
                im_feat = self.temporal_net(im_feat)
            
            if self._temporal_attn is not None:
                attn_logits = self._temporal_attn(im_feat)   # [B, T, 1]
                attn = torch.softmax(attn_logits, dim=1)     # normalize over time
                im_feat = im_feat * (1.0 + attn)             # residual-style gating
                

            class_logits = self._fc(im_feat)

            if self._actionness_fc is not None:
                actionness_logits = self._actionness_fc(im_feat)
                return {
                    "class_logits": class_logits,
                    "actionness_logits": actionness_logits
                }

            return {
                "class_logits": class_logits,
                "actionness_logits": None
            }

        def normalize(self, x):
            return x / 255.

        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:', sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args
        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor([1.0] + [5.0] * self._num_classes, dtype=torch.float32).to(self.device)
        epoch_loss = 0.0

        with torch.no_grad() if optimizer is None else nullcontext():
            for batch in tqdm(loader):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).long()

                with torch.cuda.amp.autocast():
                    output = self._model(frame)

                    if isinstance(output, dict):
                        class_logits_seq = output["class_logits"]   # [B, T, C+1]
                        actionness_logits = output["actionness_logits"]
                    else:
                        class_logits_seq = output
                        actionness_logits = None

                    class_logits = class_logits_seq.view(-1, self._num_classes + 1)
                    label_flat = label.view(-1)
                    if getattr(self._args, "use_focal_loss", False):
                        loss_cls = multiclass_focal_loss(
                            class_logits,
                            label_flat,
                            weight=weights,
                            gamma=self._args.focal_gamma,
                            alpha=self._args.focal_alpha,
                        )
                    else:
                        loss_cls = F.cross_entropy(
                            class_logits,
                            label_flat,
                            reduction='mean',
                            weight=weights
                        )

                    loss = loss_cls

                    # Smoothness loss
                    if getattr(self._args, "use_smoothness_loss", False):
                        probs = torch.softmax(class_logits_seq, dim=-1)   # [B, T, C+1]
                        smoothness_loss = ((probs[:, 1:, :] - probs[:, :-1, :]) ** 2).mean()
                        loss = loss + self._args.smoothness_loss_weight * smoothness_loss

                    # Keep this only if you still have actionness in the model
                    if actionness_logits is not None:
                        actionness_logits = actionness_logits.view(-1)
                        actionness_target = (label_flat != 0).float()

                        loss_act = F.binary_cross_entropy_with_logits(
                            actionness_logits,
                            actionness_target
                        )

                        loss = loss + self._args.actionness_loss_weight * loss_act

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = self._model(seq)

            class_logits = output["class_logits"]              # [B, T, C+1]
            actionness_logits = output["actionness_logits"]    # [B, T, 1] or None

            class_probs = torch.softmax(class_logits, dim=-1)  # [B, T, C+1]

            if actionness_logits is not None:
                actionness_probs = torch.sigmoid(actionness_logits)  # [B, T, 1]

                # Modulate only action classes, keep background separate
                alpha = getattr(self._args, "actionness_inference_alpha", 1.0)
                class_probs[..., 1:] = class_probs[..., 1:] * (actionness_probs ** alpha)

                # Renormalize so probabilities sum to 1 again
                class_probs = class_probs / (class_probs.sum(dim=-1, keepdim=True) + 1e-8)

            return class_probs.cpu().numpy()
