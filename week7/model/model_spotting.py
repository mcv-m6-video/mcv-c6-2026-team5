"""
File containing the main model.
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


class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch

            # ── Backbones RegNet (baseline) ──────────────────────────
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim
                self._features = features
                self._is_3d = False

            # ── X3D-M base ───────────────────────────────────────────
            elif self._feature_arch == 'x3d_m':
                x3d = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    'x3d_m', pretrained=True)
                # Quitar la head de clasificación (último bloque)
                self._features = nn.Sequential(*list(x3d.blocks[:-1]))
                self._spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
                # Dimensión de salida de X3D-M antes de la head
                self._d = 192
                self._is_3d = True

            # ── X3D-M + BiGRU ────────────────────────────────────────
            elif self._feature_arch == 'x3d_m_gru':
                x3d = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    'x3d_m', pretrained=True)
                self._features = nn.Sequential(*list(x3d.blocks[:-1]))
                self._spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
                self._d = 192
                self._is_3d = True

                gru_hidden = args.gru_hidden if hasattr(args, 'gru_hidden') else 256
                gru_layers = args.gru_layers if hasattr(args, 'gru_layers') else 2

                self._gru = nn.GRU(
                    input_size=self._d,
                    hidden_size=gru_hidden,
                    num_layers=gru_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.2 if gru_layers > 1 else 0.0
                )
                # La salida del GRU es hidden*2 (bidireccional)
                self._d = gru_hidden * 2

            else:
                raise NotImplementedError(args.feature_arch)

            # Cabeza de clasificación (compartida por todos los modelos)
            self._fc = FCLayers(self._d, args.num_classes + 1)

            # Augmentaciones (solo para modelos 2D frame a frame)
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            self.standarization = T.Compose([
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x):
            x = self.normalize(x)  # [0, 1]

            if self._is_3d:
                return self._forward_3d(x)
            else:
                return self._forward_2d(x)

        def _forward_2d(self, x):
            """Pipeline original frame a frame (RegNet)."""
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = self.augment(x)
            x = self.standarize(x)

            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d)

            im_feat = self._fc(im_feat)
            return im_feat

        def _forward_3d(self, x):
            """Pipeline X3D-M: input [B, T, C, H, W] → [B, T, C+1]."""
            batch_size, clip_len, channels, height, width = x.shape

            # X3D-M espera [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)

            if self.training:
                x = self._augment_3d(x)

            # Normalización ImageNet sobre la dimensión de canal
            x = self._standarize_3d(x)

            # Backbone X3D-M → [B, D, T', H', W']
            feat = self._features(x)

            # Spatial pooling → [B, D, T']
            feat = self._spatial_pool(feat)
            feat = feat.squeeze(-1).squeeze(-1)  # [B, D, T']

            # Si T' != clip_len, interpolamos de vuelta a clip_len
            if feat.shape[2] != clip_len:
                feat = F.interpolate(
                    feat,
                    size=clip_len,
                    mode='linear',
                    align_corners=False
                )  # [B, D, clip_len]

            feat = feat.permute(0, 2, 1)  # [B, clip_len, D]

            # GRU (solo en x3d_m_gru)
            if hasattr(self, '_gru'):
                feat, _ = self._gru(feat)  # [B, clip_len, hidden*2]

            # Clasificación
            out = self._fc(feat)  # [B, clip_len, num_classes+1]
            return out

        def _augment_3d(self, x):
            """Augmentación para input 3D [B, C, T, H, W]."""
            # Aplicamos las mismas augmentaciones frame a frame
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            for i in range(B):
                x[i] = self.augmentation(x[i])
            return x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        def _standarize_3d(self, x):
            """Normalización ImageNet para input 3D [B, C, T, H, W]."""
            mean = torch.tensor(
                [0.485, 0.456, 0.406],
                device=x.device).view(1, 3, 1, 1, 1)
            std = torch.tensor(
                [0.229, 0.224, 0.225],
                device=x.device).view(1, 3, 1, 1, 1)
            return (x - mean) / std

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
            print('Model params:',
                  sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if (torch.cuda.is_available()
                and "device" in args
                and args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args
        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None,
              lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor(
            [1.0] + [5.0] * self._num_classes,
            dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1)
                    label = label.view(-1)
                    loss = F.cross_entropy(
                        pred, label,
                        reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                         lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)
            pred = torch.softmax(pred, dim=-1)
            return pred.cpu().numpy()