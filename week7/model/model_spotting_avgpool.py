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
                self._is_unet = False
                self._has_gru = False

            # ── X3D-M base ───────────────────────────────────────────
            elif self._feature_arch == 'x3d_m':
                x3d = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    'x3d_m', pretrained=True)
                self._features = nn.Sequential(*list(x3d.blocks[:-1]))
                self._spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
                self._d = 192
                self._is_3d = True
                self._is_unet = False
                self._has_gru = False

            # ── X3D-M + BiGRU ────────────────────────────────────────
            elif self._feature_arch == 'x3d_m_gru':
                x3d = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    'x3d_m', pretrained=True)
                self._features = nn.Sequential(*list(x3d.blocks[:-1]))
                self._spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
                self._d = 192
                self._is_3d = True
                self._is_unet = False

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
                self._d = gru_hidden * 2
                self._has_gru = True

            # ── X3D-M + UNet con reducción temporal L→L'→L ──────────
            elif self._feature_arch == 'x3d_m_unet':
                x3d = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    'x3d_m', pretrained=True)

                blocks = list(x3d.blocks[:-1])
                self._stem   = blocks[0]  # out: [B, 24,  T, H/2,  W/2]
                self._stage1 = blocks[1]  # out: [B, 24,  T, H/4,  W/4]
                self._stage2 = blocks[2]  # out: [B, 48,  T, H/8,  W/8]
                self._stage3 = blocks[3]  # out: [B, 96,  T, H/16, W/16]
                self._stage4 = blocks[4]  # out: [B, 192, T, H/32, W/32]

                self._spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
                self._is_3d = True
                self._is_unet = True
                self._has_gru = False

                # Decoder con reducción temporal explícita:
                # e4: [B, 192, T=50] → MaxPool1d(2) → [B, 192, T'=25]
                # _dec3: ConvTranspose1d stride=2: T'=25 → T=50
                # Luego concat con skip e3 y refinar
                self._dec3_up = nn.ConvTranspose1d(
                    192, 96, kernel_size=4, stride=2, padding=1
                )  # [B, 192, 25] → [B, 96, 50]

                # self._dec3_refine = nn.Sequential(
                #     nn.Conv1d(96 + 96, 96, kernel_size=3, padding=1),
                #     nn.BatchNorm1d(96), nn.ReLU()
                # )  # [B, 96+96, 50] → [B, 96, 50]

                # self._dec2 = nn.Sequential(
                #     nn.Conv1d(96 + 48, 48, kernel_size=3, padding=1),
                #     nn.BatchNorm1d(48), nn.ReLU()
                # )  # [B, 96+48, 50] → [B, 48, 50]

                # self._dec1 = nn.Sequential(
                #     nn.Conv1d(48 + 24, 32, kernel_size=3, padding=1),
                #     nn.BatchNorm1d(32), nn.ReLU()
                # )  # [B, 48+24, 50] → [B, 32, 50]

                # self._d = 128

                self._dec3_refine = nn.Sequential(
                    nn.Conv1d(96 + 96, 96, kernel_size=3, padding=1),
                    nn.BatchNorm1d(96), nn.ReLU()
                )  # 192 → 96
                self._dec2 = nn.Sequential(
                    nn.Conv1d(96 + 48, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64), nn.ReLU()
                )  # 144 → 64
                self._dec1 = nn.Sequential(
                    nn.Conv1d(64 + 24, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64), nn.ReLU()
                )  # 88 → 64

                # Proyección hacia espacio del backbone (igual que Exp2)
                self._proj = nn.Linear(64, 192)

                self._d = 192  # igual que x3d_m y x3d_m_gru
                
            # ── X3D-M + UNet con reducción temporal L→L'→L + BiGRU ──
            elif self._feature_arch == 'x3d_m_unet_gru':
                x3d = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    'x3d_m', pretrained=True)

                blocks = list(x3d.blocks[:-1])
                self._stem   = blocks[0]
                self._stage1 = blocks[1]
                self._stage2 = blocks[2]
                self._stage3 = blocks[3]
                self._stage4 = blocks[4]

                self._spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
                self._is_3d = True
                self._is_unet = True
                self._has_gru = True

                self._dec3_up = nn.ConvTranspose1d(
                    192, 96, kernel_size=4, stride=2, padding=1
                )
                # self._dec3_refine = nn.Sequential(
                #     nn.Conv1d(96 + 96, 96, kernel_size=3, padding=1),
                #     nn.BatchNorm1d(96), nn.ReLU()
                # )
                # self._dec2 = nn.Sequential(
                #     nn.Conv1d(96 + 48, 48, kernel_size=3, padding=1),
                #     nn.BatchNorm1d(48), nn.ReLU()
                # )
                # self._dec1 = nn.Sequential(
                #     nn.Conv1d(48 + 24, 32, kernel_size=3, padding=1),
                #     nn.BatchNorm1d(32), nn.ReLU()
                # )

                # gru_hidden = args.gru_hidden if hasattr(args, 'gru_hidden') else 256
                # gru_layers = args.gru_layers if hasattr(args, 'gru_layers') else 2

                # self._gru = nn.GRU(
                #     input_size=128,
                #     hidden_size=gru_hidden,
                #     num_layers=gru_layers,
                #     batch_first=True,
                #     bidirectional=True,
                #     dropout=0.2 if gru_layers > 1 else 0.0
                # )
                # self._d = gru_hidden * 2

                self._dec3_refine = nn.Sequential(
                    nn.Conv1d(96 + 96, 96, kernel_size=3, padding=1),
                    nn.BatchNorm1d(96), nn.ReLU()
                )
                self._dec2 = nn.Sequential(
                    nn.Conv1d(96 + 48, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64), nn.ReLU()
                )
                self._dec1 = nn.Sequential(
                    nn.Conv1d(64 + 24, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64), nn.ReLU()
                )

                # Proyección hacia espacio del backbone
                self._proj = nn.Linear(64, 192)

                # GRU idéntico al Exp2
                gru_hidden = args.gru_hidden if hasattr(args, 'gru_hidden') else 256
                gru_layers = args.gru_layers if hasattr(args, 'gru_layers') else 2

                self._gru = nn.GRU(
                    input_size=192,  # ← igual que Exp2
                    hidden_size=gru_hidden,
                    num_layers=gru_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.2 if gru_layers > 1 else 0.0
                )
                self._d = gru_hidden * 2  # 512, igual que Exp2

            else:
                raise NotImplementedError(args.feature_arch)

            # Cabeza de clasificación (compartida por todos los modelos)
            self._fc = FCLayers(self._d, args.num_classes + 1)

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
            x = self.normalize(x)
            if self._is_3d:
                if self._is_unet:
                    return self._forward_3d_unet(x)
                else:
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
            """Pipeline X3D-M base y X3D-M+GRU."""
            batch_size, clip_len, channels, height, width = x.shape
            x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

            if self.training:
                x = self._augment_3d(x)
            x = self._standarize_3d(x)

            feat = self._features(x)          # [B, D, T', H', W']
            feat = self._spatial_pool(feat)   # [B, D, T', 1, 1]
            feat = feat.squeeze(-1).squeeze(-1)  # [B, D, T']

            if feat.shape[2] != clip_len:
                feat = F.interpolate(feat, size=clip_len,
                                     mode='linear', align_corners=False)

            feat = feat.permute(0, 2, 1)  # [B, T, D]

            if hasattr(self, '_gru'):
                feat, _ = self._gru(feat)

            return self._fc(feat)

        def _forward_3d_unet(self, x):
            """
            Pipeline X3D-M con UNet y reducción temporal explícita L→L'→L.

            Flujo temporal:
              Encoder stages: T=50 en todos (X3D-M no reduce T)
              Bottleneck: MaxPool1d(2) → T'=25  ← reducción explícita
              Decoder:    ConvTranspose1d(stride=2) → T=50  ← recuperación
              Skip connections preservan info de alta resolución temporal
            """
            batch_size, clip_len, channels, height, width = x.shape
            x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

            if self.training:
                x = self._augment_3d(x)
            x = self._standarize_3d(x)

            # ── ENCODER ─────────────────────────────────────────────
            s0 = self._stem(x)    # [B, 24,  T, H/2,  W/2]
            s1 = self._stage1(s0) # [B, 24,  T, H/4,  W/4]
            s2 = self._stage2(s1) # [B, 48,  T, H/8,  W/8]
            s3 = self._stage3(s2) # [B, 96,  T, H/16, W/16]
            s4 = self._stage4(s3) # [B, 192, T, H/32, W/32]

            # Spatial pooling → colapsa H y W, mantiene T
            def sp(feat):
                f = self._spatial_pool(feat)
                return f.squeeze(-1).squeeze(-1)  # [B, D, T]

            e4 = sp(s4)  # [B, 192, T=50]
            e3 = sp(s3)  # [B, 96,  T=50]
            e2 = sp(s2)  # [B, 48,  T=50]
            e1 = sp(s1)  # [B, 24,  T=50]

            # ── CUELLO DE BOTELLA: L=50 → L'=25 ─────────────────────
            # bottleneck = F.max_pool1d(e4, kernel_size=2, stride=2)
            bottleneck = F.avg_pool1d(e4, kernel_size=2, stride=2)

            # ── DECODER: L'=25 → L=50 ───────────────────────────────
            d3_up = self._dec3_up(bottleneck)
            if d3_up.shape[2] != e3.shape[2]:
                d3_up = F.interpolate(d3_up, size=e3.shape[2],
                                      mode='linear', align_corners=False)
            d3 = self._dec3_refine(torch.cat([d3_up, e3], dim=1))
            d2 = self._dec2(torch.cat([d3, e2], dim=1))
            d1 = self._dec1(torch.cat([d2, e1], dim=1))

            # if d1.shape[2] != clip_len:
            #     d1 = F.interpolate(d1, size=clip_len,
            #                        mode='linear', align_corners=False)

            # feat = d1.permute(0, 2, 1)

            # if self._has_gru:
            #     feat, _ = self._gru(feat)

            # return self._fc(feat)
            if d1.shape[2] != clip_len:
                d1 = F.interpolate(d1, size=clip_len,
                                mode='linear', align_corners=False)

            feat = d1.permute(0, 2, 1)  # [B, T, 64]

            # Proyección: 64 → 192 (mismo espacio que backbone X3D-M)
            feat = self._proj(feat)      # [B, T, 192]

            if self._has_gru:
                feat, _ = self._gru(feat)  # [B, T, 512]

            return self._fc(feat)

        def _augment_3d(self, x):
            """Augmentación para input 3D [B, C, T, H, W]."""
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4)
            for i in range(B):
                x[i] = self.augmentation(x[i])
            return x.permute(0, 2, 1, 3, 4)

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
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor(
            [1.0] + [5.0] * self._num_classes,
            dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device)

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1)

                    if label.dtype == torch.float32:
                        # ── TGLS: label suave [B*T, C+1] ────────────
                        # Usamos KL divergence: mide distancia entre
                        # la distribución predicha y la gaussiana suave
                        label_soft = label.view(-1, self._num_classes + 1)
                        log_pred = torch.log(
                            torch.softmax(pred, dim=-1) + 1e-8)
                        loss = F.kl_div(
                            log_pred,
                            label_soft,
                            reduction='batchmean')
                    else:
                        # ── Label duro: cross entropy normal ─────────
                        label_hard = label.view(-1)
                        loss = F.cross_entropy(
                            pred, label_hard,
                            reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                         lr_scheduler=lr_scheduler)

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
                pred = self._model(seq)
            pred = torch.softmax(pred, dim=-1)
            return pred.cpu().numpy()