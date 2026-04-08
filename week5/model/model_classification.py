"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F


#Local imports
from model.modules import BaseRGBModel, FCLayers, step




class AttentionPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        weights = self.attn(x)                  # [B, T, 1]
        weights = torch.softmax(weights, dim=1) # normalize over time
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
            dilation = 2 ** i   # 1, 2, 4
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

        # Short-range branch
        self.branch_short = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
                dilation=1
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Medium-range branch
        self.branch_medium = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=hidden_dim,
                kernel_size=5,
                padding=2,
                dilation=1
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Long-range branch
        self.branch_long = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=4,
                dilation=4
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Fuse concatenated branches
        self.fuse = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim * 3,
                out_channels=hidden_dim,
                kernel_size=1
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Optional residual projection
        self.residual = None
        if in_dim != hidden_dim:
            self.residual = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        self.out_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]

        out_short = self.branch_short(x)
        out_medium = self.branch_medium(x)
        out_long = self.branch_long(x)

        out = torch.cat([out_short, out_medium, out_long], dim=1)
        out = self.fuse(out)

        res = x if self.residual is None else self.residual(x)
        out = torch.relu(out + res)

        out = out.transpose(1, 2)
        return out
    

class TemporalShift1D(nn.Module):
    def __init__(self, fold_div=4):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x):
        """
        x: [B, T, D]
        Shift part of the feature channels along time.
        """
        B, T, D = x.shape
        fold = D // self.fold_div

        if fold == 0:
            return x

        out = torch.zeros_like(x)

        # Shift one part left: time t gets info from t+1
        out[:, :-1, :fold] = x[:, 1:, :fold]

        # Shift one part right: time t gets info from t-1
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]

        # Keep the remaining channels unchanged
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out
    
class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features
            self.temporal_head = getattr(args, 'temporal_head', 'max_pool')

            self.use_temporal_shift = getattr(args, 'use_temporal_shift', False)
            self.temporal_shift = None

            if self.use_temporal_shift:
                self.temporal_shift = TemporalShift1D(
                    fold_div=getattr(args, 'temporal_shift_fold_div', 4)
                )

            tcn_hidden_dim = args.tcn_hidden_dim if args.tcn_hidden_dim is not None else self._d

            self.temporal_net = None

            if self.temporal_head == 'max_pool':
                self.temporal_pool = None
                fc_in_dim = self._d

            elif self.temporal_head == 'attn_pool':
                self.temporal_pool = AttentionPool1D(self._d)
                fc_in_dim = self._d

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

            elif self.temporal_head == 'tcn_max_pool':
                self.temporal_net = TemporalConvNet(
                    in_dim=self._d,
                    hidden_dim=tcn_hidden_dim,
                    num_layers=args.tcn_num_layers,
                    kernel_size=args.tcn_kernel_size,
                    dropout=args.tcn_dropout
                )
                self.temporal_pool = None
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

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
                        
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d)  # [B, T, D]

            if self.temporal_shift is not None:
                im_feat = im_feat + self.temporal_shift(im_feat)
            
            if self.temporal_head == 'max_pool':
                im_feat = torch.max(im_feat, dim=1)[0]   # [B, D]

            elif self.temporal_head == 'attn_pool':
                im_feat = self.temporal_pool(im_feat)    # [B, D]

            elif self.temporal_head == 'tcn_attn':
                im_feat = self.temporal_net(im_feat)     # [B, T, H]
                im_feat = self.temporal_pool(im_feat)    # [B, H]

                
            elif self.temporal_head == 'tcn_max_pool':
                im_feat = self.temporal_net(im_feat)     # [B, T, H]
                im_feat = torch.max(im_feat, dim=1)[0]   # [B, H]
   
   
            elif self.temporal_head == 'ms_tcn_attn':
                im_feat = self.temporal_net(im_feat)     # [B, T, H]
                im_feat = self.temporal_pool(im_feat)    # [B, H]
                
            im_feat = self._fc(im_feat)

            return im_feat 
        
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
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).float()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    loss = self.compute_loss(pred, label)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.sigmoid(pred)
            
            return pred.cpu().numpy()
    
    #ADDED MULTICLIP
    def predict_logits(self, seq):
        """
        seq:
            [L, C, H, W] for one clip
            or
            [K, L, C, H, W] for multiple clips
        returns:
            torch.Tensor of shape [1, C] or [K, C]
        """
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
                logits = self._model(seq)

        return logits.detach().cpu()
