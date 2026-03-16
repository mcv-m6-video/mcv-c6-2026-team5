import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


def make_res_layer(in_channels: int, out_channels: int, num_blocks: int, stride: int):
    layers = [BasicBlock2D(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock2D(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class FPNFusionBlock(nn.Module):
    """
    Upsample top feature, concat with lateral feature, fuse with conv.
    """
    def __init__(self, lateral_channels: int, top_channels: int, out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(lateral_channels + top_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lateral: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        top_up = F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([lateral, top_up], dim=1)
        return self.fuse(x)


class BEVResNet18Decoder(nn.Module):
    """
    ResNet-18 style BEV decoder:
      input  : (B, Cg, Hg, Wg)
      output : (B, C_out, Hg, Wg)

    Downsamples BEV features multiple times, then upsamples with FPN-style fusion.
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 128,
        out_channels: int | None = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        # Optional stem to normalize channel width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # ResNet-18 style BEV encoder
        # Each layer downsamples by 2, as described in the paper
        self.layer1 = make_res_layer(base_channels, base_channels, num_blocks=2, stride=2)       # 1/2
        self.layer2 = make_res_layer(base_channels, base_channels * 2, num_blocks=2, stride=2)   # 1/4
        self.layer3 = make_res_layer(base_channels * 2, base_channels * 4, num_blocks=2, stride=2)  # 1/8
        self.layer4 = make_res_layer(base_channels * 4, base_channels * 8, num_blocks=2, stride=2)  # 1/16

        # Lateral projections for FPN
        self.lat4 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)
        self.lat3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=1)
        self.lat2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1)
        self.lat1 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.lat0 = nn.Conv2d(base_channels, base_channels, kernel_size=1)

        # Top-down fusion
        self.fuse3 = FPNFusionBlock(
            lateral_channels=base_channels * 4,
            top_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )
        self.fuse2 = FPNFusionBlock(
            lateral_channels=base_channels * 2,
            top_channels=base_channels * 4,
            out_channels=base_channels * 2,
        )
        self.fuse1 = FPNFusionBlock(
            lateral_channels=base_channels,
            top_channels=base_channels * 2,
            out_channels=base_channels,
        )
        self.fuse0 = FPNFusionBlock(
            lateral_channels=base_channels,
            top_channels=base_channels,
            out_channels=base_channels,
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cg, Hg, Wg)
        """
        x0 = self.stem(x)        # (B, base,   H,   W)
        x1 = self.layer1(x0)     # (B, base,   H/2, W/2)
        x2 = self.layer2(x1)     # (B, 2base,  H/4, W/4)
        x3 = self.layer3(x2)     # (B, 4base,  H/8, W/8)
        x4 = self.layer4(x3)     # (B, 8base,  H/16,W/16)

        p4 = self.lat4(x4)
        p3 = self.fuse3(self.lat3(x3), p4)
        p2 = self.fuse2(self.lat2(x2), p3)
        p1 = self.fuse1(self.lat1(x1), p2)
        p0 = self.fuse0(self.lat0(x0), p1)

        out = self.out_proj(p0)
        return out