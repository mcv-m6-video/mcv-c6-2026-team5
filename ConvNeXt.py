import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ConvNeXtFirstStage(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_base(weights=weights)
        self.stage0 = model.features[0]
        self.stage1 = model.features[1]

        if weights is not None:
            t = weights.transforms()
            self.mean = torch.tensor(t.mean).view(1, 3, 1, 1)
            self.std = torch.tensor(t.std).view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        x = (x - mean) / std
        x = self.stage0(x)
        x = self.stage1(x)
        return x