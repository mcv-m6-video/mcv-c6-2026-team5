import os
import sys
from pathlib import Path
import requests
import torch
from tqdm import tqdm
import cv2

current_dir = Path(__file__).resolve().parent
neuflow_repo_path = current_dir / 'neuflow_v2'

if str(neuflow_repo_path) not in sys.path:
    sys.path.insert(0, str(neuflow_repo_path))

from NeuFlow.backbone_v7 import ConvBlock

def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)

def check_model(model_name="neuflow_mixed"):
    model_dir = "src/optical_flow/neuflow_v2/"
    model_path = f"{model_dir}/{model_name}.pth"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        url = f"https://github.com/neufieldrobotics/NeuFlow_v2/raw/master/{model_name}.pth"
        download(url, model_path)

    return model_path

def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def fuse_model_conv_and_bn(model):
    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
            delattr(m, "norm1")
            delattr(m, "norm2")
            m.forward = m.forward_fuse
    return model