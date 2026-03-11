import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from src.optical_flow.utils import check_model, fuse_model_conv_and_bn

current_dir = Path(__file__).resolve().parent
neuflow_repo_path = current_dir / 'neuflow_v2'

if str(neuflow_repo_path) not in sys.path:
    sys.path.insert(0, str(neuflow_repo_path))

from NeuFlow.neuflow import NeuFlow

def initialize_neuflow(device: str = "cuda", half: bool = False, input_shape=(432, 768)):
    model = NeuFlow()
    model_path = check_model("neuflow_mixed") #neuflow_things  or neuflow_sintel
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model = fuse_model_conv_and_bn(model)
    model.eval()
    
    if half:
        model.half()

    model.init_bhwd(1, input_shape[0], input_shape[1], device, half)
    return model

def compute_neuflow(model, img1: torch.Tensor, img2: torch.Tensor, device: str = "cuda", half: bool = False) -> torch.Tensor:
    image_width = 768
    image_height = 432
    
    img1 = img1.to(device).float()
    img2 = img2.to(device).float()

    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)

    img1 = img1[:, [2, 1, 0], :, :]
    img2 = img2[:, [2, 1, 0], :, :]

    img1_resized = F.interpolate(img1, size=(image_height, image_width), mode='bilinear', align_corners=False)
    img2_resized = F.interpolate(img2, size=(image_height, image_width), mode='bilinear', align_corners=False)

    img1_t = img1_resized / 255.0
    img2_t = img2_resized / 255.0

    if half:
        img1_t = img1_t.half()
        img2_t = img2_t.half()

    with torch.inference_mode():
        flow_predictions = model(img1_t, img2_t)

    flow = flow_predictions[-1][0].permute(1, 2, 0)
    
    return flow