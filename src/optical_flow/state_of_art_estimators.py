import sys
from pathlib import Path
import torch
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

def compute_neuflow(model, img1: np.ndarray, img2: np.ndarray, device: str = "cuda", half=False) -> np.ndarray:
    image_width = 768
    image_height = 432

    img1_resized = cv2.resize(img1, (image_width, image_height))
    img2_resized = cv2.resize(img2, (image_width, image_height))

    img1_rgb = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB)
    
    
    if half:
        img1_t = torch.from_numpy(img1_rgb / 255.0).permute(2, 0, 1).half().unsqueeze(0).to(device)
        img2_t = torch.from_numpy(img2_rgb / 255.0).permute(2, 0, 1).half().unsqueeze(0).to(device)
    else:
        img1_t = torch.from_numpy(img1_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
        img2_t = torch.from_numpy(img2_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.inference_mode():
        flow_predictions = model(img1_t, img2_t)

    flow = flow_predictions[-1][0].permute(1, 2, 0).cpu().float().numpy()
    
    return flow