import torch
from torchcodec.decoders import VideoDecoder
from src.background.gaussian import RecursiveGaussian
from src.utils.post_processing import apply_morphology, get_bboxes_from_mask, merge_bboxes_by_distance
from src.data.parser import load_gt_xml
# Import the coco evaluator we made previously
from src.evaluation.coco_eval import evaluate_coco 

import numpy as np
import cv2
from tqdm import tqdm

class Evaluator:
    def __init__(self, video_path, gt_path, roi_path, split_ratio=0.25):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Data Once
        print("Loading Video and GT...")
        self.decoder = VideoDecoder(video_path, device="cpu")
        self.total_frames = self.decoder.metadata.num_frames
        self.train_len = int(self.total_frames * split_ratio)
        self.width = self.decoder.metadata.width
        self.height = self.decoder.metadata.height

        gt_boxes = load_gt_xml(gt_path)
        # Filter GT for testing phase
        self.gt_boxes_test = {k: v for k, v in gt_boxes.items() if k >= self.train_len}
        
        # Load ROI
        roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        self.roi_mask_tensor = torch.from_numpy(roi_mask).to(self.device).float() / 255.0

        # Pre-train Initialization (Since it's non-adaptive, we can technically cache the means 
        # but Recursive updates them, so we must re-init the model object every time, 
        # BUT we can keep the raw training frames in RAM if small enough. 
        # For now, let's just let the model fit() read from decoder to save RAM).

    def run_experiment(self, params):
        """
        params: dict with keys:
            alpha (float), rho (float),
            shadow_method (str),
            tau_s (float), tau_h (float), shadow_alpha (float), shadow_beta (float)
        """
        
        # 1. Setup Model
        model = RecursiveGaussian(
            alpha=params['alpha'], 
            rho=params['rho'], 
            device=self.device
        )
        
        # Train (Initialization)
        # Note: If this takes too long, you can compute initial mean/std ONCE 
        # in __init__ and pass it to the model manually.
        model.fit(self.decoder, num_train_frames=self.train_len)
        
        pred_boxes_test = {}
        
        # Prepare shadow params
        shadow_params = {
            "alpha": params.get('shadow_alpha', 0.5),
            "beta": params.get('shadow_beta', 0.9),
            "tau_s": params.get('tau_s', 60),
            "tau_h": params.get('tau_h', 40)
        }

        # 2. Inference Loop
        # We can skip tqdm here to keep optimization logs clean
        for i in range(self.train_len, self.total_frames):
            frame_tensor = self.decoder[i].to(self.device).float()
            
            fg_mask_tensor = model.apply(
                frame_tensor, 
                shadow_method=params['shadow_method'],
                shadow_params=shadow_params,
                detection_mode=params['detection_mode'],
                update_buffer=0#params['update_buffer']
                
            )
            
            # Apply ROI
            fg_mask_tensor = (fg_mask_tensor > 0) & (self.roi_mask_tensor > 0)
            
            # print(f"fg_mask_tensor:\n{fg_mask_tensor.cpu().numpy()}")
            # print(type(fg_mask_tensor))
            # mask to 
            fg_mask_uint_tensor = (fg_mask_tensor.float()*255).to(torch.uint8)
            # print(f"fg_mask_uint_tensor:\n{fg_mask_uint_tensor.cpu().numpy()}")
            # print(type(fg_mask_uint_tensor))
            # print(np.unique(fg_mask_uint_tensor.cpu().numpy()))
            # exit()
            
            # Post-processing
            # _, boxes = post_process_mask(mask_np, min_area=params.get('min_area', 150))
            cleaned_mask = apply_morphology(
                                        fg_mask_uint_tensor, 
                                        kernel_opening_size=params['kernel_opening_size'],
                                        kernel_closing_size=params['kernel_closing_size'],
                                        operation=params['morph_op'],
                                        morph_shape=params['morph_shape']
                                        )
            cleaned_mask = cleaned_mask.cpu().numpy().astype('uint8') * 255
            boxes = get_bboxes_from_mask(cleaned_mask, min_area=params.get('min_area', 150))
            boxes = merge_bboxes_by_distance(boxes, min_distance=params.get('merge_dist', 40), frame_height=self.height)
            
            if len(boxes) > 0:
                pred_boxes_test[i] = boxes

        # 3. Evaluate
        map50 = evaluate_coco(self.gt_boxes_test, pred_boxes_test, self.height, self.width)
        
        return map50