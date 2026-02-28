from ultralytics import YOLO
import torch
import numpy as np
from .base_detector import BaseDetector
import torchvision

class YoloOffTheShelfDetector(BaseDetector):
    def get_model(self):
        # Load the pre-trained YOLOv8 model (Nano version for speed, Medium for accuracy)
        model = YOLO("yolov8m.pt") 
        return model

    def predict(self, images, detection_threshold=0.5):
        """
        Run inference using YOLOv8 and adapt results to our common format.
        """
        # YOLOv8 can handle a list of images (numpy arrays or PIL) directly.
        # However, our BaseDetector moves things to device, so we might receive Tensors.
        # Ultralytics prefers paths, PIL images, or numpy arrays.
        
        # Convert Tensors back to Numpy if necessary (CHW -> HWC)
        processed_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                # Assume img is (C, H, W) and normalized 0-1
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                processed_images.append(img_np)
            else:
                processed_images.append(img)

        # Run Inference
        # verbose=False keeps the console clean
        results = self.model(processed_images, verbose=False, conf=detection_threshold)

        formatted_results = []
        
        # COCO Class ID for 'car' is 2
        YOLO_CAR_CLASS_ID = 2
        # Our Project Class ID for 'car' (as defined in loader.py)
        OUR_CAR_CLASS_ID = 1

        for result in results:
            boxes = result.boxes
            
            # Filter for cars only (cls == 2)
            car_mask = (boxes.cls == YOLO_CAR_CLASS_ID)
            
            # Apply mask
            car_boxes = boxes.xyxy[car_mask].cpu().numpy()
            car_scores = boxes.conf[car_mask].cpu().numpy()
            
            # Create labels array filled with OUR_CAR_CLASS_ID (1)
            # We ignore the original class ID (2) because our evaluator expects 1
            car_labels = np.full(len(car_scores), OUR_CAR_CLASS_ID, dtype=np.int64)

            formatted_results.append({
                'boxes': car_boxes,
                'scores': car_scores,
                'labels': car_labels
            })

        return formatted_results

    def train_step(self, images, targets, optimizer):
        # We strictly do not train the off-the-shelf model
        print("Warning: Attempting to train Off-the-shelf YOLO. Skipping.")
        return 0.0
    
class FasterRCNNOffTheShelf(BaseDetector):
    def get_model(self):
        # Load standard Faster R-CNN pre-trained on COCO
        # weights="DEFAULT" loads the best available pre-trained weights
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    def predict(self, images, detection_threshold=0.5):
        """
        Reuse the BaseDetector's predict (which handles moving tensors to GPU),
        then filter the results to only keep cars.
        """
        # 1. Get raw predictions for all COCO classes (80 classes)
        # super().predict returns list of dicts with numpy arrays
        raw_results = super().predict(images, detection_threshold)
        
        filtered_results = []
        
        # COCO Class ID for 'car' is 3
        # Your Project ID for 'car' is 1
        COCO_CAR_ID = 3
        PROJECT_CAR_ID = 1
        
        for res in raw_results:
            # Create a boolean mask where label is 'car'
            car_mask = res['labels'] == COCO_CAR_ID
            
            # Filter boxes and scores
            boxes = res['boxes'][car_mask]
            scores = res['scores'][car_mask]
            
            # Create new labels array filled with ID 1
            labels = np.full(len(scores), PROJECT_CAR_ID, dtype=np.int64)
            
            filtered_results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
            
        return filtered_results

    def train_step(self, images, targets, optimizer):
        print("Warning: Attempting to train Off-the-shelf Faster R-CNN. Skipping.")
        return 0.0