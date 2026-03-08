import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .base_detector import BaseDetector
import yaml
from ultralytics import YOLO
import torch
import numpy as np
import os

class FineTunedDetector(BaseDetector):
    def get_model(self):
        print("Initializing Faster R-CNN for Fine-Tuning...")
        
        # 1. Load the pre-trained model (trained on COCO)
        print("Loading rcnn")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # 2. Freeze the backbone layers
        # We don't want to destroy the features learned from ImageNet/COCO.
        for param in model.backbone.parameters():
            param.requires_grad = False

        # 3. Replace the Head (Classifier)
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        # num_classes = 2 (0: Background, 1: Car)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        
        return model
    
class YOLO8FineTuned(BaseDetector):
    def get_model(self):
        # Start with the pre-trained COCO model
        return YOLO("yolov8n.pt")

    def predict(self, images, detection_threshold=0.5):
        # Reusing the prediction logic from the Off-the-shelf implementation
        # converting tensors to numpy for YOLO
        processed_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                processed_images.append(img_np)
            else:
                processed_images.append(img)

        results = self.model(processed_images, verbose=False, conf=detection_threshold)
        formatted_results = []
        
        # When we fine-tune, we will have 1 class (Car). 
        # YOLO will output class 0. We map it to our Project ID 1.
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            # Map YOLO class 0 -> Project Class 1
            labels = np.full(len(scores), 1, dtype=np.int64) 
            
            formatted_results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        return formatted_results

    def train_step(self, images, targets, optimizer):
        raise NotImplementedError("YOLOv8 uses its own internal training loop. Use 'prepare_and_train' instead.")

    def prepare_and_train(self, dataset, train_indices, val_indices, epochs=5, batch_size=4):
        """
        1. Converts the dataset (XML/PyTorch) into YOLO format (.txt files).
        2. Creates a dataset.yaml file.
        3. Runs the Ultralytics training command.
        """
        print("Preparing YOLO dataset from caching...")
        
        # Define paths
        cache_dir = dataset.cache_dir
        labels_dir = os.path.join(os.path.dirname(cache_dir), "labels_yolo")
        os.makedirs(labels_dir, exist_ok=True)

        # 1. Generate Label Files
        # YOLO Format: class_id x_center y_center width height (Normalized 0-1)
        # We iterate over the dataset's ground truth dictionary directly for speed
        img_h, img_w = 1080, 1920 # S03_C010 resolution
        
        # Create text files for ALL frames (simplifies split logic later)
        for frame_id, boxes in dataset.ground_truth.items():
            txt_path = os.path.join(labels_dir, f"frame_{frame_id:04d}.txt")
            with open(txt_path, 'w') as f:
                for box in boxes:
                    # box is [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box
                    
                    # Convert to xywh normalized
                    w = x2 - x1
                    h = y2 - y1
                    x_c = x1 + (w / 2)
                    y_c = y1 + (h / 2)
                    
                    # Normalize
                    bn = [x_c/img_w, y_c/img_h, w/img_w, h/img_h]
                    
                    # Class 0 for 'car' in our custom YOLO training
                    f.write(f"0 {bn[0]:.6f} {bn[1]:.6f} {bn[2]:.6f} {bn[3]:.6f}\n")

        # 2. Create Directory Structure for Ultralytics
        # Ultralytics needs a text file listing the absolute paths of images
        def create_file_list(indices, filename):
            lines = []
            for idx in indices:
                # Image path from the dataset cache
                img_path = os.path.join(dataset.cache_dir, f"frame_{idx:04d}.jpg")
                lines.append(os.path.abspath(img_path))
            
            save_path = os.path.join(os.path.dirname(cache_dir), filename)
            with open(save_path, 'w') as f:
                f.write('\n'.join(lines))
            return save_path

        train_list = create_file_list(train_indices, "train_list.txt")
        val_list = create_file_list(val_indices, "val_list.txt")

        # 3. Create dataset.yaml
        yaml_data = {
            'path': os.path.abspath(os.path.dirname(cache_dir)),
            'train': "train_list.txt",
            'val': "val_list.txt",
            'names': {0: 'car'}
        }
        yaml_path = os.path.join(os.path.dirname(cache_dir), "aic_yolo.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)

        # 4. Run Training
        print(f"Starting YOLOv8 training for {epochs} epochs...")
        self.model.train(
            data=yaml_path, 
            epochs=epochs, 
            batch=batch_size, 
            imgsz=640,
            plots=True,
            project="models",
            name="yolo_finetuned",
            exist_ok=True # Overwrite existing
        )
        print("YOLO Training complete.")