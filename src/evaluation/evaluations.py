import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.detection.off_the_shelf import YoloOffTheShelfDetector, FasterRCNNOffTheShelf
from src.detection.fine_tuned import FineTunedDetector
from tqdm import tqdm
import torch
import numpy as np
import os
from time import time

def evaluate(detector, dataloader, save_video=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = MeanAveragePrecision().to(device) 
    detector.model.to(device)    
    detector.model.eval()
    
    print(f"Running inference on {device}...")
    
    # video setup
    if save_video:
        h, w = dataloader.dataset[0][0].shape[1:]  # Get height and width from the first image
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_video, fourcc, 10, (w, h))
    total_inference_time = 0.0
    for batch_images, batch_targets in tqdm(dataloader):
        batch_images = [img.to(device) for img in batch_images]
        metric_targets = [{k: v.cpu() for k, v in t.items()} for t in batch_targets]
        targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in batch_targets] 
        # if model is yolo
        start = time()
        with torch.no_grad():
            # if isinstance(detector, YoloOffTheShelfDetector):    
            #     # We use the predict method to get results in our common format
            #     preds = detector.predict(batch_images)
            # else:
            # Predict
            preds = detector.predict(batch_images)
        total_inference_time += time() - start
        # preds = detector.predict(batch_images)
        tensor_preds = []
        for p in preds:
            # filter out low confidence predictions (optional, can be adjusted based on your needs)
            keep = p['scores'] > 0.05 
            boxes = p['boxes'][keep]
            scores = p['scores'][keep]
            labels = p['labels'][keep]
            
            is_coco_vehicle = (labels == 3) | (labels == 4) | (labels == 6) | (labels == 8)
            is_already_class_1 = (labels == 1)
            
            valid_mask = is_coco_vehicle | is_already_class_1
            
            # Apply mask to keep only vehicles
            final_boxes = torch.tensor(boxes[valid_mask]).to(device)
            final_scores = torch.tensor(scores[valid_mask]).to(device)
            
            # Force all labels to Class 1 to match AICityDataset Ground Truth
            final_labels = torch.ones_like(final_scores, dtype=torch.int64)

            tensor_preds.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            })
        # tensor_preds = [{k: torch.tensor(v) for k, v in p.items()} for p in tensor_preds]

        metric.update(tensor_preds, targets_gpu)
        if save_video:
            # Initialize Video Writer once we know the image size
            if out is None:
                h, w = batch_images[0].shape[-2:]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_video, fourcc, 10.0, (w, h))

            # Iterate through the BATCH to save every frame
            for i, img_tensor in enumerate(batch_images):
                # Un-normalize (0-1 -> 0-255) and Permute (C,H,W -> H,W,C)
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Convert RGB (PyTorch) to BGR (OpenCV)
                img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # -- Draw Predictions (Green) --
                current_preds = preds[i]
                for box, score in zip(current_preds['boxes'], current_preds['scores']):
                    if type(detector) == FasterRCNNOffTheShelf or type(detector) == FineTunedDetector:
                        box = box.cpu().numpy()
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Draw Box
                    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw Score Label
                    label = f"Conf: {score:.2f}"
                    cv2.putText(img_cv2, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # -- Draw Ground Truth (Red) --
                current_gt = metric_targets[i]
                for box in current_gt['boxes']:
                    x1, y1, x2, y2 = box.int().numpy()
                    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Write frame to video
                out.write(img_cv2)
        
    result = metric.compute()
    if save_video:
        out.release()
    print(f"Total Inference Time: {total_inference_time:.2f} seconds")
    return result['map_50']
                
def save_training_plots(all_folds_history, save_dir):
    epochs = range(1, len(all_folds_history[0]['train_loss']) + 1)
    
    # Extract data [folds, epochs]
    train_losses = np.array([[float(l) for l in h['train_loss']] for h in all_folds_history])
    val_losses = np.array([[float(l) for l in h['val_loss']] for h in all_folds_history])
    val_maps = np.array([[float(m) for m in h['val_map50']] for h in all_folds_history])

    # Calculate statistics
    mean_train_loss = np.mean(train_losses, axis=0)
    std_train_loss = np.std(train_losses, axis=0)
    
    mean_val_loss = np.mean(val_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)
    
    mean_map = np.mean(val_maps, axis=0)
    std_map = np.std(val_maps, axis=0)

    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # PLOT 1: Train & Val Losses
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    # Train Loss
    plt.plot(epochs, mean_train_loss, color='tab:red', marker='o', label='Train Loss')
    plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color='tab:red', alpha=0.2)
    
    # Val Loss
    plt.plot(epochs, mean_val_loss, color='tab:orange', marker='^', linestyle='--', label='Val Loss')
    plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color='tab:orange', alpha=0.2)
    
    plt.title(f'Cross-Validation Loss\n(Mean ± Std over {len(all_folds_history)} folds)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    loss_plot_path = os.path.join(save_dir, "cv_losses.png")
    plt.savefig(loss_plot_path, facecolor='white', transparent=False)
    plt.close() # Close figure to avoid memory leaks

    # ==========================================
    # PLOT 2: Validation mAP@50
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    # Val mAP@50
    plt.plot(epochs, mean_map, color='tab:blue', marker='s', label='Val mAP@50')
    plt.fill_between(epochs, mean_map - std_map, mean_map + std_map, color='tab:blue', alpha=0.2)
    
    plt.title(f'Cross-Validation mAP@50\n(Mean ± Std over {len(all_folds_history)} folds)')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@50')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    map_plot_path = os.path.join(save_dir, "cv_map50.png")
    plt.savefig(map_plot_path, facecolor='white', transparent=False)
    plt.close() # Close figure

    print(f"\nPlots saved successfully:")
    print(f" - Losses: {loss_plot_path}")
    print(f" - mAP@50: {map_plot_path}")