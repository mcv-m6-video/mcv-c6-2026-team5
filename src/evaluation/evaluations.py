import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import torch
import numpy as np
import os

def evaluate(detector, dataloader, save_video=False):
    metric = MeanAveragePrecision()
    print("Running inference...")
    h, w = dataloader.dataset[0][0].shape[1:]  # Get height and width from the first image
    if save_video:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_video, fourcc, 10, (w, h))
    
    for batch_images, batch_targets in tqdm(dataloader):
        # Predict
        preds = detector.predict(batch_images)
        tensor_preds = [{k: torch.tensor(v) for k, v in p.items()} for p in preds]
        # Format for TorchMetrics (needs dicts)
        # Preds are already list of dicts from our BaseDetector
        # Targets need to be moved to CPU for metric calculation
        target_cpu = [{k: v.cpu() for k, v in t.items()} for t in batch_targets]
        
        metric_targets = [{k: v.cpu() for k, v in t.items()} for t in batch_targets]
        metric.update(tensor_preds, target_cpu)
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
    return result['map_50']
                
                
def save_training_plots(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Loss on the left Y-axis
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, history['train_loss'], color=color, marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a twin axis for mAP@50 on the right Y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val mAP@50', color=color)
    ax2.plot(epochs, history['val_map50'], color=color, marker='s', label='mAP@50')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Loss and Validation mAP@50')
    fig.tight_layout()
    
    plot_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()