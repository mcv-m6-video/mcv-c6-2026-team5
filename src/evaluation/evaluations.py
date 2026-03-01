import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import torch
import numpy as np
import os

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
    
    for batch_images, batch_targets in tqdm(dataloader):
        batch_images = [img.to(device) for img in batch_images]
        targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]     
        # Predict
        with torch.no_grad():
            preds = detector.model(batch_images)
        # preds = detector.predict(batch_images)
        tensor_preds = []
        for p in preds:
            # filter out low confidence predictions (optional, can be adjusted based on your needs)
            keep = p['scores'] > 0.05 
            tensor_preds.append({k: v[keep] for k, v in p.items()})
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
                
def save_training_plots(all_folds_history, save_dir):
    epochs = range(1, len(all_folds_history['train_loss']) + 1)
    # print(all_folds_history)
    
    # Convert list of dicts to numpy arrays [folds, epochs]
    
    train_losses = all_folds_history['train_loss']
    val_maps = all_folds_history['val_map50']

    # Calculate statistics
    mean_loss = np.mean(train_losses, axis=0)
    std_loss = np.std(train_losses, axis=0)
    mean_map = np.mean(val_maps, axis=0)
    std_map = np.std(val_maps, axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Loss
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, mean_loss, color=color, marker='o', label='Mean Loss')
    ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot mAP@50
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val mAP@50', color=color)
    ax2.plot(epochs, mean_map, color=color, marker='s', label='Mean mAP@50')
    ax2.fill_between(epochs, mean_map - std_map, mean_map + std_map, color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Cross-Validation Metrics (Mean ± Std over {len(all_folds_history)} folds)')
    fig.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "cv_training_metrics.png")
    plt.savefig(plot_path)
    print(f"Aggregated plot saved to {plot_path}")
    plt.show()