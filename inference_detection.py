import torch
import argparse
from src.data.loader import AICityDataset, collate_fn
from src.detection.off_the_shelf import FasterRCNNOffTheShelf, YoloOffTheShelfDetector
# from src.detection.fine_tuned import FineTunedDetector
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import numpy as np

def evaluate(detector, dataloader, save_video="results/inference.mp4"):
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

def main():
    MODE = 'faster-rcnn-off-shelf'  # or 'fine-tuned'
    WEIGHTS = "models/fine_tuned_model.pth"  # Path to fine-tuned weights if using that mode
    OUTPUT_VIDEO = "results/inference.mp4"  # Path to save inference visualization video False to skip saving video

    VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    XML_GT_PATH = "data/gt/ai_challenge_s03_c010-full_annotation.xml"
    # 1. Load Data
    # Use the VALIDATION set (from strategy A or B)
    dataset = AICityDataset(VIDEO_PATH, xml_path=XML_GT_PATH)
    
    # Simple split for inference (e.g., last 75% as per Strategy A/Week 1)
    split_point = int(len(dataset) * 0.25)
    val_sub = torch.utils.data.Subset(dataset, range(split_point, len(dataset)))
    
    val_loader = torch.utils.data.DataLoader(
        val_sub, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 2. Load Model
    if MODE == 'yolo-off-shelf':
        print("Loading Off-the-Shelf Model...")
        detector = YoloOffTheShelfDetector()
    elif MODE == 'faster-rcnn-off-shelf':
        print("Loading Off-the-Shelf Faster R-CNN Model...")
        detector = FasterRCNNOffTheShelf()
    else:
        print(f"Loading Fine-Tuned Model from {WEIGHTS}")
        detector = FineTunedDetector()
        # Load the weights we trained in the other script
        detector.model.load_state_dict(torch.load(WEIGHTS))
        detector.model.to(detector.device)

    # 3. Evaluate
    mAP_50 = evaluate(detector, val_loader, save_video=OUTPUT_VIDEO)
    print(f"Results for {MODE}:")
    print(f"mAP_50: {mAP_50.item():.4f}")

if __name__ == "__main__":
    main()