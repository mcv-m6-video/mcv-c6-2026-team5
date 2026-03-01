import torch
import argparse
from src.data.loader import AICityDataset, collate_fn
from src.detection.off_the_shelf import FasterRCNNOffTheShelf, YoloOffTheShelfDetector
from src.detection.fine_tuned import FineTunedDetector
from src.evaluation.evaluations import evaluate
from tqdm import tqdm
import numpy as np

def main():
    MODE = 'faster-rcnn-off-shelf'  # or 'fine-tuned'
    MODE = 'fine-tuned'
    WEIGHTS = "models/fine_tuned_rcnn.pth"  # Path to fine-tuned weights if using that mode
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