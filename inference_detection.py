import torch
import argparse
from src.data.loader import AICityDataset, collate_fn
from src.detection.off_the_shelf import FasterRCNNOffTheShelf, YoloOffTheShelfDetector
from src.detection.fine_tuned import FineTunedDetector
from src.evaluation.evaluations import evaluate
from src.data.splitter import DataSplitter
from tqdm import tqdm
import numpy as np

def main():
    # parser
    parser = argparse.ArgumentParser(description="Inference and Evaluation for Object Detection on AICity Dataset")
    # batch size
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    # folds
    parser.add_argument('--folds', type=int, default=4, help='Number of folds for cross-validation')
    # split strategy
    parser.add_argument('--split_strategy', type=str, default='B', choices=['A', 'B', 'C'], help='Data splitting strategy: A (random) or B (sequential)')
    # save video
    parser.add_argument('--save_video', type=str, default="results/inference.mp4", help='Path to save inference visualization video. Set to False to skip saving video.')
    # mode
    parser.add_argument('--mode', type=str, default='fine-tuned', choices=['yolo-off-shelf', 'faster-rcnn-off-shelf', 'fine-tuned'], help='Inference mode: yolo-off-shelf, faster-rcnn-off-shelf, or fine-tuned')
    args = parser.parse_args()
    MODE = 'faster-rcnn-off-shelf'  # or 'fine-tuned'
    MODE = 'fine-tuned'
    MODE = args.mode
    WEIGHTS = "models/fine_tuned_rcnn.pth"  # Path to fine-tuned weights if using that mode
    OUTPUT_VIDEO = "results/inference.mp4"  # Path to save inference visualization video False to skip saving video
    if args.save_video.lower() == 'false':
        OUTPUT_VIDEO = False
    else:
        OUTPUT_VIDEO = args.save_video  # Path to save inference visualization video False to skip saving video

    SPLIT_STRATEGY = args.split_strategy  # 'A', 'B', or 'C'
    FOLDS = args.folds
    FOLD_TO_EVAL = 0      # Index of the fold to evaluate (0 to FOLDS-1)


    VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    XML_GT_PATH = "data/gt/ai_challenge_s03_c010-full_annotation.xml"
    # 1. Load Data
    # Use the VALIDATION set (from strategy A or B)
    dataset = AICityDataset(VIDEO_PATH, xml_path=XML_GT_PATH)
    
    splitter = DataSplitter(len(dataset))
    splits = list(splitter.get_split(strategy=SPLIT_STRATEGY, k=FOLDS))
    
    all_maps = []
    
    # fold_idx = 0 if SPLIT_STRATEGY == 'A' else FOLD_TO_EVAL
    # _, val_idx = splits[fold_idx]
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.get_split(strategy=SPLIT_STRATEGY, k=FOLDS)):

        print(len(train_idx), len(val_idx))
        # Simple split for inference (e.g., last 75% as per Strategy A/Week 1)
        val_sub = torch.utils.data.Subset(dataset, val_idx)
        
        val_loader = torch.utils.data.DataLoader(
            val_sub, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
        all_maps.append(mAP_50.item())
        
        if SPLIT_STRATEGY == 'A':
            break
    print(f"\nSummary of mAP_50 across folds: {all_maps}")
    mean_map = np.mean(all_maps)
    print(f"\n========================================")
    print(f"Final Results for {MODE} (Strategy {SPLIT_STRATEGY}):")
    print(f"Mean mAP_50 over {len(all_maps)} folds: {mean_map:.4f}")
    if len(all_maps) > 1:
        print(f"Std mAP_50: {np.std(all_maps):.4f}")
    print(f"========================================\n")

    
if __name__ == "__main__":
    main()