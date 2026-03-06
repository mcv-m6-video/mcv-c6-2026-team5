from tqdm import tqdm

import numpy as np
import torch
import os
import argparse
from sklearn.model_selection import KFold
from src.data.loader import AICityDataset, collate_fn
from src.detection.fine_tuned import FineTunedDetector, YOLO8FineTuned
from src.data.splitter import DataSplitter
from src.evaluation.evaluations import save_training_plots, evaluate


# arg parser
parser = argparse.ArgumentParser(description="Train Object Detection Model on AICity Dataset")
# epochs
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
# batch size
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
# folds
parser.add_argument('--folds', type=int, default=4, help='Number of folds for cross-validation')
# split strategy
parser.add_argument('--split_strategy', type=str, default='B', choices=['A', 'B', 'C'], help='Data splitting strategy: A (random) or B (sequential)')

# --- CONFIGURATION ---
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
XML_PATH = "data/gt/ai_challenge_s03_c010-full_annotation.xml"
SAVE_DIR = "models"
BATCH_SIZE = parser.parse_args().batch_size # YOLO can handle larger batches usually
NUM_EPOCHS = parser.parse_args().epochs
SPLIT_STRATEGY = parser.parse_args().split_strategy
FOLDS = parser.parse_args().folds
MODEL = 'rcnn'  # or 'yolo'

def main():
    # 1. Load Data
    full_dataset = AICityDataset(video_path=VIDEO_PATH, xml_path=XML_PATH)
    len_dataset = len(full_dataset)
    # 2. Split Data
    splitter = DataSplitter(len_dataset)
    # print([train, val for train, val in splitter.get_split(strategy=SPLIT_STRATEGY, k=4)])
    # train_idx, val_idx = [(train, val) for train, val in splitter.get_split(strategy=SPLIT_STRATEGY, k=4)][0]
    
    # metrics
    all_folds_history = []
    best_map = 0
    for i, (train_idx, val_idx) in enumerate(splitter.get_split(strategy=SPLIT_STRATEGY, k=FOLDS)):
        print(f"Fold {i+1} - Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
        # 3. Initialize Model
        if MODEL == 'yolo':
            print("--- Mode: YOLOv8 Fine-Tuning ---")
            detector = YOLO8FineTuned()

            # YOLO handles its own loop and data loading
            detector.prepare_and_train(
                dataset=full_dataset,
                train_indices=train_idx,
                val_indices=val_idx,
                epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE
            )

            # Save mechanism is handled by Ultralytics (saved to models/yolo_finetuned/weights/best.pt)
            print(f"Best YOLO model saved to models/yolo_finetuned/weights/best.pt")

        elif MODEL == 'rcnn':
            print("--- Mode: Faster R-CNN Fine-Tuning ---")
            # Standard PyTorch Loop for R-CNN
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            print(f"Using device: {device}")
            train_sub = torch.utils.data.Subset(full_dataset, train_idx)
            val_sub = torch.utils.data.Subset(full_dataset, val_idx)
            
            train_loader = torch.utils.data.DataLoader(
                train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn
            )
            val_loader = torch.utils.data.DataLoader(
                val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn
            )
            
            history = {
                'train_loss': [],
                'val_loss': [],
                'val_map50': []
            }

            detector = FineTunedDetector()
            detector.model.to(device)
            detector.model.train()

            params = [p for p in detector.model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=3,
                gamma=0.1
            )
            print("Starting training...")
            for epoch in range(NUM_EPOCHS):
                epoch_train_loss = 0
                for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                    loss = detector.train_step(images, targets, optimizer)
                    epoch_train_loss += loss
                lr_scheduler.step()            
                avg_train_loss = epoch_train_loss / len(train_loader)
                history['train_loss'].append(avg_train_loss)

                # 4. Validation Step (mAP@50)
                detector.model.train() # train to output loss
                all_maps = []
                epoch_val_loss = 0
                with torch.no_grad():
                    for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val Loss]"):
                        # Move to device
                        images = [img.to(device) for img in images]
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        
                        # Model returns a dict of losses in train mode
                        loss_dict = detector.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        epoch_val_loss += losses.item()
                    print("Evaluating on validation set...")
                    mAP = evaluate(detector, val_loader, save_video=False) 
                # avg_val_map = np.mean(all_maps) if all_maps else (0.1 + epoch*0.04) # Mocking progress
                avg_val_loss = epoch_val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)    
                history['val_map50'].append(mAP)
                
                print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f} | mAP@50: {mAP:.4f}")

            all_folds_history.append(history)

        # Save manually
        
        torch.save(detector.model.state_dict(), f"{SAVE_DIR}/fine_tuned_rcnn_fold_{i}_strat_{SPLIT_STRATEGY}.pth")
    save_training_plots(all_folds_history, f"{SAVE_DIR}/training_plots_strat_{SPLIT_STRATEGY}")

if __name__ == "__main__":
    main()