import torch
import os
import argparse
from sklearn.model_selection import KFold
from src.data.loader import AICityDataset, collate_fn
from src.detection.fine_tuned import FineTunedDetector, YOLO8FineTuned
from src.data.splitter import DataSplitter

# --- CONFIGURATION ---
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
XML_PATH = "data/gt/ai_challenge_s03_c010-full_annotation.xml"
SAVE_DIR = "models"
BATCH_SIZE = 4 # YOLO can handle larger batches usually
NUM_EPOCHS = 5
SPLIT_STRATEGY = 'A'
MODEL = 'rcnn'  # or 'yolo'

def main():
    # 1. Load Data
    full_dataset = AICityDataset(video_path=VIDEO_PATH, xml_path=XML_PATH)
    len_dataset = len(full_dataset)
    # 2. Split Data
    splitter = DataSplitter(len_dataset)
    train_idx, val_idx = splitter.get_split(strategy=SPLIT_STRATEGY, k=4)

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

        train_sub = torch.utils.data.Subset(full_dataset, train_idx)
        train_loader = torch.utils.data.DataLoader(
            train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn
        )

        detector = FineTunedDetector()
        detector.model.to(device)
        detector.model.train()

        params = [p for p in detector.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0
            for images, targets in train_loader:
                loss = detector.train_step(images, targets, optimizer)
                epoch_loss += loss
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")

        # Save manually
        torch.save(detector.model.state_dict(), f"{SAVE_DIR}/fine_tuned_rcnn.pth")

if __name__ == "__main__":
    main()