
# Week 6: Advanced Action Spotting on SoccerNet

This folder contains the codebase for the continuation of **Project 2: Sports Video Analysis**. The focus of this week is on advancing Temporal Action Spotting on the **SoccerNet Ball Action Spotting (SN-BAS-2025)** dataset. We transition from clip-level classification to precise temporal spotting by implementing frame-level probability predictions, advanced temporal architectures, and extensive post-processing techniques (NMS).

## Checkpoints of Best Models
* Access to this shared Drive folder for the best models: https://drive.google.com/drive/folders/1b0Fe95kcJf9bDk4NaBILYh-AnUrjxH9H?usp=drive_link

## 📁 Repository Structure

* `config/`: Contains JSON configuration files for various spotting experiments (Baseline, NMS ablations, Lightweight Attention, Temporal Shift, Multi-Stage TCNs).
* `data/` & `dataset/`: Dataloaders and dataset parsers optimized for frame-by-frame spotting sequences.
* `model/`: PyTorch model architectures (`model_spotting.py`) including advanced TCNs, lightweight temporal attention, and temporal shift modules.
* `util/`: Helper functions for I/O and spotting-specific evaluation metrics (mAP, mAP@10).

## ⚙️ Setup and Data Preparation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and Extract Data:**
   If not already done in Week 5, download the dataset and extract RGB frames:
   ```bash
   python download_frames_snb.py
   python extract_frames_snb.py
   ```

## 🚀 Training & Evaluation

The training pipeline is controlled via JSON configuration files. 

**⚠️ Important Note on Data Loading:** Make sure your config has `"store_mode": "store"` on the very first run to generate and cache the clips. After this initial run, set `"store_mode": "load"` to dramatically speed up all subsequent executions.

**Train a Spotting Model (e.g., Lightweight Attention):**
```bash
python main_spotting.py --model F2_lightweight_att
```

### Computing MACs & Parameters
To evaluate the computational cost (Multiply-Accumulates and Params) of a specific spotting configuration, including the new temporal heads:
```bash
python compute_macs_spotting.py --model F2_lightweight_att
```

## 🎥 Qualitative Analysis (Animated Spotting Curves)

For action spotting, we provide an advanced visualization script (`qualitative_spotting.py`) that generates three outputs per clip, including a synchronized dashboard:
1. A video with a live frame-by-frame prediction HUD.
2. A bar chart of the maximum peak confidences.
3. **Animated Temporal Curves** showing the exact probability evolution over time synchronized directly underneath the video playback.

**Generate random animated comparisons:**
```bash
python qualitative_spotting.py --model_base A0_baseline --model_best F2_lightweight_att --num_clips 5
```

**Smart Search for Temporal Peak Improvements:**
Automatically scan the dataset to generate clips *only* where your best model successfully peaked for the Ground Truth action (>50%) but the baseline completely missed it:
```bash
python qualitative_spotting.py --model_base A0_baseline --model_best F2_lightweight_att --find_improvement --num_clips 3
```