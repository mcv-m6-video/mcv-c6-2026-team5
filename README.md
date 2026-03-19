# MCV-C6-2026-Team5: Video Surveillance for Road Traffic Monitoring

Final Presentation Link: https://docs.google.com/presentation/d/17pmLHEouEzi7AYdO3XKHkM3L-wP7AdiHgRJMXHYQkKw/edit?usp=sharing

This repository contains the implementation, code, and resources for a multi-week computer vision project focused on video surveillance and road traffic monitoring. The project is organized to provide modularity as it evolves from basic background modeling to advanced tracking and optical flow estimation.

## Team Members
* Álvaro Díaz
* Bernat Medina
* Maiol Sabater
* Gerard Vilaplana

## Project Repository Structure

The project is divided into weekly modules, each building upon the previous work with specific source code and experimental scripts.

### Core Directory (General Structure)
* **`src/`**: Main source code directory containing sub-modules for background modeling, data parsing, and evaluation metrics.
    * `background/`: Base classes and Gaussian/SOTA methods for background subtraction.
    * `data/`: Utilities for dataset parsing and loading.
    * `evaluation/`: Modules for COCO-style metrics, IoU, mAP, and tracking-specific scores (HOTA, IDF1).
    * `utils/`: Post-processing and shadow removal tools.
* **`experiments/`**: Main scripts for running evaluations and tracking ablations.
* **`results/`**: Storage for quantitative tables, visualizations, and inference videos.

## Weekly Overview

### Week 1: Background Modeling
Focuses on initial background subtraction and model optimization.
* **Key Scripts**:
    * `gridsearch_alpha.py`: Grid search for hyperparameter tuning of the alpha parameter in background subtraction.
    * `optimize.py`: General model parameter optimization.
    * `run_zbs.py`: Executes experiments using the Zero Buffering Strategy (ZBS).

### Week 2: Object Detection & Tracking
Introduces fine-tuned object detection (YOLOv8, Faster R-CNN) and tracking algorithms (Max IoU, Kalman Filter).
* **Key Scripts**:
    * `train_detection.py`: Trains detection models using K-fold cross-validation.
    * `run_tracking.py`: Executes object tracking and generates visualization videos.
    * `bayesian_search/`: Contains Optuna-based scripts to optimize tracker hyperparameters.

### Week 3: Optical Flow & MTSC Tracking
Implements optical flow estimation (PyFlow, RAFT, MaskFlowNet) and Multi-Target Single-Camera (MTSC) tracking.
* **Key Components**:
    * `pyflow_estimator.py`: Optical flow computation via PyFlow.
    * `of_tracker.py`: Enhanced tracking using optical flow to predict bounding box positions.
    * `run_mtsc_eval.py`: Evaluates the best tracking algorithms on the AI City Challenge dataset.

## Execution Guide

### Object Detection Training (Week 2)
```bash
python train_detection.py --epochs 15 --batch_size 32 --folds 4 --split_strategy B
```
* **`--folds`**: Number of folds for cross-validation (default: 4).
* **`--split_strategy`**: Strategy A (random) or B (sequential).

### Running & Evaluating Trackers
To process video and assign track IDs:
```bash
python run_tracking.py --tracker kalman --model_path models/fine_tuned_rcnn.pth --conf_thresh 0.5 --iou_thresh 0.3
```
To output performance metrics (HOTA/IDF1):
```bash
python evaluate_tracking.py --tracker kalman --conf_thresh 0.5 --iou_thresh 0.3 --max_age 3
```

### Optical Flow Evaluation (Week 3)
Scripts in the `experiments/` directory allow for the computation of MSEN (Mean Square Error in Non-occluded areas) and PEPN (Percentage of Erroneous Pixels).


## Data Management
The project utilizes the following datasets:
* **KITTI**: Sequence 45 for optical flow evaluation.
* **AI City Challenge**: Sequences S01, S03, and S04 for MTSC tracking.
* **Note**: All data must be deleted after the module finishes as per project guidelines.
