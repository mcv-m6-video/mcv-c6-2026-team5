# Week2 C6

By team5 composed by:
Álvaro Díaz, Bernat Medina, Maiol Sabater, Gerard Vilaplana


## Repository Structure

### Root Directory

* `train_detection.py`: Main script to train and fine-tune object detection models (YOLOv8 or Faster R-CNN) using K-fold cross-validation.
* `inference_detection.py`: Runs inference and evaluation (mAP@50) for object detection on the validation set using off-the-shelf or fine-tuned models.
* `run_tracking.py`: Executes object tracking (Max IoU or Kalman Filter) using a specified detection model and generates an output video with bounding boxes and track IDs.
* `evaluate_tracking.py`: Evaluates tracking performance against ground truth data, calculating metrics such as HOTA and IDF1 scores.
* `generate_detections.py`: Script to extract and cache raw object detections (likely to speed up the hyperparameter search).
* `.gitignore`: Specifies intentionally untracked files to ignore in Git.

### `/src` Directory (Core Modules)

* **`data/`**: Handles dataset loading and processing.
* `loader.py`: Contains the `AICityDataset` class for loading frames and caching XML annotations.
* `splitter.py`: Logic for splitting data into training and validation sets (e.g., K-fold random or sequential splitting).


* **`detection/`**: Contains object detection architectures.
* `base_detector.py`: Base class interface for all detectors.
* `fine_tuned.py`: Implementations for fine-tuned detectors like Faster R-CNN and YOLO.
* `off_the_shelf.py`: Pre-trained models ready for zero-shot inference.


* **`evaluation/`**: Functions for assessing performance.
* `evaluations.py`: General object detection evaluations.
* `tracking_metrics.py`: Computes tracking-specific scores (HOTA, IDF1).


* **`tracking/`**: Contains the object tracking algorithms.
* `base_tracker.py`: Standard interface for tracking algorithms.
* `iou_tracker.py`: Implementation of the Maximum Intersection-over-Union (Max IoU) Tracker.
* `kalman_tracker.py`: Implementation of a Tracker that utilizes Kalman Filters for motion estimation.



### `/bayesian_search` Directory

* `optimize_iou_tracker.py`: Uses the Optuna framework to run Bayesian hyperparameter optimization to find the best confidence threshold, IoU threshold, and max age for the Max IoU Tracker.
* `optimize_kalman_tracker.py`: Uses Optuna to find the best hyperparameters (including process and measurement noise scales in log space) for the Kalman Tracker.

### `/plots` Directory

* Contains various scripts to generate statistical performance visualizations, including K-fold boxplots (`boxplots_kfold.py`), noise scale heatmaps (`heatmap_log_noise.py`), and model comparisons (`HOTA_IDF1_iou_kalman_comparison.py`, `plot_off_the_shelf_comparative.py`).

---

## How to Execute Main Scripts

### 1. Training Object Detection

This script runs the training loop for the selected object detection model.

```bash
python train_detection.py --epochs 15 --batch_size 32 --folds 4 --split_strategy B

```

* **Arguments**:
* `--epochs`: Number of training epochs (default: 15).
* `--batch_size`: Batch size for training (default: 32).
* `--folds`: Number of folds for cross-validation (default: 4).
* `--split_strategy`: Splitting strategy A (random) or B (sequential) (default: B).



### 2. Inference & Evaluation for Object Detection

This script evaluates the detection models, calculating the mAP_50 metrics.

```bash
python inference_detection.py --mode fine-tuned --weights models/fine_tuned_rcnn.pth --batch_size 4 --folds 4 --split_strategy B

```

* **Arguments**:
* `--mode`: `yolo-off-shelf`, `faster-rcnn-off-shelf`, or `fine-tuned` (default: fine-tuned).
* `--weights`: Path to fine-tuned weights file (default: models/fine_tuned_rcnn.pth).
* `--save_video`: Path to save visualization (default: results/inference.mp4). Use `False` to disable.



### 3. Running Tracking (Max IoU & Kalman)

This script processes the video to assign track IDs to objects using your chosen tracking algorithm and saves a visualization.

```bash
python run_tracking.py --tracker kalman --model_path models/fine_tuned_rcnn.pth --conf_thresh 0.5 --iou_thresh 0.3 --max_age 1

```

* **Arguments**:
* `--tracker`: Choose `iou` or `kalman` (default: iou).
* `--conf_thresh`: Detection confidence threshold (default: 0.5).
* `--iou_thresh`: Minimum IoU to assign a track ID (default: 0.3).
* `--max_age`: Frames to keep a track alive without detections (default: 1).
* `--proc_noise` / `--meas_noise`: Configuration for Kalman Tracker noise scales (default: 1.0).



### 4. Evaluating Tracking Performance

Evaluates the predictions of the trackers to calculate final HOTA and IDF1 tracking metrics.

```bash
python evaluate_tracking.py --tracker kalman --conf_thresh 0.5 --iou_thresh 0.3 --max_age 3

```

* **Arguments**: Accepts the same core hyperparameter tuning arguments as `run_tracking.py` (`--tracker`, `--conf_thresh`, `--iou_thresh`, `--max_age`, `--proc_noise`, `--meas_noise`) but outputs metric tables instead of a video.

### 5. Bayesian Hyperparameter Optimization

These scripts use cached raw detections to quickly evaluate thousands of hyperparameter combinations and output the best settings as a JSON file.

* **For the Max IoU Tracker**:
```bash
python optimize_iou_tracker.py --trials 100

```


* **For the Kalman Tracker**:
```bash
python optimize_kalman_tracker.py --trials 100

```


* **Arguments**:
* `--trials`: The number of optimization trials Optuna should run (default: 100).