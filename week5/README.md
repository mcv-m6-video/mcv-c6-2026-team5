# Week 5: Action Spotting on SoccerNet

This folder contains the codebase for **Project 2: Sports Video Analysis**. The goal of this week is to perform Temporal Action Spotting on the **SoccerNet Ball Action Spotting** dataset, correctly predicting the timestamp and class of specific actions (Pass, Drive, Shot, etc.) from untrimmed video broadcasts.

## Checkpoints of Best Models
* Access to this shared Drive folder for the best models: https://drive.google.com/drive/folders/1b0Fe95kcJf9bDk4NaBILYh-AnUrjxH9H?usp=drive_link

## 📁 Repository Structure

* `config/`: Contains JSON configuration files for different experiments (Baseline, Strides, Attention Pooling, Multiclip, RGB+Flow, etc.).
* `data/` & `dataset/`: Dataloaders, dataset parsers, and splits management.
* `model/`: PyTorch model architectures including Temporal Convolutional Networks (TCN) and different temporal heads.
* `util/`: Helper functions for I/O, evaluation metrics (mAP, AP10), and dataset processing.

## ⚙️ Setup and Data Preparation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and Extract Data:**
   We provide scripts to automatically download the dataset, extract RGB frames, and compute Optical Flow representations.
   ```bash
   python download_frames_snb.py
   python extract_frames_snb.py
   python extract_OF.py
   ```

3. **Exploratory Data Analysis (EDA):**
   To check class imbalances and dataset statistics, run:
   ```bash
   python eda_soccernet.py --split train
   ```

## 🚀 Training & Evaluation

The training pipeline is controlled via JSON configuration files. This allows for easily reproducible experiments without changing command-line arguments manually.

**Train the Baseline Model:**
```bash
python main_classification.py --model baseline
```

**Train a Multi-Modal (RGB + Flow) Model:**
```bash
python main_classification_rgbflow.py --model rgbflow_tcn
```

### Computing MACs & Parameters
To evaluate the computational cost (Multiply-Accumulates and Params) of a specific model configuration:
```bash
python compute_macs.py --model baseline
python compute_macs.py --model head_3layer_tcn
```

## 🎥 Qualitative Analysis (Video Generation)

We provide a robust script to generate side-by-side qualitative comparisons between models. It overlays the Ground Truth, Baseline predictions, and Best Model predictions directly onto the video frames.

**Generate random comparison clips:**
```bash
python qualitative_analysis_v2.py --model_base baseline --model_best head_3layer_tcn --num_clips 5
```

**Smart Search for Improvements:**
You can use the `--find_improvement` flag to automatically scan the dataset and generate clips *only* where your best model predicted the correct action, but the baseline failed. You can also filter by a specific target class.
```bash
# Find 3 clips where the best model correctly spots a "Shot" (Class 5) but the baseline fails
python qualitative_analysis_v2.py --model_base baseline --model_best multiclip_max_BEST_s4 --target_class 5 --num_clips 3 --find_improvement
```