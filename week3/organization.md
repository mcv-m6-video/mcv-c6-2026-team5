# Video Surveillance for Road Traffic Monitoring - Week 3

[cite_start]This repository contains the implementations for optical flow estimation and multi-target single-camera (MTSC) tracking[cite: 46, 57, 58]. 

## Directory Structure

* `data/`
    * [cite_start]`kitti/`: Contains Sequence 45 (image_0) used for optical flow evaluation[cite: 173].
    * [cite_start]`ai_city_challenge/`: Contains Sequences S01, S03, and S04 for MTSC tracking[cite: 240, 247, 250]. [cite_start]*Note: Data must be deleted after the module finishes[cite: 244].*
* `src/`
    * `optical_flow/`
        * [cite_start]`pyflow_estimator.py`: Implementation for computing optical flow using PyFlow[cite: 121].
        * [cite_start]`state_of_art_estimators.py`: Wrappers for newer learning-based methods like RAFT, MaskFlowNet, or Perceiver IO[cite: 128, 135, 136, 137].
        * [cite_start]`flow_metrics.py`: Functions to compute Mean Square Error in Non-occluded areas (MSEN) and Percentage of Erroneous Pixels in Non-occluded areas (PEPN)[cite: 171].
    * `tracking/`
        * [cite_start]`base_tracker.py`: Your baseline object tracking algorithm from previous weeks[cite: 191, 197].
        * [cite_start]`of_tracker.py`: Enhanced tracking algorithm that uses optical flow to predict bounding box positions in subsequent frames[cite: 204].
        * [cite_start]`tracking_metrics.py`: Evaluation scripts to compute IDF1 and HOTA metrics[cite: 223].
    * `utils/`
        * [cite_start]`visualization.py`: Tools for generating qualitative results for the final slides[cite: 248, 251].
* `experiments/`
    * [cite_start]`run_optical_flow_eval.py`: Script to compute MSEN, PEPN, and inference runtime for all optical flow methods[cite: 169, 172].
    * [cite_start]`run_tracking_ablation.py`: Script to compare IDF1/HOTA results of the tracker with and without optical flow[cite: 197].
    * [cite_start]`run_mtsc_eval.py`: Main script to evaluate the best tracking algorithm on the AI City Challenge sequences (SEQ01 and SEQ03) across all cameras[cite: 222, 225, 247, 250].
* `results/`
    * [cite_start]`optical_flow/`: Directory to store quantitative tables and flow visualizations[cite: 169, 188].
    * [cite_start]`tracking/`: Directory to store the tracking metric outputs per camera and qualitative tracking visualizations[cite: 248, 249].