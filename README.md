# MCV C6 - Video Analysis (Team 5)

This repository contains the source code, experiments, and results for the **C6: Video Analysis** course of the Master in Computer Vision (MCV). The coursework is divided into two main projects: Video Surveillance (Tracking) and Sports Video Analysis (Action Spotting).

## 👥 Team 5 Members

* Álvaro Díaz
* Bernat Medina
* Maiol Sabater
* Gerard Vilaplana

## Presentation Link

* [Final Presentation](https://docs.google.com/presentation/d/1jIR1OEuPBhrsNZia6PKogbipJuzMgmVGlLhO0GB-l0g/edit?usp=sharing)

## Checkpoints of Best Models

* Week5-7 - Best Model Ball Action: https://drive.google.com/drive/folders/1b0Fe95kcJf9bDk4NaBILYh-AnUrjxH9H?usp=drive_link

## 🚀 Projects Overview

### Project 1: Intelligent Video Surveillance (Weeks 1-4)
Focuses on building a complete pipeline for object detection, tracking, and Multi-Target Multi-Camera (MTMC) tracking for traffic monitoring scenarios.

* **[Week 1: Background Estimation & Object Detection](./week1)** Exploration of background subtraction models (MOG, MOG2, LSBP) and evaluation of off-the-shelf vs. fine-tuned object detectors (YOLO, Mask R-CNN).
* **[Week 2: Object Tracking](./week2)** Implementation of single-camera object tracking using IoU Tracker and Kalman Filters, alongside hyperparameter optimization.
* **[Week 3: Advanced Tracking & Optical Flow](./week3)** Integration of Optical Flow to improve tracking robustness and deep learning-based tracking refinement.
* **[Week 4: MTMC Tracking & ReID](./week4)** Expansion of tracking to multiple cameras (Multi-Target Multi-Camera) using metric learning (Re-Identification) and global graph matching.

### Project 2: Sports Video Analysis (Weeks 5-6)
Focuses on temporal action spotting within long, untrimmed sports broadcast videos.

* **[Week 5: Action Classification on SoccerNet](./week5)** Initial exploration using clip-level action classification. Implementation of Temporal Convolutional Networks (TCNs), Multi-clip fusion strategies, Class-Aware Sampling, and Multi-modal architectures (RGB + Optical Flow).

* **[Week 6: Action Spotting](./week6)** Transition to precise frame-level temporal action spotting. Implementation of Non-Maximum Suppression (NMS) ablations for peak detection, Temporal Shift modules, Lightweight Temporal Attention, and animated temporal probability visualizations.

* **[Week 7: Action Spotting](./week7)** Implementation of UNet with 3XD-M backbone for action spotting, increasing the performance from the past week. Techniques such as TGLS and temporal reduction were also implemented.

## 🛠️ General Setup & Installation

Each week contains its own specific requirements and execution instructions, but generally, the environment can be set up using:

```bash
# Clone the repository
git clone [https://github.com/your-org/mcv-c6-2026-team5.git](https://github.com/your-org/mcv-c6-2026-team5.git)
cd mcv-c6-2026-team5

# It is recommended to create a virtual environment
python -m venv venv
source venv/bin/activate
```
---

