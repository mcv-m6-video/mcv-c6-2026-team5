# MCV-C6-2026-Team5

This repository contains code and resources for a computer vision project, organized for modularity and clarity. Below is an overview of the main folders and selected scripts.

## Folder Structure

### `src/`
This is the main source code directory, structured as follows:

- **background/**
  - `base.py`: Base classes and interfaces for background modeling.
  - `gaussian.py`: Implements Gaussian background subtraction methods.
  - `sota.py`: State-of-the-art background modeling algorithms.

- **configs_sota/**
  - `config.json`: Configuration file for SOTA models and experiments.

- **data/**
  - `parser.py`: Utilities for parsing datasets and annotations.

- **evaluation/**
  - `coco_eval.py`: COCO-style evaluation metrics.
  - `iou.py`: Custom intersection-over-Union calculations.
  - `map.py`: custom mean Average Precision evaluation.
  - `sota_conf_filtering.py`: Confidence filtering for SOTA models.
  - `sota_eval.py`: Evaluation routines for SOTA models.

- **utils/**
  - `post_processing.py`: Post-processing utilities for predictions.
  - `shadow_removal.py`: Functions for removing shadows from images.

- **visualization/**
  - `debugger.py`: Visualization and debugging tools for model outputs.


## Key Scripts

- **gridsearch_alpha.py**
  - Performs grid search over the alpha parameter for model optimization. Useful for hyperparameter tuning to find the best alpha value for background subtraction or related tasks.

- **optimize.py**
  - Contains routines for optimizing model parameters, possibly using techniques like Optuna or other optimization libraries. Automates the search for the best configuration.

- **run_evaluation.py**
  - Main entry point for evaluating models. Runs evaluation metrics on predictions, aggregates results, and outputs performance statistics.

- **run_zbs.py**
  - Executes experiments or evaluations using the Zero Buffering Strategy (ZBS), a specific approach or baseline for comparison in the project.


For more details on usage or contributing, please refer to comments in the code or contact the maintainers.
