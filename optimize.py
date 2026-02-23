import optuna
import pandas as pd
import os
import csv
from run_evaluation import Evaluator 

# --- CONFIG ---
VIDEO_PATH = "data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
GT_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
ROI_PATH = "data/AICity_data/AICity_data/train/S03/c010/roi.jpg"
STUDY_NAME = "task1_recursive_gaussian"
STORAGE_DB = "sqlite:///optuna_study.db"  # Saves progress to a file
CSV_LOG_FILE = "results/optimization_results.csv"

# Initialize Evaluator ONCE
evaluator = Evaluator(VIDEO_PATH, GT_PATH, ROI_PATH, split_ratio=0.25)

def save_to_csv(params, score, filename):
    """Appends a single trial's result to a CSV file."""
    file_exists = os.path.isfile(filename)
    
    # Combine params and score into one dict
    row_data = params.copy()
    row_data['mAP'] = score
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

def objective(trial):
    # 1. Sample Hyperparameters
    params = {
        # Background Model
        'alpha': trial.suggest_float('alpha', 1.5, 6.0),
        'rho': trial.suggest_float('rho', 0.005, 0.2, log=True),
        
        # Shadow Removal (HSV)
        'shadow_method': "hsv",
        'tau_s': trial.suggest_int('tau_s', 20, 90),
        'tau_h': trial.suggest_int('tau_h', 20, 90), # 0-180
        'shadow_alpha': trial.suggest_float('shadow_alpha', 0.4, 0.7),
        'shadow_beta': trial.suggest_float('shadow_beta', 0.8, 1.0),
        
        # Post-Processing
        'morph_kernel': trial.suggest_int('morph_kernel', 3, 7, step=2),
        'morph_op': trial.suggest_categorical('morph_op', ["open", "close", "open_close", "close_open"]),
        'min_area': trial.suggest_int('min_area', 50, 400),
        'merge_dist': trial.suggest_int('merge_dist', 30, 80)
    }
    
    # 2. Run Experiment
    try:
        print(f"\n--- Trial {trial.number} ---")
        print(f"Params: {params}")
        
        mAP = evaluator.run_experiment(params)
        
        if mAP is None: mAP = 0.0
        
        print(f"Result: mAP = {mAP:.4f}")
        
        # 3. Save to CSV immediately
        save_to_csv(params, mAP, CSV_LOG_FILE)
        
        return mAP
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

if __name__ == "__main__":
    # Create persistent study
    # load_if_exists=True allows you to CTRL+C and restart later without losing data
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_DB, 
        direction="maximize",
        load_if_exists=True
    )
    
    print("Starting optimization... Press Ctrl+C to stop.")
    # Run 100 trials (approx 2-3 hours depending on speed)
    study.optimize(objective, n_trials=100) 

    print("\n--- Best Trial ---")
    print(study.best_trial.params)