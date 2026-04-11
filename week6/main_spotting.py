#!/usr/bin/env python3
"""
File containing the main training script.
"""

#Standard imports
import argparse
import torch
import os
import numpy as np
import random
import time  # <-- Added for timing
import csv   # <-- Added for saving metrics to CSV
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

# WandB and Thop imports
import wandb
from thop import profile

#Local imports
from util.io import load_json, store_json
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model.model_spotting import Model


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # --- Initialize wandb and log Configuration file ---
    wandb.init(
        project="C6-Spotting",
        name=args.model,
        config=config  # Automatically logs all parameters from the JSON
    )
    # ---------------------------------------------------

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True) # Ensure save_dir exists for CSVs

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Model
    model = Model(args=args)

    # --- Calculate MACs and Parameters using thop ---
    print("Calculating MACs and Parameters...")
    dummy_input = torch.randn(1, args.clip_len, 3, 224, 398).to(args.device)
    macs, params = profile(model._model, inputs=(dummy_input, ), verbose=False)
    wandb.run.summary["MACs"] = macs
    wandb.run.summary["Parameters"] = params
    print(f"MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
    # ------------------------------------------------

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = float('inf')
        epoch = 0

        # --- Initialize Train Metrics CSV ---
        train_csv_path = os.path.join(args.save_dir, 'metrics_train.csv')
        with open(train_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'epoch_time_sec'])
        # ------------------------------------

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            # --- Start Epoch Timer and VRAM Tracker ---
            start_time = time.time()
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            # ------------------------------------------

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)
            
            val_loss = model.epoch(val_loader)

            # --- Calculate Epoch Time and Max VRAM ---
            epoch_time = time.time() - start_time
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if args.device == "cuda" else 0.0
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_per_epoch_sec": epoch_time,
                "vram_used_mb": vram_mb
            })
            # -----------------------------------------

            # --- Save Train Metrics to CSV ---
            with open(train_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, epoch_time])
            # ---------------------------------

            better = False
            if val_loss < best_criterion:
                best_criterion = val_loss
                better = True
            
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f} | Time: {:.2f}s | Peak VRAM: {:.2f} MB'.format(
                epoch, train_loss, val_loss, epoch_time, vram_mb))
            if better:
                print('New best mAP epoch!')

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss
            })

            if args.save_dir is not None:
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save( model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt') )

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    # --- Start Inference Timer ---
    inference_start = time.time()
    # -----------------------------

    # Evaluation on test split
    map_score, ap_score = evaluate(model, test_data, nms_window = 5)

    # --- Log Inference Time ---
    inference_time = time.time() - inference_start
    wandb.run.summary["inference_time_sec"] = inference_time
    print(f"Inference Time: {inference_time:.2f} seconds")
    # --------------------------

    # Report results per-class in table
    table = []
    ap_logs = {}
    for i, class_name in enumerate(classes.keys()):
        ap_val = ap_score[i] * 100
        table.append([class_name, f"{ap_val:.2f}"])
        ap_logs[f"AP/{class_name}"] = ap_val  # Log individual AP per class

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    # Report average results in table
    avg_table = [["Mean (mAP12)", f"{map_score*100:.2f}"]]
    headers = ["", "Average Precision"]
    print(tabulate(avg_table, headers, tablefmt="grid"))
    
    # --- Calculate AP10 ---
    if len(ap_score) >= 10:
        ap10_score = np.mean(ap_score[:10])
    else:
        ap10_score = map_score
    # ----------------------

    # --- Log Final Metrics to wandb (mAP12 and mAP10) ---
    ap_logs["mAP12"] = map_score * 100
    ap_logs["mAP10"] = ap10_score * 100
        
    wandb.log(ap_logs)
    wandb.run.summary["mAP12"] = ap_logs["mAP12"]
    wandb.run.summary["mAP10"] = ap_logs["mAP10"]
    wandb.finish()
    # ----------------------------------------------------

    # --- Save Eval Metrics to CSV ---
    eval_csv_path = os.path.join(args.save_dir, 'metrics_eval.csv')
    with open(eval_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'average_precision'])
        for i, class_name in enumerate(classes.keys()):
            writer.writerow([class_name, ap_score[i]])  # Saving raw float as requested
        writer.writerow(['mean', map_score])
        writer.writerow(['AP10', ap10_score])
    # --------------------------------

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


if __name__ == '__main__':
    main(get_args())