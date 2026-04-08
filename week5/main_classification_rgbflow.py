#!/usr/bin/env python3
"""
Main training script for RGB + Optical Flow classification.
"""

# Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tabulate import tabulate
import csv
import time

# Local imports
from util.io import load_json, store_json
from util.eval_classification_rgbflow import evaluate_rgbflow
from dataset.datasets_rgbflow import get_datasets_rgbflow
from model.model_classification_rgbflow import Model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def update_args(args, config):
    # Basic paths / data
    args.frame_dir = config['frame_dir']
    args.flow_dir = config['flow_dir']
    args.save_dir = config['save_dir'] + '/' + args.model
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.stride = config['stride']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']

    # Model / optimization
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']

    # Loss
    args.loss_type = config.get('loss_type', 'bce')
    args.class_aware_sampling = config.get('class_aware_sampling', False)

    # Temporal head
    args.temporal_head = config.get('temporal_head', 'tcn_maxpool')
    args.tcn_num_layers = config.get('tcn_num_layers', 3)
    args.tcn_kernel_size = config.get('tcn_kernel_size', 3)
    args.tcn_hidden_dim = config.get('tcn_hidden_dim', None)
    args.tcn_dropout = config.get('tcn_dropout', 0.2)

    # RGB + Flow specifics
    args.fusion_alpha = config.get('fusion_alpha', 0.5)
    args.flow_clip_value = config.get('flow_clip_value', 20.0)

    # Optional explicit checkpoint override
    args.checkpoint_path = config.get('checkpoint_path', None)

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


def compute_pos_weight(train_data, num_classes):
    pos_counts = np.zeros(num_classes, dtype=np.float64)
    total_samples = 0

    for i in range(len(train_data)):
        sample = train_data[i]
        label = sample['label']

        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        pos_counts += label
        total_samples += 1

    neg_counts = total_samples - pos_counts
    pos_counts = np.maximum(pos_counts, 1.0)

    pos_weight = neg_counts / pos_counts
    return pos_weight.tolist()


def compute_sample_weights(train_data, num_classes):
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for i in range(len(train_data)):
        label = train_data[i]['label']
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        class_counts += label

    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / np.sqrt(class_counts)

    sample_weights = []

    for i in range(len(train_data)):
        label = train_data[i]['label']
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        pos_indices = np.where(label > 0)[0]

        if len(pos_indices) > 0:
            weight = np.max(class_weights[pos_indices])
        else:
            weight = 1.0

        sample_weights.append(weight)

    print("Sample weights stats:",
          np.min(sample_weights),
          np.max(sample_weights),
          np.mean(sample_weights))

    return sample_weights


def main(args):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets
    classes, train_data, val_data, test_data = get_datasets_rgbflow(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    # Prepare loss args BEFORE creating model
    args.pos_weight = None
    if (not args.only_test) and args.loss_type == 'weighted_bce':
        args.pos_weight = compute_pos_weight(train_data, args.num_classes)
        print('Computed pos_weight from training set:')
        print(args.pos_weight)

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    if (not args.only_test) and args.class_aware_sampling:
        print("Using class-aware sampling")

        sample_weights = compute_sample_weights(train_data, args.num_classes)
        sample_weights = torch.DoubleTensor(sample_weights)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_data,
            sampler=sampler,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            prefetch_factor=(2 if args.num_workers > 0 else None),
            worker_init_fn=worker_init_fn
        )

    else:
        train_loader = DataLoader(
            train_data,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
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

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)

        losses = []
        best_criterion = float('inf')
        epoch = 0

        train_csv_path = os.path.join(args.save_dir, 'metrics_train.csv')

        # Create CSV header
        with open(train_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'epoch_time_sec'])

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):
            start_time = time.time()

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)

            val_loss = model.epoch(val_loader)
            epoch_time = time.time() - start_time

            better = False
            if val_loss < best_criterion:
                best_criterion = val_loss
                better = True

            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
                epoch, train_loss, val_loss))

            with open(train_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, epoch_time])

            if better:
                print('New best epoch based on val loss!')

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt'))

    print('START INFERENCE')
    print('Loading best checkpoint')

    if args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        ckpt_path = os.path.join(ckpt_dir, 'checkpoint_best.pt')

    print(f'Checkpoint path: {ckpt_path}')
    model.load(torch.load(ckpt_path))

    # Evaluation on test split
    ap_score = evaluate_rgbflow(model, test_data)

    eval_csv_path = os.path.join(args.save_dir, 'metrics_eval.csv')

    with open(eval_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['class', 'average_precision'])

        # Per-class AP
        for i, class_name in enumerate(classes.keys()):
            writer.writerow([class_name, ap_score[i]])

        # Mean AP
        writer.writerow(['mean', np.mean(ap_score)])

        valid_classes = [i for i, c in enumerate(classes.keys()) if c not in ['FREE KICK', 'GOAL']]
        ap10 = np.mean([ap_score[i] for i in valid_classes])

        writer.writerow(['AP10', ap10])

    # Report results per-class in table
    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i]*100:.2f}"])

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    # Report average results in table
    avg_table = [["Average", f"{np.mean(ap_score)*100:.2f}"]]
    headers = ["", "Average Precision"]

    print(tabulate(avg_table, headers, tablefmt="grid"))
    print(f"AP10: {ap10*100:.2f}")

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


if __name__ == '__main__':
    main(get_args())