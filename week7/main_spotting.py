#!/usr/bin/env python3
"""
File containing the main training script.
"""

import argparse
import torch
import os
import numpy as np
import random
import time
import csv
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

import wandb
from thop import profile

from util.io import load_json, store_json
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model.model_spotting import Model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def update_args(args, config):
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model
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

    # NMS and smoothing options
    args.nms_window = config.get('nms_window', 5)
    args.nms_type = config.get('nms_type', 'hard')
    args.nms_thresh = config.get('nms_thresh', 0.05)
    args.smoothing = config.get('smoothing', None)
    args.smoothing_window = config.get('smoothing_window', 3)
    args.soft_nms_sigma = config.get('soft_nms_sigma', 1.0)

    # Checkpoint / rerun
    args.rerun_path = config.get('rerun_path', None)
    args.checkpoint_path = config.get('checkpoint_path', None)

    # X3D options
    args.gru_hidden = config.get('gru_hidden', 256)
    args.gru_layers = config.get('gru_layers', 2)

    # Early stopping 
    args.early_stopping_patience = config.get('early_stopping_patience', None)

    return args


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs),
    ])


def main(args):
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    wandb.init(project="C6-Spotting-W7", name=args.model, config=config, entity="Team5-C5")

    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(worker_id):
        random.seed(worker_id + epoch * 100)

    train_loader = DataLoader(
        train_data,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn,
    )

    model = Model(args=args)

    print("Calculating MACs and Parameters...")
    dummy_input = torch.randn(1, args.clip_len, 3, 224, 398).to(args.device)
    macs, params = profile(model._model, inputs=(dummy_input,), verbose=False)
    wandb.run.summary["MACs"] = macs
    wandb.run.summary["Parameters"] = params
    print(f"MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)

        losses = []
        best_criterion = -1.0
        epoch = 0

        # ── Early stopping ───────────────────────────────────────────
        patience = args.early_stopping_patience          
        epochs_no_improve = 0                            

        train_csv_path = os.path.join(args.save_dir, 'metrics_train.csv')
        with open(train_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss',
                'val_mAP_1s', 'val_mAP_05s', 'epoch_time_sec'
            ])

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):
            start_time = time.time()
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)

            val_loss = model.epoch(val_loader)

            val_mAP_1s, val_mAP_05s, _, _ = evaluate(
                model, val_data,
                nms_window=args.nms_window,
                nms_type=args.nms_type,
                nms_thresh=args.nms_thresh,
                smoothing=args.smoothing,
                smoothing_window=args.smoothing_window,
                soft_nms_sigma=args.soft_nms_sigma,
            )

            epoch_time = time.time() - start_time
            vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                       if args.device == "cuda" else 0.0)

            better = val_mAP_1s > best_criterion
            if better:
                best_criterion = val_mAP_1s
                epochs_no_improve = 0                    
            else:
                epochs_no_improve += 1                   

            print('[Epoch {}] Train loss: {:0.5f} | Val loss: {:0.5f} | '
                  'Val mAP@1s: {:.2f} | Val mAP@0.5s: {:.2f} | '
                  'Time: {:.2f}s | Peak VRAM: {:.2f} MB{}'.format(
                epoch, train_loss, val_loss,
                val_mAP_1s * 100, val_mAP_05s * 100,
                epoch_time, vram_mb,
                '  ← BEST' if better else ''))

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mAP_1s": val_mAP_1s * 100,
                "val_mAP_05s": val_mAP_05s * 100,
                "time_per_epoch_sec": epoch_time,
                "vram_used_mb": vram_mb,
            })

            with open(train_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, train_loss, val_loss,
                    val_mAP_1s * 100, val_mAP_05s * 100, epoch_time
                ])

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss,
                'val_mAP_1s': val_mAP_1s * 100,
                'val_mAP_05s': val_mAP_05s * 100,
            })

            if args.save_dir is not None:
                store_json(os.path.join(args.save_dir, 'loss.json'),
                           losses, pretty=True)
                if better:
                    torch.save(
                        model.state_dict(),
                        os.path.join(ckpt_dir, 'checkpoint_best.pt'))

            # ── Early stopping check ─────────────────────────────────
            if patience is not None and epochs_no_improve >= patience:  
                print(f'Early stopping triggered at epoch {epoch} '        
                      f'(no improvement for {patience} epochs)')           
                break                                                       

    print('START INFERENCE')
    if args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        ckpt_path = os.path.join(ckpt_dir, 'checkpoint_best.pt')

    print(f"Loading checkpoint from: {ckpt_path}")
    model.load(torch.load(ckpt_path))

    inference_start = time.time()
    test_mAP_1s, test_mAP_05s, test_ap_1s, test_ap_05s = evaluate(
        model, test_data,
        nms_window=args.nms_window,
        nms_type=args.nms_type,
        nms_thresh=args.nms_thresh,
        smoothing=args.smoothing,
        smoothing_window=args.smoothing_window,
        soft_nms_sigma=args.soft_nms_sigma,
    )
    inference_time = time.time() - inference_start
    wandb.run.summary["inference_time_sec"] = inference_time
    print(f"Inference Time: {inference_time:.2f} seconds")

    # ── Tabla per-class ──────────────────────────────────────────────
    table = []
    ap_logs = {}
    for i, class_name in enumerate(classes.keys()):
        ap_1s = test_ap_1s[i] * 100
        ap_05s = test_ap_05s[i] * 100
        table.append([class_name, f"{ap_1s:.2f}", f"{ap_05s:.2f}"])
        ap_logs[f"AP_1s/{class_name}"] = ap_1s
        ap_logs[f"AP_05s/{class_name}"] = ap_05s

    headers = ["Class", "AP@1s", "AP@0.5s"]
    print(tabulate(table, headers, tablefmt="grid"))

    # mAP10 (excluyendo Free Kick y Goal, clases 10 y 11)
    test_mAP10_1s  = float(np.mean(test_ap_1s[:10]))  * 100
    test_mAP10_05s = float(np.mean(test_ap_05s[:10])) * 100

    avg_table = [
        ["mAP12 @ 1s",   f"{test_mAP_1s  * 100:.2f}"],
        ["mAP12 @ 0.5s", f"{test_mAP_05s * 100:.2f}"],
        ["mAP10 @ 1s",   f"{test_mAP10_1s:.2f}"],
        ["mAP10 @ 0.5s", f"{test_mAP10_05s:.2f}"],
    ]
    headers = ["Metric", "Value"]
    print(tabulate(avg_table, headers, tablefmt="grid"))

    ap_logs.update({
        "mAP12_1s":   test_mAP_1s  * 100,
        "mAP12_05s":  test_mAP_05s * 100,
        "mAP10_1s":   test_mAP10_1s,
        "mAP10_05s":  test_mAP10_05s,
    })
    wandb.log(ap_logs)
    wandb.run.summary["mAP12_1s"]  = test_mAP_1s  * 100
    wandb.run.summary["mAP12_05s"] = test_mAP_05s * 100
    wandb.run.summary["mAP10_1s"]  = test_mAP10_1s
    wandb.run.summary["mAP10_05s"] = test_mAP10_05s
    wandb.finish()

    # ── CSV de resultados test ───────────────────────────────────────
    eval_base_dir = args.rerun_path if args.rerun_path is not None else args.save_dir
    os.makedirs(eval_base_dir, exist_ok=True)
    eval_csv_path = os.path.join(eval_base_dir, 'metrics_eval.csv')

    with open(eval_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'AP_1s', 'AP_05s'])
        for i, class_name in enumerate(classes.keys()):
            writer.writerow([
                class_name,
                test_ap_1s[i] * 100,
                test_ap_05s[i] * 100
            ])
        writer.writerow(['mAP12', test_mAP_1s  * 100, test_mAP_05s * 100])
        writer.writerow(['mAP10', test_mAP10_1s,       test_mAP10_05s])

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


if __name__ == '__main__':
    main(get_args())