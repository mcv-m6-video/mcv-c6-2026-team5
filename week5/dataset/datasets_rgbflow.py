"""
File containing the function to load RGB+Flow datasets.
"""

import os
from util.dataset import load_classes
from dataset.frame_rgbflow import ActionSpotDatasetRGBFlow

DEFAULT_STRIDE = 2
DEFAULT_OVERLAP = 0.9


def get_datasets_rgbflow(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = args.stride if "stride" in args else DEFAULT_STRIDE
    overlap = args.overlap if "overlap" in args else DEFAULT_OVERLAP

    dataset_kwargs = {
        'stride': stride,
        'overlap': overlap,
        'dataset': args.dataset,
        'labels_dir': args.labels_dir,
        'task': args.task,
    }

    print('Dataset size:', dataset_len)

    train_data = ActionSpotDatasetRGBFlow(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.flow_dir, args.store_dir, args.store_mode,
        args.clip_len, dataset_len, **dataset_kwargs)
    train_data.print_info()

    val_data = ActionSpotDatasetRGBFlow(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.flow_dir, args.store_dir, args.store_mode,
        args.clip_len, dataset_len // 4, **dataset_kwargs)
    val_data.print_info()

    dataset_kwargs['overlap'] = 0

    test_data = ActionSpotDatasetRGBFlow(
        classes, os.path.join('data', args.dataset, 'test.json'),
        args.frame_dir, args.flow_dir, args.store_dir, args.store_mode,
        args.clip_len, None, pad_len=0, **dataset_kwargs)
    test_data.print_info()

    return classes, train_data, val_data, test_data