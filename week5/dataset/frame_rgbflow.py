#!/usr/bin/env python3

from util.io import load_json
import os
import random
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import pickle
import json
from pathlib import Path

DEFAULT_PAD_LEN = 5
FPS_SN = 25


class ActionSpotDatasetRGBFlow(Dataset):

    def __init__(
            self,
            classes,
            game_file,
            frame_dir,
            flow_dir,
            store_dir,
            store_mode,
            clip_len,
            dataset_len,
            stride=1,
            overlap=1,
            pad_len=DEFAULT_PAD_LEN,
            dataset='soccernetball',
            labels_dir=None,
            task='classification'
    ):
        self._src_file = game_file
        self._games = load_json(game_file)
        self._split = game_file.split('/')[-1].split('.')[0]
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._games)}
        self._dataset = dataset
        assert dataset == 'soccernetball'

        self._frame_dir = frame_dir
        self._flow_dir = flow_dir
        self._store_dir = store_dir
        self._store_mode = store_mode
        assert store_mode in ['store', 'load']

        self._clip_len = clip_len
        self._stride = stride
        assert clip_len > 0
        assert stride > 0
        assert overlap >= 0 and overlap <= 1

        self._clip_sampling_step = 1 if overlap == 1 else int((1 - overlap) * clip_len * stride)
        self._pad_len = pad_len
        assert pad_len >= 0

        self._labels_dir = labels_dir
        self._task = task
        assert task in ['classification', 'spotting']

        self._frame_reader = FrameReader(frame_dir, dataset=dataset)
        self._flow_reader = FlowChunkReader(flow_dir)

        if self._store_mode == 'store':
            self._store_clips()
        elif self._store_mode == 'load':
            self._load_clips()

        if dataset_len is None:
            self._dataset_len = len(self._frame_paths)
        else:
            self._dataset_len = dataset_len

        self._total_len = len(self._frame_paths)

    def _store_clips(self):
        self._frame_paths = []
        self._labels_store = []

        for video in tqdm(self._games):
            video_len = int(video['num_frames'])

            video_half = 1
            labels_file = load_json(
                os.path.join(self._labels_dir, video['video'] + '/Labels-ball.json')
            )['annotations']

            for base_idx in range(
                -self._pad_len * self._stride,
                max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride),
                self._clip_sampling_step
            ):
                frames_paths = self._frame_reader.load_paths(
                    video['video'],
                    base_idx,
                    base_idx + self._clip_len * self._stride,
                    stride=self._stride
                )

                labels = []

                for event in labels_file:
                    event_half = int(event['gameTime'][0])
                    if event_half == video_half:
                        event_frame = int(int(event['position']) / 1000 * FPS_SN)
                        label_idx = (event_frame - base_idx) // self._stride

                        if 0 <= label_idx < self._clip_len:
                            label = self._class_dict[event['label']]
                            labels.append({'label': label, 'label_idx': label_idx})

                if frames_paths[1] != -1:
                    self._frame_paths.append(frames_paths)
                    self._labels_store.append(labels)

        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'SPLIT' + self._split)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)

        print('Stored clips to ' + store_path)

    def _load_clips(self):
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'SPLIT' + self._split)

        with open(store_path + '/frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + '/labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)

        print('Loaded clips from ' + store_path)

    def _build_labels_from_store(self, dict_label):
        if self._task == 'spotting':
            labels = np.zeros(self._clip_len, np.int64)
            for label in dict_label:
                labels[label['label_idx']] = label['label']

        elif self._task == 'classification':
            labels = np.zeros(len(self._class_dict), np.int64)
            for label in dict_label:
                labels[label['label'] - 1] = 1

        return labels

    def _get_by_index(self, idx):
        frames_path = self._frame_paths[idx]
        dict_label = self._labels_store[idx]

        rgb = self._frame_reader.load_frames(frames_path, pad=True, stride=self._stride)
        flow = self._flow_reader.load_clip(frames_path, clip_len=self._clip_len, stride=self._stride)

        labels = self._build_labels_from_store(dict_label)

        return {
            'frame': rgb,
            'flow': flow,
            'contains_event': int(np.sum(labels) > 0),
            'label': labels
        }

    def _get_one(self):
        idx = random.randint(0, self._total_len - 1)
        return self._get_by_index(idx)

    def __getitem__(self, idx):
        if self._split == 'train':
            return self._get_one()
        else:
            return self._get_by_index(idx)

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._games)


class FrameReader:

    def __init__(self, frame_dir, dataset):
        self._frame_dir = frame_dir
        self.dataset = dataset

    def read_frame(self, frame_path):
        return torchvision.io.read_image(frame_path)

    def load_paths(self, video_name, start, end, stride=1):
        path = os.path.join(self._frame_dir, video_name)

        found_start = -1
        pad_start = 0
        pad_end = 0
        for frame_num in range(start, end, stride):
            if frame_num < 0:
                pad_start += 1
                continue

            if pad_end > 0:
                pad_end += 1
                continue

            frame_path = os.path.join(path, 'frame' + str(frame_num) + '.jpg')
            base_path = path
            ndigits = -1

            exist_frame = os.path.exists(frame_path)
            if exist_frame and found_start == -1:
                found_start = frame_num

            if not exist_frame:
                pad_end += 1

        ret = [base_path, found_start, pad_start, pad_end, ndigits, (end - start) // stride]
        return ret

    def load_frames(self, paths, pad=False, stride=1):
        base_path, start, pad_start, pad_end, ndigits, length = paths

        ret = []
        if ndigits == -1:
            path = os.path.join(base_path, 'frame')
            _ = [ret.append(self.read_frame(path + str(start + j * stride) + '.jpg'))
                 for j in range(length - pad_start - pad_end)]
        else:
            path = base_path + '/'
            _ = [ret.append(self.read_frame(path + str(start + j * stride).zfill(ndigits) + '.jpg'))
                 for j in range(length - pad_start - pad_end)]

        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        if pad_start > 0 or (pad and pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0)
            )

        return ret


class FlowChunkReader:
    def __init__(self, flow_dir):
        self._flow_dir = Path(flow_dir)
        self._chunk_cache = {}
        self._video_meta_cache = {}

    def _load_video_metadata(self, relative_video_dir):
        key = str(relative_video_dir)
        if key not in self._video_meta_cache:
            video_dir = self._flow_dir / relative_video_dir
            with open(video_dir / 'metadata.json', 'r') as f:
                meta = json.load(f)

            num_stored = meta['num_stored_flow_frames']
            chunk_size = meta['chunk_size']

            self._video_meta_cache[key] = {
                'meta': meta,
                'num_stored': num_stored,
                'chunk_size': chunk_size
            }

        return self._video_meta_cache[key]

    def _load_chunk(self, relative_video_dir, chunk_id):
        key = (str(relative_video_dir), chunk_id)
        if key not in self._chunk_cache:
            chunk_path = self._flow_dir / relative_video_dir / f'chunk_{chunk_id:05d}.npz'
            data = np.load(chunk_path)
            self._chunk_cache[key] = {
                'flow': data['flow'],
                'frame_idx': data['frame_idx']
            }
        return self._chunk_cache[key]

    def _decode_flow(self, flow_uint8, clip_value):
        flow = flow_uint8.astype(np.float32) / 255.0
        flow = flow * (2.0 * clip_value) - clip_value
        return torch.from_numpy(flow).permute(2, 0, 1).float()

    def load_clip(self, frame_paths, clip_len, stride):
        base_path, start, pad_start, pad_end, ndigits, length = frame_paths

        base_path = Path(base_path)
        relative_video_dir = Path(*base_path.parts[-3:])

        info = self._load_video_metadata(relative_video_dir)
        meta = info['meta']
        chunk_size = meta['chunk_size']
        clip_value = meta['clip_value']
        rgb_sample_stride = meta['rgb_sample_stride']

        frames = []
        valid_len = length - pad_start - pad_end

        for j in range(valid_len):
            frame_idx = start + j * stride

            if frame_idx < 0 or (frame_idx % rgb_sample_stride != 0):
                flow = torch.zeros((2, meta['height'], meta['width']), dtype=torch.float32)
            else:
                stored_index = frame_idx // rgb_sample_stride
                chunk_id = stored_index // chunk_size
                local_idx = stored_index % chunk_size

                chunk = self._load_chunk(relative_video_dir, chunk_id)
                flow_uint8 = chunk['flow'][local_idx]
                flow = self._decode_flow(flow_uint8, clip_value)

            frames.append(flow)

        ret = torch.stack(frames, dim=0)

        if pad_start > 0 or pad_end > 0:
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, pad_start, pad_end)
            )

        return ret


def _print_info_helper(src_file, labels):
    num_frames = sum([x['num_frames'] for x in labels])
    print('{} : {} videos, {} frames'.format(src_file, len(labels), num_frames))