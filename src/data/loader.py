import os
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm

class AICityDataset(Dataset):
    def __init__(self, video_path, xml_path, force_extract=False):
        """
        Args:
            video_path (str): Path to the .mp4 or .avi file.
            xml_path (str): Path to the .xml or .txt annotation file.
            force_extract (bool): If True, re-extracts frames even if folder exists.
        """
        self.video_path = video_path
        self.xml_path = xml_path
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.cache_dir = os.path.join(os.path.dirname(video_path), "frames_cache", video_name)
        
        if force_extract or not os.path.exists(self.cache_dir):
            self._extract_frames()
        else:
            if len(os.listdir(self.cache_dir)) == 0:
                self._extract_frames()
            else:
                print(f"Loading frames from existing cache: {self.cache_dir}")

        self.imgs = list(sorted(os.listdir(self.cache_dir)))
        
        if self.xml_path.endswith('.txt'):
            self.ground_truth = self._load_gt_txt(self.xml_path)
        else:
            self.ground_truth = self._load_gt_xml(self.xml_path)

    def _extract_frames(self):
        print(f"Extracting frames from {self.video_path} to {self.cache_dir}...")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
            
        frame_idx = 0
        pbar = tqdm()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            filename = os.path.join(self.cache_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, frame)
            
            frame_idx += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        print("Extraction complete.")

    def _load_gt_txt(self, txt_path):
        if not os.path.exists(txt_path):
            print(f"Warning: GT file {txt_path} not found.")
            return defaultdict(list)

        gt_boxes = defaultdict(list)
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                # Fallback for space-separated files
                if len(parts) < 6:
                    parts = line.strip().split()
                if len(parts) < 6:
                    continue
                
                # Align MOT 1-indexed frames with 0-indexed dataset caching
                frame_id = int(parts[0]) - 1
                track_id = int(parts[1])
                
                left = float(parts[2])
                top = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                
                x1 = left
                y1 = top
                x2 = left + width
                y2 = top + height
                
                gt_boxes[frame_id].append([x1, y1, x2, y2, track_id])
                
        return gt_boxes

    def _load_gt_xml(self, xml_path, exclude_parked=True):
        if not os.path.exists(xml_path):
            print(f"Warning: GT file {xml_path} not found.")
            return defaultdict(list)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        gt_boxes = defaultdict(list)

        for track in root.findall('track'):
            if track.attrib.get('label') not in ['car']:
                continue
            track_id = int(track.attrib['id'])

            for box in track.findall('box'):
                if box.attrib.get('outside') == '1':
                    continue
                
                frame_id = int(box.attrib['frame'])
                
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])
                
                gt_boxes[frame_id].append([xtl, ytl, xbr, ybr, track_id])

        return gt_boxes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.cache_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        frame_data = self.ground_truth.get(idx, [])
        
        target = {}
        target['image_id'] = torch.tensor([idx])
        
        if len(frame_data) > 0:
            frame_data = torch.tensor(frame_data, dtype=torch.float32)
            
            boxes = frame_data[:, :4] 
            ids = frame_data[:, 4]    
            
            target['boxes'] = boxes
            target['track_id'] = ids.to(torch.int64) 
            target['labels'] = torch.ones((len(boxes),), dtype=torch.int64)
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['track_id'] = torch.zeros((0,), dtype=torch.int64) 
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))