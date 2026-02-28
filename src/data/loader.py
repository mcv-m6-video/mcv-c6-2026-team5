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
            xml_path (str): Path to the .xml annotation file.
            force_extract (bool): If True, re-extracts frames even if folder exists.
        """
        self.video_path = video_path
        self.xml_path = xml_path
        
        # 1. Setup Cache Directory
        # We create a folder with the same name as the video to store extracted frames
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.cache_dir = os.path.join(os.path.dirname(video_path), "frames_cache", video_name)
        
        # 2. Extract Frames if they don't exist
        if force_extract or not os.path.exists(self.cache_dir):
            self._extract_frames()
        else:
            # Check if cache looks empty
            if len(os.listdir(self.cache_dir)) == 0:
                self._extract_frames()
            else:
                print(f"Loading frames from existing cache: {self.cache_dir}")

        # 3. List all frame files (sorted)
        self.imgs = list(sorted(os.listdir(self.cache_dir)))
        
        # 4. Load Annotations
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
                
            # Save as formatted filename (e.g., frame_0000.jpg)
            # This matches the frame_id used in the XML parser
            filename = os.path.join(self.cache_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, frame)
            
            frame_idx += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        print("Extraction complete.")

    def _load_gt_xml(self, xml_path, exclude_parked=True):
        """
        Parses the XML using the specific logic provided.
        """
        if not os.path.exists(xml_path):
            print(f"Warning: GT file {xml_path} not found.")
            return defaultdict(list)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        gt_boxes = defaultdict(list)

        for track in root.findall('track'):
            # Only cars & bikes
            if track.attrib.get('label') not in ['car', 'bike']:
                continue

            for box in track.findall('box'):
                # LOGIC FIX: Standard CVAT uses outside='1' for invisible/left frame.
                # If outside='0', it IS visible. 
                # We skip ONLY if it is outside (== '1').
                if box.attrib.get('outside') == '1':
                    continue
                if box.attrib.get('occluded') == '1':
                    continue
                
                # Check parked attribute
                # is_parked = False
                # for attr in box.findall('attribute'):
                #     if attr.attrib.get('name') == 'parked':
                #         if attr.text == 'true':
                #             is_parked = True
                #             break

                # if exclude_parked and is_parked:
                #     continue

                frame_id = int(box.attrib['frame'])
                
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])
                
                # PyTorch/Faster-RCNN expects [x1, y1, x2, y2].
                # We store [x1, y1, x2, y2] here to be compatible with the model.
                gt_boxes[frame_id].append([xtl, ytl, xbr, ybr])

        return gt_boxes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 1. Load Image from Cache
        img_name = self.imgs[idx]
        img_path = os.path.join(self.cache_dir, img_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to Tensor (C, H, W) normalized 0-1
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 2. Get Targets
        # The XML parser uses frame_id (int). 
        # We assume file "frame_0000.jpg" corresponds to frame_id 0.
        frame_id = idx 
        boxes = self.ground_truth.get(frame_id, [])
        
        target = {}
        target['image_id'] = torch.tensor([idx])
        
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            
            # Area is required for evaluation
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            
            # ISCROWD (0 = regular object)
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            
            target['boxes'] = boxes_tensor
            # Class 1 = Car (0 is background)
            target['labels'] = torch.ones((len(boxes),), dtype=torch.int64)
            target['area'] = area
            target['iscrowd'] = iscrowd
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))