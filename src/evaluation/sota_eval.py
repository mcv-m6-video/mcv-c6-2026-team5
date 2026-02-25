import xml.etree.ElementTree as ET
from collections import defaultdict

from coco_eval import evaluate_coco
import cv2
import numpy as np


import xml.etree.ElementTree as ET

xml_path = "/ghome/group05/bernat/mcv-c6-2026-team5/data/ai_challenge_s03_c010-full_annotation.xml"
tree = ET.parse(xml_path)
root = tree.getroot()

print("Root tag:", root.tag)
print("First-level children tags:", [c.tag for c in list(root)[:20]])

# find any track-like elements anywhere
tracks = root.findall(".//track")
print("Num tracks found with .//track:", len(tracks))
if tracks:
    print("Example track attrs:", tracks[0].attrib)

# list unique labels if tracks exist
labels = set()
for t in tracks:
    if "label" in t.attrib:
        labels.add(t.attrib["label"])
print("Labels:", sorted(labels)[:50])



def load_gt_xml(xml_path, exclude_parked=True):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_boxes = defaultdict(list)

    for track in root.findall('track'):

        # Only cars&bikes
        if track.attrib.get('label') not in ['car', 'bike']:
            continue

        for box in track.findall('box'):

            # Skip if outside frame
            if box.attrib.get('outside') == '1':
                continue
            
            is_parked = False
            for attr in box.findall('attribute'):
                if attr.attrib.get('name') == 'parked':
                    if attr.text == 'true':
                        is_parked = True
                        break

            if exclude_parked and is_parked:
                continue

            frame_id = int(box.attrib['frame'])

            xtl = float(box.attrib['xtl'])
            ytl = float(box.attrib['ytl'])
            xbr = float(box.attrib['xbr'])
            ybr = float(box.attrib['ybr'])

            w = xbr - xtl
            h = ybr - ytl

            gt_boxes[frame_id].append([xtl, ytl, w, h])

    return gt_boxes

def load_pred_txt(path: str, has_score: bool = True):
    """
    Reads prediction txt with format:
      frame x y w h conf
    Returns:
      pred_boxes: dict[frame_id] = list of [x,y,w,h]
      pred_scores: dict[frame_id] = list of score (optional)
    """
    pred_boxes = {}
    pred_scores = {} if has_score else None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if has_score:
                if len(parts) != 6:
                    raise ValueError(f"Expected 6 columns (frame x y w h conf) but got {len(parts)} in: {line}")
                frame, x, y, w, h, s = parts
                s = float(s)
            else:
                if len(parts) != 5:
                    raise ValueError(f"Expected 5 columns (frame x y w h) but got {len(parts)} in: {line}")
                frame, x, y, w, h = parts
                s = 1.0

            frame = int(frame)
            box = [float(x), float(y), float(w), float(h)]

            pred_boxes.setdefault(frame, []).append(box)
            if pred_scores is not None:
                pred_scores.setdefault(frame, []).append(float(s))

    return pred_boxes, pred_scores




def get_video_hw(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return H, W

if __name__ == "__main__":
    video_path = "/ghome/group05/bernat/mcv-c6-2026-team5/data/AICity_data/AICity_data/train/S03/c010/vdo.avi"

    gt_xml = "/ghome/group05/bernat/mcv-c6-2026-team5/data/ai_challenge_s03_c010-full_annotation.xml"  
    pred_txt = "/ghome/group05/gerard/mcv-c6-2026-team5/results/task3_bg_sota_min_hw_15/dets_mog2_open5_close15.txt"
    print("pred_txt:", pred_txt)

    gt_boxes = load_gt_xml(gt_xml, exclude_parked=True)
    pred_boxes, pred_scores = load_pred_txt(pred_txt, has_score=True)

    H, W = get_video_hw(video_path)

    map50 = evaluate_coco(gt_boxes, pred_boxes, H, W)
        # --- Debug counts ---
    gt_frames = sorted(gt_boxes.keys())
    pred_frames = sorted(pred_boxes.keys())

    print("GT frames:", (gt_frames[0], gt_frames[-1]) if gt_frames else "EMPTY")
    print("Pred frames:", (pred_frames[0], pred_frames[-1]) if pred_frames else "EMPTY")

    num_gt = sum(len(v) for v in gt_boxes.values())
    num_pred = sum(len(v) for v in pred_boxes.values())

    print("Total GT boxes:", num_gt)
    print("Total Pred boxes:", num_pred)

    inter = set(gt_boxes.keys()) & set(pred_boxes.keys())
    print("Frame intersection size:", len(inter))
    if inter:
        sample = sorted(list(inter))[:5]
        print("Sample intersecting frames:", sample, "GT boxes:", len(gt_boxes[sample[0]]), "Pred boxes:", len(pred_boxes[sample[0]]))
    print("AP@0.5 =", map50)