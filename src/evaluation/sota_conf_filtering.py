import os
import csv
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import cv2

from coco_eval import evaluate_coco


def load_gt_xml(xml_path, exclude_parked=True):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_boxes = defaultdict(list)

    for track in root.findall(".//track"):
        label = (track.attrib.get("label") or "").strip().lower()

        # Only cars & bikes (adjust if your GT uses bicycle)
        if label not in ["car", "bike", "bicycle"]:
            continue

        for box in track.findall("box"):
            if box.attrib.get("outside") == "1":
                continue

            is_parked = False
            for attr in box.findall("attribute"):
                if attr.attrib.get("name") == "parked" and (attr.text or "").strip().lower() == "true":
                    is_parked = True
                    break

            if exclude_parked and is_parked:
                continue

            frame_id = int(float(box.attrib["frame"]))
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            gt_boxes[frame_id].append([xtl, ytl, xbr - xtl, ybr - ytl])

    return gt_boxes


def load_pred_txt(path: str, has_score: bool = True):
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
                    raise ValueError(f"Expected 6 columns but got {len(parts)} in: {line}")
                frame, x, y, w, h, s = parts
                s = float(s)
            else:
                if len(parts) != 5:
                    raise ValueError(f"Expected 5 columns but got {len(parts)} in: {line}")
                frame, x, y, w, h = parts
                s = 1.0

            frame = int(frame)
            box = [float(x), float(y), float(w), float(h)]

            pred_boxes.setdefault(frame, []).append(box)
            if pred_scores is not None:
                pred_scores.setdefault(frame, []).append(s)

    return pred_boxes, pred_scores


def get_video_hw(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return H, W


def filter_txt_by_conf(in_txt: str, out_txt: str, conf_thr: float):
    with open(in_txt, "r") as f_in, open(out_txt, "w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 6 and float(parts[5]) >= conf_thr:
                f_out.write(line)


def append_row(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "subtype", "mAP50"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    video_path = "/ghome/group05/bernat/mcv-c6-2026-team5/data/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    gt_xml = "/ghome/group05/bernat/mcv-c6-2026-team5/data/ai_challenge_s03_c010-full_annotation.xml"
    csv_file = "/ghome/group05/gerard/mcv-c6-2026-team5/results/results.csv"

    files = [
        ("mog",  "/ghome/group05/gerard/mcv-c6-2026-team5/results/task3_bg_sota_min_hw_25/dets_mog_open5_close15.txt"),
        ("mog2", "/ghome/group05/gerard/mcv-c6-2026-team5/results/task3_bg_sota_min_hw_25/dets_mog2_open5_close15.txt"),
        ("lsbp", "/ghome/group05/gerard/mcv-c6-2026-team5/results/task3_bg_sota_roi/dets_lsbp.txt"),
    ]

    gt_boxes = load_gt_xml(gt_xml, exclude_parked=True)
    H, W = get_video_hw(video_path)

    for conf in np.round(np.arange(0.0, 1.01, 0.1), 1):
        for model_name, in_txt in files:
            out_txt = in_txt.replace(".txt", f"_conf_{conf:.1f}.txt")
            filter_txt_by_conf(in_txt, out_txt, float(conf))

            pred_boxes, pred_scores = load_pred_txt(out_txt, has_score=True)
            map50 = evaluate_coco(gt_boxes, pred_boxes, H, W)  # pass scores if your eval supports it

            # Append to CSV (no overwrite)
            append_row(csv_file, {
                "model": model_name,
                "subtype": f"conf_{conf:.1f}",
                "mAP50": float(map50),
            })

            print(f"{model_name} conf={conf:.1f} -> AP@0.5={map50:.4f}")