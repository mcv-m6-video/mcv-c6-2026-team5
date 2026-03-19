import os
import csv
from pathlib import Path
from PIL import Image


DATASET_ROOT = Path("/home/group05/bernat/mcv-c6-2026-team5/data/AI_CITY_CHALLENGE_2022_TRAIN/train/")
IMAGES_TRAIN_DIR = Path("/ghome/group05/maiol/mcv-c6-2026-team5/yolo_train/yolo_dataset/images/train")
IMAGES_VAL_DIR = Path("/ghome/group05/maiol/mcv-c6-2026-team5/yolo_train/yolo_dataset/images/val")

OUTPUT_ROOT = Path("./reid_crops")
CSV_OUT = Path("./reid_annotations.csv")

MIN_W = 10
MIN_H = 10

# safest default unless you know IDs are global across cameras
USE_GLOBAL_IDS = True


def parse_gt_line(line: str):
    """
    MOT-style line:
    frame,id,x,y,w,h,conf,-1,-1,-1
    """
    parts = line.strip().split(",")
    if len(parts) < 6:
        return None

    frame_id = int(parts[0])
    track_id = int(parts[1])
    x = float(parts[2])
    y = float(parts[3])
    w = float(parts[4])
    h = float(parts[5])

    conf = 1.0
    if len(parts) >= 7:
        conf = float(parts[6])

    return frame_id, track_id, x, y, w, h, conf


def clamp_box(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(0, min(int(round(x2)), img_w))
    y2 = max(0, min(int(round(y2)), img_h))
    return x1, y1, x2, y2


def build_identity(seq_name: str, track_id: int):
    if USE_GLOBAL_IDS:
        return str(track_id)
    return f"{seq_name}_{track_id}"


def build_possible_frame_names(seq_name: str, frame_id: int):
    """
    Your files look like:
      S01_c001_frame_0000.jpg

    We do not know yet whether gt frame 1 maps to 0000 or 0001,
    so try both frame_id and frame_id - 1.
    """
    candidates = []

    # try direct mapping
    if frame_id >= 0:
        for ext in [".jpg", ".png", ".jpeg"]:
            candidates.append(f"S01_{seq_name}_frame_{frame_id:04d}{ext}")
            candidates.append(f"S02_{seq_name}_frame_{frame_id:04d}{ext}")
            candidates.append(f"S03_{seq_name}_frame_{frame_id:04d}{ext}")

    # try off-by-one mapping
    fid2 = frame_id - 1
    if fid2 >= 0:
        for ext in [".jpg", ".png", ".jpeg"]:
            candidates.append(f"S01_{seq_name}_frame_{fid2:04d}{ext}")
            candidates.append(f"S02_{seq_name}_frame_{fid2:04d}{ext}")
            candidates.append(f"S03_{seq_name}_frame_{fid2:04d}{ext}")

    return candidates


def find_frame_path(seq_name: str, frame_id: int):
    """
    Search both train/ and val/ centralized image folders.
    """
    candidates = build_possible_frame_names(seq_name, frame_id)

    for fname in candidates:
        p = IMAGES_TRAIN_DIR / fname
        if p.exists():
            return p

        p = IMAGES_VAL_DIR / fname
        if p.exists():
            return p

    return None


def find_all_gt_files(dataset_root: Path):
    """
    Recursively find all gt.txt files and infer sequence name from parent folders.
    Accepts layouts like:
      dataset/c001/gt/gt.txt
      dataset/train/c001/gt/gt.txt
      dataset/S01/c001/gt/gt.txt
    """
    gt_files = list(dataset_root.rglob("gt.txt"))
    results = []

    for gt_path in gt_files:
        # prefer the directory just above 'gt'
        # example: dataset/c001/gt/gt.txt -> seq_name = c001
        parts = gt_path.parts
        if len(parts) < 3:
            continue

        if gt_path.parent.name != "gt":
            continue

        seq_name = gt_path.parent.parent.name
        if not seq_name.startswith("c"):
            # skip weird matches
            continue

        results.append((seq_name, gt_path))

    results.sort(key=lambda x: x[0])
    return results


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    num_saved = 0
    num_skipped = 0
    num_missing_frames = 0

    gt_entries = find_all_gt_files(DATASET_ROOT)

    if not gt_entries:
        print("No gt.txt files found.")
        return

    print(f"Found {len(gt_entries)} sequences with gt.txt")

    image_cache = {}

    for seq_name, gt_path in gt_entries:
        seq_output_dir = OUTPUT_ROOT / seq_name
        seq_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {seq_name} from {gt_path}")

        with open(gt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parsed = parse_gt_line(line)
            if parsed is None:
                num_skipped += 1
                continue

            frame_id, track_id, x, y, w, h, conf = parsed

            if conf <= 0:
                num_skipped += 1
                continue

            if w < MIN_W or h < MIN_H:
                num_skipped += 1
                continue

            frame_path = find_frame_path(seq_name, frame_id)
            if frame_path is None:
                num_missing_frames += 1
                num_skipped += 1
                continue

            if frame_path not in image_cache:
                image_cache[frame_path] = Image.open(frame_path).convert("RGB")

            img = image_cache[frame_path]
            img_w, img_h = img.size

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, img_w, img_h)

            if x2 <= x1 or y2 <= y1:
                num_skipped += 1
                continue

            crop = img.crop((x1, y1, x2, y2))

            identity = build_identity(seq_name, track_id)

            crop_name = f"{seq_name}_f{frame_id:04d}_id{track_id:04d}.jpg"
            crop_path = seq_output_dir / crop_name
            crop.save(crop_path, quality=95)

            split = "train" if str(frame_path).startswith(str(IMAGES_TRAIN_DIR)) else "val"

            rows.append([
                str(crop_path),
                identity,
                seq_name,
                frame_id,
                track_id,
                split,
                str(frame_path)
            ])
            num_saved += 1

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "identity",
            "sequence",
            "frame",
            "track_id",
            "split",
            "source_frame_path"
        ])
        writer.writerows(rows)

    print("Done.")
    print(f"Saved crops:     {num_saved}")
    print(f"Skipped rows:    {num_skipped}")
    print(f"Missing frames:  {num_missing_frames}")
    print(f"CSV written to:  {CSV_OUT}")


if __name__ == "__main__":
    main()