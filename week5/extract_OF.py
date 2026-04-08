#!/usr/bin/env python3

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

cv2.setNumThreads(4)

def natural_frame_key(name: str):
    stem = Path(name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def list_frame_files(video_dir: Path):
    files = [
        f for f in video_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    files.sort(key=lambda p: natural_frame_key(p.name))
    return files


def find_frame_dirs(root: Path):
    frame_dirs = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        try:
            has_frames = any(
                f.is_file()
                and f.name.startswith("frame")
                and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                for f in p.iterdir()
            )
        except Exception:
            has_frames = False

        if has_frames:
            frame_dirs.append(p)

    frame_dirs.sort()
    return frame_dirs


def read_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def compute_farneback(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


def quantize_flow(flow, clip_value=20.0):
    flow = np.clip(flow, -clip_value, clip_value)
    flow = (flow + clip_value) / (2.0 * clip_value)
    flow = np.clip(flow * 255.0, 0, 255).astype(np.uint8)
    return flow


def save_chunk(video_out_dir: Path, chunk_id: int, flow_list, frame_idx_list):
    if len(flow_list) == 0:
        return

    flow_arr = np.stack(flow_list, axis=0)     # [N, H, W, 2], uint8
    frame_idx_arr = np.asarray(frame_idx_list, dtype=np.int32)

    out_path = video_out_dir / f"chunk_{chunk_id:05d}.npz"
    np.savez_compressed(out_path, flow=flow_arr, frame_idx=frame_idx_arr)


def process_video(
    video_dir: Path,
    frames_root: Path,
    out_root: Path,
    flow_stride: int = 5,
    rgb_sample_stride: int = 2,
    chunk_size: int = 256,
    clip_value: float = 20.0,
    overwrite: bool = False,
):
    frame_files = list_frame_files(video_dir)
    if len(frame_files) == 0:
        return

    relative_video_dir = video_dir.relative_to(frames_root)
    video_out_dir = out_root / relative_video_dir
    video_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = video_out_dir / "metadata.json"
    if metadata_path.exists() and not overwrite:
        print(f"[SKIP] {relative_video_dir} already processed.")
        return

    first_gray = read_gray(frame_files[0])
    h, w = first_gray.shape[:2]
    num_frames = len(frame_files)

    metadata = {
        "relative_video_dir": str(relative_video_dir),
        "num_rgb_frames": num_frames,
        "height": h,
        "width": w,
        "flow_stride": flow_stride,
        "rgb_sample_stride": rgb_sample_stride,
        "chunk_size": chunk_size,
        "clip_value": clip_value,
        "storage_dtype": "uint8",
        "storage_format": "npz_compressed_chunks",
        "flow_definition": f"flow aligned to frame t, computed from frame t-{flow_stride} to frame t",
        "stored_frame_rule": f"only frames with index t % {rgb_sample_stride} == 0 are stored",
    }

    gray_cache = {}

    def get_gray(i):
        if i not in gray_cache:
            gray_cache[i] = read_gray(frame_files[i])
        return gray_cache[i]

    flow_chunk = []
    idx_chunk = []
    chunk_id = 0
    stored_count = 0

    
    for t in tqdm(range(num_frames), desc=str(relative_video_dir), leave=False):
        if t % rgb_sample_stride != 0:
            continue

        if t - flow_stride < 0:
            flow = np.zeros((h, w, 2), dtype=np.float32)
        else:
            prev_gray = get_gray(t - flow_stride)
            curr_gray = get_gray(t)
            flow = compute_farneback(prev_gray, curr_gray)

        flow_q = quantize_flow(flow, clip_value=clip_value)

        flow_chunk.append(flow_q)
        idx_chunk.append(natural_frame_key(frame_files[t].name))
        stored_count += 1

        if len(flow_chunk) >= chunk_size:
            save_chunk(video_out_dir, chunk_id, flow_chunk, idx_chunk)
            chunk_id += 1
            flow_chunk = []
            idx_chunk = []

        min_needed = max(0, t - flow_stride - 2)
        keys_to_delete = [k for k in gray_cache if k < min_needed]
        for k in keys_to_delete:
            del gray_cache[k]

    if len(flow_chunk) > 0:
        save_chunk(video_out_dir, chunk_id, flow_chunk, idx_chunk)
        chunk_id += 1

    metadata["num_stored_flow_frames"] = stored_count
    metadata["num_chunks"] = chunk_id

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--flow_stride", type=int, default=5)
    parser.add_argument("--rgb_sample_stride", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--clip_value", type=float, default=20.0)
    parser.add_argument("--video_filter", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)

    if not frames_root.exists():
        raise FileNotFoundError(f"frames_root does not exist: {frames_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    video_dirs = find_frame_dirs(frames_root)

    if args.video_filter is not None:
        video_dirs = [p for p in video_dirs if args.video_filter in str(p)]

    print(f"Found {len(video_dirs)} frame directories")
    print(f"frames_root       : {frames_root}")
    print(f"out_root          : {out_root}")
    print(f"flow_stride       : {args.flow_stride}")
    print(f"rgb_sample_stride : {args.rgb_sample_stride}")
    print(f"chunk_size        : {args.chunk_size}")
    print(f"clip_value        : {args.clip_value}")

    for video_dir in tqdm(video_dirs, desc="Videos"):
        try:
            process_video(
                video_dir=video_dir,
                frames_root=frames_root,
                out_root=out_root,
                flow_stride=args.flow_stride,
                rgb_sample_stride=args.rgb_sample_stride,
                chunk_size=args.chunk_size,
                clip_value=args.clip_value,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {video_dir}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()