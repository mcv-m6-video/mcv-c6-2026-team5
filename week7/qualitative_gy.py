import pickle
import os
import argparse
import numpy as np
import cv2
import torchvision
import imageio.v2 as imageio


def find_clips_for_class(labels_store, target_class, max_clips=10):
    """
    Return up to max_clips indices containing the target class.
    """
    indices = []
    for i, clip_labels in enumerate(labels_store):
        if any(l["label"] == target_class for l in clip_labels):
            indices.append(i)
        if len(indices) >= max_clips:
            break
    return indices


def build_frame_paths(paths_metadata, stride):
    """
    Reconstruct actual frame paths from stored metadata.
    """
    base_path, start, pad_start, pad_end, ndigits, length = paths_metadata

    actual_frame_paths = [None] * pad_start
    for j in range(length - pad_start - pad_end):
        frame_idx = start + j * stride
        if ndigits == -1:
            fname = f"frame{frame_idx}.jpg"
        else:
            fname = f"{str(frame_idx).zfill(ndigits)}.jpg"
        actual_frame_paths.append(os.path.join(base_path, fname))
    actual_frame_paths += [None] * pad_end

    return actual_frame_paths


def get_valid_frame_shape(actual_frame_paths):
    """
    Find shape of first valid frame.
    """
    for p in actual_frame_paths:
        if p is not None and os.path.exists(p):
            img = torchvision.io.read_image(p)
            return img.shape
    return None


def generate_gt_gif(
    idx,
    class_name,
    target_class,
    frame_paths,
    labels_store,
    output_dir,
    clip_len=50,
    stride=2,
    fps=25.0,
):
    """
    Generate one GT-only qualitative GIF.
    """
    paths_metadata = frame_paths[idx]
    clip_labels = labels_store[idx]

    if not paths_metadata or paths_metadata[1] == -1:
        print(f"Skipping clip {idx}: invalid metadata")
        return

    actual_frame_paths = build_frame_paths(paths_metadata, stride)
    valid_shape = get_valid_frame_shape(actual_frame_paths)
    if valid_shape is None:
        print(f"Skipping clip {idx}: no valid frames found")
        return

    print(f"[GENERATING] Clip {idx} for class {class_name}")

    OUT_WIDTH = 800
    HUD_HEIGHT = 120

    # Read first valid frame
    first_valid_path = next((p for p in actual_frame_paths if p is not None and os.path.exists(p)), None)
    if first_valid_path is None:
        print(f"Skipping clip {idx}: no readable frame")
        return

    first_frame = cv2.imread(first_valid_path)
    orig_h, orig_w, _ = first_frame.shape

    scaled_video_h = int(orig_h * (OUT_WIDTH / orig_w))
    total_canvas_h = HUD_HEIGHT + scaled_video_h

    playback_fps = fps / stride
    frame_duration = 1.0 / playback_fps  # seconds per GIF frame

    out_path = os.path.join(output_dir, f"clip_{idx}_{class_name.replace(' ', '_')}_GT.gif")

    target_gt_frames = [lbl["label_idx"] for lbl in clip_labels if lbl["label"] == target_class]

    gif_frames = []

    for frame_idx, p in enumerate(actual_frame_paths):
        canvas = np.zeros((total_canvas_h, OUT_WIDTH, 3), dtype=np.uint8)

        # A. Video frame
        frame = cv2.imread(p) if p is not None and os.path.exists(p) else np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        frame_resized = cv2.resize(frame, (OUT_WIDTH, scaled_video_h))
        canvas[HUD_HEIGHT:, :] = frame_resized

        # B. HUD
        cv2.putText(
            canvas,
            f"Frame {frame_idx}/{clip_len}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"Target: {class_name}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"GT frames: {target_gt_frames}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # Highlight GT near the event
        for gt_fr in target_gt_frames:
            if abs(gt_fr - frame_idx) <= 2:
                cv2.putText(
                    canvas,
                    f"--> {class_name} <--",
                    (OUT_WIDTH // 2 - 140, HUD_HEIGHT + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

        # Convert BGR (OpenCV) -> RGB (GIF/imageio)
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        gif_frames.append(canvas_rgb)

    imageio.mimsave(out_path, gif_frames, duration=frame_duration, loop=0)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate GT qualitative GIFs.")
    parser.add_argument(
        "--store_dir",
        type=str,
        default="/ghome/group05/datasets/WEEK7/SN-BAS-2025_savedata/splits",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--clip_len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--num_clips_per_class", type=int, default=10)
    args = parser.parse_args()

    # Label mapping based on your order
    class_to_label = {
        "Pass": 1,
        "Drive": 2,
        "Header": 3,
        "High Pass": 4,
        "Out": 5,
        "Cross": 6,
        "Throw In": 7,
        "Shot": 8,
        "Ball Player Block": 9,
        "Player Succesful Tackle": 10,
        "Free Kick": 11,
        "Goal": 12,
    }

    target_classes = ["Goal", "Pass", "Free Kick"]

    store_path = os.path.join(args.store_dir, f"LEN{args.clip_len}SPLIT{args.split}")

    with open(os.path.join(store_path, "frame_paths.pkl"), "rb") as f:
        frame_paths = pickle.load(f)

    with open(os.path.join(store_path, "labels.pkl"), "rb") as f:
        labels_store = pickle.load(f)

    output_dir = os.path.join("qualitative_results", f"GT_ONLY_GIFS_{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    for class_name in target_classes:
        target_class = class_to_label[class_name]
        indices = find_clips_for_class(labels_store, target_class, max_clips=args.num_clips_per_class)

        if len(indices) == 0:
            print(f"No clips found for class: {class_name}")
            continue

        print(f"\nFound {len(indices)} clips for class {class_name}")

        for idx in indices:
            generate_gt_gif(
                idx=idx,
                class_name=class_name,
                target_class=target_class,
                frame_paths=frame_paths,
                labels_store=labels_store,
                output_dir=output_dir,
                clip_len=args.clip_len,
                stride=args.stride,
                fps=args.fps,
            )

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()