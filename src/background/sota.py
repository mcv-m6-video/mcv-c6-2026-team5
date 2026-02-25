"""
Task 3 (State-of-the-art BG subtraction) — NO TRACKING, SINGLE CLASS.
Methods:
  - MOG  (KaewTraKulPong et al.)  -> cv2.bgsegm.createBackgroundSubtractorMOG
  - MOG2 (Zivkovic et al.)       -> cv2.createBackgroundSubtractorMOG2
  - LSBP (Guo et al.)            -> cv2.bgsegm.createBackgroundSubtractorLSBP

Pipeline per frame:
  frame -> subtractor -> FG mask -> postprocess -> connected components -> bboxes

Notes aligned with instructions:
  - Do not consider IDs (no tracking)
  - Use single class (all detections are "car"/"vehicle"/"foreground object")
  - Do not consider parked cars: evaluation GT includes moving objects only; we just detect FG blobs.

"""

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Tuple, Optional

import json
import argparse
from types import SimpleNamespace
from itertools import product

def load_config(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Convert dict → object with dot access
    return SimpleNamespace(**config_dict)

def as_list(x):
    return x if isinstance(x, list) else [x]

# -----------------------------
# Background subtractor factory
# -----------------------------
def make_subtractor(method: str, history: int, var_threshold: float, detect_shadows: bool):
    method = method.lower()
    if method == "mog":
        if not hasattr(cv2, "bgsegm") or not hasattr(cv2.bgsegm, "createBackgroundSubtractorMOG"):
            raise RuntimeError("MOG requires opencv-contrib-python (cv2.bgsegm.createBackgroundSubtractorMOG).")
        try:
            return cv2.bgsegm.createBackgroundSubtractorMOG(history=history)
        except TypeError:
            return cv2.bgsegm.createBackgroundSubtractorMOG()

    if method == "mog2":
        return cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

    if method == "lsbp":
        if not hasattr(cv2, "bgsegm") or not hasattr(cv2.bgsegm, "createBackgroundSubtractorLSBP"):
            raise RuntimeError("LSBP requires opencv-contrib-python (cv2.bgsegm.createBackgroundSubtractorLSBP).")
        return cv2.bgsegm.createBackgroundSubtractorLSBP()

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# FG post-processing + bboxes
# -----------------------------
def fg_to_binary(
    fg: np.ndarray,
    method: str,
    remove_shadows: bool,
    shadow_value: int = 127,
    thr: int = 200,
) -> np.ndarray:
    """Convert FG mask to binary (0/255)."""
    if fg.ndim == 3:
        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

    if remove_shadows and method.lower() == "mog2":
        fg = np.where(fg == shadow_value, 0, fg).astype(np.uint8)

    _, bin_mask = cv2.threshold(fg, thr, 255, cv2.THRESH_BINARY)
    return bin_mask


def postprocess_mask(
    bin_mask: np.ndarray,
    kernel_open: int,
    kernel_close: int,
    min_area: int
) -> np.ndarray:
    """Morph + remove small CC."""
    mask = bin_mask

    if kernel_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if kernel_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Filter small components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def mask_to_bboxes(
    mask: np.ndarray,
    min_w: int,
    min_h: int,
    max_w: Optional[int],
    max_h: Optional[int],
    aspect_min: float,
    aspect_max: float
) -> List[Tuple[int, int, int, int, float]]:
    """
    Return bboxes as (x, y, w, h, conf).
    conf here is a simple proxy: fraction of FG pixels inside bbox (0..1).
    (You can also output 1.0 for all boxes if you prefer.)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    H, W = mask.shape[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h:
            continue
        if max_w is not None and w > max_w:
            continue
        if max_h is not None and h > max_h:
            continue

        asp = w / float(h)
        if asp < aspect_min or asp > aspect_max:
            continue

        x2 = min(W, x + w)
        y2 = min(H, y + h)
        roi = mask[y:y2, x:x2]
        fg_frac = float(np.count_nonzero(roi)) / float(roi.size) if roi.size else 0.0

        dets.append((x, y, w, h, fg_frac))

    # optional: sort by "confidence" descending
    dets.sort(key=lambda t: t[4], reverse=True)
    return dets


# -----------------------------
# Run one method
# -----------------------------
def run(
    video_path: str,
    method: str,
    out_txt: str,
    out_video: Optional[str],
    show: bool,
    roi_path: Optional[str],
    args
):
    # Open video
    cap = cv2.VideoCapture(int(video_path) if video_path.isdigit() else video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    roi_mask = None
    if roi_path is not None:
        roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            raise RuntimeError(f"Could not read ROI mask: {roi_path}")

        # Ensure same size as frame
        if roi_mask.shape[:2] != (H, W):
            roi_mask = cv2.resize(roi_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # Binarize to {0,255}
        roi_mask = (roi_mask > 0).astype(np.uint8) * 255

    elif roi_path is None:
        print(f"[{method}] No ROI mask provided.")

    subtractor = make_subtractor(method, args.history, args.var_threshold, args.detect_shadows)

    writer = None
    if out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))

    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    f = open(out_txt, "w")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fg = subtractor.apply(frame, learningRate=args.learning_rate)


        bin_mask = fg_to_binary(
            fg, method=method,
            remove_shadows=args.remove_shadows,
            thr=args.fg_threshold
        )


        if roi_mask is not None:
            bin_mask = cv2.bitwise_and(bin_mask, roi_mask)
        if frame_idx == 0 and roi_mask is not None:
            print(f"[{method}] ROI coverage: {np.count_nonzero(roi_mask) / roi_mask.size:.3f}", flush=True)
            
        clean = postprocess_mask(
            bin_mask,
            kernel_open=args.kernel_open,
            kernel_close=args.kernel_close,
            min_area=args.min_area
        )

        dets = mask_to_bboxes(
            clean,
            min_w=args.min_w, min_h=args.min_h,
            max_w=args.max_w, max_h=args.max_h,
            aspect_min=args.aspect_min,
            aspect_max=args.aspect_max
        )

        # Write detections (single class). NO IDs.
        # Format: frame x y w h conf
        for (x, y, w, h, conf) in dets:
            f.write(f"{frame_idx} {x} {y} {w} {h} {conf:.4f}\n")

        # Visualize (optional)
        if writer or show:
            vis = frame.copy()

            if args.draw_mask:
                mask_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
                small = cv2.resize(mask_bgr, (W // 4, H // 4))
                vis[0:small.shape[0], 0:small.shape[1]] = small

            for (x, y, w, h, conf) in dets:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis, f"{conf:.2f}", (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(vis, f"Method: {method.upper()}  Frame: {frame_idx}",
                        (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            if writer:
                writer.write(vis)
            if show:
                cv2.imshow(f"BG Subtraction - {method}", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

        frame_idx += 1

    f.close()
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def main():
    print("Starting BG subtraction SOTA methods...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json")
    print("Parsing arguments...")
    args_cli = parser.parse_args()

    args = load_config(args_cli.config)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.method == "all":
        methods = ["mog", "mog2", "lsbp"]
    elif args.method == "mogs":
        methods = ["mog", "mog2"]
    else:
        methods = [args.method]

    open_list = as_list(getattr(args, "kernel_open", 0))
    close_list = as_list(getattr(args, "kernel_close", 0))

    for m in methods:
        for ko, kc in product(open_list, close_list):
            print(f"Running method: {m} | open={ko} close={kc}")

            # temporarily override for this run
            args.kernel_open = int(ko)
            args.kernel_close = int(kc)

            tag = f"open{ko}_close{kc}"
            out_txt = os.path.join(args.out_dir, f"dets_{m}_{tag}.txt")
            out_vid = os.path.join(args.out_dir, f"vis_{m}_{tag}.mp4") if args.save_video else None

            try:
                run(args.video, m, out_txt, out_vid, args.show, args.roi_mask, args)
                print(f"[OK] {m} ({tag}): wrote {out_txt}" + (f" and {out_vid}" if out_vid else ""))
            except Exception as e:
                print(f"[WARN] {m} ({tag}) failed: {e}")

if __name__ == "__main__":
    main()