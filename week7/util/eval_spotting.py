"""
File containing main evaluation functions.
"""

# Standard imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from SoccerNet.Evaluation.ActionSpotting import average_mAP
import json
import os

# Local imports
from dataset.frame import FPS_SN

INFERENCE_BATCH_SIZE = 4


def evaluate(model, dataset,
             batch_size=INFERENCE_BATCH_SIZE,
             nms_window=5,
             nms_type='hard',
             nms_thresh=0.05,
             smoothing=None,
             smoothing_window=3,
             soft_nms_sigma=1.0):

    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(dataset._class_dict)), np.float32),
            np.zeros(video_len, np.int32))

    for clip in tqdm(DataLoader(
            dataset, num_workers=batch_size * 2, pin_memory=True,
            batch_size=batch_size
    )):
        batch_pred_scores = model.predict(clip['frame'])

        for i in range(clip['frame'].shape[0]):
            video = clip['video'][i]
            scores, support = pred_dict[video]
            pred_scores = batch_pred_scores[i]
            start = clip['start'][i].item()
            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0
            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores[:, 1:]
            support[start:end] += (pred_scores.sum(axis=1) != 0) * 1

    detections_numpy = []
    for video, video_len, _ in dataset.videos:
        scores, support = pred_dict[video]
        support[support == 0] = 1
        scores = scores / support[:, np.newaxis]

        # Smoothing opcional (mean) antes del NMS
        if smoothing == 'mean' and smoothing_window > 1:
            scores = _apply_smoothing(scores, smoothing_window)

        if nms_type == 'soft':
            pred = apply_soft_NMS(scores, nms_window, soft_nms_sigma, nms_thresh)
        else:  # hard (default)
            pred = apply_NMS(scores, nms_window, nms_thresh)

        detections_numpy.append(pred)

    targets_numpy = []
    closests_numpy = []
    for video, video_len, _ in dataset.videos:
        targets = np.zeros(
            (video_len, len(dataset._class_dict)), np.float32)
        labels = json.load(open(
            os.path.join(dataset._labels_dir, video, 'Labels-ball.json')))

        for annotation in labels["annotations"]:
            event = dataset._class_dict[annotation["label"]]
            frame = int(
                FPS_SN / dataset._stride * (int(annotation["position"]) / 1000))
            frame = min(frame, video_len - 1)
            targets[frame, event - 1] = 1

        targets_numpy.append(targets)

        closest_numpy = np.zeros(targets.shape) - 1
        for c in np.arange(targets.shape[-1]):
            indexes = np.where(targets[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0],
                           (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = targets[indexes[i], c]
        closests_numpy.append(closest_numpy)

    fps = FPS_SN / dataset._stride

    # ── Tolerancia 1s ────────────────────────────────────────────────
    mAP_1s, AP_1s, _, _, _, _ = average_mAP(
        targets_numpy, detections_numpy, closests_numpy,
        fps, deltas=np.array([1.0])
    )

    # ── Tolerancia 0.5s ──────────────────────────────────────────────
    mAP_05s, AP_05s, _, _, _, _ = average_mAP(
        targets_numpy, detections_numpy, closests_numpy,
        fps, deltas=np.array([0.5])
    )

    return mAP_1s, mAP_05s, AP_1s, AP_05s


def _apply_smoothing(scores, window):
    """Mean smoothing over temporal dimension."""
    smoothed = np.copy(scores)
    half = window // 2
    for t in range(scores.shape[0]):
        t_start = max(0, t - half)
        t_end = min(scores.shape[0], t + half + 1)
        smoothed[t] = scores[t_start:t_end].mean(axis=0)
    return smoothed


def apply_NMS(predictions, window, thresh=0.0):
    nf, nc = predictions.shape
    for i in range(nc):
        aux = predictions[:, i].copy()
        aux2 = np.zeros(nf) - 1
        while np.max(aux) >= thresh:
            max_value = np.max(aux)
            max_index = np.argmax(aux)
            nms_from = int(np.maximum(-(window / 2) + max_index, 0))
            nms_to = int(np.minimum(max_index + int(window / 2), len(aux)))
            aux[nms_from:nms_to] = -1
            aux2[max_index] = max_value
        predictions[:, i] = aux2
    return predictions


def apply_soft_NMS(predictions, window, sigma=1.0, thresh=0.0):
    """Soft-NMS: decay surrounding scores with gaussian instead of zeroing."""
    nf, nc = predictions.shape
    result = np.zeros_like(predictions) - 1
    for i in range(nc):
        aux = predictions[:, i].copy()
        while np.max(aux) >= thresh:
            max_value = np.max(aux)
            max_index = np.argmax(aux)
            result[max_index, i] = max_value
            # Gaussian decay around peak
            for t in range(nf):
                dist = abs(t - max_index)
                if dist <= window:
                    weight = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                    aux[t] *= (1 - weight)
            aux[max_index] = -1
    return result