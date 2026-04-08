import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score


def evaluate_multiclip(model, dataset, offsets, aggregation='mean'):
    all_scores = []
    all_labels = []

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        label = sample['label']

        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        clips = []
        for off in offsets:
            clip = dataset.get_clip_with_offset(idx, off)
            clips.append(clip)

        clips = torch.stack(clips, dim=0)   # [K, L, C, H, W]

        logits = model.predict_logits(clips)   # [K, C]

        if aggregation == 'mean':
            agg_logits = logits.mean(dim=0)

        elif aggregation == 'max':
            agg_logits = logits.max(dim=0).values

        elif aggregation == 'weighted_mean':
            k = logits.shape[0]
            if k == 3:
                weights = torch.tensor([0.25, 0.50, 0.25], dtype=logits.dtype)
            elif k == 5:
                weights = torch.tensor([0.10, 0.20, 0.40, 0.20, 0.10], dtype=logits.dtype)
            else:
                weights = torch.ones(k, dtype=logits.dtype) / k
            weights = weights / weights.sum()
            agg_logits = (logits * weights[:, None]).sum(dim=0)

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        probs = torch.sigmoid(agg_logits).cpu().numpy()

        all_scores.append(probs)
        all_labels.append(label)

    all_scores = np.asarray(all_scores)
    all_labels = np.asarray(all_labels)

    ap = []
    for c in range(all_labels.shape[1]):
        y_true = all_labels[:, c]
        y_score = all_scores[:, c]

        if np.sum(y_true) == 0:
            ap.append(0.0)
        else:
            ap.append(average_precision_score(y_true, y_score))

    return np.asarray(ap, dtype=np.float32)