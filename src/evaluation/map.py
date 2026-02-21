import numpy as np
import random
from .iou import compute_iou

def compute_ap_pascal_voc_11_point(recall, precision):
    """
    Computes AP using the 11-point interpolation method (Pascal VOC 2010).
    AP = mean precision at recall levels [0, 0.1, ..., 1.0]
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            # "The precision at each recall level r is interpolated by taking the 
            # maximum precision measured for a method for which the corresponding 
            # recall exceeds r" 
            p = np.max(precision[recall >= t])
        ap += p
    return ap / 11.0

def evaluate_detections(gt_boxes_by_frame, pred_boxes_by_frame, iou_thresh=0.5):
    """
    Calculates Precision and Recall for a SINGLE ranked list of detections.
    """
    # Flatten all predictions into a single list: [frame_id, x, y, w, h, score]
    all_preds = []
    for frame_id, boxes in pred_boxes_by_frame.items():
        for box in boxes:
            all_preds.append({'frame': frame_id, 'bbox': box['bbox'], 'score': box['score']})

    # Sort predictions by confidence score (High -> Low)
    all_preds.sort(key=lambda x: x['score'], reverse=True)

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    
    # Track which GT boxes have already been detected (to avoid double counting)
    # detected_gt[frame_id][gt_idx] = True/False
    detected_gt = {fid: [False]*len(boxes) for fid, boxes in gt_boxes_by_frame.items()}
    total_gt = sum(len(boxes) for boxes in gt_boxes_by_frame.values())

    if total_gt == 0: return 0.0

    # Match Predictions to GT
    for i, pred in enumerate(all_preds):
        frame_id = pred['frame']
        pred_box = pred['bbox']
        
        gt_list = gt_boxes_by_frame.get(frame_id, [])
        best_iou = -1
        best_gt_idx = -1

        # Find best matching GT box
        for idx, gt_box in enumerate(gt_list):
            if detected_gt[frame_id][idx]: continue # Already matched
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        # Determine TP or FP 
        if best_iou >= iou_thresh:
            tp[i] = 1
            detected_gt[frame_id][best_gt_idx] = True
        else:
            fp[i] = 1

    # Compute Cumulative Precision and Recall
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-6) # Add epsilon

    return compute_ap_pascal_voc_11_point(recall, precision)

def compute_map_randomized(gt_boxes, pred_boxes, n_runs=10, iou_thresh=0.5):
    """
    Evaluates mAP for a detector WITHOUT confidence scores (Task 1.2).
    Generates N random rankings and averages the AP.
    """
    ap_sum = 0.0
    
    for run in range(n_runs):
        # Assign random scores to predictions
        preds_with_scores = {}
        for fid, boxes in pred_boxes.items():
            preds_with_scores[fid] = []
            for box in boxes:
                # Assign random float between 0 and 1
                preds_with_scores[fid].append({'bbox': box, 'score': random.random()})
        
        ap = evaluate_detections(gt_boxes, preds_with_scores, iou_thresh)
        ap_sum += ap
        print(f"Run {run+1}/{n_runs}: AP = {ap:.4f}")
        
    return ap_sum / n_runs