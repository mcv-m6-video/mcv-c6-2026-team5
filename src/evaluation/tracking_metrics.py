import numpy as np
# Adjust import based on your folder structure
try:
    from trackeval.metrics import HOTA, Identity
except ImportError:
    try:
        from src.evaluation.metrics import HOTA, Identity # adjustments for our path
    except:
        from metrics import HOTA, Identity

def _compute_iou_matrix(gt_boxes, pred_boxes):
    """Calculates IoU matrix (N_gt x N_pred) for a single frame."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    x1g, y1g, x2g, y2g = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    x1p, y1p, x2p, y2p = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

    xx1 = np.maximum(x1g[:, None], x1p)
    yy1 = np.maximum(y1g[:, None], y1p)
    xx2 = np.minimum(x2g[:, None], x2p)
    yy2 = np.minimum(y2g[:, None], y2p)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h
    
    area_gt = (x2g - x1g) * (y2g - y1g)
    area_pred = (x2p - x1p) * (y2p - y1p)
    union = area_gt[:, None] + area_pred - intersection
    
    return intersection / (union + 1e-10)

def calculate_tracking_metrics(gt_list, pred_list):
    """
    Computes HOTA and IDF1 with ID remapping.
    """
    # 1. REMAP IDs TO 0..N-1
    # Collect all unique IDs first
    all_gt_ids = set()
    all_pred_ids = set()
    
    for frame in gt_list:
        for obj in frame:
            all_gt_ids.add(obj['id'])
            
    for frame in pred_list:
        for obj in frame:
            all_pred_ids.add(obj['id'])
            
    # Create mapping dictionaries {Real_ID: Mapped_Index}
    # Sorting ensures determinism
    gt_id_map = {old_id: new_idx for new_idx, old_id in enumerate(sorted(list(all_gt_ids)))}
    pred_id_map = {old_id: new_idx for new_idx, old_id in enumerate(sorted(list(all_pred_ids)))}

    # 2. Initialize Metric Classes
    hota_metric = HOTA()
    idf1_metric = Identity()
    
    # 3. Prepare Data
    data = {
        'gt_ids': [],         
        'tracker_ids': [],    
        'similarity_scores': [], 
        'num_gt_dets': 0,
        'num_tracker_dets': 0,
        'num_gt_ids': max(1, len(all_gt_ids)),      # Use mapped counts
        'num_tracker_ids': max(1, len(all_pred_ids))
    }
    
    # 4. Process Frames with Mapped IDs
    for t in range(len(gt_list)):
        gt_objs = gt_list[t]
        pred_objs = pred_list[t]
        
        # Extract and Map GT
        if len(gt_objs) > 0:
            gt_boxes = np.array([o['bbox'] for o in gt_objs])
            # APPLY MAP HERE
            gt_ids = np.array([gt_id_map[o['id']] for o in gt_objs], dtype=int)
        else:
            gt_boxes = np.empty((0, 4))
            gt_ids = np.empty((0), dtype=int)
            
        # Extract and Map Preds
        if len(pred_objs) > 0:
            pred_boxes = np.array([o['bbox'] for o in pred_objs])
            # APPLY MAP HERE
            pred_ids = np.array([pred_id_map[o['id']] for o in pred_objs], dtype=int)
        else:
            pred_boxes = np.empty((0, 4))
            pred_ids = np.empty((0), dtype=int)

        data['gt_ids'].append(gt_ids)
        data['tracker_ids'].append(pred_ids)
        data['num_gt_dets'] += len(gt_ids)
        data['num_tracker_dets'] += len(pred_ids)
        
        data['similarity_scores'].append(_compute_iou_matrix(gt_boxes, pred_boxes))

    # 5. Run Evaluation
    hota_res = hota_metric.eval_sequence(data)
    idf1_res = idf1_metric.eval_sequence(data)
    
    # Return dictionary
    return {
        'HOTA': np.mean(hota_res['HOTA']),
        'IDF1': idf1_res['IDF1']
    }