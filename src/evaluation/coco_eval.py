import os
import json
import tempfile
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def convert_to_coco_gt(gt_boxes, all_frame_ids, height, width):
    """
    Converts ground truth dictionary to COCO format.
    Ensures ALL frames are registered as images, even if they have no annotations.
    """
    images = []
    annotations = []
    ann_id = 1
    
    # Sort frame IDs to keep things organized
    sorted_ids = sorted(list(all_frame_ids))
    
    for frame_id in sorted_ids:
        # 1. Add Image Info (Required for ALL frames)
        images.append({
            "id": int(frame_id),
            "width": int(width),
            "height": int(height),
            "file_name": f"frame_{frame_id:04d}.jpg"
        })
        
        # 2. Add Annotations (Only if this frame has GT boxes)
        if frame_id in gt_boxes:
            boxes = gt_boxes[frame_id]
            for box in boxes:
                x, y, w, h = box
                annotations.append({
                    "id": ann_id,
                    "image_id": int(frame_id),
                    "category_id": 1,  # 'car'
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
            
    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car"}]
    }
    return dataset

def convert_to_coco_dt(pred_boxes):
    """
    Converts prediction dictionary to COCO results format.
    """
    results = []
    
    for frame_id, boxes in pred_boxes.items():
        for box in boxes:
            x, y, w, h = box
            results.append({
                "image_id": int(frame_id),
                "category_id": 1,
                "bbox": [x, y, w, h],
                "score": 1.0  # Default score for binary detectors
            })
            
    return results

def evaluate_coco(gt_boxes, pred_boxes, height, width):
    """
    Runs COCO evaluation.
    Returns the mAP@0.5 stats.
    """
    # 1. Identify ALL frames relevant to this evaluation
    # This union is crucial: it captures frames with GT cars AND frames where we predicted cars
    all_frame_ids = set(gt_boxes.keys()) | set(pred_boxes.keys())
    
    # 2. Convert GT to COCO JSON format (Passing the full set of frames)
    coco_gt_dict = convert_to_coco_gt(gt_boxes, all_frame_ids, height, width)
    
    # 3. Save GT to a temporary JSON file (Required by COCO API)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(coco_gt_dict, tmp)
        gt_json_path = tmp.name

    try:
        # 4. Initialize COCO Ground Truth
        coco_gt = COCO(gt_json_path)
        
        # 5. Convert Predictions and Load Results
        coco_dt_list = convert_to_coco_dt(pred_boxes)
        
        # Handle case where no predictions exist
        if len(coco_dt_list) == 0:
            print("No predictions found!")
            return 0.0

        coco_dt = coco_gt.loadRes(coco_dt_list)
        
        # 6. Run Evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = sorted(list(all_frame_ids)) # Evaluate on all frames
        
        # Suppress prints if you want silent running (optional)
        # import sys, io
        # original_stdout = sys.stdout
        # sys.stdout = io.StringIO()
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # sys.stdout = original_stdout # Restore print
        
        # Extract mAP@0.5 (Index 1 in stats)
        # stats[0] = AP @ 0.50:0.95
        # stats[1] = AP @ 0.50
        map50 = coco_eval.stats[1]
        
        return map50

    finally:
        # Cleanup temp file
        if os.path.exists(gt_json_path):
            os.remove(gt_json_path)