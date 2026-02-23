import cv2
import numpy as np
from src.evaluation.iou import compute_iou

def visualize_comparison(frame, gt_boxes, pred_boxes):
    """
    Draws GT (Green) and Pred (Red) boxes. 
    Displays IoU for overlapping boxes.
    """
    img = frame.copy()
    if img.shape[0] == 3: img = img.transpose(1, 2, 0) # Handle Torch (C,H,W)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.shape[2] == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Handle Grayscale

    # Draw Ground Truth (Green)
    # Format assumed: [x, y, w, h]
    for box in gt_boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "GT", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw Predictions (Red) & Calculate IoU
    for pred in pred_boxes:
        px, py, pw, ph = map(int, pred)
        best_iou = 0
        
        # Check IoU against all GTs to find the best match
        for gt in gt_boxes:
            iou = compute_iou(pred, gt)
            if iou > best_iou: best_iou = iou
        
        # Color logic: Red if bad IoU, Yellow if borderline, Blue if good
        color = (0, 0, 255) # Red (Fail)
        if best_iou >= 0.5: color = (255, 0, 0) # Blue (Pass)
        elif best_iou > 0.3: color = (0, 255, 255) # Yellow (Close)

        cv2.rectangle(img, (px, py), (px + pw, py + ph), color, 2)
        cv2.putText(img, f"IoU: {best_iou:.2f}", (px, py + ph + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, f"Area: {pw*ph}", (px, py + ph + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    return img