def compute_iou(boxA, boxB):
    """
    Computes IoU between two boxes [x, y, w, h].
    """
    # Convert [x, y, w, h] -> [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute intersection area
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    
    # reject non-overlapping boxes
    if interWidth <=0 or interHeight <=0 :
        return -1.0
    interArea = interWidth * interHeight

    # Compute union area
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0: return 0
    return interArea / unionArea