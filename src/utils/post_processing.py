import cv2
import numpy as np

def post_process_mask(mask_np, min_area=100):
    """
    Cleans up the binary mask and extracts bounding boxes.
    
    Args:
        mask_np (np.ndarray): Binary mask (0 or 255), shape (H, W).
        min_area (int): Minimum pixel area to consider a blob a valid object.
        
    Returns:
        cleaned_mask (np.ndarray): The mask after morphology.
        bboxes (list): List of bounding boxes [x, y, w, h].
    """
    
    # 1. Morphological Operations
    # Kernel size affects how much noise is removed. 3x3 to 5x5 is standard.
    # The slides mention "thin lines" to close objects.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Opening: Erosion -> Dilation (Removes small noise)
    cleaned_mask = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    
    # Closing: Dilation -> Erosion (Fills holes inside objects)
    # Only useful if objects are fragmented.
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    
    # 2. Find Connected Components
    # connectivity=8 means diagonal pixels are connected
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    bboxes = []
    
    # Loop through all detected components
    # Note: Label 0 is the Background, so we skip it (start range at 1)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter out small noise blobs
        if area >= min_area:
            bboxes.append([x, y, w, h])
            
    return cleaned_mask, bboxes