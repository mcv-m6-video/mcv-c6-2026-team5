import cv2
import numpy as np

def apply_morphology(mask_np, kernel_opening_size=5, kernel_closing_size=20, operation="open_close"):
    """
    Applies morphological operations to clean the mask.
    
    Args:
        mask_np: Binary mask.
        kernel_size: Size of the structuring element (e.g., 3, 5, 7).
        operation: String describing the workflow:
                   - "open": Erosion -> Dilation
                   - "close": Dilation -> Erosion
                   - "open_close": Open then Close (Standard for noise removal + hole filling)
                   - "close_open": Close then Open
    """
    # 1. Create Kernel
    # MORPH_ELLIPSE is generally better for natural objects (cars) than RECT
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_opening_size, kernel_opening_size))
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_closing_size, kernel_closing_size))
    
    cleaned_mask = mask_np.copy()
    
    # 2. Apply Operations based on the param
    if operation == "open":
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, opening_kernel)
    elif operation == "close":
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, closing_kernel)
    elif operation == "open_close":
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, opening_kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, closing_kernel)
    elif operation == "close_open":
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, closing_kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, opening_kernel)
        
    return cleaned_mask

def get_bboxes_from_mask(mask_np, min_area=100):
    """
    Extracts bounding boxes from a binary mask using Connected Components.
    Does NOT perform morphology.
    """
    # Find Connected Components
    # connectivity=8 means diagonal pixels are connected
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
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
            
    return bboxes

def merge_bboxes_by_distance(bboxes, min_distance=30, frame_height=None):
    """
    Merges bounding boxes that are closer than `min_distance` pixels.
    
    Args:
        bboxes: List of [x, y, w, h]
        min_distance: Base distance threshold to merge.
        frame_height: If provided, scales distance based on Y position (Perspective).
        
    Returns:
        merged_bboxes: List of [x, y, w, h]
    """
    if len(bboxes) == 0:
        return []

    # Convert to [x1, y1, x2, y2] for easier math
    boxes = np.array(bboxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # We will keep merging until no more boxes can be merged
    while True:
        num_boxes = len(boxes)
        merged = False
        new_boxes = []
        used_indices = set()
        
        for i in range(num_boxes):
            if i in used_indices: continue
            
            # Start a new cluster with box i
            current_x1, current_y1, current_x2, current_y2 = x1[i], y1[i], x2[i], y2[i]
            used_indices.add(i)
            
            # Check against all other boxes
            for j in range(i + 1, num_boxes):
                if j in used_indices: continue
                
                # Determine dynamic threshold based on Y position (Perspective)
                # If frame_height is None, use fixed distance
                threshold = min_distance
                if frame_height:
                    # Scale: Bottom (y=H) -> 1.0 * dist, Top (y=0) -> 0.2 * dist
                    # Center of the box pair
                    cy = (current_y1 + y1[j]) / 2
                    scale = 0.2 + 0.8 * (cy / frame_height)
                    threshold = min_distance * scale

                # Calculate distance between box i and box j
                # Horizontal distance
                dist_x = max(0, x1[j] - current_x2, current_x1 - x2[j])
                # Vertical distance
                dist_y = max(0, y1[j] - current_y2, current_y1 - y2[j])
                
                # If close enough, MERGE THEM
                if dist_x < threshold and dist_y < threshold:
                    current_x1 = min(current_x1, x1[j])
                    current_y1 = min(current_y1, y1[j])
                    current_x2 = max(current_x2, x2[j])
                    current_y2 = max(current_y2, y2[j])
                    
                    used_indices.add(j)
                    merged = True # We found a merge, so we might need another pass
            
            new_boxes.append([current_x1, current_y1, current_x2, current_y2])
        
        # Update list for next iteration
        # Convert back to arrays for the loop
        new_boxes = np.array(new_boxes)
        if len(new_boxes) == 0: break 
        
        x1 = new_boxes[:, 0]
        y1 = new_boxes[:, 1]
        x2 = new_boxes[:, 2]
        y2 = new_boxes[:, 3]
        boxes = np.column_stack([x1, y1, x2-x1, y2-y1]) # Back to xywh just for length check
        
        if not merged:
            break # No merges happened in this pass, we are done
            
    # Convert final [x1, y1, x2, y2] back to [x, y, w, h]
    final_bboxes = []
    for i in range(len(x1)):
        final_bboxes.append([
            int(x1[i]), 
            int(y1[i]), 
            int(x2[i] - x1[i]), 
            int(y2[i] - y1[i])
        ])
        
    return final_bboxes