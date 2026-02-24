import cv2
import numpy as np
import torch
import torch.nn.functional as F

# def apply_morphology(mask_np, kernel_opening_size=5, kernel_closing_size=10, operation="open_close", morph_shape="ellipse"):
#     """
#     Applies morphological operations to clean the mask.
    
#     Args:
#         mask_np: Binary mask.
#         kernel_size: Size of the structuring element (e.g., 3, 5, 7).
#         operation: String describing the workflow:
#                    - "open": Erosion -> Dilation
#                    - "close": Dilation -> Erosion
#                    - "open_close": Open then Close (Standard for noise removal + hole filling)
#                    - "close_open": Close then Open
#     """
#     if morph_shape == "rect":
#         shape = cv2.MORPH_RECT
#     elif morph_shape == "cross":
#         shape = cv2.MORPH_CROSS
#     else:
#         shape = cv2.MORPH_ELLIPSE

#     # 1. Create Kernel
#     # MORPH_ELLIPSE is generally better for natural objects (cars) than RECT
#     opening_kernel = cv2.getStructuringElement(shape, (kernel_opening_size, kernel_opening_size))
#     closing_kernel = cv2.getStructuringElement(shape, (kernel_closing_size, kernel_closing_size))
    
#     cleaned_mask = mask_np.copy()
    
#     # 2. Apply Operations based on the param
#     if operation == "open":
#         cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, opening_kernel)
#     elif operation == "close":
#         cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, closing_kernel)
#     elif operation == "open_close":
#         cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, opening_kernel)
#         cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, closing_kernel)
#     elif operation == "close_open":
#         cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, closing_kernel)
#         cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, opening_kernel)
        
#     return cleaned_mask

def _get_kernel(size, shape, device):
    """
    Creates a binary kernel for GPU morphology.
    """
    if size % 2 == 0: size += 1 # Ensure odd size
    
    # Create the grid
    coords = torch.arange(size, device=device).float() - (size - 1) / 2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    
    if shape == "rect":
        kernel = torch.ones((size, size), device=device)
    elif shape == "cross":
        kernel = torch.zeros((size, size), device=device)
        mid = size // 2
        kernel[mid, :] = 1
        kernel[:, mid] = 1
    else: # ellipse (Euclidean distance)
        dist = x**2 + y**2
        radius = (size - 1) / 2
        kernel = (dist <= radius**2).float()
        
    return kernel.view(1, 1, size, size)

def apply_morphology(mask_tensor, operation, kernel_opening_size=None, kernel_closing_size=None, morph_shape="ellipse", kernel_size=5):
    """
    Performs morphology on a GPU Tensor.
    
    Args:
        operation: "open", "close", "open_close", "close_open"
        kernel_size: Base size (used if specific open/close sizes aren't provided)
        kernel_size_open: Specific size for opening
        kernel_size_close: Specific size for closing
    """
    # Handle sizes
    k_open = kernel_opening_size if kernel_opening_size else kernel_size
    k_close = kernel_closing_size if kernel_closing_size else kernel_size
    
    if not k_open: k_open = 5
    if not k_close: k_close = 5

    # Ensure 4D shape (B, C, H, W)
    original_shape = mask_tensor.shape
    if mask_tensor.ndim == 2:
        x = mask_tensor.unsqueeze(0).unsqueeze(0).float()
    elif mask_tensor.ndim == 3:
        x = mask_tensor.unsqueeze(1).float()
    else:
        x = mask_tensor.float()

    def get_ops(k_size):
        """Returns dilate/erode functions for a specific kernel size"""
        if k_size <= 1: 
            return lambda t: t, lambda t: t
            
        pad = k_size // 2
        kernel = _get_kernel(k_size, morph_shape, mask_tensor.device)
        kernel_sum = kernel.sum()

        def dilate(t):
            out = F.conv2d(t, kernel, padding=pad)
            return (out > 0).float()

        def erode(t):
            out = F.conv2d(t, kernel, padding=pad)
            return (out >= kernel_sum - 0.1).float()
            
        return dilate, erode

    # -- EXECUTE OPERATIONS --
    
    if operation == "open":
        dilate, erode = get_ops(k_open)
        # Opening: Erode -> Dilate
        x = dilate(erode(x))
        
    elif operation == "close":
        dilate, erode = get_ops(k_close)
        # Closing: Dilate -> Erode
        x = erode(dilate(x))
        
    elif operation == "open_close":
        # 1. Open (Remove Noise)
        dilate_o, erode_o = get_ops(k_open)
        x = dilate_o(erode_o(x))
        
        # 2. Close (Fill Holes)
        dilate_c, erode_c = get_ops(k_close)
        x = erode_c(dilate_c(x))

    elif operation == "close_open":
        # 1. Close
        dilate_c, erode_c = get_ops(k_close)
        x = erode_c(dilate_c(x))
        
        # 2. Open
        dilate_o, erode_o = get_ops(k_open)
        x = dilate_o(erode_o(x))

    # Restore shape
    if len(original_shape) == 2:
        return x.squeeze(0).squeeze(0) > 0.5
    else:
        return x > 0.5

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