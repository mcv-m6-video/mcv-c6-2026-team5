import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_annotated_flow_legend(output_path='optical_flow_legend.png', size=300):
    # Create a grid of x and y coordinates centered at 0
    y, x = np.mgrid[-size/2:size/2, -size/2:size/2]
    
    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(x, y)
    
    # Mask to keep only the circular region
    radius = size / 2
    mask = mag <= radius
    
    # Generate HSV image identical to the tracking script
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = np.clip(mag * 255.0 / radius, 0, 255).astype(np.uint8)
    hsv[..., 1] = 255 # White background for zero magnitude
    
    # Convert to RGB for matplotlib
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Add alpha channel for transparency outside the circle
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(mask, 255, 0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    
    # extent=[-1, 1, 1, -1] inverses the Y-axis to match standard image coordinates 
    # where Y increases as you go DOWN the image.
    ax.imshow(rgba, extent=[-1, 1, 1, -1]) 
    
    # Add directional labels
    ax.text(1.1, 0, 'Right', va='center', ha='left', fontsize=12, fontweight='bold')
    ax.text(-1.1, 0, 'Left', va='center', ha='right', fontsize=12, fontweight='bold')
    ax.text(0, 1.1, 'Down', va='top', ha='center', fontsize=12, fontweight='bold')
    ax.text(0, -1.1, 'Up', va='bottom', ha='center', fontsize=12, fontweight='bold')
    
    ax.axis('off')
    
    # Save with transparent background
    plt.savefig(output_path, bbox_inches='tight', transparent=True)
    print(f"Legend saved to {output_path}")

if __name__ == "__main__":
    generate_annotated_flow_legend()