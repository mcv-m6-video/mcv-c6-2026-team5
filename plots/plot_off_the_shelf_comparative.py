import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
models = ['Faster R-CNN\n(ResNet50)', 'YOLOv8\nMedium', 'YOLOv8\nNano']
map_scores = [0.8288, 0.6475, 0.5872]   # Replace with your actual mAP50 scores
fps_scores = [35, 71, 79]        # Replace with your actual FPS values

# Colors for aesthetics
bar_color = '#1f77b4'  # Blue for mAP
line_color = '#d62728' # Red for FPS

# --- 2. Create Plot ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar width
width = 0.5
x = np.arange(len(models))

# --- 3. Left Axis (Bar Plot for mAP) ---
bars = ax1.bar(x, map_scores, width, label='mAP@50', color=bar_color, alpha=0.8)

# Formatting Left Axis
ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
ax1.set_ylabel('mAP@50', fontsize=12, color=bar_color, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=bar_color)
ax1.set_ylim(0, 1.1)  # mAP is usually 0-1

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', color=bar_color, fontsize=10, fontweight='bold')

# --- 4. Right Axis (Line Plot for FPS) ---
ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis

# Plot FPS as a line with markers
line = ax2.plot(x, fps_scores, label='FPS', color=line_color, linewidth=3, marker='o', markersize=10)

# Formatting Right Axis
ax2.set_ylabel('Frames Per Second (FPS)', fontsize=12, color=line_color, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=line_color)
ax2.set_ylim(0, max(fps_scores) * 1.2)  # Give some headroom for FPS

# Add value labels for points
for i, txt in enumerate(fps_scores):
    ax2.annotate(f'{txt}', (x[i], fps_scores[i]), 
                 textcoords="offset points", xytext=(0,10), 
                 ha='center', color=line_color, fontweight='bold')

# --- 5. Final Touches ---
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
plt.title('Task 1.1: Accuracy (mAP) vs. Speed (FPS) Trade-off', fontsize=14, pad=20)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()

# Save or Show
plt.savefig('results/task1_1_comparison.png', dpi=300, transparent=True)
