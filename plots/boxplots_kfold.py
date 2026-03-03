import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
# mAP@50 scores for each strategy
map_A = [0.9771]
map_B = [0.9779, 0.9838, 0.9286, 0.9365]
map_C = [0.9851, 0.9913, 0.9843, 0.9852]

data = [map_A, map_B, map_C]
labels = ['Strategy A\n(Static Split)', 'Strategy B\n(Block K-Fold)', 'Strategy C\n(Random K-Fold)']

# --- 2. Plot Setup ---
fig, ax = plt.subplots(figsize=(10, 7))

# Create boxplot
box_colors = ['#cccccc', '#1f77b4', '#ff7f0e']
bplot = ax.boxplot(data, patch_artist=True, labels=labels, 
                   widths=0.5, medianprops=dict(color="black", linewidth=2.5))

# Add colors to boxes
for patch, color in zip(bplot['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# --- 3. Overlay individual data points (Scatter) ---
# This is great for slides so people see exactly how many folds there are
for i, d in enumerate(data):
    # Add some random jitter to x-axis so points don't overlap completely
    x = np.random.normal(i + 1, 0.04, size=len(d))
    ax.scatter(x, d, alpha=0.9, color='black', edgecolor='white', s=100, zorder=3)

# --- 4. Aesthetics & Large Fonts for Slides ---
# Adjust font sizes
TITLE_SIZE = 20
LABEL_SIZE = 16
TICK_SIZE = 14

ax.set_title('Cross-Validation Strategies Variance (mAP@50)', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
ax.set_ylabel('mAP@50', fontsize=LABEL_SIZE, fontweight='bold')

# Tick parameters
ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax.set_ylim(0.92, 1.0) # Set limits to make the variance visible

# Grid for easier reading
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
ax.set_axisbelow(True)

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()

# Save the plot
plt.savefig('results/plots/cross_validation_variance.png', dpi=300, transparent=True)
plt.show()