import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Setup ---
trackers = [' IoU', 'Kalman']
hota_scores = [0.8850, 0.8849]
idf1_scores = [0.9200, 0.9243]

x = np.arange(len(trackers))
width = 0.35  # the width of the bars

# Colors
color_hota = "#5d8040"  # Blue
color_idf1 = '#ff7f0e'  # Orange

# --- 2. Create Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, hota_scores, width, label='HOTA', color=color_hota)
rects2 = ax.bar(x + width/2, idf1_scores, width, label='IDF1', color=color_idf1)

# --- 3. Aesthetics & Large Fonts for Slides ---
TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 14

ax.set_ylabel('Score', fontsize=LABEL_SIZE, fontweight='bold')
ax.set_title('Tracking Performance (IoU vs. Kalman)', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(trackers, fontsize=TICK_SIZE, fontweight='bold')
ax.tick_params(axis='y', labelsize=TICK_SIZE)

# Set y-axis limits to zoom in on the differences
ax.set_ylim(0.85, 0.93)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# --- 4. Add Value Labels on Bars ---
def autolabel(rects):
    """Attach a text label above each bar, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

ax.legend(fontsize=LABEL_SIZE, loc='upper left', frameon=False)

plt.tight_layout()

# Save the plot
plt.savefig('results/plots/task2_2_tracker_comparison.png', dpi=300, transparent=True)