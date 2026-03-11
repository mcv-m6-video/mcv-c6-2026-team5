import matplotlib.pyplot as plt
import numpy as np

trackers = ['Max IoU', 'Kalman Filter', 'NeuFlow (Optimized)']
hota = [0.7346, 0.7357, 0.7523]
idf1 = [0.7307, 0.7330, 0.7437]

x = np.arange(len(trackers))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, hota, width, label='HOTA', color='#2ca02c', alpha=0.85)
rects2 = ax.bar(x + width/2, idf1, width, label='IDF1', color='#9467bd', alpha=0.85)

ax.set_ylabel('Metric Score', fontsize=12, fontweight='bold')
ax.set_title('Tracking Pipeline Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(trackers, fontsize=12)
ax.set_ylim(0.70, 0.77) 
ax.legend(loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Add value labels
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

fig.tight_layout()
plt.savefig('results/plots/task1_2_tracker_comparison.png', dpi=300, transparent=True)