import matplotlib.pyplot as plt
import numpy as np

methods = ['PyFlow\n(Fast)','PyFlow\n(Def)', 'NeuFlow\n(Things)', 'NeuFlow\n(Sintel)',  'NeuFlow\n(Mix)']
msen_rgb = [1.0491, 0.9932, 0.7780, 0.6641, 0.6490]
pepn_rgb = [8.60, 8.27, 5.28, 3.85, 3.48]
runtime = [3.1081, 7.5839, 0.0216, 0.0227, 0.0226]

fig, ax1 = plt.subplots(figsize=(10, 6))

fig.subplots_adjust(right=0.75)

x = np.arange(len(methods))
width = 0.35 

color1 = "#3b86c4"
ax1.set_ylabel('MSEN', color=color1, fontsize=12, fontweight='bold')
bars1 = ax1.bar(x - width/2, msen_rgb, width, label='MSEN', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, max(msen_rgb) * 1.25) 
# ax1.bar_label(bars1, fmt='%.4f', padding=3, fontsize=9, color=color1, fontweight='bold', rotation=90)

ax2 = ax1.twinx()
color2 = "#db7e14"
ax2.set_ylabel('PEPN (%)', color=color2, fontsize=12, fontweight='bold')
bars2 = ax2.bar(x + width/2, pepn_rgb, width, label='PEPN', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, max(pepn_rgb) * 1.25)
# ax2.bar_label(bars2, fmt='%.2f', padding=3, fontsize=9, color=color2, fontweight='bold', rotation=90)

ax3 = ax1.twinx()
ax3.spines.right.set_position(("axes", 1.15))
color3 = "#0ac20a"
ax3.set_ylabel('Runtime (s)', color=color3, fontsize=12, fontweight='bold')
lines = ax3.plot(x, runtime, color=color3, marker='s', linestyle='--', linewidth=2.5, markersize=8, label='Runtime')
ax3.tick_params(axis='y', labelcolor=color3)
ax3.set_ylim(0, max(runtime) * 1.25)

for i, val in enumerate(runtime):
    ax3.annotate(f'{val:.4f}s', (x[i], val), textcoords="offset points", xytext=(0, 10), 
                 ha='center', va='bottom', fontsize=10, color=color3, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11)
ax1.set_title('Optical Flow Performance Comparison', fontsize=14, fontweight='bold', pad=35)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

plots = [bars1, bars2] + lines
labels = [p.get_label() for p in plots]
ax1.legend(plots, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)

plt.savefig('results/plots/task1_1_metrics_detailed.png', dpi=300, bbox_inches='tight', transparent=True)