import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('results/optimization_results_kalman.csv')

# Transform coordinates to log10 space for mathematically sound triangulation
log_proc = np.log10(df['proc_noise'])
log_meas = np.log10(df['meas_noise'])

fig, ax = plt.subplots(figsize=(8, 6))

cntr = ax.tricontourf(
    log_proc, 
    log_meas, 
    df['value'], 
    levels=14, 
    cmap='viridis'
)

ax.scatter(
    log_proc, 
    log_meas, 
    c='black', 
    s=20, 
    alpha=0.6, 
    label='Sampled Points'
)

fig.colorbar(cntr, ax=ax, label='Objective Value (0.5HOTA * 0.5IDF1)')

# Explicitly label the axes to indicate the log10 scale
ax.set_xlabel('log10(Process Noise)')
ax.set_ylabel('log10(Measurement Noise)')
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
ax.set_title('Process vs. Measurement Noise', fontsize=18, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('results/plots/heatmap_noise_log.png', dpi=300, bbox_inches='tight', transparent=True)