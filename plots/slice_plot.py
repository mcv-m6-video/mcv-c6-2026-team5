import pandas as pd
import matplotlib.pyplot as plt

# We read the generated CSV with our optimization results
df = pd.read_csv('results/optimization_results_kalman.csv')

# We isolate the parameter columns by excluding 'trial' and 'value'
param_cols = [col for col in df.columns if col not in ['trial', 'value', 'proc_noise', 'meas_noise']]
num_params = len(param_cols)

# We create a figure with subplots for each parameter
fig, axes = plt.subplots(1, num_params, figsize=(15, 4), sharey=True)

for i, col in enumerate(param_cols):
    if 'noise' in col.lower():
        continue
    # We map the objective value to the color map to emphasize the top performers
    scatter = axes[i].scatter(
        df[col], 
        df['value'], 
        c=df['value'], 
        cmap='viridis', 
        alpha=0.8, 
        edgecolors='w', 
        s=60
    )
    axes[i].set_xlabel(col)
    
    if i == 0:
        axes[i].set_ylabel('Objective Value\n(0.5HOTA * 0.5IDF1)')
        
    axes[i].grid(True, linestyle='--', alpha=0.6)
    # if noise metric log scale is used, we can set y-axis to log scale
    if 'noise' in col.lower():
        axes[i].set_xscale('log')

plt.suptitle('Hyperparameter Slice Plot (Kalman tracking)', y=1.05, fontsize=18, fontweight='bold')
plt.tight_layout()
# bigger axis labels for presentation
for ax in axes:
    ax.xaxis.label.set_size(15)
    # ax.xaxis.label.set_weight('bold')
    ax.yaxis.label.set_size(15)
    # ax.yaxis.label.set_weight('bold')
# bigger axis numbers
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=15)

# We save the figure directly for the presentation slides
plt.savefig('results/plots/slice_plot_kalman.png', dpi=300, bbox_inches='tight', transparent=True)