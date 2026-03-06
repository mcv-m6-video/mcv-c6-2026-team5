import matplotlib.pyplot as plt
import numpy as np
"""
This script creates a bar chart comparing the mAP of two 
models (e.g., Off-the-shelf RCNN vs Fine-tuned RCNN) and visually 
highlights the improvement.
"""


# --- USER INPUTS ---
# Replace these values with your actual results
METRIC_LOW = 0.8288  # Example: Result from worse model (e.g., Non-Adaptive, Off-the-shelf)
METRIC_HIGH = 0.9779  # Example: Result from your best model (e.g., Adaptive, Fine-tuned)
LABEL_LOW = "Off-the-shelf RCNN"
LABEL_HIGH = "Fine-Tuned RCNN"
METRIC_LABEL = "mAP @ IoU 0.5"

OUTPUT_FILE = "results/plots/improvement_finetuned_plot.png"
TITLE = "Impact of fine-tuning RCNN on AICity Dataset"

# --- PLOT STYLE ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def plot_improvement():
    # 1. Calculate Improvement
    improvement_pct = ((METRIC_HIGH - METRIC_LOW) / METRIC_LOW) * 100
    
    # Data Setup
    labels = [LABEL_LOW, LABEL_HIGH]
    values = [METRIC_LOW, METRIC_HIGH]
    colors = ['#95a5a6', '#2ecc71'] # Grey for old, Green for new
    
    # 2. Create Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # 3. Add Value Labels on Top of Bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01, 
            f'{height:.4f}', 
            ha='center', 
            va='bottom', 
            fontweight='bold',
            fontsize=16,
            color='black'
        )

    # 4. Draw the Improvement Annotation
    # Coordinates
    x_base = bars[0].get_x() + bars[0].get_width() / 2
    x_adapt = bars[1].get_x() + bars[1].get_width() / 2
    y_base = values[0]
    y_adapt = values[1]
    
    # Draw a dashed line from Baseline top across to Adaptive column
    ax.plot([x_base, x_adapt], [y_base, y_base], color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Draw a double-headed arrow or bracket indicating the gap
    ax.annotate(
        text='', 
        xy=(x_adapt, y_adapt), 
        xytext=(x_adapt, y_base), 
        arrowprops=dict(arrowstyle='<->', color='black', lw=2)
    )
    
    # Add the Percentage Text (Centered in the gap)
    mid_y = (y_base + y_adapt) / 2
    ax.text(
        x_adapt + 0.05, # Shift text slightly to the right of the arrow
        mid_y, 
        f'+{improvement_pct:.1f}%\nImprovement', 
        ha='left', 
        va='center', 
        color='#27ae60', 
        fontweight='bold', 
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#27ae60", alpha=0.9)
    )

    # 5. Final Aesthetics
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title(TITLE, pad=20, fontweight='bold')
    ax.set_ylim(0, 1) # Add headroom for labels
    
    # Add grid behind bars
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, transparent=True)
    print(f"Plot saved to {OUTPUT_FILE}")
    # plt.show()

if __name__ == "__main__":
    plot_improvement()