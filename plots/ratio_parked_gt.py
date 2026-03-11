import matplotlib.pyplot as plt
import numpy as np

def plot_simple_ratio(cameras, ratios, title, filename, use_log=False):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    c_ratio = '#ff7f0e'
    bars = ax.bar(cameras, ratios, color=c_ratio, alpha=0.8)
    
    if use_log:
        ax.set_yscale('symlog', linthresh=4.8)
        ax.set_ylabel('False Negative Ratio (Log Scale)', fontsize=10, fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                    f'{height:.1f}x', ha='center', va='bottom', 
                    color=c_ratio, fontweight='bold', fontsize=9)
    else:
        ax.set_ylabel('FP/GT Ratio', fontsize=10, fontweight='bold')
        ax.bar_label(bars, fmt='%.2fx', padding=3, fontsize=9, color=c_ratio, fontweight='bold')
        ax.set_ylim(0, max(ratios) * 1.2)
        
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    fig.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close(fig)

cameras_s01 = ['c001', 'c002', 'c003', 'c004', 'c005']
ratio_s01 = [1.22, 0.54, 0.60, 0.40, 0.21]

cameras_s03 = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
ratio_s03 = [3.62, 4.69, 4.75, 2.47, 2.36, 977.84]

plot_simple_ratio(cameras_s01, ratio_s01, 'Annotation Bias (SEQ01)', 'results/plots/task2_1_seq01_ratio.png')
plot_simple_ratio(cameras_s03, ratio_s03, 'Annotation Bias (SEQ03)', 'results/plots/task2_2_seq03_ratio.png', use_log=True)