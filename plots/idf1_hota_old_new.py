import matplotlib.pyplot as plt
import numpy as np

def plot_sequence_metrics_clean(cameras, hota_old, idf1_old, hota_new, idf1_new, title, filename, figsize=(10.63, 7.35)):
    mean_hota_old, mean_hota_new = np.mean(hota_old), np.mean(hota_new)
    mean_idf1_old, mean_idf1_new = np.mean(idf1_old), np.mean(idf1_new)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.95)

    c_old = 'gray'
    c_hota = '#d62728'
    c_idf1 = '#1f77b4'

    x = np.arange(len(cameras))
    width = 0.35

    # --- HOTA Plot ---
    bars_hota_old = ax1.bar(x - width/2, hota_old, width, color=c_old, label='HOTA (Old)', alpha=0.7)
    bars_hota_new = ax1.bar(x + width/2, hota_new, width, color=c_hota, label='HOTA (New)')
    
    ax1.bar_label(bars_hota_old, fmt='%.4f', label_type='center', fontsize=13, color='black', rotation=90)
    ax1.bar_label(bars_hota_new, fmt='%.4f', label_type='center', fontsize=13, color='black', fontweight='bold', rotation=90)
    
    ax1.axhline(mean_hota_new, color=c_hota, linestyle='--', alpha=0.8, label=f'New Mean: {mean_hota_new:.4f}')
    ax1.axhline(mean_hota_old, color=c_old, linestyle='--', alpha=0.8, label=f'Old Mean: {mean_hota_old:.4f}')
    
    ax1.set_ylabel('HOTA Score', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=False, fontsize=12)
    ax1.margins(y=0.15) 

    # --- IDF1 Plot ---
    bars_idf1_old = ax2.bar(x - width/2, idf1_old, width, color=c_old, label='IDF1 (Old)', alpha=0.7)
    bars_idf1_new = ax2.bar(x + width/2, idf1_new, width, color=c_idf1, label='IDF1 (New)')
    
    ax2.bar_label(bars_idf1_old, fmt='%.4f', label_type='center', fontsize=13, color='black', rotation=90)
    ax2.bar_label(bars_idf1_new, fmt='%.4f', label_type='center', fontsize=13, color='black', fontweight='bold', rotation=90)
    
    ax2.axhline(mean_idf1_new, color=c_idf1, linestyle='--', alpha=0.8, label=f'New Mean: {mean_idf1_new:.4f}')
    ax2.axhline(mean_idf1_old, color=c_old, linestyle='--', alpha=0.8, label=f'Old Mean: {mean_idf1_old:.4f}')
    
    ax2.set_ylabel('IDF1 Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Camera Sequence', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', frameon=False, fontsize=12)
    ax2.margins(y=0.15)

    for ax in [ax1, ax2]:
        ax.set_xticks(x)
        ax.set_xticklabels(cameras)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-0.5, len(cameras)-0.5)

    fig.tight_layout()
    plt.savefig(filename, dpi=300, transparent=False, bbox_inches='tight')
    plt.close(fig)

# --- SEQ01 Example Data ---
cameras_s01 = ['c001', 'c002', 'c003', 'c004', 'c005']
hota_s01_old = [0.3336, 0.4053, 0.4331, 0.4530, 0.5151]
idf1_s01_old = [0.4599, 0.6094, 0.6358, 0.6879, 0.7463]
hota_s01_new = [0.4159, 0.4979, 0.4735, 0.5560, 0.4354]
idf1_s01_new = [0.5705, 0.7551, 0.7131, 0.8030, 0.6511]

cameras_s03 = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
hota_s03_old = [0.1894, 0.1694, 0.0609, 0.2814, 0.2883, 0.0177]
idf1_s03_old = [0.1918, 0.1602, 0.0276, 0.3346, 0.3151, 0.0013]
hota_s03_new = [0.2735, 0.1697, 0.1855, 0.2709, 0.3301, 0.0221]
idf1_s03_new = [0.3487, 0.1949, 0.1349, 0.3285, 0.3933, 0.0020]

plot_sequence_metrics_clean(cameras_s01, hota_s01_old, idf1_s01_old, hota_s01_new, idf1_s01_new, 
                            'Generalization Across Views (Old vs New)', 
                            'results/plots/task2_1_seq01_metrics_clean.png', figsize=(12.63, 7.35))

plot_sequence_metrics_clean(cameras_s03, hota_s03_old, idf1_s03_old, hota_s03_new, idf1_s03_new,
                            'Metric Degradation in SEQ03 (Old vs New)',
                            'results/plots/task2_2_seq03_metrics_clean.png', figsize=(12.63, 7.35))