import matplotlib.pyplot as plt
import numpy as np

def plot_sequence_metrics(cameras, hota, idf1, title, xlabel, filename, ylim_bottom, ylim_top):
    mean_hota = np.mean(hota)
    mean_idf1 = np.mean(idf1)

    fig, ax = plt.subplots(figsize=(10, 6))

    c_hota = '#d62728'
    c_idf1 = '#1f77b4'
    
    ax.plot(cameras, hota, marker='o', markersize=9, linestyle='-', linewidth=2.5, label='HOTA', color=c_hota)
    ax.plot(cameras, idf1, marker='s', markersize=9, linestyle='-', linewidth=2.5, label='IDF1', color=c_idf1)

    ax.axhline(mean_hota, color=c_hota, linestyle='-.', linewidth=1.5, alpha=0.6)
    ax.axhline(mean_idf1, color=c_idf1, linestyle='-.', linewidth=1.5, alpha=0.6)

    ax.text(len(cameras) - 0.7, mean_hota - 0.025, f'Mean: {mean_hota:.4f}', 
            color=c_hota, fontsize=11, fontweight='bold', ha='right', va='bottom')
    ax.text(len(cameras) - 0.7, mean_idf1 + 0.005, f'Mean: {mean_idf1:.4f}', 
            color=c_idf1, fontsize=11, fontweight='bold', ha='right', va='bottom')

    for i, (h, idf) in enumerate(zip(hota, idf1)):
        ax.annotate(f'{h:.4f}', (i, h), textcoords="offset points", xytext=(0, -16), 
                    ha='center', fontsize=9, color=c_hota, fontweight='medium')
        ax.annotate(f'{idf:.4f}', (i, idf), textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontsize=9, color=c_idf1, fontweight='medium')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(ylim_bottom, ylim_top)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=True)

    fig.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close(fig)

# cameras_s01 = ['c001', 'c002', 'c003', 'c004', 'c005']
# hota_s01 = [0.3336, 0.4053, 0.4331, 0.4530, 0.5151]
# idf1_s01 = [0.4599, 0.6094, 0.6358, 0.6879, 0.7463]
# plot_sequence_metrics(cameras_s01, hota_s01, idf1_s01, 
#                       'Generalization Across Views', 'Camera Sequence (SEQ01)', 
#                       'results/plots/task2_1_seq01_metrics.png', 0.25, 0.85)

# cameras_s03 = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
# hota_s03 = [0.1894, 0.1694, 0.0609, 0.2814, 0.2883, 0.0177]
# idf1_s03 = [0.1918, 0.1602, 0.0276, 0.3346, 0.3151, 0.0013]
# plot_sequence_metrics(cameras_s03, hota_s03, idf1_s03, 
#                       'Metric Degradation in SEQ03', 'Camera Sequence (SEQ03)', 
#                       'results/plots/task2_2_seq03_metrics.png', -0.05, 0.45)
# --- SEQ01: Generalization Across Views ---
cameras_s01 = ['c001', 'c002', 'c003', 'c004', 'c005']
# Extracted from logs: HOTA (0.4159, 0.4979, 0.4735, 0.5560, 0.4354)
hota_s01 = [0.4159, 0.4979, 0.4735, 0.5560, 0.4354]
# Extracted from logs: IDF1 (0.5705, 0.7551, 0.7131, 0.8030, 0.6511)
idf1_s01 = [0.5705, 0.7551, 0.7131, 0.8030, 0.6511]

plot_sequence_metrics(cameras_s01, hota_s01, idf1_s01, 
                      'Generalization Across Views', 'Camera Sequence (SEQ01)', 
                      'results/plots/task2_1_seq01_metrics.png', 0.35, 0.90)

# --- SEQ03: Metric Degradation ---
cameras_s03 = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
# Extracted from logs: HOTA (0.2735, 0.1697, 0.1855, 0.2709, 0.3301, 0.0221)
hota_s03 = [0.2735, 0.1697, 0.1855, 0.2709, 0.3301, 0.0221]
# Extracted from logs: IDF1 (0.3487, 0.1949, 0.1349, 0.3285, 0.3933, 0.0020)
idf1_s03 = [0.3487, 0.1949, 0.1349, 0.3285, 0.3933, 0.0020]

plot_sequence_metrics(cameras_s03, hota_s03, idf1_s03, 
                      'Metric Degradation in SEQ03', 'Camera Sequence (SEQ03)', 
                      'results/plots/task2_2_seq03_metrics.png', 0.00, 0.50)