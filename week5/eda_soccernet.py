import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from util.io import load_json
from util.dataset import load_classes

def main():
    parser = argparse.ArgumentParser(description="EDA for SoccerNet Ball Action Spotting")
    parser.add_argument('--data_dir', type=str, default='data/soccernetball', help='Path to dataset directory')
    parser.add_argument('--labels_dir', type=str, default='/ghome/group05/datasets/SoccerNet/SN-BAS-2025', help='Path to the raw labels folder')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    # Load classes
    classes_dict = load_classes(os.path.join(args.data_dir, 'class.txt'))
    
    # Load split videos
    split_file = os.path.join(args.data_dir, f'{args.split}.json')
    if not os.path.exists(split_file):
        print(f"Error: {split_file} not found.")
        return
        
    videos = load_json(split_file)
    print(f"Analyzing {len(videos)} videos in the '{args.split}' split...")

    class_counts = Counter()
    actions_per_video = []

    # Parse all labels
    for video in videos:
        video_name = video['video']
        labels_path = os.path.join(args.labels_dir, video_name, 'Labels-ball.json')
        
        if not os.path.exists(labels_path):
            continue
            
        annotations = load_json(labels_path).get('annotations', [])
        actions_per_video.append(len(annotations))
        
        for event in annotations:
            class_counts[event['label']] += 1

    # --- Print Statistical Summary ---
    total_actions = sum(class_counts.values())
    print("\n--- EDA SUMMARY ---")
    print(f"Total Actions: {total_actions}")
    print(f"Average actions per video: {np.mean(actions_per_video):.1f}")
    print(f"Max actions in a single video: {np.max(actions_per_video)}")
    print(f"Min actions in a single video: {np.min(actions_per_video)}")
    
    print("\n--- CLASS DISTRIBUTION ---")
    # Sort classes by frequency for a cleaner read
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_percentages = {}
    
    for class_name, count in sorted_classes:
        percentage = (count / total_actions) * 100
        class_percentages[class_name] = percentage
        print(f"{class_name:<20}: {count:>6} ({percentage:>5.1f}%)")

    # --- NEW: Extract Copy-Pasteable Percentages ---
    print("\n--- PERCENTAGES FOR PLOTTING SCRIPT ---")
    # Sort by class name/ID to match the default order in the metrics_eval.csv
    sorted_by_key = sorted(class_percentages.items())
    ordered_classes = [k for k, v in sorted_by_key]
    list_of_percentages = [round(v, 2) for k, v in sorted_by_key]
    
    print(f"Ordered Classes: {ordered_classes}")
    print(f"Percentages:     {list_of_percentages}")
    print("-> Copy the 'Percentages' list above into the 'placeholder_percentages' variable in your plotting script!")

    # --- Plotting ---
    names = [x[0] for x in sorted_classes]
    counts = [x[1] for x in sorted_classes]

    # Updated to match your custom styling
    plt.figure(figsize=(12, 6))
    
    plt.bar(names, counts, color='#A1C9F4', edgecolor='black', linewidth=0.8, zorder=3)
    
    plt.title(f'Action Class Imbalance ({args.split.upper()} Split)', fontsize=15, fontweight='bold', pad=14)
    plt.xlabel('Action Classes', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Occurrences', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    plt.gca().set_axisbelow(True)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()

    os.makedirs('eda_results', exist_ok=True)
    out_path = f"eda_results/class_distribution_{args.split}.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved class distribution plot to {out_path}")

if __name__ == '__main__':
    main()