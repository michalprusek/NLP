#!/usr/bin/env python3
"""
Compare HbBoPs experiments with different grid sizes (10x25 vs 25x25).
Generates comprehensive visualization similar to HbBoPs Improved 2.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score


def load_ground_truth(jsonl_path):
    """Load ground truth results from JSONL file."""
    gt_data = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            key = (entry['instruction_id'], entry['exemplar_id'])
            gt_data[key] = entry['error_rate']
    return gt_data


def compute_ndcg_at_k(y_true, y_pred, k=None):
    """Compute NDCG@k score."""
    if k is None:
        k = len(y_true)
    # Convert to relevance scores (lower error = higher relevance)
    relevance_true = 1 - np.array(y_true)
    relevance_pred = 1 - np.array(y_pred)

    # Reshape for sklearn
    relevance_true = relevance_true.reshape(1, -1)
    relevance_pred = relevance_pred.reshape(1, -1)

    return ndcg_score(relevance_true, relevance_pred, k=min(k, len(y_true)))


def compute_top_k_overlap(hbbops_errors, gt_errors, k):
    """Compute percentage overlap of top-K prompts."""
    hbbops_top_k = set(np.argsort(hbbops_errors)[:k])
    gt_top_k = set(np.argsort(gt_errors)[:k])
    overlap = len(hbbops_top_k & gt_top_k)
    return (overlap / k) * 100


def group_by_fidelity(prompts):
    """Group prompts by their max_fidelity level."""
    fidelity_groups = defaultdict(list)
    for p in prompts:
        fidelity_groups[p['max_fidelity']].append(p['diff_pp'])
    return fidelity_groups


def create_comparison_plot(data_10x25, data_25x25, gt_data, output_path):
    """Create comprehensive comparison plot."""
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.4)

    # Colors
    color_10x25 = '#FF9966'  # Orange
    color_25x25 = '#B266FF'  # Purple

    # ============== Panel 1: Accuracy Difference by Fidelity ==============
    ax1 = fig.add_subplot(gs[0, :])

    fid_10x25 = group_by_fidelity(data_10x25['all_evaluated_prompts'])
    fid_25x25 = group_by_fidelity(data_25x25['all_evaluated_prompts'])

    # Get all unique fidelity levels
    all_fidelities = sorted(set(list(fid_10x25.keys()) + list(fid_25x25.keys())))

    # Prepare data for box plots
    positions_10x25 = []
    positions_25x25 = []
    data_10x25_boxes = []
    data_25x25_boxes = []
    labels = []

    for i, fid in enumerate(all_fidelities):
        if fid in fid_10x25:
            positions_10x25.append(i - 0.2)
            data_10x25_boxes.append(fid_10x25[fid])
        if fid in fid_25x25:
            positions_25x25.append(i + 0.2)
            data_25x25_boxes.append(fid_25x25[fid])
        labels.append(str(fid))

    # Create box plots
    bp1 = ax1.boxplot(data_10x25_boxes, positions=positions_10x25, widths=0.3,
                       patch_artist=True, showfliers=True,
                       boxprops=dict(facecolor=color_10x25, alpha=0.7),
                       medianprops=dict(color='darkred', linewidth=2),
                       whiskerprops=dict(color=color_10x25),
                       capprops=dict(color=color_10x25))

    bp2 = ax1.boxplot(data_25x25_boxes, positions=positions_25x25, widths=0.3,
                       patch_artist=True, showfliers=True,
                       boxprops=dict(facecolor=color_25x25, alpha=0.7),
                       medianprops=dict(color='darkred', linewidth=2),
                       whiskerprops=dict(color=color_25x25),
                       capprops=dict(color=color_25x25))

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xticks(range(len(all_fidelities)))
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Fidelity Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Diff (HbBoPs - GT) in pp', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Difference by Fidelity Level', fontsize=14, fontweight='bold')
    ax1.legend([bp1["boxes"][0], bp2["boxes"][0]],
               [f'10×25 (250)', f'25×25 (625)'],
               loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ============== Panel 2: Computational Efficiency ==============
    ax2 = fig.add_subplot(gs[1, 0])

    efficiencies = [
        data_25x25['llm_calls']['efficiency_ratio'],
        data_10x25['llm_calls']['efficiency_ratio']
    ]
    hbbops_calls = [
        data_25x25['llm_calls']['hbbops_total'],
        data_10x25['llm_calls']['hbbops_total']
    ]

    bars = ax2.bar([0, 1], efficiencies, color=[color_25x25, color_10x25], alpha=0.7, width=0.6)

    # Add annotations
    for i, (bar, eff, hb_call) in enumerate(zip(bars, efficiencies, hbbops_calls)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{eff:.1f}x\n({hb_call/1000:.1f}K)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Efficiency (GT calls / HbBoPs calls)', fontsize=11, fontweight='bold')
    ax2.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['25×25 (625)', '10×25 (250)'])
    ax2.set_ylim(0, max(efficiencies) * 1.2)
    ax2.grid(True, alpha=0.3, axis='y')

    # ============== Panel 3: Rank Correlation ==============
    ax3 = fig.add_subplot(gs[1, 1])

    # Compute correlations for both datasets
    correlations = []
    for data, label in [(data_25x25, '25×25 (625)'), (data_10x25, '10×25 (250)')]:
        hbbops_errors = [p['hbbops_error'] for p in data['all_evaluated_prompts']]
        gt_errors = [p['gt_error'] for p in data['all_evaluated_prompts']]

        spearman_corr, _ = spearmanr(hbbops_errors, gt_errors)
        kendall_corr, _ = kendalltau(hbbops_errors, gt_errors)

        correlations.append({
            'label': label,
            'spearman': spearman_corr,
            'kendall': kendall_corr
        })

    # Plot correlations
    x = np.arange(len(correlations))
    width = 0.35

    spearman_vals = [c['spearman'] for c in correlations]
    kendall_vals = [c['kendall'] for c in correlations]

    ax3.bar(x - width/2, spearman_vals, width, label='Spearman ρ',
            color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, kendall_vals, width, label='Kendall τ',
            color=color_10x25, alpha=0.7)

    # Add value labels
    for i, (sp, kd) in enumerate(zip(spearman_vals, kendall_vals)):
        ax3.text(i - width/2, sp + 0.01, f'{sp:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(i + width/2, kd + 0.01, f'{kd:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
    ax3.set_title('Rank Correlation with Ground Truth', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c['label'] for c in correlations])
    ax3.legend()
    ax3.set_ylim(0.8, 1.0)
    ax3.grid(True, alpha=0.3, axis='y')

    # ============== Panel 4: Top-K Overlap ==============
    ax4 = fig.add_subplot(gs[1, 2])

    k_values_overlap = [5, 10, 25, 50, 100]

    # Create table data
    overlap_data = []
    for k in k_values_overlap:
        # 10x25
        hbbops_errors = [p['hbbops_error'] for p in data_10x25['all_evaluated_prompts']]
        gt_errors = [p['gt_error'] for p in data_10x25['all_evaluated_prompts']]
        overlap_10x25 = compute_top_k_overlap(hbbops_errors, gt_errors, k)

        # 25x25
        hbbops_errors = [p['hbbops_error'] for p in data_25x25['all_evaluated_prompts']]
        gt_errors = [p['gt_error'] for p in data_25x25['all_evaluated_prompts']]
        overlap_25x25 = compute_top_k_overlap(hbbops_errors, gt_errors, k)

        overlap_data.append((k, overlap_25x25, overlap_10x25))

    # Plot as grouped bar chart
    x = np.arange(len(k_values_overlap))
    width = 0.35

    overlap_25x25_vals = [d[1] for d in overlap_data]
    overlap_10x25_vals = [d[2] for d in overlap_data]

    bars1 = ax4.bar(x - width/2, overlap_25x25_vals, width,
                    label='25×25 (625)', color=color_25x25, alpha=0.7)
    bars2 = ax4.bar(x + width/2, overlap_10x25_vals, width,
                    label='10×25 (250)', color=color_10x25, alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax4.set_ylabel('Overlap (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Top-K Overlap (HbBoPs ∩ GT)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('K', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'K={k}' for k in k_values_overlap])
    ax4.legend()
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')

    # Overall title
    fig.suptitle('HbBoPs Grid Size Comparison\n(25×25 = 625 prompts vs 10×25 = 250 prompts)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def main():
    """Main execution."""
    base_dir = Path(__file__).parent.parent  # /home/prusek/NLP
    results_dir = base_dir / "hbbops" / "results"
    datasets_dir = base_dir / "datasets" / "hbbops"

    # Load data
    print("Loading experiment results...")
    with open(results_dir / "10x25_20251203_005743.json", 'r') as f:
        data_10x25 = json.load(f)

    with open(results_dir / "25x25_20251203_194153.json", 'r') as f:
        data_25x25 = json.load(f)

    print("Loading ground truth...")
    gt_data = load_ground_truth(datasets_dir / "full_grid_combined.jsonl")

    print(f"10×25: {len(data_10x25['all_evaluated_prompts'])} prompts evaluated")
    print(f"25×25: {len(data_25x25['all_evaluated_prompts'])} prompts evaluated")
    print(f"Ground truth: {len(gt_data)} total combinations")

    # Create comparison plot
    output_path = results_dir / "grid_size_comparison.png"
    print("\nGenerating comparison plot...")
    create_comparison_plot(data_10x25, data_25x25, gt_data, output_path)

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
