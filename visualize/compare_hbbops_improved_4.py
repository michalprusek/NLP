#!/usr/bin/env python3
"""
Generate comparison visualizations for HbBoPs Improved 4.
1. Grid comparison (10x25 vs 25x25)
2. Full method comparison (Original, Improved, Improved 2, Improved 3, Improved 4)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau


def load_ground_truth(jsonl_path):
    """Load ground truth results from JSONL file."""
    gt_data = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            key = (entry['instruction_id'], entry['exemplar_id'])
            gt_data[key] = entry['error_rate']
    return gt_data


def parse_log_for_evaluations(log_path):
    """Parse log file to extract evaluation cache."""
    evaluations = {}
    with open(log_path, 'r') as f:
        for line in f:
            if "Evaluated prompt" in line and "with error" in line:
                parts = line.strip().split()
                try:
                    prompt_idx = int(parts[parts.index('prompt') + 1])
                    error = float(parts[parts.index('error') + 1])
                    fidelity = int(parts[parts.index('fidelity') + 1])
                    key = (prompt_idx, fidelity)
                    evaluations[key] = error
                except (ValueError, IndexError):
                    continue
    return evaluations


def build_comparison_data(evaluations, gt_data, instruction_mapping, exemplar_mapping, num_inst, num_ex):
    """Build comparison data structure from evaluations."""
    prompt_max_fidelity = {}
    for (prompt_idx, fidelity), error in evaluations.items():
        if prompt_idx not in prompt_max_fidelity or fidelity > prompt_max_fidelity[prompt_idx][0]:
            prompt_max_fidelity[prompt_idx] = (fidelity, error)

    all_prompts = []
    for prompt_idx, (max_fid, error) in prompt_max_fidelity.items():
        inst_id = prompt_idx // num_ex
        ex_id = prompt_idx % num_ex

        if inst_id < len(instruction_mapping) and ex_id < len(exemplar_mapping):
            orig_inst = instruction_mapping[inst_id]
            orig_ex = exemplar_mapping[ex_id]

            gt_error = gt_data.get((orig_inst, orig_ex))
            if gt_error is not None:
                diff = (error - gt_error) * 100
                all_prompts.append({
                    'sel_inst': inst_id,
                    'sel_ex': ex_id,
                    'orig_inst': orig_inst,
                    'orig_ex': orig_ex,
                    'max_fidelity': max_fid,
                    'hbbops_error': error,
                    'gt_error': gt_error,
                    'diff_pp': diff
                })

    hbbops_calls = sum(fidelity for (_, fidelity) in evaluations.keys())
    gt_calls = num_inst * num_ex * 1319

    return {
        'all_evaluated_prompts': all_prompts,
        'llm_calls': {
            'hbbops_total': hbbops_calls,
            'gt_total': gt_calls,
            'efficiency_ratio': gt_calls / hbbops_calls if hbbops_calls > 0 else 0
        },
        'total_prompts_in_grid': num_inst * num_ex
    }


def group_by_fidelity(prompts):
    """Group prompts by their max_fidelity level."""
    fidelity_groups = defaultdict(list)
    for p in prompts:
        fidelity_groups[p['max_fidelity']].append(p['diff_pp'])
    return fidelity_groups


def compute_top_k_overlap(hbbops_errors, gt_errors, k):
    """Compute percentage overlap of top-K prompts."""
    hbbops_top_k = set(np.argsort(hbbops_errors)[:k])
    gt_top_k = set(np.argsort(gt_errors)[:k])
    overlap = len(hbbops_top_k & gt_top_k)
    return (overlap / k) * 100


def create_grid_comparison_plot(data_10x25, data_25x25, output_path, title_suffix=""):
    """Create grid size comparison plot (10x25 vs 25x25)."""
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.4)

    color_10x25 = '#FF9966'  # Orange
    color_25x25 = '#B266FF'  # Purple

    # Panel 1: Accuracy Difference by Fidelity
    ax1 = fig.add_subplot(gs[0, :])

    fid_10x25 = group_by_fidelity(data_10x25['all_evaluated_prompts'])
    fid_25x25 = group_by_fidelity(data_25x25['all_evaluated_prompts'])

    all_fidelities = sorted(set(list(fid_10x25.keys()) + list(fid_25x25.keys())))

    positions_10x25 = []
    positions_25x25 = []
    data_10x25_boxes = []
    data_25x25_boxes = []

    for i, fid in enumerate(all_fidelities):
        if fid in fid_10x25:
            positions_10x25.append(i - 0.2)
            data_10x25_boxes.append(fid_10x25[fid])
        if fid in fid_25x25:
            positions_25x25.append(i + 0.2)
            data_25x25_boxes.append(fid_25x25[fid])

    if data_10x25_boxes:
        bp1 = ax1.boxplot(data_10x25_boxes, positions=positions_10x25, widths=0.3,
                          patch_artist=True, showfliers=True,
                          boxprops=dict(facecolor=color_10x25, alpha=0.7),
                          medianprops=dict(color='darkred', linewidth=2))

    if data_25x25_boxes:
        bp2 = ax1.boxplot(data_25x25_boxes, positions=positions_25x25, widths=0.3,
                          patch_artist=True, showfliers=True,
                          boxprops=dict(facecolor=color_25x25, alpha=0.7),
                          medianprops=dict(color='darkred', linewidth=2))

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xticks(range(len(all_fidelities)))
    ax1.set_xticklabels([str(f) for f in all_fidelities], rotation=45, ha='right')
    ax1.set_xlabel('Fidelity Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Diff (HbBoPs - GT) in pp', fontsize=12, fontweight='bold')
    ax1.set_title(f'HbBoPs Improved 4: Accuracy Difference by Fidelity Level{title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend([plt.Rectangle((0,0),1,1, facecolor=color_10x25, alpha=0.7),
                plt.Rectangle((0,0),1,1, facecolor=color_25x25, alpha=0.7)],
               ['10x25 (250)', '25x25 (625)'], loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Computational Efficiency
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

    for i, (bar, eff, hb_call) in enumerate(zip(bars, efficiencies, hbbops_calls)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{eff:.1f}x\n({hb_call/1000:.1f}K)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Efficiency (GT calls / HbBoPs calls)', fontsize=11, fontweight='bold')
    ax2.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['25x25 (625)', '10x25 (250)'])
    ax2.set_ylim(0, max(efficiencies) * 1.3)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Rank Correlation
    ax3 = fig.add_subplot(gs[1, 1])

    correlations = []
    for data, label in [(data_25x25, '25x25'), (data_10x25, '10x25')]:
        hbbops_errors = [p['hbbops_error'] for p in data['all_evaluated_prompts']]
        gt_errors = [p['gt_error'] for p in data['all_evaluated_prompts']]

        spearman_corr, _ = spearmanr(hbbops_errors, gt_errors)
        kendall_corr, _ = kendalltau(hbbops_errors, gt_errors)

        correlations.append({
            'label': label,
            'spearman': spearman_corr,
            'kendall': kendall_corr
        })

    x = np.arange(len(correlations))
    width = 0.35

    spearman_vals = [c['spearman'] for c in correlations]
    kendall_vals = [c['kendall'] for c in correlations]

    ax3.bar(x - width/2, spearman_vals, width, label='Spearman rho', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, kendall_vals, width, label='Kendall tau', color=color_10x25, alpha=0.7)

    for i, (sp, kd) in enumerate(zip(spearman_vals, kendall_vals)):
        ax3.text(i - width/2, sp + 0.01, f'{sp:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(i + width/2, kd + 0.01, f'{kd:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
    ax3.set_title('Rank Correlation with Ground Truth', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c['label'] for c in correlations])
    ax3.legend()
    ax3.set_ylim(0.8, 1.0)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Top-K Overlap
    ax4 = fig.add_subplot(gs[1, 2])

    k_values = [5, 10, 25, 50, 100]

    overlap_data = []
    for k in k_values:
        hbbops_errors_10 = [p['hbbops_error'] for p in data_10x25['all_evaluated_prompts']]
        gt_errors_10 = [p['gt_error'] for p in data_10x25['all_evaluated_prompts']]
        overlap_10x25 = compute_top_k_overlap(hbbops_errors_10, gt_errors_10, min(k, len(hbbops_errors_10)))

        hbbops_errors_25 = [p['hbbops_error'] for p in data_25x25['all_evaluated_prompts']]
        gt_errors_25 = [p['gt_error'] for p in data_25x25['all_evaluated_prompts']]
        overlap_25x25 = compute_top_k_overlap(hbbops_errors_25, gt_errors_25, min(k, len(hbbops_errors_25)))

        overlap_data.append((k, overlap_25x25, overlap_10x25))

    x = np.arange(len(k_values))
    width = 0.35

    overlap_25x25_vals = [d[1] for d in overlap_data]
    overlap_10x25_vals = [d[2] for d in overlap_data]

    bars1 = ax4.bar(x - width/2, overlap_25x25_vals, width, label='25x25', color=color_25x25, alpha=0.7)
    bars2 = ax4.bar(x + width/2, overlap_10x25_vals, width, label='10x25', color=color_10x25, alpha=0.7)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_ylabel('Overlap (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Top-K Overlap (HbBoPs intersection GT)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('K', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'K={k}' for k in k_values])
    ax4.legend()
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle('HbBoPs Improved 4: Grid Size Comparison\n(Multi-Fidelity GP + Top 75% Fidelity Filtering)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved grid comparison to {output_path}")
    plt.close()


def calculate_all_metrics(data):
    """Calculate all comparison metrics for a method."""
    prompt_list = data['all_evaluated_prompts']

    hbbops_accs = [(1 - p['hbbops_error']) * 100 for p in prompt_list]
    gt_accs = [(1 - p['gt_error']) * 100 for p in prompt_list]
    diffs = [p['diff_pp'] for p in prompt_list]

    hbbops_accs = np.array(hbbops_accs)
    gt_accs = np.array(gt_accs)
    diffs = np.array(diffs)

    llm_calls = data.get('llm_calls', {})

    return {
        'n_evaluated': len(prompt_list),
        'n_total': data.get('total_prompts_in_grid', 250),
        'coverage_pct': len(prompt_list) / data.get('total_prompts_in_grid', 250) * 100,
        'llm_calls': llm_calls.get('hbbops_total', 0),
        'gt_calls': llm_calls.get('gt_total', 0),
        'efficiency': llm_calls.get('efficiency_ratio', 0),
        'mean_diff': np.mean(diffs),
        'mean_abs_diff': np.mean(np.abs(diffs)),
        'spearman_rho': spearmanr(hbbops_accs, gt_accs).statistic if len(hbbops_accs) > 1 else 0,
        'kendall_tau': kendalltau(hbbops_accs, gt_accs).statistic if len(hbbops_accs) > 1 else 0,
    }


def create_full_comparison_figure(all_data, all_metrics, output_path):
    """Create comprehensive comparison figure for all methods."""
    methods = list(all_data.keys())
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']  # Blue, Green, Orange, Pink, Purple

    fig = plt.figure(figsize=(18, 10))

    # Panel 1: Box plots by fidelity (top row)
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)

    all_fidelities = set()
    for data in all_data.values():
        diffs_by_fid = group_by_fidelity(data['all_evaluated_prompts'])
        all_fidelities.update(diffs_by_fid.keys())
    fidelities = sorted(all_fidelities)

    positions = []
    box_data = []
    box_colors = []

    for i, fid in enumerate(fidelities):
        for j, (method, data) in enumerate(all_data.items()):
            diffs_by_fid = group_by_fidelity(data['all_evaluated_prompts'])
            if fid in diffs_by_fid and len(diffs_by_fid[fid]) > 0:
                pos = i * (len(methods) + 1) + j
                positions.append(pos)
                box_data.append(diffs_by_fid[fid])
                box_colors.append(colors[j % len(colors)])

    if box_data:
        bp = ax1.boxplot(box_data, positions=positions, patch_artist=True, widths=0.8)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Fidelity Level', fontsize=12)
    ax1.set_ylabel('Diff (HbBoPs - GT) in pp', fontsize=12)
    ax1.set_title('Accuracy Difference by Fidelity Level', fontsize=14, fontweight='bold')

    tick_positions = [i * (len(methods) + 1) + len(methods)//2 for i in range(len(fidelities))]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([str(f) for f in fidelities], rotation=45, ha='right')

    legend_patches = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7) for i in range(len(methods))]
    ax1.legend(legend_patches, methods, loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Efficiency comparison (top right)
    ax2 = plt.subplot2grid((2, 3), (0, 2))

    x = np.arange(len(methods))
    efficiency = [all_metrics[m]['efficiency'] for m in methods]
    llm_calls = [all_metrics[m]['llm_calls'] / 1000 for m in methods]

    bars = ax2.bar(x, efficiency, color=colors[:len(methods)], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Efficiency (GT/HbBoPs calls)', fontsize=11)
    ax2.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)

    for i, (bar, calls) in enumerate(zip(bars, llm_calls)):
        ax2.annotate(f'{calls:.1f}K',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Rank correlation (bottom left)
    ax3 = plt.subplot2grid((2, 3), (1, 0))

    x = np.arange(len(methods))
    width = 0.35

    spearman = [all_metrics[m]['spearman_rho'] for m in methods]
    kendall = [all_metrics[m]['kendall_tau'] for m in methods]

    bars1 = ax3.bar(x - width/2, spearman, width, label='Spearman rho', color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, kendall, width, label='Kendall tau', color='coral', alpha=0.8)

    ax3.set_ylabel('Correlation Coefficient', fontsize=11)
    ax3.set_title('Rank Correlation with GT', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax3.legend(loc='lower right')
    ax3.set_ylim(0.85, 1.0)
    ax3.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    # Panel 4: Mean |Diff| comparison (bottom center)
    ax4 = plt.subplot2grid((2, 3), (1, 1))

    mean_abs_diffs = [all_metrics[m]['mean_abs_diff'] for m in methods]
    bars = ax4.bar(x, mean_abs_diffs, color=colors[:len(methods)], alpha=0.7, edgecolor='black')

    ax4.set_ylabel('Mean |Diff| (pp)', fontsize=11)
    ax4.set_title('Mean Absolute Difference', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)

    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel 5: Coverage (bottom right)
    ax5 = plt.subplot2grid((2, 3), (1, 2))

    coverage = [all_metrics[m]['coverage_pct'] for m in methods]
    bars = ax5.bar(x, coverage, color=colors[:len(methods)], alpha=0.7, edgecolor='black')

    ax5.set_ylabel('Coverage (%)', fontsize=11)
    ax5.set_title('Prompt Coverage', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)

    for bar, n in zip(bars, [all_metrics[m]['n_evaluated'] for m in methods]):
        height = bar.get_height()
        ax5.annotate(f'{n}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=10)
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3, axis='y')

    plt.suptitle('HbBoPs Methods Comprehensive Comparison (10x25 Grid)\nOriginal -> Improved -> Improved 2 -> Improved 3 -> Improved 4',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved full comparison to {output_path}")
    plt.close()


def main():
    base_dir = Path('/home/prusek/NLP')
    gt_path = base_dir / 'datasets' / 'hbbops' / 'full_grid_combined.jsonl'

    print("Loading ground truth...")
    gt_data = load_ground_truth(gt_path)

    # Instruction mappings
    instruction_mapping_10 = [8, 20, 15, 7, 14, 11, 3, 22, 18, 1]
    instruction_mapping_25 = list(range(25))
    exemplar_mapping = list(range(25))

    # Load improved_4 results
    print("\nLoading HbBoPs Improved 4 results...")

    # 10x25
    log_10x25 = base_dir / 'hbbops_improved_4' / 'results' / '10x25' / 'hbbops_improved4_20251208_095841.log'
    evaluations_10x25 = parse_log_for_evaluations(log_10x25)
    data_10x25 = build_comparison_data(evaluations_10x25, gt_data, instruction_mapping_10, exemplar_mapping, 10, 25)
    print(f"  10x25: {len(data_10x25['all_evaluated_prompts'])} prompts, {data_10x25['llm_calls']['hbbops_total']} LLM calls")

    # 25x25
    log_25x25 = base_dir / 'hbbops_improved_4' / 'results' / '25x25' / 'hbbops_improved4_20251208_095841.log'
    evaluations_25x25 = parse_log_for_evaluations(log_25x25)
    data_25x25 = build_comparison_data(evaluations_25x25, gt_data, instruction_mapping_25, exemplar_mapping, 25, 25)
    print(f"  25x25: {len(data_25x25['all_evaluated_prompts'])} prompts, {data_25x25['llm_calls']['hbbops_total']} LLM calls")

    # Create grid comparison
    print("\nCreating grid comparison plot...")
    grid_output = base_dir / 'hbbops_improved_4' / 'results' / 'hbbops_improved_4_grid_comparison.png'
    create_grid_comparison_plot(data_10x25, data_25x25, grid_output)

    # Load other methods for full comparison (10x25 only)
    print("\nLoading other HbBoPs methods for full comparison...")

    all_data = {}
    all_metrics = {}

    # Load existing comparison JSONs
    comparison_files = {
        'Original': base_dir / 'hbbops' / 'results' / '10x25_20251203_005743.json',
        'Improved': base_dir / 'hbbops_improved' / 'results' / '10x25_20251203_023331.json',
        'Improved 2': base_dir / 'hbbops_improved_2' / 'results' / '10x25_20251203_095921.json',
    }

    for name, path in comparison_files.items():
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            all_data[name] = data
            all_metrics[name] = calculate_all_metrics(data)
            print(f"  Loaded {name}: {all_metrics[name]['n_evaluated']} prompts")

    # Add Improved 3
    log_improved3 = base_dir / 'hbbops_improved_3' / 'results' / '10x25' / 'hbbops_improved3_20251207_235402.log'
    if log_improved3.exists():
        evaluations_improved3 = parse_log_for_evaluations(log_improved3)
        data_improved3 = build_comparison_data(evaluations_improved3, gt_data, instruction_mapping_10, exemplar_mapping, 10, 25)
        all_data['Improved 3'] = data_improved3
        all_metrics['Improved 3'] = calculate_all_metrics(data_improved3)
        print(f"  Added Improved 3: {all_metrics['Improved 3']['n_evaluated']} prompts")

    # Add Improved 4
    all_data['Improved 4'] = data_10x25
    all_metrics['Improved 4'] = calculate_all_metrics(data_10x25)
    print(f"  Added Improved 4: {all_metrics['Improved 4']['n_evaluated']} prompts")

    # Create full comparison
    print("\nCreating full comparison plot...")
    full_output = base_dir / 'visualize' / 'hbbops_full_comparison.png'
    create_full_comparison_figure(all_data, all_metrics, full_output)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Method':<15} {'Prompts':<10} {'LLM Calls':<12} {'Efficiency':<12} {'Spearman':<12} {'Mean|Diff|':<12}")
    print("-" * 90)
    for method in all_metrics:
        m = all_metrics[method]
        print(f"{method:<15} {m['n_evaluated']:<10} {m['llm_calls']:<12,} {m['efficiency']:<12.1f}x {m['spearman_rho']:<12.4f} {m['mean_abs_diff']:<12.2f}")
    print("=" * 90)


if __name__ == '__main__':
    main()
