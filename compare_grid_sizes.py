#!/usr/bin/env python3
"""Comparison of HbBoPs Improved 2 on different grid sizes (25x25 vs 10x25)."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict


def load_results(json_path: Path) -> dict:
    """Load comparison results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def calculate_ndcg(hbbops_ranks: list, gt_ranks: list, k: int = None) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    if k is None:
        k = len(hbbops_ranks)

    max_gt_rank = max(gt_ranks)
    relevance = [max_gt_rank - r + 1 for r in gt_ranks]

    sorted_pairs = sorted(zip(hbbops_ranks, relevance), key=lambda x: x[0])

    dcg = 0.0
    for i, (_, rel) in enumerate(sorted_pairs[:k]):
        dcg += rel / np.log2(i + 2)

    ideal_rels = sorted(relevance, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels[:k]):
        idcg += rel / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_top_k_overlap(hbbops_data: list, gt_data: dict, k: int) -> float:
    """Calculate Top-K Overlap: intersection of HbBoPs top-K and GT top-K."""
    gt_sorted = sorted(gt_data.items(), key=lambda x: x[1], reverse=True)
    gt_top_k = set(p[0] for p in gt_sorted[:k])

    hbbops_sorted = sorted(hbbops_data, key=lambda x: x['hbbops_acc'], reverse=True)
    hbbops_top_k = set(d['prompt_id'] for d in hbbops_sorted[:k])

    overlap = len(gt_top_k & hbbops_top_k)
    return overlap / k


def calculate_all_metrics(data: dict) -> dict:
    """Calculate all comparison metrics."""
    prompt_list = data['all_evaluated_prompts']

    hbbops_accs = []
    gt_accs = []
    diffs = []
    prompt_ids = []

    for p in prompt_list:
        hbbops_acc = (1 - p['hbbops_error']) * 100
        gt_acc = (1 - p['gt_error']) * 100
        hbbops_accs.append(hbbops_acc)
        gt_accs.append(gt_acc)
        diffs.append(p['diff_pp'])
        prompt_ids.append(f"({p['sel_inst']},{p['sel_ex']})")

    hbbops_accs = np.array(hbbops_accs)
    gt_accs = np.array(gt_accs)
    diffs = np.array(diffs)

    hbbops_ranks = stats.rankdata(-hbbops_accs)
    gt_ranks = stats.rankdata(-gt_accs)

    gt_data = {pid: acc for pid, acc in zip(prompt_ids, gt_accs)}

    prompt_comparisons = [
        {'prompt_id': pid, 'hbbops_acc': hacc, 'gt_acc': gacc}
        for pid, hacc, gacc in zip(prompt_ids, hbbops_accs, gt_accs)
    ]

    llm_calls = data.get('llm_calls', {})
    hbbops_calls = llm_calls.get('hbbops_total', 0)
    gt_calls = llm_calls.get('gt_total', 0)

    n_total = data.get('total_prompts_in_grid', 250)

    metrics = {
        'n_evaluated': len(prompt_list),
        'n_total': n_total,
        'coverage_pct': len(prompt_list) / n_total * 100,

        'llm_calls': hbbops_calls,
        'gt_calls': gt_calls,
        'efficiency': gt_calls / hbbops_calls if hbbops_calls > 0 else 0,

        'mean_diff': np.mean(diffs),
        'mean_abs_diff': np.mean(np.abs(diffs)),
        'median_abs_diff': np.median(np.abs(diffs)),
        'std_diff': np.std(diffs),
        'max_diff': np.max(diffs),
        'min_diff': np.min(diffs),

        'spearman_rho': stats.spearmanr(hbbops_accs, gt_accs).statistic,
        'kendall_tau': stats.kendalltau(hbbops_accs, gt_accs).statistic,

        'ndcg_10': calculate_ndcg(hbbops_ranks.tolist(), gt_ranks.tolist(), 10),
        'ndcg_25': calculate_ndcg(hbbops_ranks.tolist(), gt_ranks.tolist(), 25),
        'ndcg_50': calculate_ndcg(hbbops_ranks.tolist(), gt_ranks.tolist(), 50),
        'ndcg_all': calculate_ndcg(hbbops_ranks.tolist(), gt_ranks.tolist()),

        'overlap_5': calculate_top_k_overlap(prompt_comparisons, gt_data, 5),
        'overlap_10': calculate_top_k_overlap(prompt_comparisons, gt_data, 10),
        'overlap_25': calculate_top_k_overlap(prompt_comparisons, gt_data, 25),
        'overlap_50': calculate_top_k_overlap(prompt_comparisons, gt_data, 50),
        'overlap_100': calculate_top_k_overlap(prompt_comparisons, gt_data, min(100, len(prompt_list))),

        'best_hbbops': prompt_ids[np.argmax(hbbops_accs)],
        'best_gt': prompt_ids[np.argmax(gt_accs)],
    }

    return metrics


def get_diffs_by_fidelity(data: dict) -> dict:
    """Extract diff_pp values grouped by max fidelity level."""
    by_fidelity = defaultdict(list)
    for p in data['all_evaluated_prompts']:
        fidelity = p.get('max_fidelity', 0)
        by_fidelity[fidelity].append(p['diff_pp'])
    return dict(by_fidelity)


def print_summary_table(all_metrics: dict):
    """Print formatted comparison table."""
    methods = list(all_metrics.keys())

    print("\n" + "=" * 90)
    print("HbBoPs Improved 2: GRID SIZE COMPARISON (25x25 vs 10x25)")
    print("=" * 90)

    header = f"{'Metric':<35}" + "".join(f"{m:>27}" for m in methods)
    print(header)
    print("-" * 90)

    print(f"{'Grid Size (prompts)':<35}" + "".join(
        f"{all_metrics[m]['n_total']:>27,}" for m in methods))
    print(f"{'Prompts Evaluated':<35}" + "".join(
        f"{all_metrics[m]['n_evaluated']:>20} ({all_metrics[m]['coverage_pct']:.1f}%)"
        for m in methods))
    print(f"{'HbBoPs LLM Calls':<35}" + "".join(
        f"{all_metrics[m]['llm_calls']:>27,}" for m in methods))
    print(f"{'GT LLM Calls':<35}" + "".join(
        f"{all_metrics[m]['gt_calls']:>27,}" for m in methods))
    print(f"{'Efficiency (GT/HbBoPs)':<35}" + "".join(
        f"{all_metrics[m]['efficiency']:>27.2f}x" for m in methods))

    print("-" * 90)
    print("RANK CORRELATION")
    print(f"{'Spearman ρ':<35}" + "".join(
        f"{all_metrics[m]['spearman_rho']:>27.4f}" for m in methods))
    print(f"{'Kendall τ':<35}" + "".join(
        f"{all_metrics[m]['kendall_tau']:>27.4f}" for m in methods))

    print("-" * 90)
    print("NDCG (Normalized DCG)")
    for k in [10, 25, 50, 'all']:
        key = f'ndcg_{k}'
        label = f'NDCG@{k}'
        print(f"{label:<35}" + "".join(
            f"{all_metrics[m][key]:>27.4f}" for m in methods))

    print("-" * 90)
    print("TOP-K OVERLAP (HbBoPs top-K ∩ GT top-K)")
    for k in [5, 10, 25, 50, 100]:
        key = f'overlap_{k}'
        label = f'Top-{k} Overlap'
        print(f"{label:<35}" + "".join(
            f"{all_metrics[m][key]:>27.1%}" for m in methods))

    print("-" * 90)
    print("ERROR METRICS (pp)")
    print(f"{'Mean Diff':<35}" + "".join(
        f"{all_metrics[m]['mean_diff']:>27.2f}" for m in methods))
    print(f"{'Mean |Diff|':<35}" + "".join(
        f"{all_metrics[m]['mean_abs_diff']:>27.2f}" for m in methods))
    print(f"{'Median |Diff|':<35}" + "".join(
        f"{all_metrics[m]['median_abs_diff']:>27.2f}" for m in methods))
    print(f"{'Std Dev':<35}" + "".join(
        f"{all_metrics[m]['std_diff']:>27.2f}" for m in methods))

    print("-" * 90)
    print("BEST PROMPT")
    print(f"{'Best by HbBoPs':<35}" + "".join(
        f"{all_metrics[m]['best_hbbops']:>27}" for m in methods))
    print(f"{'Best by GT':<35}" + "".join(
        f"{all_metrics[m]['best_gt']:>27}" for m in methods))
    print(f"{'Match?':<35}" + "".join(
        f"{'✓ YES':>27}" if all_metrics[m]['best_hbbops'] == all_metrics[m]['best_gt']
        else f"{'✗ NO':>27}" for m in methods))

    print("=" * 90)


def create_comparison_figure(all_data: dict, all_metrics: dict, output_path: Path):
    """Create comparison figure (2x3 grid)."""
    methods = list(all_data.keys())
    colors = ['#9C27B0', '#FF5722']  # Purple for 25x25, Deep Orange for 10x25

    fig = plt.figure(figsize=(18, 10))

    # 1. Box plots by fidelity (top row, spans 2 columns)
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)

    all_fidelities = set()
    for data in all_data.values():
        diffs_by_fid = get_diffs_by_fidelity(data)
        all_fidelities.update(diffs_by_fid.keys())
    fidelities = sorted(all_fidelities)

    positions = []
    box_data = []
    box_colors = []

    for i, fid in enumerate(fidelities):
        for j, (method, data) in enumerate(all_data.items()):
            diffs_by_fid = get_diffs_by_fidelity(data)
            if fid in diffs_by_fid:
                pos = i * 3 + j
                positions.append(pos)
                box_data.append(diffs_by_fid[fid])
                box_colors.append(colors[j])

    bp = ax1.boxplot(box_data, positions=positions, patch_artist=True, widths=0.8)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Fidelity Level', fontsize=12)
    ax1.set_ylabel('Diff (HbBoPs - GT) in pp', fontsize=12)
    ax1.set_title('Accuracy Difference by Fidelity Level', fontsize=14, fontweight='bold')

    tick_positions = [i * 3 + 0.5 for i in range(len(fidelities))]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(fidelities)

    legend_patches = [plt.Rectangle((0,0),1,1, facecolor=c, alpha=0.7) for c in colors]
    ax1.legend(legend_patches, methods, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Efficiency comparison (top right)
    ax2 = plt.subplot2grid((2, 3), (0, 2))

    x = np.arange(len(methods))
    efficiency = [all_metrics[m]['efficiency'] for m in methods]
    llm_calls = [all_metrics[m]['llm_calls'] / 1000 for m in methods]

    bars = ax2.bar(x, efficiency, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Efficiency (GT calls / HbBoPs calls)', fontsize=11)
    ax2.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right')

    for i, (bar, eff, calls) in enumerate(zip(bars, efficiency, llm_calls)):
        ax2.annotate(f'{eff:.1f}x\n({calls:.1f}K)',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Rank correlation metrics (bottom left)
    ax3 = plt.subplot2grid((2, 3), (1, 0))

    x = np.arange(len(methods))
    width = 0.35

    spearman = [all_metrics[m]['spearman_rho'] for m in methods]
    kendall = [all_metrics[m]['kendall_tau'] for m in methods]

    bars1 = ax3.bar(x - width/2, spearman, width, label='Spearman ρ', color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, kendall, width, label='Kendall τ', color='coral', alpha=0.8)

    ax3.set_ylabel('Correlation Coefficient', fontsize=11)
    ax3.set_title('Rank Correlation with Ground Truth', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=15, ha='right')
    ax3.legend(loc='lower right')
    ax3.set_ylim(0.8, 1.0)
    ax3.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    # 4. NDCG comparison (bottom center)
    ax4 = plt.subplot2grid((2, 3), (1, 1))

    x = np.arange(4)
    width = 0.35
    k_values = ['@10', '@25', '@50', '@all']

    ndcg_bars = []
    for i, (method, color) in enumerate(zip(methods, colors)):
        ndcg_vals = [all_metrics[method][f'ndcg_{k.replace("@", "")}'] for k in k_values]
        bars = ax4.bar(x + i*width - width/2, ndcg_vals, width, label=method, color=color, alpha=0.8)
        ndcg_bars.append((bars, ndcg_vals))

    ax4.set_ylabel('NDCG Score', fontsize=11)
    ax4.set_title('NDCG at Various K', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_values)
    ax4.legend(loc='lower right')
    ax4.set_ylim(0.95, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')

    for bars, vals in ndcg_bars:
        for bar, val in zip(bars, vals):
            ax4.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, rotation=90)

    # 5. Overlap@K (bottom right)
    ax5 = plt.subplot2grid((2, 3), (1, 2))

    x = np.arange(5)
    width = 0.35
    k_values = [5, 10, 25, 50, 100]

    overlap_bars = []
    for i, (method, color) in enumerate(zip(methods, colors)):
        overlap_vals = [all_metrics[method][f'overlap_{k}'] * 100 for k in k_values]
        bars = ax5.bar(x + i*width - width/2, overlap_vals, width, label=method, color=color, alpha=0.8)
        overlap_bars.append((bars, overlap_vals))

    ax5.set_ylabel('Overlap (%)', fontsize=11)
    ax5.set_title('Top-K Overlap (HbBoPs ∩ GT)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'K={k}' for k in k_values])
    ax5.legend(loc='lower right')
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3, axis='y')

    for bars, vals in overlap_bars:
        for bar, val in zip(bars, vals):
            ax5.annotate(f'{val:.0f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle('HbBoPs Improved 2: Grid Size Comparison\n(25×25 = 625 prompts vs 10×25 = 250 prompts)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved to: {output_path}")
    plt.close()


def main():
    results_files = {
        '25×25 (625)': Path('/home/prusek/NLP/hbbops_improved_2/results/25x25_20251203_154129.json'),
        '10×25 (250)': Path('/home/prusek/NLP/hbbops_improved_2/results/10x25_20251203_095921.json'),
    }

    all_data = {}
    all_metrics = {}

    print("Loading results...")
    for name, path in results_files.items():
        if path.exists():
            all_data[name] = load_results(path)
            all_metrics[name] = calculate_all_metrics(all_data[name])
            print(f"  ✓ Loaded {name}: {all_metrics[name]['n_evaluated']} prompts evaluated")
        else:
            print(f"  ✗ File not found: {path}")

    if not all_data:
        print("No data loaded!")
        return

    print_summary_table(all_metrics)

    output_path = Path('/home/prusek/NLP/hbbops_grid_comparison.png')
    create_comparison_figure(all_data, all_metrics, output_path)


if __name__ == '__main__':
    main()
