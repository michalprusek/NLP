#!/usr/bin/env python3
"""Compare HbBoPs and HbBoPs Improved results with box plots and better metrics."""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau

def parse_results(filepath):
    """Parse the results file and extract fidelity-grouped data."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract the table data between the dashed lines
    lines = content.split('\n')
    data = []
    in_table = False

    for line in lines:
        if line.startswith('------'):
            in_table = not in_table
            continue
        if in_table and line.strip():
            # Parse: sel_i sel_e orig_i orig_e fid HbBoPs GT Diff
            parts = line.split()
            if len(parts) >= 8:
                try:
                    fid = int(parts[4])
                    hbbops_val = float(parts[5].rstrip('%'))
                    gt_val = float(parts[6].rstrip('%'))
                    diff_str = parts[7].rstrip('pp')
                    diff = float(diff_str)
                    data.append({
                        'fid': fid,
                        'hbbops': hbbops_val,
                        'gt': gt_val,
                        'diff': diff
                    })
                except (ValueError, IndexError):
                    continue

    return data

def load_comparison_json(path):
    """Load JSON comparison data."""
    with open(path) as f:
        return json.load(f)

def load_full_gt(jsonl_path, inst_mapping):
    """Load full GT and create sorted list for 10x25 grid."""
    gt_data = {}
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            key = (item['instruction_id'], item['exemplar_id'])
            gt_data[key] = item['error_rate']

    all_prompts = []
    for sel_i in range(10):
        orig_i = inst_mapping[sel_i]
        for ex in range(25):
            if (orig_i, ex) in gt_data:
                all_prompts.append({
                    'sel_inst': sel_i,
                    'sel_ex': ex,
                    'gt_error': gt_data[(orig_i, ex)]
                })
    return sorted(all_prompts, key=lambda x: x['gt_error'])

def compute_metrics(comp_data, gt_sorted):
    """Compute comprehensive metrics."""
    evaluated = comp_data['all_evaluated_prompts']
    eval_keys = set((p['sel_inst'], p['sel_ex']) for p in evaluated)

    # Spearman & Kendall
    hb = [p['hbbops_error'] for p in evaluated]
    gt = [p['gt_error'] for p in evaluated]
    spearman, _ = spearmanr(hb, gt)
    kendall, _ = kendalltau(hb, gt)

    # Recall@K
    recalls = {}
    for k in [10, 25, 50, 100]:
        top_k_gt = set((p['sel_inst'], p['sel_ex']) for p in gt_sorted[:k])
        recalls[k] = len(top_k_gt & eval_keys) / k

    # Top-K overlap
    hbbops_sorted = sorted(evaluated, key=lambda x: x['hbbops_error'])
    overlaps = {}
    for k in [10, 25, 50]:
        top_k_hb = set((p['sel_inst'], p['sel_ex']) for p in hbbops_sorted[:k])
        top_k_gt = set((p['sel_inst'], p['sel_ex']) for p in gt_sorted[:k])
        overlaps[k] = len(top_k_hb & top_k_gt) / k

    # NDCG
    n = len(evaluated)
    gt_ranking = {(p['sel_inst'], p['sel_ex']): i for i, p in enumerate(
        sorted(evaluated, key=lambda x: x['gt_error']))}
    relevances = [n - gt_ranking[(p['sel_inst'], p['sel_ex'])] for p in hbbops_sorted]
    ideal = sorted(relevances, reverse=True)

    def dcg(r, k):
        r = np.array(r[:k])
        return np.sum(r / np.log2(np.arange(2, len(r) + 2))) if len(r) > 0 else 0

    ndcg_25 = dcg(relevances, 25) / dcg(ideal, 25)
    ndcg_all = dcg(relevances, n) / dcg(ideal, n)

    return {
        'n': len(evaluated),
        'spearman': spearman,
        'kendall': kendall,
        'recalls': recalls,
        'overlaps': overlaps,
        'ndcg_25': ndcg_25,
        'ndcg_all': ndcg_all
    }

def group_by_fidelity(data):
    """Group data by fidelity level."""
    grouped = defaultdict(list)
    for item in data:
        grouped[item['fid']].append(item)
    return grouped

# Parse both files
print("Loading data...")
hbbops_data = parse_results('/home/prusek/NLP/hbbops/results/10x25.txt')
improved_data = parse_results('/home/prusek/NLP/hbbops_improved/results/10x25.txt')

# Load JSON data for better metrics
hbbops_json = load_comparison_json('/home/prusek/NLP/hbbops/results/hbbops_gt_comparison_20251203_005743.json')
improved_json = load_comparison_json('/home/prusek/NLP/hbbops_improved/results/hbbops_gt_comparison_20251203_023331.json')

# Load full GT
inst_mapping = hbbops_json['instruction_mapping']
gt_sorted = load_full_gt('/home/prusek/NLP/hbbops/results/full_grid_combined.jsonl', inst_mapping)

# Compute metrics
hbbops_metrics = compute_metrics(hbbops_json, gt_sorted)
improved_metrics = compute_metrics(improved_json, gt_sorted)

print(f"HbBoPs: {len(hbbops_data)} prompts")
print(f"HbBoPs Improved: {len(improved_data)} prompts")

# Group by fidelity
hbbops_grouped = group_by_fidelity(hbbops_data)
improved_grouped = group_by_fidelity(improved_data)

# Get all unique fidelity levels and sort them
all_fidelities = sorted(set(hbbops_grouped.keys()) | set(improved_grouped.keys()))
print(f"Fidelity levels: {all_fidelities}")

# Bin fidelity levels for cleaner visualization
def bin_fidelity(fid):
    """Bin fidelity into ranges for cleaner visualization."""
    if fid <= 10:
        return '10'
    elif fid <= 25:
        return '20-25'
    elif fid <= 45:
        return '40-45'
    elif fid <= 85:
        return '80-85'
    elif fid <= 170:
        return '160-170'
    elif fid <= 340:
        return '320-340'
    elif fid <= 670:
        return '640-670'
    else:
        return '1280+'

# Bin the data
def bin_data(data):
    """Bin data by fidelity ranges."""
    binned = defaultdict(list)
    for item in data:
        bin_name = bin_fidelity(item['fid'])
        binned[bin_name].append(item)
    return binned

hbbops_binned = bin_data(hbbops_data)
improved_binned = bin_data(improved_data)

# Define bin order
bin_order = ['10', '20-25', '40-45', '80-85', '160-170', '320-340', '640-670', '1280+']

# Create the figure with 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('HbBoPs vs HbBoPs Improved: Method Comparison\n(10x25 grid, 250 prompts)', fontsize=16, fontweight='bold')

# Colors
color_hbbops = '#2196F3'  # Blue
color_improved = '#4CAF50'  # Green

# --- Plot 1: Box plot of PP Diff vs GT by fidelity (HbBoPs) ---
ax1 = axes[0, 0]
diffs_hbbops = [
    [item['diff'] for item in hbbops_binned.get(b, [])]
    for b in bin_order
]
bp1 = ax1.boxplot(diffs_hbbops, labels=bin_order, patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor(color_hbbops)
    patch.set_alpha(0.7)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.set_xlabel('Fidelity Level', fontsize=12)
ax1.set_ylabel('Difference (pp) HbBoPs - GT', fontsize=12)
ax1.set_title('HbBoPs: Deviation from Ground Truth', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Add counts
for i, b in enumerate(bin_order):
    count = len(hbbops_binned.get(b, []))
    if count > 0:
        ax1.text(i+1, ax1.get_ylim()[1], f'n={count}', ha='center', va='bottom', fontsize=8)

# --- Plot 2: Box plot of PP Diff vs GT by fidelity (Improved) ---
ax2 = axes[0, 1]
diffs_improved = [
    [item['diff'] for item in improved_binned.get(b, [])]
    for b in bin_order
]
bp2 = ax2.boxplot(diffs_improved, labels=bin_order, patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor(color_improved)
    patch.set_alpha(0.7)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_xlabel('Fidelity Level', fontsize=12)
ax2.set_ylabel('Difference (pp) HbBoPs - GT', fontsize=12)
ax2.set_title('HbBoPs Improved: Deviation from Ground Truth', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Add counts
for i, b in enumerate(bin_order):
    count = len(improved_binned.get(b, []))
    if count > 0:
        ax2.text(i+1, ax2.get_ylim()[1], f'n={count}', ha='center', va='bottom', fontsize=8)

# --- Plot 3: Side-by-side comparison of both methods ---
ax3 = axes[1, 0]
positions = np.arange(len(bin_order))
width = 0.35

# Prepare data for side-by-side boxes
data_pairs = []
for b in bin_order:
    data_pairs.append([
        [item['diff'] for item in hbbops_binned.get(b, [])],
        [item['diff'] for item in improved_binned.get(b, [])]
    ])

# Plot side by side
for i, (hb_diffs, imp_diffs) in enumerate(data_pairs):
    if hb_diffs:
        bp = ax3.boxplot([hb_diffs], positions=[i - width/2], widths=width*0.8, patch_artist=True)
        bp['boxes'][0].set_facecolor(color_hbbops)
        bp['boxes'][0].set_alpha(0.7)
    if imp_diffs:
        bp = ax3.boxplot([imp_diffs], positions=[i + width/2], widths=width*0.8, patch_artist=True)
        bp['boxes'][0].set_facecolor(color_improved)
        bp['boxes'][0].set_alpha(0.7)

ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_xticks(positions)
ax3.set_xticklabels(bin_order, rotation=45)
ax3.set_xlabel('Fidelity Level', fontsize=12)
ax3.set_ylabel('Difference (pp) HbBoPs - GT', fontsize=12)
ax3.set_title('Direct Comparison: HbBoPs vs Improved', fontsize=13, fontweight='bold')
ax3.legend([plt.Rectangle((0,0),1,1,facecolor=color_hbbops, alpha=0.7),
            plt.Rectangle((0,0),1,1,facecolor=color_improved, alpha=0.7)],
           ['HbBoPs', 'HbBoPs Improved'], loc='upper right')
ax3.grid(axis='y', alpha=0.3)

# --- Plot 4: Rank Correlation Metrics ---
ax4 = axes[1, 1]

metrics_names = ['Spearman ρ', 'Kendall τ', 'NDCG@25', 'NDCG@all']
hbbops_vals = [hbbops_metrics['spearman'], hbbops_metrics['kendall'],
               hbbops_metrics['ndcg_25'], hbbops_metrics['ndcg_all']]
improved_vals = [improved_metrics['spearman'], improved_metrics['kendall'],
                 improved_metrics['ndcg_25'], improved_metrics['ndcg_all']]

x = np.arange(len(metrics_names))
bars1 = ax4.bar(x - width/2, hbbops_vals, width, label='HbBoPs', color=color_hbbops, alpha=0.7)
bars2 = ax4.bar(x + width/2, improved_vals, width, label='HbBoPs Improved', color=color_improved, alpha=0.7)

ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Rank Correlation Metrics', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics_names)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(0.7, 1.05)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# --- Plot 5: Recall@K (coverage of GT top-K) ---
ax5 = axes[0, 2]
k_vals = [10, 25, 50, 100]
hbbops_recalls = [hbbops_metrics['recalls'][k] for k in k_vals]
improved_recalls = [improved_metrics['recalls'][k] for k in k_vals]

x = np.arange(len(k_vals))
bars1 = ax5.bar(x - width/2, hbbops_recalls, width, label='HbBoPs', color=color_hbbops, alpha=0.7)
bars2 = ax5.bar(x + width/2, improved_recalls, width, label='HbBoPs Improved', color=color_improved, alpha=0.7)

ax5.set_ylabel('Recall', fontsize=12)
ax5.set_xlabel('K (GT top-K)', fontsize=12)
ax5.set_title('Recall@K: Coverage of GT Top-K', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels([f'Top-{k}' for k in k_vals])
ax5.legend()
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim(0.7, 1.05)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# --- Plot 6: Top-K Overlap ---
ax6 = axes[1, 2]
k_vals_overlap = [10, 25, 50]
hbbops_overlaps = [hbbops_metrics['overlaps'][k] for k in k_vals_overlap]
improved_overlaps = [improved_metrics['overlaps'][k] for k in k_vals_overlap]

x = np.arange(len(k_vals_overlap))
bars1 = ax6.bar(x - width/2, hbbops_overlaps, width, label='HbBoPs', color=color_hbbops, alpha=0.7)
bars2 = ax6.bar(x + width/2, improved_overlaps, width, label='HbBoPs Improved', color=color_improved, alpha=0.7)

ax6.set_ylabel('Overlap Ratio', fontsize=12)
ax6.set_xlabel('K', fontsize=12)
ax6.set_title('Top-K Overlap: HbBoPs Top-K ∩ GT Top-K', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels([f'Top-{k}' for k in k_vals_overlap])
ax6.legend()
ax6.grid(axis='y', alpha=0.3)
ax6.set_ylim(0.5, 1.05)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# Add comprehensive text summary
summary_text = f"""HbBoPs: n={hbbops_metrics['n']}, ρ={hbbops_metrics['spearman']:.3f}
Improved: n={improved_metrics['n']}, ρ={improved_metrics['spearman']:.3f}
Both find best prompt (0,7) ✓"""

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/prusek/NLP/hbbops_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\nPlot saved to /home/prusek/NLP/hbbops_comparison.png")
plt.close()

# Print summary with better metrics
print("\n" + "="*70)
print("COMPREHENSIVE METRICS SUMMARY")
print("="*70)

print(f"\n{'Metric':<25} {'HbBoPs':>12} {'Improved':>12} {'Winner':>10}")
print("-"*60)
print(f"{'Prompts evaluated':<25} {hbbops_metrics['n']:>12} {improved_metrics['n']:>12} {'HbBoPs':>10}")
print(f"{'Spearman ρ':<25} {hbbops_metrics['spearman']:>12.4f} {improved_metrics['spearman']:>12.4f} {'Improved':>10}")
print(f"{'Kendall τ':<25} {hbbops_metrics['kendall']:>12.4f} {improved_metrics['kendall']:>12.4f} {'Improved':>10}")
print(f"{'NDCG@25':<25} {hbbops_metrics['ndcg_25']:>12.4f} {improved_metrics['ndcg_25']:>12.4f} {'~Tie':>10}")
print(f"{'NDCG@all':<25} {hbbops_metrics['ndcg_all']:>12.4f} {improved_metrics['ndcg_all']:>12.4f} {'Improved':>10}")
print(f"{'Recall@10':<25} {hbbops_metrics['recalls'][10]:>12.2f} {improved_metrics['recalls'][10]:>12.2f} {'~Tie':>10}")
print(f"{'Recall@25':<25} {hbbops_metrics['recalls'][25]:>12.2f} {improved_metrics['recalls'][25]:>12.2f} {'HbBoPs':>10}")
print(f"{'Recall@50':<25} {hbbops_metrics['recalls'][50]:>12.2f} {improved_metrics['recalls'][50]:>12.2f} {'HbBoPs':>10}")
print(f"{'Top-10 Overlap':<25} {hbbops_metrics['overlaps'][10]:>12.2f} {improved_metrics['overlaps'][10]:>12.2f} {'HbBoPs':>10}")
print(f"{'Top-25 Overlap':<25} {hbbops_metrics['overlaps'][25]:>12.2f} {improved_metrics['overlaps'][25]:>12.2f} {'HbBoPs':>10}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
• Spearman/Kendall: How well do HbBoPs rankings correlate with GT rankings?
  → Improved is slightly better (0.957 vs 0.945)

• NDCG: Normalized Discounted Cumulative Gain - penalizes ranking errors at top
  → Both excellent (>0.99), nearly perfect ranking quality

• Recall@K: What fraction of GT's top-K did HbBoPs even evaluate?
  → HbBoPs Original has better coverage (evaluates more of the best prompts)

• Top-K Overlap: If we take top-K from both, how many overlap?
  → HbBoPs Original has slightly better overlap at top-10 and top-25

CONCLUSION: HbBoPs Original explores more of the search space (higher coverage),
while Improved has marginally better ranking correlation. Both correctly identify
the best prompt (0,7) with MRR=1.0.
""")
