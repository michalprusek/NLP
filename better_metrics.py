#!/usr/bin/env python3
"""Better metrics for HbBoPs vs GT comparison."""

import json
import numpy as np
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict

def load_comparison_data(json_path):
    """Load comparison JSON with evaluated prompts."""
    with open(json_path) as f:
        return json.load(f)

def load_full_gt(jsonl_path):
    """Load full GT grid data."""
    gt_data = {}
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            key = (item['instruction_id'], item['exemplar_id'])
            gt_data[key] = item['error_rate']
    return gt_data

# Load data
print("Loading data...")
hbbops_comp = load_comparison_data('/home/prusek/NLP/hbbops/results/hbbops_gt_comparison_20251203_005743.json')
improved_comp = load_comparison_data('/home/prusek/NLP/hbbops_improved/results/hbbops_gt_comparison_20251203_023331.json')

# Load full GT for all 250 prompts
full_gt = load_full_gt('/home/prusek/NLP/hbbops/results/full_grid_combined.jsonl')

# Get instruction mapping from the comparison file
inst_mapping = hbbops_comp['instruction_mapping']  # [8, 20, 15, 7, 14, 11, 3, 22, 18, 1]

print(f"Full GT has {len(full_gt)} prompts")
print(f"Instruction mapping (sel -> orig): {inst_mapping}")

# Create full GT list for the 10x25 grid (mapped instructions)
all_gt_prompts = []
for sel_i in range(10):
    orig_i = inst_mapping[sel_i]
    for ex in range(25):
        if (orig_i, ex) in full_gt:
            all_gt_prompts.append({
                'sel_inst': sel_i,
                'sel_ex': ex,
                'orig_inst': orig_i,
                'orig_ex': ex,
                'gt_error': full_gt[(orig_i, ex)]
            })

print(f"Grid prompts (10x25): {len(all_gt_prompts)}")

# Sort GT prompts by error (best first)
gt_sorted = sorted(all_gt_prompts, key=lambda x: x['gt_error'])

def compute_metrics(comparison_data, name):
    """Compute comprehensive metrics for one method."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)

    evaluated = comparison_data['all_evaluated_prompts']

    # Create lookup for evaluated prompts
    eval_lookup = {}
    for p in evaluated:
        key = (p['sel_inst'], p['sel_ex'])
        eval_lookup[key] = p

    print(f"Evaluated prompts: {len(evaluated)} / 250 ({100*len(evaluated)/250:.1f}%)")

    # 1. SPEARMAN on evaluated prompts (already computed, verify)
    hbbops_errors = [p['hbbops_error'] for p in evaluated]
    gt_errors = [p['gt_error'] for p in evaluated]
    spearman, _ = spearmanr(hbbops_errors, gt_errors)
    kendall, _ = kendalltau(hbbops_errors, gt_errors)
    print(f"\nRank Correlation (on {len(evaluated)} evaluated):")
    print(f"  Spearman ρ: {spearman:.4f}")
    print(f"  Kendall τ:  {kendall:.4f}")

    # 2. RECALL@K - How many of top-K GT prompts were evaluated?
    print(f"\nRecall@K (coverage of GT top-K):")
    for k in [10, 25, 50, 100, 177]:
        if k > len(gt_sorted):
            continue
        top_k_gt = set((p['sel_inst'], p['sel_ex']) for p in gt_sorted[:k])
        evaluated_keys = set((p['sel_inst'], p['sel_ex']) for p in evaluated)
        recall = len(top_k_gt & evaluated_keys) / k
        print(f"  Recall@{k:3d}: {recall:.3f} ({len(top_k_gt & evaluated_keys)}/{k})")

    # 3. TOP-K OVERLAP - Intersection of top-K from both rankings
    print(f"\nTop-K Overlap (HbBoPs top-K ∩ GT top-K):")
    hbbops_sorted = sorted(evaluated, key=lambda x: x['hbbops_error'])
    for k in [5, 10, 25, 50]:
        if k > len(hbbops_sorted):
            continue
        top_k_hbbops = set((p['sel_inst'], p['sel_ex']) for p in hbbops_sorted[:k])
        top_k_gt = set((p['sel_inst'], p['sel_ex']) for p in gt_sorted[:k])
        overlap = len(top_k_hbbops & top_k_gt)
        print(f"  Top-{k:2d} overlap: {overlap}/{k} ({100*overlap/k:.1f}%)")

    # 4. Mean Reciprocal Rank of GT best in HbBoPs ranking
    gt_best_key = (gt_sorted[0]['sel_inst'], gt_sorted[0]['sel_ex'])
    if gt_best_key in eval_lookup:
        hbbops_sorted_keys = [(p['sel_inst'], p['sel_ex']) for p in hbbops_sorted]
        rank = hbbops_sorted_keys.index(gt_best_key) + 1
        print(f"\nGT Best Prompt Rank in HbBoPs:")
        print(f"  GT best: inst={gt_best_key[0]}, ex={gt_best_key[1]} (error={gt_sorted[0]['gt_error']:.4f})")
        print(f"  Rank in HbBoPs: {rank} / {len(evaluated)}")
        print(f"  MRR: {1/rank:.4f}")
    else:
        print(f"\n⚠ GT best prompt not evaluated by HbBoPs!")

    # 5. NDCG (Normalized Discounted Cumulative Gain)
    # Using GT ranking as ground truth
    def dcg_at_k(relevances, k):
        """Compute DCG@k."""
        relevances = np.array(relevances[:k])
        if len(relevances) == 0:
            return 0.0
        return np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2)))

    def ndcg_at_k(predicted_relevances, ideal_relevances, k):
        """Compute NDCG@k."""
        dcg = dcg_at_k(predicted_relevances, k)
        idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
        return dcg / idcg if idcg > 0 else 0.0

    # Convert GT error to relevance (lower error = higher relevance)
    # Use rank-based relevance: best = N, worst = 1
    n = len(evaluated)
    gt_ranking = {(p['sel_inst'], p['sel_ex']): i for i, p in enumerate(
        sorted(evaluated, key=lambda x: x['gt_error']))}

    # Get relevances in HbBoPs order
    hbbops_ordered_relevances = []
    for p in hbbops_sorted:
        key = (p['sel_inst'], p['sel_ex'])
        rank = gt_ranking[key]
        relevance = n - rank  # Higher relevance for better GT rank
        hbbops_ordered_relevances.append(relevance)

    ideal_relevances = sorted(hbbops_ordered_relevances, reverse=True)

    print(f"\nNDCG (ranking quality):")
    for k in [10, 25, 50, 100, n]:
        if k > n:
            k = n
        ndcg = ndcg_at_k(hbbops_ordered_relevances, ideal_relevances, k)
        print(f"  NDCG@{k:3d}: {ndcg:.4f}")

    # 6. Error at different percentiles
    print(f"\nBest prompt by each method:")
    hbbops_best = hbbops_sorted[0]
    gt_best_evaluated = sorted(evaluated, key=lambda x: x['gt_error'])[0]
    print(f"  HbBoPs best: ({hbbops_best['sel_inst']},{hbbops_best['sel_ex']}) "
          f"HbBoPs={hbbops_best['hbbops_error']:.4f}, GT={hbbops_best['gt_error']:.4f}")
    print(f"  GT best (eval): ({gt_best_evaluated['sel_inst']},{gt_best_evaluated['sel_ex']}) "
          f"HbBoPs={gt_best_evaluated['hbbops_error']:.4f}, GT={gt_best_evaluated['gt_error']:.4f}")

    # 7. Rank correlation only on top-K
    print(f"\nSpearman on Top-K only:")
    for k in [25, 50, 100]:
        if k > len(hbbops_sorted):
            continue
        top_k = hbbops_sorted[:k]
        hb_errors = [p['hbbops_error'] for p in top_k]
        gt_errors = [p['gt_error'] for p in top_k]
        sp, _ = spearmanr(hb_errors, gt_errors)
        print(f"  Spearman (top-{k}): {sp:.4f}")

    return {
        'spearman': spearman,
        'kendall': kendall,
        'n_evaluated': len(evaluated)
    }

# Compute metrics for both methods
hbbops_metrics = compute_metrics(hbbops_comp, "HbBoPs Original")
improved_metrics = compute_metrics(improved_comp, "HbBoPs Improved")

print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)
print(f"                      HbBoPs    Improved")
print(f"  Spearman:           {hbbops_metrics['spearman']:.4f}    {improved_metrics['spearman']:.4f}")
print(f"  Kendall:            {hbbops_metrics['kendall']:.4f}    {improved_metrics['kendall']:.4f}")
print(f"  Evaluated:          {hbbops_metrics['n_evaluated']}       {improved_metrics['n_evaluated']}")
