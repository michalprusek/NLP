#!/usr/bin/env python3
"""
Compare acquisition functions for flow-guided Bayesian optimization.

Evaluates UCB, LCB, EI, PI on gradient quality metrics:
1. Gradient improvement rate (does following gradient improve acquisition?)
2. Gradient magnitude statistics
3. Gradient stability (variance across samples)

Thompson Sampling is evaluated separately for batch selection.

Usage:
    uv run python scripts/compare_acquisition_functions.py \
        --instructions datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt \
        --output results/acquisition_comparison.json
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

from src.ecoflow.gp_surrogate import SonarGPSurrogate
from src.ecoflow.acquisition_functions import (
    compute_acquisition_gradient_metrics,
    thompson_sampling_select,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_embeddings(path: str, device: str = "cuda"):
    """Load instruction embeddings and accuracies."""
    data = torch.load(path, weights_only=False)
    return (
        data['embeddings'].to(device),
        data['accuracies'].to(device),
        data.get('instructions', [])
    )


def run_acquisition_comparison(
    X: torch.Tensor,
    y: torch.Tensor,
    n_folds: int = 5,
    alphas: list = None,
    device: str = "cuda",
) -> dict:
    """
    Run k-fold comparison of acquisition functions.

    Returns dict with metrics for each acquisition function across folds.
    """
    if alphas is None:
        alphas = [0.5, 1.0, 1.96, 2.0]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {
        'n_samples': X.shape[0],
        'n_folds': n_folds,
        'alphas': alphas,
        'acquisitions': {},
    }

    all_fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X.cpu().numpy())):
        logger.info(f"Fold {fold + 1}/{n_folds}")

        train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
        test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

        train_X, train_y = X[train_idx], y[train_idx]
        test_X, test_y = X[test_idx], y[test_idx]

        # Fit GP
        gp = SonarGPSurrogate(D=X.shape[1], device=device)
        gp.fit(train_X, train_y)

        fold_results = {}
        for alpha in alphas:
            metrics = compute_acquisition_gradient_metrics(
                gp.model, test_X, test_y, alpha=alpha
            )
            fold_results[f'alpha_{alpha}'] = metrics

        all_fold_results.append(fold_results)

    # Aggregate across folds
    for alpha in alphas:
        alpha_key = f'alpha_{alpha}'
        for acq in ['ucb', 'lcb', 'ei', 'pi']:
            values = [r[alpha_key][acq] for r in all_fold_results]

            results['acquisitions'][f'{acq}_{alpha_key}'] = {
                'gradient_norm_mean': {
                    'mean': float(np.mean([v['gradient_norm_mean'] for v in values])),
                    'std': float(np.std([v['gradient_norm_mean'] for v in values])),
                },
                'gradient_improvement_rate': {
                    'mean': float(np.mean([v['gradient_improvement_rate'] for v in values])),
                    'std': float(np.std([v['gradient_improvement_rate'] for v in values])),
                },
            }

    return results


def evaluate_thompson_sampling(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_sizes: list = None,
    n_trials: int = 10,
    device: str = "cuda",
) -> dict:
    """
    Evaluate Thompson Sampling for batch selection.

    Measures: selected batch quality (mean accuracy of selected points)
    """
    if batch_sizes is None:
        batch_sizes = [4, 8, 16]

    # Use first half for training, second half as "candidates"
    n_train = X.shape[0] // 2
    train_X, train_y = X[:n_train], y[:n_train]
    cand_X, cand_y = X[n_train:], y[n_train:]

    # Fit GP
    gp = SonarGPSurrogate(D=X.shape[1], device=device)
    gp.fit(train_X, train_y)

    results = {}
    for batch_size in batch_sizes:
        batch_qualities = []

        for trial in range(n_trials):
            selected_idx = thompson_sampling_select(
                gp.model, cand_X, n_select=batch_size, seed=42 + trial
            )
            selected_y = cand_y[selected_idx]
            batch_qualities.append(float(selected_y.mean()))

        results[f'batch_{batch_size}'] = {
            'mean_selected_accuracy': {
                'mean': float(np.mean(batch_qualities)),
                'std': float(np.std(batch_qualities)),
            },
            'baseline_random': float(cand_y.mean()),  # Random selection baseline
        }

    return results


def plot_acquisition_comparison(results: dict, output_path: str) -> None:
    """
    Generate paper-ready visualization of acquisition function comparison.

    Creates grouped bar chart showing gradient improvement rate for each
    acquisition function across different alpha values.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data for plotting
    acq_names = ['ucb', 'lcb', 'ei', 'pi']
    alphas = results['alphas']

    x = np.arange(len(acq_names))
    width = 0.2
    colors = sns.color_palette("husl", len(alphas))

    for i, alpha in enumerate(alphas):
        gir_means = []
        gir_stds = []
        for acq in acq_names:
            key = f'{acq}_alpha_{alpha}'
            if key in results['acquisitions']:
                data = results['acquisitions'][key]['gradient_improvement_rate']
                gir_means.append(data['mean'])
                gir_stds.append(data['std'])
            else:
                gir_means.append(0)
                gir_stds.append(0)

        offset = (i - len(alphas)/2 + 0.5) * width
        ax.bar(x + offset, gir_means, width, yerr=gir_stds,
               label=f'alpha={alpha}', color=colors[i], capsize=3)

    ax.set_xlabel('Acquisition Function', fontsize=12)
    ax.set_ylabel('Gradient Improvement Rate', fontsize=12)
    ax.set_title('Acquisition Function Comparison for Flow Guidance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in acq_names])
    ax.legend(title='Alpha')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved acquisition comparison figure to {output_path}")


def plot_thompson_sampling(ts_results: dict, output_path: str) -> None:
    """
    Generate paper-ready visualization of Thompson Sampling batch selection.

    Creates bar chart showing mean selected accuracy vs random baseline.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    batch_sizes = sorted([int(k.split('_')[1]) for k in ts_results.keys()])
    x = np.arange(len(batch_sizes))

    ts_means = []
    ts_stds = []
    baselines = []

    for bs in batch_sizes:
        data = ts_results[f'batch_{bs}']
        ts_means.append(data['mean_selected_accuracy']['mean'])
        ts_stds.append(data['mean_selected_accuracy']['std'])
        baselines.append(data['baseline_random'])

    width = 0.35
    ax.bar(x - width/2, ts_means, width, yerr=ts_stds, label='Thompson Sampling',
           color=sns.color_palette("husl")[0], capsize=3)
    ax.bar(x + width/2, baselines, width, label='Random Baseline',
           color=sns.color_palette("husl")[3], alpha=0.7)

    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Mean Selected Accuracy', fontsize=12)
    ax.set_title('Thompson Sampling Batch Selection Quality', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Thompson Sampling figure to {output_path}")


def analyze_results(gradient_results: dict, ts_results: dict) -> dict:
    """
    Analyze results and generate recommendations.

    Returns dict with analysis and recommendations.
    """
    analysis = {
        'gradient_analysis': {},
        'thompson_analysis': {},
        'recommendations': [],
    }

    # Find best acquisition for each alpha
    alphas = gradient_results['alphas']

    for alpha in alphas:
        alpha_key = f'alpha_{alpha}'
        best_acq = None
        best_gir = 0

        for acq in ['ucb', 'lcb', 'ei', 'pi']:
            key = f'{acq}_{alpha_key}'
            if key in gradient_results['acquisitions']:
                gir = gradient_results['acquisitions'][key]['gradient_improvement_rate']['mean']
                if gir > best_gir:
                    best_gir = gir
                    best_acq = acq

        analysis['gradient_analysis'][alpha_key] = {
            'best_acquisition': best_acq,
            'best_gradient_improvement_rate': best_gir,
        }

    # Thompson Sampling analysis
    for bs_key, data in ts_results.items():
        ts_acc = data['mean_selected_accuracy']['mean']
        baseline = data['baseline_random']
        improvement = ts_acc - baseline
        relative_improvement = improvement / baseline if baseline > 0 else 0

        analysis['thompson_analysis'][bs_key] = {
            'thompson_accuracy': ts_acc,
            'random_baseline': baseline,
            'improvement': improvement,
            'relative_improvement_pct': relative_improvement * 100,
        }

    # Generate recommendations
    ucb_girs = []
    lcb_girs = []
    ei_girs = []
    pi_girs = []

    for alpha in alphas:
        alpha_key = f'alpha_{alpha}'
        for acq, gir_list in [('ucb', ucb_girs), ('lcb', lcb_girs), ('ei', ei_girs), ('pi', pi_girs)]:
            key = f'{acq}_{alpha_key}'
            if key in gradient_results['acquisitions']:
                gir_list.append(gradient_results['acquisitions'][key]['gradient_improvement_rate']['mean'])

    avg_ucb = np.mean(ucb_girs) if ucb_girs else 0
    avg_lcb = np.mean(lcb_girs) if lcb_girs else 0
    avg_ei = np.mean(ei_girs) if ei_girs else 0
    avg_pi = np.mean(pi_girs) if pi_girs else 0

    # Rank acquisitions
    acq_scores = [('UCB', avg_ucb), ('LCB', avg_lcb), ('EI', avg_ei), ('PI', avg_pi)]
    acq_scores.sort(key=lambda x: x[1], reverse=True)

    analysis['recommendations'].append(
        f"Best acquisition for flow guidance: {acq_scores[0][0]} (GIR={acq_scores[0][1]:.2%})"
    )
    analysis['recommendations'].append(
        f"Second best: {acq_scores[1][0]} (GIR={acq_scores[1][1]:.2%})"
    )

    if avg_ucb > 0.7 or avg_lcb > 0.7:
        analysis['recommendations'].append(
            "UCB/LCB confirmed as best for flow guidance (smooth gradients, high GIR)"
        )

    # Thompson Sampling recommendation
    ts_improvements = [v['relative_improvement_pct'] for v in analysis['thompson_analysis'].values()]
    if np.mean(ts_improvements) > 5:
        analysis['recommendations'].append(
            f"Thompson Sampling recommended for batch selection (+{np.mean(ts_improvements):.1f}% vs random)"
        )

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Compare acquisition functions for flow-guided Bayesian optimization"
    )
    parser.add_argument(
        "--instructions",
        default="datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt",
        help="Path to instruction embeddings with accuracies"
    )
    parser.add_argument(
        "--output",
        default="results/acquisition_comparison.json",
        help="Path to output JSON file"
    )
    parser.add_argument("--device", default="cuda:0", help="Device for computation")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds for CV")

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.output).parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embeddings...")
    X, y, texts = load_embeddings(args.instructions, args.device)
    logger.info(f"Loaded {X.shape[0]} instructions with {X.shape[1]}D embeddings")
    logger.info(f"Accuracy range: [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")

    # Run gradient comparison
    logger.info("Running acquisition function comparison...")
    gradient_results = run_acquisition_comparison(
        X, y, n_folds=args.n_folds, device=args.device
    )

    # Run Thompson Sampling evaluation
    logger.info("Evaluating Thompson Sampling for batch selection...")
    ts_results = evaluate_thompson_sampling(X, y, device=args.device)

    # Analyze and generate recommendations
    logger.info("Analyzing results...")
    analysis = analyze_results(gradient_results, ts_results)

    # Combine results
    results = {
        'gradient_comparison': gradient_results,
        'thompson_sampling': ts_results,
        'analysis': analysis,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")

    # Generate paper-ready figures
    figure_path = str(figures_dir / 'acquisition_comparison.png')
    plot_acquisition_comparison(gradient_results, figure_path)

    ts_figure_path = str(figures_dir / 'thompson_sampling.png')
    plot_thompson_sampling(ts_results, ts_figure_path)

    # Print summary
    print("\n" + "="*60)
    print("ACQUISITION FUNCTION COMPARISON RESULTS")
    print("="*60)

    print("\nGradient Improvement Rate (higher = better for flow guidance):")
    for key, data in gradient_results['acquisitions'].items():
        gir = data['gradient_improvement_rate']
        gnm = data['gradient_norm_mean']
        print(f"  {key:20s}: GIR={gir['mean']:.2%} +/- {gir['std']:.2%}, ||grad||={gnm['mean']:.4f}")

    print("\nThompson Sampling Batch Selection:")
    for key, data in ts_results.items():
        acc = data['mean_selected_accuracy']
        baseline = data['baseline_random']
        print(f"  {key:10s}: {acc['mean']:.3f} +/- {acc['std']:.3f} (baseline: {baseline:.3f})")

    print("\n" + "-"*60)
    print("RECOMMENDATIONS")
    print("-"*60)
    for rec in analysis['recommendations']:
        print(f"  * {rec}")
    print("="*60)


if __name__ == "__main__":
    main()
