"""Statistical significance testing for flow matching experiments.

This module provides tools for statistical analysis of multi-seed experiments:
- Paired t-tests and Wilcoxon signed-rank tests
- Confidence interval computation
- Results aggregation across seeds
- Bootstrap confidence intervals

Usage:
    from study.flow_matching.significance import (
        aggregate_seed_results,
        compare_methods,
        compute_ci,
    )

    # Aggregate results from multiple seeds
    agg = aggregate_seed_results(results_per_seed)
    print(f"Mean: {agg['mean']:.4f} +/- {agg['ci_95']:.4f}")

    # Compare two methods
    comparison = compare_methods(method_a_scores, method_b_scores)
    print(f"p-value: {comparison['p_value']:.4f}")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation.

    Args:
        values: List of values.

    Returns:
        (mean, std) tuple.
    """
    arr = np.array(values)
    return float(arr.mean()), float(arr.std(ddof=1))


def compute_ci(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute confidence interval using t-distribution.

    Args:
        values: List of values from multiple seeds.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (mean, ci_low, ci_high) tuple.
    """
    arr = np.array(values)
    n = len(arr)
    mean = arr.mean()
    sem = stats.sem(arr)  # Standard error of mean

    # t-value for confidence level
    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * sem

    return float(mean), float(mean - margin), float(mean + margin)


def compute_bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    More robust for small sample sizes.

    Args:
        values: List of values.
        confidence: Confidence level.
        n_bootstrap: Number of bootstrap samples.
        random_state: Random seed for reproducibility.

    Returns:
        (mean, ci_low, ci_high) tuple.
    """
    rng = np.random.RandomState(random_state)
    arr = np.array(values)
    n = len(arr)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        bootstrap_means.append(sample.mean())

    bootstrap_means = np.array(bootstrap_means)

    # Percentile CI
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(arr.mean()), float(ci_low), float(ci_high)


def paired_ttest(
    method_a: List[float],
    method_b: List[float],
) -> Dict[str, float]:
    """Perform paired t-test between two methods.

    Use when comparing two methods on the same test problems.

    Args:
        method_a: Scores for method A across seeds.
        method_b: Scores for method B across seeds.

    Returns:
        Dict with t_statistic, p_value, effect_size (Cohen's d).
    """
    arr_a = np.array(method_a)
    arr_b = np.array(method_b)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Length mismatch: {len(arr_a)} vs {len(arr_b)}")

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(arr_a, arr_b)

    # Cohen's d effect size
    diff = arr_a - arr_b
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "effect_size_d": float(d),
        "mean_diff": float(diff.mean()),
        "significant_p05": p_value < 0.05,
        "significant_p01": p_value < 0.01,
    }


def wilcoxon_test(
    method_a: List[float],
    method_b: List[float],
) -> Dict[str, float]:
    """Perform Wilcoxon signed-rank test (non-parametric).

    Use when normality assumption may not hold or for small samples.

    Args:
        method_a: Scores for method A.
        method_b: Scores for method B.

    Returns:
        Dict with statistic, p_value.
    """
    arr_a = np.array(method_a)
    arr_b = np.array(method_b)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Length mismatch: {len(arr_a)} vs {len(arr_b)}")

    try:
        stat, p_value = stats.wilcoxon(arr_a, arr_b)
    except ValueError as e:
        # Can fail if all differences are zero
        logger.warning(f"Wilcoxon test failed: {e}")
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_p05": False,
            "significant_p01": False,
        }

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_p05": p_value < 0.05,
        "significant_p01": p_value < 0.01,
    }


def compare_methods(
    method_a: List[float],
    method_b: List[float],
    names: Tuple[str, str] = ("A", "B"),
) -> Dict:
    """Comprehensive comparison of two methods.

    Runs both t-test and Wilcoxon test, reports all statistics.

    Args:
        method_a: Scores for method A.
        method_b: Scores for method B.
        names: Names for the methods (for reporting).

    Returns:
        Comprehensive comparison dict with all statistics.
    """
    mean_a, std_a = compute_mean_std(method_a)
    mean_b, std_b = compute_mean_std(method_b)

    _, ci_low_a, ci_high_a = compute_ci(method_a)
    _, ci_low_b, ci_high_b = compute_ci(method_b)

    ttest = paired_ttest(method_a, method_b)
    wilcox = wilcoxon_test(method_a, method_b)

    return {
        names[0]: {
            "mean": mean_a,
            "std": std_a,
            "ci_95_low": ci_low_a,
            "ci_95_high": ci_high_a,
        },
        names[1]: {
            "mean": mean_b,
            "std": std_b,
            "ci_95_low": ci_low_b,
            "ci_95_high": ci_high_b,
        },
        "ttest": ttest,
        "wilcoxon": wilcox,
        "winner": names[0] if mean_a < mean_b else names[1],  # Assuming lower is better (loss)
    }


def aggregate_seed_results(
    results_per_seed: Dict[int, Dict],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Aggregate results from multiple seeds into summary statistics.

    Args:
        results_per_seed: Dict mapping seed -> result dict.
        metric_keys: Metrics to aggregate (default: all numeric values).

    Returns:
        Dict mapping metric -> {mean, std, ci_95_low, ci_95_high, values}.
    """
    if not results_per_seed:
        return {}

    # Determine metrics to aggregate
    sample_result = next(iter(results_per_seed.values()))
    if metric_keys is None:
        metric_keys = [
            k for k, v in sample_result.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

    aggregated = {}
    for metric in metric_keys:
        values = [
            results_per_seed[seed][metric]
            for seed in sorted(results_per_seed.keys())
            if metric in results_per_seed[seed]
        ]

        if not values:
            continue

        mean, ci_low, ci_high = compute_ci(values)
        _, std = compute_mean_std(values)

        aggregated[metric] = {
            "mean": mean,
            "std": std,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "ci_95": (ci_high - ci_low) / 2,  # Half-width for +/- notation
            "n_seeds": len(values),
            "values": values,
        }

    return aggregated


def load_experiment_results(
    checkpoint_dir: str | Path,
    pattern: str = "*.json",
) -> Dict[str, Dict]:
    """Load experiment results from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing result JSON files.
        pattern: Glob pattern for result files.

    Returns:
        Dict mapping run_name -> results.
    """
    checkpoint_dir = Path(checkpoint_dir)
    results = {}

    for path in checkpoint_dir.glob(pattern):
        with open(path) as f:
            data = json.load(f)
            run_name = path.stem
            results[run_name] = data

    return results


def format_ci_string(mean: float, ci: float, precision: int = 4) -> str:
    """Format mean +/- CI as string.

    Args:
        mean: Mean value.
        ci: CI half-width.
        precision: Decimal places.

    Returns:
        Formatted string like "0.1234 +/- 0.0012".
    """
    return f"{mean:.{precision}f} +/- {ci:.{precision}f}"


def create_results_table(
    experiments: Dict[str, Dict[str, Dict]],
    metric: str = "best_val_loss",
    sort_by_mean: bool = True,
) -> str:
    """Create markdown table of results.

    Args:
        experiments: Dict mapping method_name -> aggregated results.
        metric: Metric to display.
        sort_by_mean: Sort by mean value (ascending).

    Returns:
        Markdown table string.
    """
    rows = []
    for name, agg in experiments.items():
        if metric in agg:
            data = agg[metric]
            rows.append({
                "Method": name,
                "Mean": data["mean"],
                "Std": data["std"],
                "CI (95%)": data["ci_95"],
                "n": data["n_seeds"],
            })

    if sort_by_mean:
        rows.sort(key=lambda x: x["Mean"])

    # Build markdown table
    headers = ["Method", "Mean", "Std", "CI (95%)", "n"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        line = f"| {row['Method']} | {row['Mean']:.4f} | {row['Std']:.4f} | +/- {row['CI (95%)']:.4f} | {row['n']} |"
        lines.append(line)

    return "\n".join(lines)


def run_all_pairwise_comparisons(
    experiments: Dict[str, List[float]],
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """Run pairwise comparisons between all methods.

    Args:
        experiments: Dict mapping method_name -> list of scores across seeds.
        alpha: Significance level.

    Returns:
        Dict with pairwise comparison results.
    """
    methods = list(experiments.keys())
    comparisons = {}

    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1:]:
            key = f"{method_a}_vs_{method_b}"
            comparisons[key] = compare_methods(
                experiments[method_a],
                experiments[method_b],
                names=(method_a, method_b),
            )

    return comparisons


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Statistical Significance Testing Example")
    print("=" * 60)

    # Simulated results from 3 seeds
    icfm_losses = [0.0432, 0.0445, 0.0438]
    otcfm_losses = [0.0401, 0.0412, 0.0405]
    sigvp_losses = [0.0425, 0.0430, 0.0428]

    print("\n--- Aggregated Results ---")
    for name, values in [("I-CFM", icfm_losses), ("OT-CFM", otcfm_losses), ("SI-GVP", sigvp_losses)]:
        mean, ci_low, ci_high = compute_ci(values)
        ci_half = (ci_high - ci_low) / 2
        print(f"{name}: {format_ci_string(mean, ci_half)}")

    print("\n--- Method Comparison (I-CFM vs OT-CFM) ---")
    comparison = compare_methods(icfm_losses, otcfm_losses, names=("I-CFM", "OT-CFM"))
    print(f"Mean diff: {comparison['ttest']['mean_diff']:.4f}")
    print(f"t-test p-value: {comparison['ttest']['p_value']:.4f}")
    print(f"Wilcoxon p-value: {comparison['wilcoxon']['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {comparison['ttest']['effect_size_d']:.2f}")
    print(f"Significant at p<0.05: {comparison['ttest']['significant_p05']}")

    print("\n--- Pairwise Comparisons ---")
    experiments = {
        "I-CFM": icfm_losses,
        "OT-CFM": otcfm_losses,
        "SI-GVP": sigvp_losses,
    }
    pairwise = run_all_pairwise_comparisons(experiments)
    for key, result in pairwise.items():
        print(f"\n{key}:")
        print(f"  Winner: {result['winner']}")
        print(f"  t-test p={result['ttest']['p_value']:.4f}")
