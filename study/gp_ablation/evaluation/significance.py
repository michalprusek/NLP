"""Statistical significance testing for GP ablation study.

Provides:
- Paired t-test and Wilcoxon signed-rank test
- Bootstrap confidence intervals
- Effect size (Cohen's d)
- Win rate computation
- Comparison table generation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def compute_significance(
    metric_A: np.ndarray,
    metric_B: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """Compute statistical significance between two methods.

    Uses paired t-test and Wilcoxon signed-rank test.

    Args:
        metric_A: Metric values for method A [n_seeds].
        metric_B: Metric values for method B [n_seeds].
        alpha: Significance level.

    Returns:
        Dict with p-values, effect size, win rate, confidence intervals.
    """
    metric_A = np.asarray(metric_A)
    metric_B = np.asarray(metric_B)

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(metric_A, metric_B)

    # Wilcoxon signed-rank (non-parametric)
    # Handle case where differences are all zero
    diff = metric_A - metric_B
    if np.allclose(diff, 0):
        w_stat, p_wilcoxon = 0, 1.0
    else:
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(metric_A, metric_B)
        except ValueError:
            # Too few samples
            w_stat, p_wilcoxon = 0, 1.0

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(metric_A) - 1) * metric_A.std()**2 +
         (len(metric_B) - 1) * metric_B.std()**2) /
        (len(metric_A) + len(metric_B) - 2)
    )
    if pooled_std < 1e-8:
        cohens_d = 0.0
    else:
        cohens_d = (metric_A.mean() - metric_B.mean()) / pooled_std

    # Win rate (fraction where A > B)
    win_rate = (metric_A > metric_B).mean()
    tie_rate = (np.abs(metric_A - metric_B) < 1e-8).mean()

    # Bootstrap 95% CI for mean difference
    ci_low, ci_high = compute_bootstrap_ci(diff)

    # Is the difference significant?
    significant = p_ttest < alpha and p_wilcoxon < alpha

    return {
        "t_stat": t_stat,
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
        "cohens_d": cohens_d,
        "win_rate": win_rate,
        "tie_rate": tie_rate,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "mean_diff": diff.mean(),
        "std_diff": diff.std(),
        "significant": significant,
    }


def compute_bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        data: Data samples [n].
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap samples.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    data = np.asarray(data)
    n = len(data)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        bootstrap_means.append(data[idx].mean())

    bootstrap_means = np.array(bootstrap_means)

    # Percentile method
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return ci_low, ci_high


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        magnitude = "negligible"
    elif d_abs < 0.5:
        magnitude = "small"
    elif d_abs < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    direction = "better" if d > 0 else "worse"
    return f"{magnitude} ({direction})"


def generate_comparison_table(
    results: Dict[str, Dict[str, List[float]]],
    baseline: str,
    metrics: Optional[List[str]] = None,
) -> Dict:
    """Generate comparison table for all methods vs baseline.

    Args:
        results: Dict of {method: {metric: [values across seeds]}}.
        baseline: Name of baseline method.
        metrics: Metrics to compare (default: all).

    Returns:
        Dict with comparison statistics for each (method, metric) pair.
    """
    if metrics is None:
        # Get all metrics from first method
        first_method = next(iter(results.values()))
        metrics = list(first_method.keys())

    comparison = {}

    if baseline not in results:
        raise ValueError(f"Baseline '{baseline}' not in results")

    baseline_results = results[baseline]

    for method, method_results in results.items():
        if method == baseline:
            continue

        comparison[method] = {}

        for metric in metrics:
            if metric not in baseline_results or metric not in method_results:
                continue

            baseline_vals = np.array(baseline_results[metric])
            method_vals = np.array(method_results[metric])

            # Compute significance
            sig = compute_significance(method_vals, baseline_vals)

            comparison[method][metric] = {
                "method_mean": method_vals.mean(),
                "method_std": method_vals.std(),
                "baseline_mean": baseline_vals.mean(),
                "baseline_std": baseline_vals.std(),
                "improvement": method_vals.mean() - baseline_vals.mean(),
                "improvement_pct": (
                    (method_vals.mean() - baseline_vals.mean()) /
                    (abs(baseline_vals.mean()) + 1e-8) * 100
                ),
                **sig,
            }

    return comparison


def format_result_with_ci(
    values: np.ndarray,
    confidence: float = 0.95,
) -> str:
    """Format result as mean ± CI for publication.

    Args:
        values: Values across seeds [n_seeds].
        confidence: Confidence level.

    Returns:
        Formatted string like "0.832 ± 0.015".
    """
    values = np.asarray(values)
    mean = values.mean()

    if len(values) > 1:
        ci_low, ci_high = compute_bootstrap_ci(values, confidence)
        ci_half = (ci_high - ci_low) / 2
        return f"{mean:.3f} ± {ci_half:.3f}"
    else:
        return f"{mean:.3f}"


def rank_methods(
    results: Dict[str, Dict[str, List[float]]],
    metric: str,
    higher_is_better: bool = True,
) -> List[Tuple[str, float, float]]:
    """Rank methods by a metric.

    Args:
        results: Dict of {method: {metric: [values across seeds]}}.
        metric: Metric to rank by.
        higher_is_better: If True, higher values rank higher.

    Returns:
        List of (method, mean, std) tuples, sorted by mean.
    """
    rankings = []

    for method, method_results in results.items():
        if metric not in method_results:
            continue

        values = np.array(method_results[metric])
        rankings.append((method, values.mean(), values.std()))

    # Sort by mean (descending if higher is better)
    rankings.sort(key=lambda x: x[1], reverse=higher_is_better)

    return rankings
