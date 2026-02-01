"""Regret-based metrics for Bayesian Optimization performance.

Metrics for evaluating how well the BO finds the optimum:
- Simple Regret: Gap to optimum at each step
- Cumulative Regret: Sum of suboptimalities
- AUC Regret: Area under regret curve
- Steps to threshold: Iterations to reach X% of optimum
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_simple_regret(
    Y_trajectory: torch.Tensor,
    Y_optimal: Optional[float] = None,
) -> torch.Tensor:
    """Compute simple regret at each step.

    Simple regret = Y_optimal - best_found_so_far

    Lower is better. Measures how close we are to the optimum.

    Args:
        Y_trajectory: Observed values at each step [T].
        Y_optimal: True optimal value. If None, uses max(Y_trajectory).

    Returns:
        Simple regret at each step [T].
    """
    Y = Y_trajectory.cpu() if torch.is_tensor(Y_trajectory) else torch.tensor(Y_trajectory)

    # Best found so far (cumulative max)
    best_so_far = torch.cummax(Y, dim=0).values

    if Y_optimal is None:
        Y_optimal = Y.max().item()

    simple_regret = Y_optimal - best_so_far
    return simple_regret


def compute_cumulative_regret(
    Y_trajectory: torch.Tensor,
    Y_optimal: Optional[float] = None,
) -> torch.Tensor:
    """Compute cumulative regret at each step.

    Cumulative regret = sum(Y_optimal - Y_t) for t=1..T

    Lower is better. Measures total suboptimality over time.

    Args:
        Y_trajectory: Observed values at each step [T].
        Y_optimal: True optimal value. If None, uses max(Y_trajectory).

    Returns:
        Cumulative regret at each step [T].
    """
    Y = Y_trajectory.cpu() if torch.is_tensor(Y_trajectory) else torch.tensor(Y_trajectory)

    if Y_optimal is None:
        Y_optimal = Y.max().item()

    instantaneous_regret = Y_optimal - Y
    cumulative_regret = torch.cumsum(instantaneous_regret, dim=0)

    return cumulative_regret


def compute_auc_regret(
    Y_trajectory: torch.Tensor,
    Y_optimal: Optional[float] = None,
) -> float:
    """Compute Area Under Regret Curve.

    Integral of simple regret over iterations. Lower is better.

    Args:
        Y_trajectory: Observed values at each step [T].
        Y_optimal: True optimal value.

    Returns:
        AUC regret value.
    """
    simple_regret = compute_simple_regret(Y_trajectory, Y_optimal)
    return np.trapz(simple_regret.numpy())


def compute_steps_to_threshold(
    Y_trajectory: torch.Tensor,
    threshold_pct: float = 0.95,
    Y_optimal: Optional[float] = None,
) -> int:
    """Compute number of steps to reach threshold percentage of optimum.

    Args:
        Y_trajectory: Observed values at each step [T].
        threshold_pct: Target percentage of optimum (e.g., 0.95 for 95%).
        Y_optimal: True optimal value.

    Returns:
        Number of steps to reach threshold, or T if not reached.
    """
    Y = Y_trajectory.cpu() if torch.is_tensor(Y_trajectory) else torch.tensor(Y_trajectory)

    if Y_optimal is None:
        Y_optimal = Y.max().item()

    target = threshold_pct * Y_optimal
    best_so_far = torch.cummax(Y, dim=0).values

    reached = torch.where(best_so_far >= target)[0]
    if len(reached) > 0:
        return reached[0].item() + 1  # 1-indexed
    return len(Y)


def compute_improvement_rate(
    Y_trajectory: torch.Tensor,
) -> float:
    """Compute fraction of steps that improved on previous best.

    Higher is better. Measures exploration efficiency.

    Args:
        Y_trajectory: Observed values at each step [T].

    Returns:
        Improvement rate in [0, 1].
    """
    Y = Y_trajectory.cpu() if torch.is_tensor(Y_trajectory) else torch.tensor(Y_trajectory)

    if len(Y) < 2:
        return 0.0

    best_so_far = torch.cummax(Y, dim=0).values

    # Count improvements (excluding first point)
    improvements = (best_so_far[1:] > best_so_far[:-1]).float().sum()
    return (improvements / (len(Y) - 1)).item()


def compute_all_regret_metrics(
    Y_trajectory: torch.Tensor,
    Y_optimal: Optional[float] = None,
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """Compute all regret-based metrics.

    Args:
        Y_trajectory: Observed values at each step [T].
        Y_optimal: True optimal value.
        thresholds: Percentage thresholds to compute steps for.

    Returns:
        Dict with all regret metrics.
    """
    if thresholds is None:
        thresholds = [0.90, 0.95, 0.99]

    Y = Y_trajectory.cpu() if torch.is_tensor(Y_trajectory) else torch.tensor(Y_trajectory)

    if Y_optimal is None:
        Y_optimal = Y.max().item()

    simple_regret = compute_simple_regret(Y, Y_optimal)
    cumulative_regret = compute_cumulative_regret(Y, Y_optimal)

    result = {
        "final_simple_regret": simple_regret[-1].item(),
        "final_cumulative_regret": cumulative_regret[-1].item(),
        "auc_regret": compute_auc_regret(Y, Y_optimal),
        "final_best": torch.cummax(Y, dim=0).values[-1].item(),
        "improvement_rate": compute_improvement_rate(Y),
        "y_optimal": Y_optimal,
    }

    # Steps to various thresholds
    for pct in thresholds:
        key = f"steps_to_{int(pct * 100)}"
        result[key] = compute_steps_to_threshold(Y, pct, Y_optimal)

    return result


def compute_regret_trajectory(
    Y_trajectory: torch.Tensor,
    Y_optimal: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Compute full regret trajectories for plotting.

    Args:
        Y_trajectory: Observed values at each step [T].
        Y_optimal: True optimal value.

    Returns:
        Dict with simple_regret, cumulative_regret, best_so_far trajectories.
    """
    Y = Y_trajectory.cpu() if torch.is_tensor(Y_trajectory) else torch.tensor(Y_trajectory)

    if Y_optimal is None:
        Y_optimal = Y.max().item()

    return {
        "simple_regret": compute_simple_regret(Y, Y_optimal),
        "cumulative_regret": compute_cumulative_regret(Y, Y_optimal),
        "best_so_far": torch.cummax(Y, dim=0).values,
        "instantaneous": Y,
    }
