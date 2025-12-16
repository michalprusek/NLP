"""
Aggregation functions for combining scores across multiple models.

Provides different strategies for computing a single score from per-model scores,
used to evaluate prompt universality across frontier LLMs.
"""
from typing import Dict, Optional

import numpy as np


def aggregate_scores(
    error_rates: Dict[str, float],
    strategy: str = "weighted_softmin",
    temperature: float = 0.1,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Aggregate error rates from multiple models into a single score.

    Args:
        error_rates: Dict mapping model_name -> error_rate (0.0 to 1.0)
        strategy: Aggregation strategy:
            - "average": Simple weighted mean
            - "minimum": Worst-case (max error, min accuracy)
            - "weighted_softmin": Smooth approximation to minimum
            - "harmonic": Harmonic mean of accuracies
        temperature: For weighted_softmin, controls smoothness
            - T -> 0: approaches minimum (conservative)
            - T -> inf: approaches average (balanced)
            - Default 0.1 gives moderate penalty to weak models
        weights: Optional per-model weights (must sum to 1.0)
            If None, uses uniform weights

    Returns:
        Aggregated error rate (lower is better, 0.0 to 1.0)

    Examples:
        >>> errors = {"model_a": 0.1, "model_b": 0.2, "model_c": 0.15}
        >>> aggregate_scores(errors, "average")  # 0.15
        >>> aggregate_scores(errors, "minimum")  # 0.2 (worst case)
        >>> aggregate_scores(errors, "weighted_softmin", temperature=0.1)  # ~0.18
    """
    if not error_rates:
        raise ValueError("error_rates cannot be empty")

    models = list(error_rates.keys())
    errors = np.array([error_rates[m] for m in models])

    # Clamp error rates to [0, 1] (GP predictions can exceed bounds)
    errors = np.clip(errors, 0.0, 1.0)

    # Set up weights
    if weights is not None:
        w = np.array([weights[m] for m in models])
    else:
        w = np.ones(len(errors)) / len(errors)

    # Compute aggregation based on strategy
    if strategy == "average":
        return float(np.sum(w * errors))

    elif strategy == "minimum":
        # For error rates, max error = worst case
        # This is the most conservative strategy
        return float(np.max(errors))

    elif strategy == "weighted_softmin":
        # Softmax on positive errors gives more weight to higher errors
        # This is a smooth approximation to maximum (worst-case for errors)
        #
        # softmax(e/T) * e = sum of errors weighted by their softmax
        # As T -> 0, this approaches max(errors)
        # As T -> inf, this approaches mean(errors)
        exp_weights = np.exp(errors / temperature)
        softmax_weights = exp_weights / np.sum(exp_weights)
        return float(np.sum(softmax_weights * errors))

    elif strategy == "harmonic":
        # Harmonic mean of accuracies, converted back to error
        # Penalizes low performers more than arithmetic mean
        accuracies = 1.0 - errors

        # Handle edge case: if any accuracy is 0, harmonic mean is 0
        if np.any(accuracies <= 0):
            return 1.0  # Worst possible error

        harmonic_acc = len(accuracies) / np.sum(1.0 / accuracies)
        return float(1.0 - harmonic_acc)

    else:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy}. "
            f"Use: average, minimum, weighted_softmin, harmonic"
        )


def aggregate_accuracies(
    accuracies: Dict[str, float],
    strategy: str = "weighted_softmin",
    temperature: float = 0.1,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Aggregate accuracies from multiple models into a single score.

    Convenience wrapper that converts accuracies to error rates,
    aggregates, and converts back.

    Args:
        accuracies: Dict mapping model_name -> accuracy (0.0 to 1.0)
        strategy: Same as aggregate_scores
        temperature: Same as aggregate_scores
        weights: Same as aggregate_scores

    Returns:
        Aggregated accuracy (higher is better, 0.0 to 1.0)
    """
    error_rates = {m: 1.0 - acc for m, acc in accuracies.items()}
    aggregated_error = aggregate_scores(error_rates, strategy, temperature, weights)
    return 1.0 - aggregated_error


def compute_bounds_aggregated(
    model_bounds: Dict[str, Dict[str, float]],
    strategy: str = "weighted_softmin",
    temperature: float = 0.1,
) -> tuple[float, float]:
    """
    Aggregate confidence bounds across models.

    For sequential testing, we need to aggregate per-model confidence intervals
    into a single interval for decision making.

    Args:
        model_bounds: Dict mapping model_name -> {"lower": float, "upper": float}
            where lower/upper are accuracy bounds
        strategy: Aggregation strategy
        temperature: For weighted_softmin

    Returns:
        Tuple of (aggregated_lower_accuracy, aggregated_upper_accuracy)
    """
    # Convert accuracy bounds to error bounds
    lower_errors = {m: 1.0 - b["upper"] for m, b in model_bounds.items()}
    upper_errors = {m: 1.0 - b["lower"] for m, b in model_bounds.items()}

    # Aggregate error bounds
    agg_lower_error = aggregate_scores(lower_errors, strategy, temperature)
    agg_upper_error = aggregate_scores(upper_errors, strategy, temperature)

    # Convert back to accuracy bounds
    agg_lower_acc = 1.0 - agg_upper_error
    agg_upper_acc = 1.0 - agg_lower_error

    return agg_lower_acc, agg_upper_acc
