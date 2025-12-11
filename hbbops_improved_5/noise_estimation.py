"""
Noise Estimation Utilities for HbBoPs Improved 5

This module provides robust variance estimation for binomial data:
1. Standard binomial variance: p(1-p)/n
2. Wilson score variance for extreme p values

NO LOGIT TRANSFORM - GP models accuracy directly in [0, 1] space.
"""

import torch
from typing import Tuple


def wilson_score_variance(p: float, n: int, z: float = 1.96) -> float:
    """
    Compute variance using Wilson score interval bounds.

    The Wilson score provides a robust variance estimate when p is near 0 or 1,
    where the standard binomial variance p(1-p)/n approaches zero and causes
    numerical instability.

    The Wilson score interval is: p_tilde +/- delta
    where:
        p_tilde = (p + z^2/(2n)) / (1 + z^2/n)
        delta = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)

    We use (delta/z)^2 as a robust variance estimate.

    Args:
        p: Observed accuracy/proportion in [0, 1]
        n: Sample size (fidelity)
        z: Z-score for confidence interval (1.96 for 95% CI)

    Returns:
        Robust variance estimate (always > 0)
    """
    if n == 0:
        return 1.0  # Maximum uncertainty

    z2 = z * z
    denom = 1.0 + z2 / n

    # Half-width of Wilson interval
    inner = p * (1.0 - p) / n + z2 / (4.0 * n * n)
    delta_sq = inner / (denom * denom)

    # Use interval half-width as variance proxy
    return max(delta_sq * z2, 1e-8)


def compute_heteroscedastic_noise(
    accuracies: torch.Tensor,
    fidelities: torch.Tensor,
    epsilon: float = 0.001,
    use_wilson: bool = True
) -> torch.Tensor:
    """
    Compute observation noise variance for each data point.

    For binomial data: Var(p_hat) = p(1-p)/n
    Uses Wilson score for robust estimation when p is near 0 or 1.

    Args:
        accuracies: (N,) tensor of accuracy values in [0, 1]
        fidelities: (N,) tensor of sample sizes
        epsilon: Threshold for using Wilson score (p < epsilon or p > 1-epsilon)
        use_wilson: Whether to use Wilson score for extreme values

    Returns:
        (N,) tensor of noise variances
    """
    N = len(accuracies)
    variances = torch.zeros(N, dtype=accuracies.dtype, device=accuracies.device)

    for i in range(N):
        p = float(accuracies[i])
        n = int(fidelities[i])

        if n <= 0:
            variances[i] = 1.0
            continue

        if use_wilson and (p < epsilon or p > 1 - epsilon):
            # Use Wilson score for extreme values
            variances[i] = wilson_score_variance(p, n)
        else:
            # Standard binomial variance
            variances[i] = max(p * (1.0 - p) / n, 1e-8)

    return variances
