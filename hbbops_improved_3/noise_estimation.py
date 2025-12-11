"""
Noise Estimation Utilities for HbBoPs Improved 3

This module provides robust variance estimation for binomial data:
1. Wilson score variance for extreme p values
2. Delta method for transforming variance through logit function

Key insight: When using logit transform on accuracy values, the variance
must also be transformed using the Delta method:
    σ²_new = σ²_old × [f'(p)]² = 1/(n × p × (1-p))
"""

import torch
from typing import Tuple


def wilson_score_variance(p: float, n: int, z: float = 1.96) -> float:
    """
    Compute variance using Wilson score interval bounds.

    The Wilson score provides a robust variance estimate when p is near 0 or 1,
    where the standard binomial variance p(1-p)/n approaches zero and causes
    numerical instability.

    The Wilson score interval is: p_tilde ± delta
    where:
        p_tilde = (p + z²/(2n)) / (1 + z²/n)
        delta = z × sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)

    We use (delta/z)² as a robust variance estimate.

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
    # This is the Wilson score variance estimate
    return max(delta_sq * z2, 1e-8)


def compute_heteroscedastic_noise(
    accuracies: torch.Tensor,
    fidelities: torch.Tensor,
    epsilon: float = 0.001,
    use_wilson: bool = True
) -> torch.Tensor:
    """
    Compute observation noise variance for each data point (in original space).

    For binomial data: Var(p_hat) = p(1-p)/n
    Uses Wilson score for robust estimation when p is near 0 or 1.

    Args:
        accuracies: (N,) tensor of accuracy values in [0, 1]
        fidelities: (N,) tensor of sample sizes
        epsilon: Threshold for using Wilson score (p < epsilon or p > 1-epsilon)
        use_wilson: Whether to use Wilson score for extreme values

    Returns:
        (N,) tensor of noise variances in original (accuracy) space
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


def logit_transform_with_delta_method(
    accuracies: torch.Tensor,
    fidelities: torch.Tensor,
    epsilon: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply logit transform and compute transformed variance via Delta method.

    This is the critical function that correctly combines output warping
    with heteroscedastic noise. The Delta method approximation gives:

    Transform: y = log(p / (1-p))  [logit function]
    Derivative: dy/dp = 1 / (p × (1-p))

    Delta method: Var(y) ≈ Var(p) × (dy/dp)²
                        = [p(1-p)/n] × [1/(p(1-p))]²
                        = 1 / (n × p × (1-p))

    Key insight: In the original space, variance is largest at p=0.5.
    In logit space, variance is largest at the extremes (p→0 or p→1)
    because the logit function stretches the space near the boundaries.

    Args:
        accuracies: (N,) tensor of accuracy values in [0, 1]
        fidelities: (N,) tensor of sample sizes
        epsilon: Clipping bound to avoid log(0) and division by zero

    Returns:
        Tuple of:
        - y_transformed: (N,) tensor of logit-transformed values
        - variance_transformed: (N,) tensor of transformed variances
    """
    # Clip to avoid log(0) and division by zero
    p_clipped = torch.clamp(accuracies, epsilon, 1.0 - epsilon)

    # Logit transform: y = log(p / (1-p))
    y_transformed = torch.log(p_clipped / (1.0 - p_clipped))

    # Transformed variance via Delta method: σ²_new = 1 / (n × p × (1-p))
    # This is derived from: σ²_new = σ²_old × [f'(p)]²
    # where σ²_old = p(1-p)/n and f'(p) = 1/(p(1-p))
    variance_transformed = 1.0 / (fidelities * p_clipped * (1.0 - p_clipped))

    # Apply Wilson-based correction for extreme values
    for i in range(len(accuracies)):
        p = float(accuracies[i])
        n = int(fidelities[i])

        if p < epsilon or p > 1.0 - epsilon:
            # Use Wilson-based variance and transform it
            p_w = float(p_clipped[i])
            wilson_var = wilson_score_variance(p, n)

            # Delta method on Wilson variance
            deriv_sq = 1.0 / (p_w * (1.0 - p_w)) ** 2
            variance_transformed[i] = wilson_var * deriv_sq

    # Ensure minimum variance for numerical stability
    variance_transformed = torch.clamp(variance_transformed, min=1e-6)

    return y_transformed, variance_transformed


def inverse_logit(y_logit: torch.Tensor) -> torch.Tensor:
    """
    Convert logit-transformed values back to accuracy (sigmoid function).

    Args:
        y_logit: (N,) tensor of logit values

    Returns:
        (N,) tensor of accuracy values in (0, 1)
    """
    return torch.sigmoid(y_logit)


def logit(p: torch.Tensor, epsilon: float = 0.001) -> torch.Tensor:
    """
    Compute logit transform: log(p / (1-p))

    Args:
        p: (N,) tensor of probability/accuracy values
        epsilon: Clipping bound

    Returns:
        (N,) tensor of logit values
    """
    p_clipped = torch.clamp(p, epsilon, 1.0 - epsilon)
    return torch.log(p_clipped / (1.0 - p_clipped))
