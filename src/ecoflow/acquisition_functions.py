"""
Acquisition functions for Bayesian optimization in SONAR embedding space.

Implements Expected Improvement (EI), Probability of Improvement (PI), and
Thompson Sampling for comparison against LCB/UCB (already in gp_surrogate.py).

For flow guidance, LCB/UCB are preferred due to smooth gradient properties.
EI and PI are included for completeness and ablation studies.

Thompson Sampling is recommended for batch selection from discrete candidate
sets, not for continuous gradient guidance.
"""

from typing import Optional

import torch
from torch import Tensor
from botorch.models import SingleTaskGP


def expected_improvement_gradient(
    model: SingleTaskGP,
    X: Tensor,
    best_f: float,
    maximize: bool = True,
) -> Tensor:
    """
    Compute gradient of Expected Improvement acquisition function.

    EI(x) = E[max(f(x) - best_f, 0)]

    For Gaussian posterior:
    EI(x) = sigma(x) * (z * Phi(z) + phi(z))
    where z = (mu(x) - best_f) / sigma(x)

    The gradient involves derivatives of Phi (CDF) and phi (PDF) of standard normal.

    Args:
        model: Fitted GP model
        X: Input points [B, D]
        best_f: Current best observed value
        maximize: If True, maximize f; if False, minimize f

    Returns:
        grad_ei: Gradient of EI w.r.t. X [B, D]

    Note:
        EI gradient is more complex than UCB/LCB due to CDF/PDF terms.
        May have numerical issues near best_f (z approaches 0 or infinity).
    """
    device = X.device

    with torch.enable_grad():
        X_var = X.clone().requires_grad_(True)
        posterior = model.posterior(X_var)

        mean = posterior.mean.squeeze(-1)  # [B]
        var = posterior.variance.squeeze(-1)  # [B]
        std = torch.sqrt(var + 1e-6)

        # Improvement direction
        if maximize:
            improvement = mean - best_f
        else:
            improvement = best_f - mean

        z = improvement / std  # [B]

        # Standard normal CDF and PDF
        normal = torch.distributions.Normal(
            torch.zeros(1, device=device),
            torch.ones(1, device=device)
        )
        Phi_z = normal.cdf(z)  # CDF
        phi_z = torch.exp(normal.log_prob(z))  # PDF

        # EI = std * (z * Phi(z) + phi(z))
        ei = std * (z * Phi_z + phi_z)

        grad = torch.autograd.grad(
            ei.sum(),
            X_var,
            create_graph=False,
            retain_graph=False,
        )[0]

    return grad


def probability_improvement_gradient(
    model: SingleTaskGP,
    X: Tensor,
    best_f: float,
    maximize: bool = True,
) -> Tensor:
    """
    Compute gradient of Probability of Improvement acquisition function.

    PI(x) = P(f(x) > best_f) = Phi((mu(x) - best_f) / sigma(x))

    The gradient is:
    d(PI)/dx = phi(z) * d(z)/dx

    where z = (mu - best_f) / sigma

    Args:
        model: Fitted GP model
        X: Input points [B, D]
        best_f: Current best observed value
        maximize: If True, maximize f; if False, minimize f

    Returns:
        grad_pi: Gradient of PI w.r.t. X [B, D]

    Warning:
        PI gradient can be very flat (near zero) when z is large positive
        or large negative. This makes it less suitable for flow guidance.
    """
    device = X.device

    with torch.enable_grad():
        X_var = X.clone().requires_grad_(True)
        posterior = model.posterior(X_var)

        mean = posterior.mean.squeeze(-1)
        var = posterior.variance.squeeze(-1)
        std = torch.sqrt(var + 1e-6)

        if maximize:
            z = (mean - best_f) / std
        else:
            z = (best_f - mean) / std

        normal = torch.distributions.Normal(
            torch.zeros(1, device=device),
            torch.ones(1, device=device)
        )
        pi = normal.cdf(z)

        grad = torch.autograd.grad(
            pi.sum(),
            X_var,
            create_graph=False,
            retain_graph=False,
        )[0]

    return grad


def thompson_sampling_select(
    model: SingleTaskGP,
    candidates: Tensor,
    n_select: int = 1,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Select candidates using Thompson Sampling.

    Draws a sample from the GP posterior and selects candidates with
    highest sampled values. This is NOT suitable for continuous gradient
    guidance but excellent for batch selection from discrete candidate sets.

    Args:
        model: Fitted GP model
        candidates: Candidate points [N, D]
        n_select: Number of candidates to select
        seed: Random seed for reproducibility

    Returns:
        selected: Indices of selected candidates [n_select]

    Usage:
        For batch selection in BO loop:
        >>> flow_candidates = flow_model.sample(n=1000)
        >>> selected_idx = thompson_sampling_select(gp, flow_candidates, n_select=8)
        >>> batch = flow_candidates[selected_idx]
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()

    with torch.no_grad():
        posterior = model.posterior(candidates)

        # Sample from posterior
        # Shape: [1, N, 1] -> [N]
        sample = posterior.sample().squeeze()

        # Handle single candidate case
        if sample.dim() == 0:
            sample = sample.unsqueeze(0)

        # Select top n_select
        _, indices = sample.topk(min(n_select, len(sample)))

    return indices


def compute_acquisition_gradient_metrics(
    model: SingleTaskGP,
    X: Tensor,
    y: Tensor,
    alpha: float = 1.0,
) -> dict:
    """
    Compute gradient quality metrics for all acquisition functions.

    Metrics:
    - gradient_norm_mean: Average L2 norm of gradients
    - gradient_norm_std: Std of gradient norms
    - gradient_improvement_rate: % of points where following gradient improves acquisition

    Args:
        model: Fitted GP model
        X: Test points [N, D]
        y: Test targets [N] (for best_f computation)
        alpha: UCB/LCB exploration parameter

    Returns:
        dict with metrics for each acquisition function
    """
    model.eval()
    best_f = y.max().item()

    results = {}

    # LCB (already in gp_surrogate, compute here for comparison)
    with torch.enable_grad():
        X_var = X.clone().requires_grad_(True)
        posterior = model.posterior(X_var)
        mean = posterior.mean.squeeze(-1)
        std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)
        lcb = mean - alpha * std
        grad_lcb = torch.autograd.grad(lcb.sum(), X_var)[0]

    results['lcb'] = _compute_gradient_metrics(model, X, grad_lcb, 'lcb', alpha)

    # UCB
    with torch.enable_grad():
        X_var = X.clone().requires_grad_(True)
        posterior = model.posterior(X_var)
        mean = posterior.mean.squeeze(-1)
        std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)
        ucb = mean + alpha * std
        grad_ucb = torch.autograd.grad(ucb.sum(), X_var)[0]

    results['ucb'] = _compute_gradient_metrics(model, X, grad_ucb, 'ucb', alpha)

    # EI
    grad_ei = expected_improvement_gradient(model, X, best_f, maximize=True)
    results['ei'] = _compute_gradient_metrics(model, X, grad_ei, 'ei', alpha)

    # PI
    grad_pi = probability_improvement_gradient(model, X, best_f, maximize=True)
    results['pi'] = _compute_gradient_metrics(model, X, grad_pi, 'pi', alpha)

    return results


def _compute_gradient_metrics(
    model: SingleTaskGP,
    X: Tensor,
    grad: Tensor,
    acq_name: str,
    alpha: float,
) -> dict:
    """Helper to compute gradient metrics for a single acquisition function."""

    grad_norms = torch.norm(grad, dim=-1)

    # Take step in gradient direction
    step_size = 0.01
    X_stepped = X + step_size * grad / (grad_norms.unsqueeze(-1) + 1e-8)

    # Compute acquisition value before/after
    model.eval()
    with torch.no_grad():
        post_before = model.posterior(X)
        post_after = model.posterior(X_stepped)

        mean_before = post_before.mean.squeeze(-1)
        std_before = torch.sqrt(post_before.variance.squeeze(-1) + 1e-6)
        mean_after = post_after.mean.squeeze(-1)
        std_after = torch.sqrt(post_after.variance.squeeze(-1) + 1e-6)

        if acq_name in ['lcb']:
            acq_before = mean_before - alpha * std_before
            acq_after = mean_after - alpha * std_after
        elif acq_name in ['ucb']:
            acq_before = mean_before + alpha * std_before
            acq_after = mean_after + alpha * std_after
        else:
            # For EI/PI, just use mean as proxy
            acq_before = mean_before
            acq_after = mean_after

        improved = (acq_after > acq_before).float()

    return {
        'gradient_norm_mean': float(grad_norms.mean()),
        'gradient_norm_std': float(grad_norms.std()),
        'gradient_improvement_rate': float(improved.mean()),
    }
