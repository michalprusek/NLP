"""Batch selection methods for parallel Bayesian optimization.

This module implements batch selection strategies for selecting multiple
candidates per iteration, enabling parallel evaluation in BO loops.

Key Components:
1. `local_penalization_batch`: Sequential greedy batch selection with
   diversity enforcement via penalization near already-selected points
2. `PenalizedAcquisition`: Wrapper that multiplies base acquisition by
   penalty term to discourage clustering
3. `select_batch_candidates`: Convenience function for selecting from
   pre-generated candidates (main entry point for BOOptimizationLoop)
4. `estimate_lipschitz_constant`: Gradient-based Lipschitz estimation

Algorithm: Local Penalization (Gonzalez et al., 2016)
- Select first point by optimizing base acquisition (UCB)
- For subsequent points, penalize acquisition near selected points
- Penalty based on distance and GP mean difference

Reference:
    Gonzalez et al. (2016) "Batch Bayesian Optimization via Local Penalization"
    https://arxiv.org/abs/1505.08052
"""

import logging
import math
from typing import Optional, Tuple

import torch
from torch.distributions import Normal


def estimate_lipschitz_constant(
    model,
    bounds: torch.Tensor,
    n_samples: int = 500,
) -> float:
    """
    Estimate Lipschitz constant by maximizing gradient norm of GP mean.

    L = max_x ||grad mu(x)||

    This gives a conservative estimate of the function's smoothness,
    used to determine penalization radii in Local Penalization.

    Args:
        model: BoTorch GP model (SingleTaskGP or similar)
        bounds: Search space bounds [2, D], where bounds[0] is lower
                and bounds[1] is upper
        n_samples: Number of random samples for gradient estimation

    Returns:
        L: Estimated Lipschitz constant (floored at 1e-7 for stability)

    Example:
        >>> gp = create_surrogate('msr', D=1024)
        >>> gp.fit(X, Y)
        >>> L = estimate_lipschitz_constant(gp.model, bounds)
    """
    device = bounds.device
    D = bounds.shape[1]

    # Sample random points within bounds
    X = torch.rand(n_samples, D, device=device)
    X = bounds[0] + X * (bounds[1] - bounds[0])
    X.requires_grad_(True)

    # Compute gradient of GP mean
    model.eval()
    with torch.enable_grad():
        posterior = model.posterior(X)
        mean = posterior.mean.squeeze(-1)  # [n_samples]

        # Gradient for each sample via backprop
        grads = torch.autograd.grad(
            mean.sum(),
            X,
            create_graph=False,
        )[0]  # [n_samples, D]

    # Maximum gradient norm
    grad_norms = grads.norm(dim=-1)  # [n_samples]
    L = grad_norms.max().item()

    # Handle NaN/Inf from ill-conditioned GP
    if not math.isfinite(L):
        logger = logging.getLogger(__name__)
        logger.warning("Lipschitz estimate is non-finite, using default 1.0")
        L = 1.0

    # Floor for numerical stability
    return max(L, 1e-7)


class PenalizedAcquisition(torch.nn.Module):
    """
    Penalized acquisition function for Local Penalization batch selection.

    Wraps a base acquisition function and multiplies its value by a penalty
    that discourages selecting points near already-selected candidates.

    The penalized acquisition is:
        a_penalized(x) = a(x) * product_{x' in selected} hammer(x, x')

    where hammer(x, x') = 1 - phi(-(||x-x'|| - r(x')) / s)
          r(x') = |mu(x') - min_Y| / L  (penalization radius)
          phi = standard normal CDF
          s = smoothing parameter

    The hammer function transitions smoothly from 0 (at x') to 1 (far from x'),
    with the transition centered at distance r(x') from x'.

    Attributes:
        base_acqf: Base acquisition function to penalize
        X_selected: Already selected points [K, D]
        radii: Precomputed penalization radii for each selected point [K]
        s: Smoothing parameter for hammer function transition

    Args:
        base_acqf: Base acquisition function (e.g., UCB)
        X_selected: Already selected batch candidates [K, D]
        L: Estimated Lipschitz constant
        model: GP model (for computing radii from mean predictions)
        s: Smoothing parameter (default 0.2)

    Example:
        >>> base_acqf = UpperConfidenceBound(model, beta=1.96**2)
        >>> penalized = PenalizedAcquisition(base_acqf, X_selected, L, model)
        >>> value = penalized(X)  # X: [batch, q, D]
    """

    def __init__(
        self,
        base_acqf: torch.nn.Module,
        X_selected: torch.Tensor,
        L: float,
        model,
        s: float = 0.2,
    ):
        super().__init__()
        self.base_acqf = base_acqf
        self.X_selected = X_selected  # [K, D]
        self.L = L
        self.model = model
        self.s = s

        # Precompute penalization radii for selected points
        # r(x') = |mu(x') - min_Y| / L
        with torch.no_grad():
            posterior = model.posterior(X_selected)
            mu_selected = posterior.mean.squeeze(-1)  # [K]

            # Get minimum observed Y from training data
            train_Y = model.train_targets
            if train_Y.numel() > 0:
                min_Y = train_Y.min()
            else:
                min_Y = torch.tensor(0.0, device=X_selected.device)

            # Penalization radius: how far to penalize around each point
            self.radii = torch.abs(mu_selected - min_Y) / L  # [K]

            # Minimum radius for numerical stability
            self.radii = self.radii.clamp(min=0.01)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute penalized acquisition value.

        Args:
            X: Input tensor [batch_shape, q, D] where q is typically 1

        Returns:
            Penalized acquisition values [batch_shape]
        """
        # Base acquisition value
        base_val = self.base_acqf(X)  # [batch_shape]

        # Compute penalty from all selected points
        # X shape: [batch_shape, q, D], X_selected: [K, D]
        original_shape = X.shape[:-1]  # [batch_shape, q]
        X_flat = X.view(-1, X.shape[-1])  # [batch_shape*q, D]

        # Pairwise distances to selected points
        # cdist: [batch_shape*q, K]
        dists = torch.cdist(X_flat, self.X_selected)

        # Hammer function: 1 - phi(-(d - r) / s)
        # where phi is standard normal CDF
        # When d < r: -(d-r)/s > 0, phi > 0.5, hammer < 0.5 (penalty)
        # When d > r: -(d-r)/s < 0, phi < 0.5, hammer > 0.5 (no penalty)
        normal = Normal(torch.tensor(0.0, device=X.device),
                       torch.tensor(1.0, device=X.device))
        z = -(dists - self.radii.unsqueeze(0)) / self.s
        hammer = 1.0 - normal.cdf(z)  # [batch_shape*q, K]

        # Product over all selected points (multiplicative penalty)
        penalty = hammer.prod(dim=-1)  # [batch_shape*q]

        # Reshape back and aggregate over q dimension
        penalty = penalty.view(*original_shape)  # [batch_shape, q]
        if penalty.dim() > 1:
            penalty = penalty.prod(dim=-1)  # [batch_shape]

        return base_val * penalty


def local_penalization_batch(
    model,
    bounds: torch.Tensor,
    batch_size: int,
    base_acqf_class,
    acqf_kwargs: dict,
    lipschitz_constant: Optional[float] = None,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> torch.Tensor:
    """
    Select batch candidates using Local Penalization algorithm.

    Sequentially selects batch_size candidates by iteratively:
    1. Optimizing the (penalized) acquisition function
    2. Adding the selected point to the batch
    3. Creating a penalized acquisition for the next iteration

    This ensures batch diversity - later candidates are pushed away from
    earlier ones by the penalization term.

    Args:
        model: BoTorch GP model
        bounds: Search space bounds [2, D]
        batch_size: Number of candidates to select
        base_acqf_class: Base acquisition function class (e.g., qUpperConfidenceBound)
        acqf_kwargs: Keyword arguments for base acquisition function
        lipschitz_constant: Estimated Lipschitz constant (computed if None)
        num_restarts: Number of restarts for acquisition optimization
        raw_samples: Number of raw samples for initialization

    Returns:
        Batch candidates [batch_size, D]

    Example:
        >>> from botorch.acquisition import qUpperConfidenceBound
        >>> batch = local_penalization_batch(
        ...     model=gp.model,
        ...     bounds=bounds,
        ...     batch_size=4,
        ...     base_acqf_class=qUpperConfidenceBound,
        ...     acqf_kwargs={"beta": 1.96**2},
        ... )
    """
    # Lazy import to avoid circular dependencies
    from botorch.optim import optimize_acqf

    device = bounds.device

    # Estimate Lipschitz constant if not provided
    if lipschitz_constant is None:
        lipschitz_constant = estimate_lipschitz_constant(model, bounds)

    selected = []

    for i in range(batch_size):
        # Create base acquisition function
        acqf = base_acqf_class(model=model, **acqf_kwargs)

        # Apply penalization if we have already selected points
        if selected:
            X_selected = torch.stack(selected)  # [i, D]
            acqf = PenalizedAcquisition(
                base_acqf=acqf,
                X_selected=X_selected,
                L=lipschitz_constant,
                model=model,
            )

        # Optimize acquisition
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        selected.append(candidate.squeeze(0))  # [D]

    return torch.stack(selected)  # [batch_size, D]


def select_batch_candidates(
    gp,
    candidates: torch.Tensor,
    batch_size: int,
    method: str = "local_penalization",
    alpha: float = 1.96,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select diverse batch from pre-generated candidates.

    This is the main entry point for BOOptimizationLoop integration.
    More efficient than optimize_acqf when candidates are already
    generated by the flow model.

    Uses UCB acquisition with Local Penalization for diversity:
    UCB(x) = mu(x) + alpha * sigma(x)

    Args:
        gp: GP surrogate (SonarGPSurrogate, HeteroscedasticSonarGP, or similar)
            Must have .model attribute and .predict() method
        candidates: Pre-generated candidate embeddings [N, D]
        batch_size: Number of candidates to select
        method: Selection method ("local_penalization" or "greedy")
        alpha: UCB exploration weight (default 1.96 for ~95% CI)

    Returns:
        selected: Selected batch embeddings [batch_size, D]
        indices: Indices of selected candidates in original tensor [batch_size]

    Example:
        >>> gp = create_surrogate('msr', D=1024, device='cuda')
        >>> gp.fit(X_train, Y_train)
        >>> candidates = flow_model.sample(n_samples=100)
        >>> batch, indices = select_batch_candidates(gp, candidates, batch_size=4)
    """
    device = candidates.device
    N, D = candidates.shape

    # Handle edge case: batch_size >= N
    if batch_size >= N:
        return candidates, torch.arange(N, device=device)

    # Compute UCB values for all candidates
    with torch.no_grad():
        mean, std = gp.predict(candidates)
        ucb = mean + alpha * std  # [N]

    if method == "greedy":
        # Simple greedy: select top-K by UCB
        _, indices = ucb.topk(batch_size)
        return candidates[indices], indices

    elif method == "local_penalization":
        # Local Penalization for diverse batch selection
        model = gp.model

        # Estimate Lipschitz constant from GP mean gradients
        # Use candidate range as bounds for estimation
        cand_min = candidates.min(dim=0).values
        cand_max = candidates.max(dim=0).values
        # Add small margin to avoid singular bounds
        margin = 0.1 * (cand_max - cand_min).clamp(min=0.01)
        bounds = torch.stack([cand_min - margin, cand_max + margin])

        L = estimate_lipschitz_constant(model, bounds, n_samples=min(N, 200))

        # Sequentially select with penalization
        selected_indices = []
        remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
        current_ucb = ucb.clone()

        for i in range(batch_size):
            # Select best among remaining
            masked_ucb = current_ucb.clone()
            masked_ucb[~remaining_mask] = float('-inf')
            best_idx = masked_ucb.argmax()

            selected_indices.append(best_idx)
            remaining_mask[best_idx] = False

            # Apply penalization for next iteration
            if i < batch_size - 1:
                # Penalize UCB values near selected point
                x_selected = candidates[best_idx]  # [D]

                # Compute distances from remaining candidates to selected point
                dists = torch.norm(candidates - x_selected.unsqueeze(0), dim=-1)  # [N]

                # Penalization radius based on GP mean difference
                with torch.no_grad():
                    mu_selected = mean[best_idx]
                    min_Y = gp.model.train_targets.min()
                    radius = abs(mu_selected - min_Y) / L
                    radius = max(radius.item(), 0.01)  # minimum radius

                # Hammer function penalty
                s = 0.2  # smoothing
                normal = Normal(torch.tensor(0.0, device=device),
                               torch.tensor(1.0, device=device))
                z = -(dists - radius) / s
                penalty = 1.0 - normal.cdf(z)  # [N]

                # Apply multiplicative penalty to UCB
                current_ucb = current_ucb * penalty

        indices = torch.tensor(selected_indices, device=device)
        return candidates[indices], indices

    else:
        raise ValueError(f"Unknown method: {method}. Use 'local_penalization' or 'greedy'.")


__all__ = [
    "estimate_lipschitz_constant",
    "PenalizedAcquisition",
    "local_penalization_batch",
    "select_batch_candidates",
]
