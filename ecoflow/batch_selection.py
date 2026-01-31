"""Batch selection for Bayesian optimization with Local Penalization.

This module implements batch selection strategies for selecting multiple
candidates per iteration in BO loops. Uses Local Penalization (Gonzalez et al.,
2016) to ensure batch diversity by penalizing points near already-selected ones.

Reference:
    Gonzalez et al. (2016) "Batch Bayesian Optimization via Local Penalization"
    https://arxiv.org/abs/1505.08052
"""

import logging
import math
from typing import Tuple

import torch
from torch.distributions import Normal

logger = logging.getLogger(__name__)


def _estimate_lipschitz_constant(
    model,
    bounds: torch.Tensor,
    n_samples: int = 200,
) -> float:
    """
    Estimate Lipschitz constant of GP mean via gradient sampling.

    L = max_x ||grad mu(x)||

    Args:
        model: BoTorch GP model
        bounds: Search space bounds [2, D]
        n_samples: Number of random samples for gradient estimation

    Returns:
        Estimated Lipschitz constant (minimum 1e-7)
    """
    device = bounds.device
    D = bounds.shape[1]

    X = torch.rand(n_samples, D, device=device)
    X = bounds[0] + X * (bounds[1] - bounds[0])
    X.requires_grad_(True)

    model.eval()
    with torch.enable_grad():
        posterior = model.posterior(X)
        mean = posterior.mean.squeeze(-1)

        grads = torch.autograd.grad(mean.sum(), X, create_graph=False)[0]

    grad_norms = grads.norm(dim=-1)
    L = grad_norms.max().item()

    if not math.isfinite(L):
        logger.error(
            "Lipschitz estimate is non-finite. Using fallback L=1.0. "
            "Batch selection diversity may be compromised."
        )
        return 1.0

    return max(L, 1e-7)


def select_batch_candidates(
    gp,
    candidates: torch.Tensor,
    batch_size: int,
    method: str = "local_penalization",
    alpha: float = 1.96,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select diverse batch from pre-generated candidates using Local Penalization.

    Uses UCB acquisition with sequential penalization for diversity:
    UCB(x) = mu(x) + alpha * sigma(x)

    Args:
        gp: GP surrogate with .model attribute and .predict() method
        candidates: Pre-generated candidate embeddings [N, D]
        batch_size: Number of candidates to select
        method: "local_penalization" for diverse selection, "greedy" for top-K UCB
        alpha: UCB exploration weight (default 1.96 for ~95% CI)

    Returns:
        selected: Selected batch embeddings [batch_size, D]
        indices: Indices of selected candidates in original tensor [batch_size]
    """
    device = candidates.device
    N, D = candidates.shape

    if batch_size >= N:
        return candidates, torch.arange(N, device=device)

    with torch.no_grad():
        mean, std = gp.predict(candidates)
        ucb = mean + alpha * std

    if method == "greedy":
        _, indices = ucb.topk(batch_size)
        return candidates[indices], indices

    if method == "local_penalization":
        model = gp.model

        # Compute Lipschitz constant for penalization radius
        if hasattr(gp, "_embed") and hasattr(gp, "S"):
            candidates_for_lipschitz = gp._embed(candidates)
        else:
            candidates_for_lipschitz = candidates

        cand_min = candidates_for_lipschitz.min(dim=0).values
        cand_max = candidates_for_lipschitz.max(dim=0).values
        margin = 0.1 * (cand_max - cand_min).clamp(min=0.01)
        bounds = torch.stack([cand_min - margin, cand_max + margin])

        L = _estimate_lipschitz_constant(model, bounds, n_samples=min(N, 200))

        # Sequential selection with penalization
        selected_indices = []
        remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
        current_ucb = ucb.clone()

        for i in range(batch_size):
            masked_ucb = current_ucb.clone()
            masked_ucb[~remaining_mask] = float("-inf")
            best_idx = masked_ucb.argmax()

            selected_indices.append(best_idx)
            remaining_mask[best_idx] = False

            if i < batch_size - 1:
                x_selected = candidates[best_idx]
                dists = torch.norm(candidates - x_selected.unsqueeze(0), dim=-1)

                with torch.no_grad():
                    mu_selected = mean[best_idx]
                    min_Y = gp.model.train_targets.min()
                    radius = abs(mu_selected - min_Y) / L
                    radius = max(radius.item(), 0.01)

                # Hammer function penalty
                s = 0.2
                normal = Normal(
                    torch.tensor(0.0, device=device),
                    torch.tensor(1.0, device=device),
                )
                z = -(dists - radius) / s
                penalty = 1.0 - normal.cdf(z)

                current_ucb = current_ucb * penalty

        indices = torch.tensor(selected_indices, device=device)
        return candidates[indices], indices

    raise ValueError(f"Unknown method: {method}. Use 'local_penalization' or 'greedy'.")


__all__ = ["select_batch_candidates"]
