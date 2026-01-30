"""Flow density computation via Hutchinson trace estimator.

This module implements density estimation for flow matching models using
the instantaneous change of variables formula and Hutchinson's trace estimator.

For a continuous normalizing flow, the log density is:
    log p(z_1) = log p(z_0) - integral_0^1 tr(d v(z_t, t) / d z_t) dt

The Hutchinson trace estimator computes:
    tr(A) = E[epsilon^T A epsilon]  where epsilon ~ N(0, I)

Key Functions:
- `hutchinson_trace_estimate`: Estimate trace of Jacobian via random probes
- `compute_flow_log_density`: Compute log p(z) by backward ODE integration
- `filter_by_flow_density`: Filter samples to keep high-density (on-manifold) ones

Note on Normalization:
    If the flow model was trained on normalized data, embeddings must be
    normalized before density computation. The flow operates in normalized
    space, so comparing density across different normalization scales is
    meaningless.

Reference:
    Chen et al. (2018) "Neural Ordinary Differential Equations"
    Grathwohl et al. (2019) "FFJORD: Free-form Continuous Dynamics for
        Scalable Reversible Generative Models"
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def hutchinson_trace_estimate(
    velocity_net: nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    n_hutchinson: int = 1,
) -> torch.Tensor:
    """
    Estimate tr(d v / d z) using Hutchinson trace estimator.

    The Hutchinson estimator computes:
        tr(A) = E[epsilon^T A epsilon]  where epsilon ~ N(0, I)

    For the Jacobian d v / d z, we compute:
        epsilon^T (dv/dz) epsilon = grad(v dot epsilon, z) dot epsilon

    This requires only vector-Jacobian products (vjp), which are efficient
    via autograd.

    Args:
        velocity_net: Velocity network v(z, t) -> [B, D]
        z: Input positions [B, D]
        t: Scalar timestep tensor (will be expanded to batch)
        n_hutchinson: Number of random probes for averaging (default 1)

    Returns:
        trace_estimate: Estimated trace of Jacobian [B]

    Example:
        >>> vel_net = VelocityNetwork(input_dim=1024, hidden_dim=512)
        >>> z = torch.randn(8, 1024)
        >>> t = torch.tensor(0.5)
        >>> trace = hutchinson_trace_estimate(vel_net, z, t, n_hutchinson=5)
    """
    B, D = z.shape
    device = z.device

    trace_estimates = []

    for _ in range(n_hutchinson):
        # Gaussian random probe vector
        epsilon = torch.randn(B, D, device=device)

        # Clone z and enable gradients for autograd
        z_var = z.clone().requires_grad_(True)

        # Expand scalar t to batch dimension
        t_expanded = t.expand(B) if t.dim() == 0 else t

        # Forward pass: v(z, t)
        v = velocity_net(z_var, t_expanded)  # [B, D]

        # Compute v dot epsilon
        v_dot_eps = (v * epsilon).sum(dim=-1)  # [B]

        # Gradient of (v dot epsilon) w.r.t z gives epsilon^T (dv/dz)
        grad_v_eps = torch.autograd.grad(
            v_dot_eps.sum(),
            z_var,
            create_graph=False,
            retain_graph=False,
        )[0]  # [B, D]

        # Trace estimate: (epsilon^T (dv/dz)) dot epsilon
        trace_est = (epsilon * grad_v_eps).sum(dim=-1)  # [B]
        trace_estimates.append(trace_est)

    # Average over Hutchinson samples
    trace_estimate = torch.stack(trace_estimates).mean(dim=0)  # [B]
    return trace_estimate


def compute_flow_log_density(
    flow_model,
    z_final: torch.Tensor,
    num_steps: int = 50,
    n_hutchinson: int = 1,
) -> torch.Tensor:
    """
    Compute log p(z) for flow model samples using backward ODE integration.

    Integrates backward from t=1 (data space) to t=0 (noise space):
        log p(z_1) = log p(z_0) - integral_0^1 tr(Jacobian) dt

    Since we integrate backward:
        z_{t-dt} = z_t - v(z_t, t) * dt
        log_density += tr(dv/dz) * dt

    At t=0, we have standard Gaussian prior:
        log p(z_0) = -0.5 * ||z_0||^2 - D/2 * log(2*pi)

    Args:
        flow_model: FlowMatchingModel with velocity_net attribute
        z_final: Samples at t=1 (data space) [B, D]
        num_steps: Number of integration steps (default 50)
        n_hutchinson: Number of Hutchinson samples per step (default 1)

    Returns:
        log_density: Log probability of each sample [B]

    Note:
        If flow_model operates in normalized space, z_final should be
        the normalized embeddings (before denormalization).

    Example:
        >>> flow = FlowMatchingModel(vel_net)
        >>> z = torch.randn(8, 1024)  # samples at t=1
        >>> log_p = compute_flow_log_density(flow, z, num_steps=50)
    """
    B, D = z_final.shape
    device = z_final.device

    velocity_net = flow_model.velocity_net
    velocity_net.eval()

    # Start from z at t=1, integrate backward to t=0
    z = z_final.clone()
    dt = 1.0 / num_steps

    # Accumulate negative trace integral (= sum of traces * dt)
    # Note: integral from 0 to 1 of tr(dv/dz) dt
    # But we integrate backward, so we add trace at each step
    accumulated_trace = torch.zeros(B, device=device)

    with torch.enable_grad():
        for i in range(num_steps):
            # Time decreases from 1 to 0
            t = torch.tensor(1.0 - i * dt, device=device)

            # Estimate trace of Jacobian
            trace = hutchinson_trace_estimate(
                velocity_net, z, t, n_hutchinson=n_hutchinson
            )

            # Accumulate trace integral
            accumulated_trace = accumulated_trace + trace * dt

            # Backward Euler step (no grad needed for trajectory)
            with torch.no_grad():
                t_expanded = t.expand(B)
                v = velocity_net(z, t_expanded)
                z = z - v * dt

    # z is now at t=0, should be approximately standard Gaussian
    # Log probability under standard Gaussian prior
    log_p_z0 = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

    # Change of variables: log p(z_1) = log p(z_0) - integral tr(dv/dz) dt
    log_density = log_p_z0 - accumulated_trace

    return log_density


def filter_by_flow_density(
    flow_model,
    embeddings: torch.Tensor,
    percentile: float = 10.0,
    min_samples: int = 4,
    num_steps: int = 30,
    n_hutchinson: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter embeddings by flow model density, keeping high-density samples.

    Low-density samples under the flow model are likely off-manifold and
    will decode poorly. This provides a complementary filter to L2-r.

    The percentile threshold rejects the bottom X% by log density.
    For example, percentile=25.0 keeps the top 75% of samples.

    Args:
        flow_model: FlowMatchingModel for density computation
        embeddings: Input embeddings [N, D]
        percentile: Reject samples below this percentile (default 10.0)
        min_samples: Minimum number of samples to keep (default 4)
        num_steps: ODE integration steps for density (default 30)
        n_hutchinson: Hutchinson samples per step (default 1)

    Returns:
        filtered_embeddings: High-density samples [M, D] where M >= min_samples
        log_densities: Log densities for ALL input samples [N]

    Note on Normalization:
        If flow_model operates in normalized space (has norm_stats),
        embeddings are automatically normalized before density computation.
        The returned filtered_embeddings are in the ORIGINAL (unnormalized)
        space to maintain consistency with the caller.

    Example:
        >>> flow = FlowMatchingModel(vel_net, norm_stats={'mean': ..., 'std': ...})
        >>> candidates = guided_sampler.sample(n_samples=64)
        >>> filtered, densities = filter_by_flow_density(flow, candidates, percentile=25)
        >>> print(f"Kept {len(filtered)}/{len(candidates)} samples")
    """
    N, D = embeddings.shape
    device = embeddings.device

    # If flow operates in normalized space, normalize embeddings for density computation
    embeddings_for_density = embeddings
    if hasattr(flow_model, 'norm_stats') and flow_model.norm_stats is not None:
        mean = flow_model.norm_stats['mean'].to(device)
        std = flow_model.norm_stats['std'].to(device)
        # Normalize: (x - mean) / std
        embeddings_for_density = (embeddings - mean) / std

    # Compute log densities
    log_densities = compute_flow_log_density(
        flow_model,
        embeddings_for_density,
        num_steps=num_steps,
        n_hutchinson=n_hutchinson,
    )

    # Compute threshold based on percentile
    # percentile=25 means reject bottom 25%, keep top 75%
    threshold = torch.quantile(log_densities, percentile / 100.0)

    # Create mask for samples above threshold (high density)
    high_density_mask = log_densities >= threshold

    # Ensure we keep at least min_samples
    n_above_threshold = high_density_mask.sum().item()
    if n_above_threshold < min_samples and N >= min_samples:
        # Keep top min_samples by density instead
        _, top_indices = log_densities.topk(min_samples)
        high_density_mask = torch.zeros(N, dtype=torch.bool, device=device)
        high_density_mask[top_indices] = True

    # Return filtered embeddings in ORIGINAL space (not normalized)
    filtered_embeddings = embeddings[high_density_mask]

    return filtered_embeddings, log_densities


__all__ = [
    "hutchinson_trace_estimate",
    "compute_flow_log_density",
    "filter_by_flow_density",
]
