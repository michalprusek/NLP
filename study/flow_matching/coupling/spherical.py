"""Spherical Flow Matching coupling using SLERP interpolation.

For data that lies on a hypersphere (like normalized SONAR embeddings),
SLERP provides geodesic (shortest path on sphere) interpolation.

This creates synergy with ArcCosine GP kernels which also operate on the sphere.

References:
- Shoemake (1985) "Animating rotation with quaternion curves"
- Chen & Lipman (2023) "Riemannian Flow Matching on General Geometries"
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def slerp_interpolate(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spherical Linear Interpolation (SLERP) and tangent velocity.

    SLERP follows the geodesic (great circle) path on the unit hypersphere:
    x_t = sin((1-t)*θ)/sin(θ) * x0 + sin(t*θ)/sin(θ) * x1

    The velocity (derivative w.r.t. t) is:
    v_t = -θ*cos((1-t)*θ)/sin(θ) * x0 + θ*cos(t*θ)/sin(θ) * x1

    Args:
        x0: Source samples (normalized noise) [B, D].
        x1: Target samples (normalized data) [B, D].
        t: Timesteps [B].
        eps: Small value for numerical stability.

    Returns:
        x_t: SLERP interpolated samples [B, D] (on unit sphere).
        v_t: Tangent velocity [B, D] (in tangent plane of sphere at x_t).
    """
    # Ensure inputs are normalized
    x0 = F.normalize(x0, p=2, dim=-1)
    x1 = F.normalize(x1, p=2, dim=-1)

    # Compute angle theta between x0 and x1
    # cos(θ) = x0 · x1
    dot = (x0 * x1).sum(dim=-1, keepdim=True)
    dot = dot.clamp(-1 + eps, 1 - eps)  # Numerical stability
    theta = torch.acos(dot)  # [B, 1]

    # sin(θ) for SLERP coefficients
    sin_theta = torch.sin(theta)

    # Handle edge case: when theta ≈ 0 (x0 ≈ x1), use linear interpolation
    # This avoids division by zero
    small_angle = sin_theta.abs() < eps

    # Expand t for broadcasting: [B] -> [B, 1]
    t_expanded = t.unsqueeze(-1)

    # SLERP coefficients for position x_t
    # c0 = sin((1-t)*θ) / sin(θ)
    # c1 = sin(t*θ) / sin(θ)
    c0 = torch.sin((1 - t_expanded) * theta) / (sin_theta + eps)
    c1 = torch.sin(t_expanded * theta) / (sin_theta + eps)

    # SLERP coefficients for velocity v_t (derivative w.r.t. t)
    # dc0/dt = -θ * cos((1-t)*θ) / sin(θ)
    # dc1/dt = θ * cos(t*θ) / sin(θ)
    dc0 = -theta * torch.cos((1 - t_expanded) * theta) / (sin_theta + eps)
    dc1 = theta * torch.cos(t_expanded * theta) / (sin_theta + eps)

    # Compute SLERP interpolation and velocity
    x_t = c0 * x0 + c1 * x1
    v_t = dc0 * x0 + dc1 * x1

    # For small angles, fall back to linear interpolation
    if small_angle.any():
        x_t_linear = (1 - t_expanded) * x0 + t_expanded * x1
        v_t_linear = x1 - x0
        x_t = torch.where(small_angle, x_t_linear, x_t)
        v_t = torch.where(small_angle, v_t_linear, v_t)

    # Normalize x_t to ensure it stays on unit sphere
    x_t = F.normalize(x_t, p=2, dim=-1)

    return x_t, v_t


class SphericalCoupling:
    """Spherical Flow Matching coupling using SLERP.

    Uses geodesic interpolation on the unit hypersphere instead of
    Euclidean linear interpolation. This is ideal for:
    - Normalized embeddings (SONAR, word2vec, etc.)
    - Use with ArcCosine or geodesic GP kernels
    - Generating samples that naturally lie on the sphere

    The flow learns to rotate vectors along great circles rather than
    interpolating through the interior of the sphere.

    Training:
        - x_t = SLERP(x0, x1, t) (position on geodesic)
        - v_t = d/dt SLERP(x0, x1, t) (tangent velocity)

    Inference:
        - Integrate dx/dt = v(x, t) while projecting x onto sphere
        - Or use exponential map integration for pure geodesic flow
    """

    def __init__(self, normalize_inputs: bool = True):
        """Initialize spherical coupling.

        Args:
            normalize_inputs: Whether to normalize x0, x1 to unit sphere.
                             Set True for unnormalized data (recommended).
        """
        self.normalize_inputs = normalize_inputs

    def sample(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample t, x_t, v_t for spherical velocity matching.

        Args:
            x0: Source samples (noise) [B, D].
            x1: Target samples (data) [B, D].

        Returns:
            t: Uniformly sampled timesteps [B].
            x_t: SLERP interpolated samples [B, D] (on unit sphere).
            v_t: Target tangent velocity [B, D].
        """
        # Normalize inputs to unit sphere if requested
        if self.normalize_inputs:
            x0 = F.normalize(x0, p=2, dim=-1)
            x1 = F.normalize(x1, p=2, dim=-1)

        # Sample time uniformly
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)

        # Compute SLERP interpolation and velocity
        x_t, v_t = slerp_interpolate(x0, x1, t)

        return t, x_t, v_t


class SphericalOTCoupling:
    """Spherical Flow Matching with Optimal Transport coupling.

    Combines OT pairing (from OT-CFM) with SLERP interpolation.
    This should give the best of both worlds:
    - OT: Straight paths in batch (minimal crossing)
    - SLERP: Geodesic paths on sphere (geometrically correct)
    """

    def __init__(
        self,
        reg: float = 0.5,
        normalize_cost: bool = True,
        normalize_inputs: bool = True,
    ):
        """Initialize spherical OT coupling.

        Args:
            reg: Sinkhorn regularization (lower = more OT, slower).
            normalize_cost: Normalize cost matrix by max value.
            normalize_inputs: Normalize x0, x1 to unit sphere.
        """
        self.reg = reg
        self.normalize_cost = normalize_cost
        self.normalize_inputs = normalize_inputs

    def _compute_ot_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Compute OT plan using geodesic distance on sphere."""
        # Normalize for distance computation
        x0_n = F.normalize(x0, p=2, dim=-1)
        x1_n = F.normalize(x1, p=2, dim=-1)

        # Geodesic distance: arccos(x0 · x1)
        cos_sim = x0_n @ x1_n.T
        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        cost = torch.acos(cos_sim)  # [B, B] geodesic distances

        if self.normalize_cost:
            cost = cost / (cost.max() + 1e-6)

        # Sinkhorn algorithm
        K = torch.exp(-cost / self.reg)
        n = x0.shape[0]
        u = torch.ones(n, device=x0.device) / n
        v = torch.ones(n, device=x0.device) / n

        for _ in range(50):  # Sinkhorn iterations
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)

        # Transport plan
        plan = torch.diag(u) @ K @ torch.diag(v)
        return plan

    def sample(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample with OT pairing and SLERP interpolation."""
        # Compute OT plan
        plan = self._compute_ot_plan(x0, x1)

        # Sample pairs according to plan
        indices = torch.multinomial(plan.flatten(), x0.shape[0], replacement=True)
        i_indices = indices // x0.shape[0]
        j_indices = indices % x0.shape[0]

        x0_paired = x0[i_indices]
        x1_paired = x1[j_indices]

        # Normalize if requested
        if self.normalize_inputs:
            x0_paired = F.normalize(x0_paired, p=2, dim=-1)
            x1_paired = F.normalize(x1_paired, p=2, dim=-1)

        # Sample time and compute SLERP
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        x_t, v_t = slerp_interpolate(x0_paired, x1_paired, t)

        return t, x_t, v_t
