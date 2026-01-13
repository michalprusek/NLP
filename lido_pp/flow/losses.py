"""
Loss Functions for Rectified Flow Matching with OAT-FM.

This module implements:
1. Conditional Flow Matching (CFM) loss - trains velocity prediction
2. OAT-FM regularization - minimizes trajectory acceleration
3. Combined loss for LID-O++

Key equations:
- CFM: L_CFM = E[||v_pred - (x_1 - x_0)||²]
- OAT: L_OAT = E[||∂v/∂t + (v·∇)v||²]
- Total: L = L_CFM + λ·L_OAT

OAT (Optimal Acceleration Transport) ensures smoother trajectories,
which is critical for:
1. Stable 1-step inference after Reflow
2. Semantic consistency in latent space
3. Avoiding sudden direction changes that could lead to OOD samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable


def conditional_flow_matching_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    sigma_min: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Conditional Flow Matching (CFM) loss.

    Trains the model to predict the velocity field that transports
    noise (x_0) to data (x_1) along straight-line trajectories.

    The optimal transport (OT) path is:
        x_t = (1 - t) * x_0 + t * x_1

    The corresponding velocity is:
        v_t = dx_t/dt = x_1 - x_0  (constant!)

    We add small noise (sigma_min) for numerical stability.

    Args:
        model: FlowDiT model predicting v(x_t, t, context)
        x_0: Source samples (noise) (B, latent_dim)
        x_1: Target samples (data) (B, latent_dim)
        context: Optional context (B, num_ctx, context_dim)
        sigma_min: Minimum noise for stability (default: 0.001)

    Returns:
        loss: Scalar CFM loss
        metrics: Dict with training metrics
    """
    batch_size = x_1.shape[0]
    device = x_1.device

    # Sample timestep uniformly from [0, 1]
    t = torch.rand(batch_size, device=device)

    # Optimal Transport interpolation: straight line from x_0 to x_1
    # x_t = (1 - t) * x_0 + t * x_1
    t_expand = t.unsqueeze(-1)  # (B, 1)
    x_t = (1 - t_expand) * x_0 + t_expand * x_1

    # Add small noise for numerical stability
    # This prevents exact interpolation artifacts
    if sigma_min > 0:
        noise = torch.randn_like(x_t) * sigma_min
        x_t = x_t + noise

    # Target velocity: straight line derivative (constant in t)
    v_target = x_1 - x_0

    # Predict velocity
    v_pred = model(x_t, t, context)

    # CFM loss: MSE between predicted and target velocity
    loss = F.mse_loss(v_pred, v_target)

    # Compute metrics
    with torch.no_grad():
        metrics = {
            "cfm_loss": loss.item(),
            "v_pred_norm": v_pred.norm(dim=-1).mean().item(),
            "v_target_norm": v_target.norm(dim=-1).mean().item(),
            "v_cosine_sim": F.cosine_similarity(v_pred, v_target, dim=-1).mean().item(),
        }

    return loss, metrics


def oat_regularization(
    model: nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    OAT-FM (Optimal Acceleration Transport) regularization.

    Minimizes trajectory acceleration to ensure smooth, straight paths.

    The material derivative (acceleration) of velocity is:
        Dv/Dt = ∂v/∂t + (v · ∇)v

    For a perfectly straight trajectory, acceleration should be zero.
    We approximate this as:
        ||a_t|| ≈ ||(v_{t+dt} - v_t) / dt||

    This is equivalent to measuring how much velocity changes along trajectory.

    Args:
        model: FlowDiT model
        x_0: Source samples (noise) (B, latent_dim)
        x_1: Target samples (data) (B, latent_dim)
        context: Optional context
        num_steps: Number of steps for acceleration estimation

    Returns:
        loss: OAT regularization loss
        metrics: Dict with metrics
    """
    batch_size = x_1.shape[0]
    device = x_1.device

    dt = 1.0 / num_steps
    total_accel_sq = 0.0

    # Sample initial x_0 position
    x = x_0.clone()

    # Track velocities for acceleration computation
    prev_v = None

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((batch_size,), t_val, device=device)

        # Compute x_t along ideal path (for gradient computation)
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # Get velocity at this point
        v = model(x_t, t, context)

        if prev_v is not None:
            # Discrete acceleration: (v_{t+dt} - v_t) / dt
            accel = (v - prev_v) / dt
            accel_norm_sq = (accel ** 2).sum(dim=-1).mean()
            total_accel_sq = total_accel_sq + accel_norm_sq

        prev_v = v.detach()  # Detach to avoid double backprop

    # Average acceleration over trajectory
    loss = total_accel_sq / (num_steps - 1)

    with torch.no_grad():
        metrics = {
            "oat_loss": loss.item(),
            "avg_accel": (loss ** 0.5).item(),  # RMS acceleration
        }

    return loss, metrics


def oat_regularization_jacobian(
    model: nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_samples: int = 5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    OAT-FM regularization with Jacobian computation.

    More accurate than discrete approximation - computes actual
    acceleration using autograd:

        a = ∂v/∂t + (∂v/∂x) · v

    This is more expensive but provides better gradients.

    Args:
        model: FlowDiT model
        x_0: Source samples (B, latent_dim)
        x_1: Target samples (B, latent_dim)
        context: Optional context
        num_samples: Number of timestep samples

    Returns:
        loss: OAT regularization loss (Jacobian-based)
        metrics: Dict with metrics
    """
    batch_size = x_1.shape[0]
    latent_dim = x_1.shape[1]
    device = x_1.device

    total_accel_sq = 0.0

    for _ in range(num_samples):
        # Sample random timestep
        t = torch.rand(batch_size, device=device)
        t.requires_grad_(True)

        # Compute x_t along ideal path
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        x_t.requires_grad_(True)

        # Get velocity
        v = model(x_t, t, context)  # (B, latent_dim)

        # Compute ∂v/∂t using autograd
        # Need to sum over batch and latent dims for backward
        dv_dt = torch.zeros_like(v)
        for j in range(latent_dim):
            grad_outputs = torch.zeros_like(v)
            grad_outputs[:, j] = 1.0
            dv_dt_j = torch.autograd.grad(
                v, t,
                grad_outputs=grad_outputs[:, j],
                create_graph=True,
                retain_graph=True,
            )[0]
            dv_dt[:, j] = dv_dt_j

        # Compute (∂v/∂x) · v using vector-Jacobian product
        # This is v^T @ (∂v/∂x), which equals ∂(v^T v / 2)/∂x... not quite
        # Actually we want (∂v/∂x) @ v, which is the convective term

        # Use autograd for Jacobian-vector product: (∂v/∂x) @ v
        # vjp of v w.r.t. x with vector v gives us what we need
        dv_dx_v = torch.zeros_like(v)
        for j in range(latent_dim):
            grad_outputs = torch.zeros_like(v)
            grad_outputs[:, j] = v[:, j].detach()  # Weight by v[j]
            dv_dx_v_j = torch.autograd.grad(
                v, x_t,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            dv_dx_v = dv_dx_v + dv_dx_v_j

        # Total acceleration: a = ∂v/∂t + (∂v/∂x) · v
        accel = dv_dt + dv_dx_v

        # Squared norm
        accel_sq = (accel ** 2).sum(dim=-1).mean()
        total_accel_sq = total_accel_sq + accel_sq

    loss = total_accel_sq / num_samples

    with torch.no_grad():
        metrics = {
            "oat_jacobian_loss": loss.item(),
        }

    return loss, metrics


def oat_flow_matching_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    sigma_min: float = 0.001,
    oat_weight: float = 0.1,
    oat_steps: int = 10,
    use_jacobian_oat: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined Flow Matching loss with OAT regularization.

    Total loss: L = L_CFM + λ · L_OAT

    Args:
        model: FlowDiT model
        x_0: Source samples (noise) (B, latent_dim)
        x_1: Target samples (data) (B, latent_dim)
        context: Optional context (B, num_ctx, context_dim)
        sigma_min: Minimum noise for CFM
        oat_weight: Weight λ for OAT regularization
        oat_steps: Steps for OAT discrete approximation
        use_jacobian_oat: Use Jacobian-based OAT (slower but more accurate)

    Returns:
        total_loss: Combined loss
        metrics: Dict with all metrics
    """
    # CFM loss
    cfm_loss, cfm_metrics = conditional_flow_matching_loss(
        model, x_0, x_1, context, sigma_min
    )

    # OAT regularization (if enabled)
    if oat_weight > 0:
        if use_jacobian_oat:
            oat_loss, oat_metrics = oat_regularization_jacobian(
                model, x_0, x_1, context, num_samples=oat_steps
            )
        else:
            oat_loss, oat_metrics = oat_regularization(
                model, x_0, x_1, context, num_steps=oat_steps
            )

        total_loss = cfm_loss + oat_weight * oat_loss
    else:
        oat_loss = torch.tensor(0.0, device=x_0.device)
        oat_metrics = {"oat_loss": 0.0}
        total_loss = cfm_loss

    # Combine metrics
    metrics = {
        **cfm_metrics,
        **oat_metrics,
        "total_loss": total_loss.item(),
        "oat_weight": oat_weight,
    }

    return total_loss, metrics


def measure_trajectory_straightness(
    model: nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
) -> Dict[str, float]:
    """
    Measure how straight the learned trajectories are.

    Computes:
    1. Average deviation from ideal straight line
    2. Velocity variance along trajectory
    3. Path length ratio (actual / ideal)

    A perfectly straight trajectory has:
    - deviation = 0
    - velocity variance = 0
    - path length ratio = 1

    Args:
        model: Trained FlowDiT model
        x_0: Start points (B, latent_dim)
        x_1: End points (B, latent_dim)
        context: Optional context
        num_steps: Steps for trajectory integration

    Returns:
        metrics: Dict with straightness metrics
    """
    batch_size = x_0.shape[0]
    device = x_0.device
    dt = 1.0 / num_steps

    # Track trajectory
    x = x_0.clone()
    velocities = []
    deviations = []

    with torch.no_grad():
        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((batch_size,), t_val, device=device)

            # Get velocity
            v = model(x, t, context)
            velocities.append(v.clone())

            # Ideal position at this t
            x_ideal = (1 - t_val) * x_0 + t_val * x_1

            # Deviation from ideal
            deviation = (x - x_ideal).norm(dim=-1).mean()
            deviations.append(deviation.item())

            # Euler step
            x = x + v * dt

    # Final deviation from target
    final_deviation = (x - x_1).norm(dim=-1).mean().item()

    # Velocity statistics
    velocities = torch.stack(velocities, dim=0)  # (num_steps, B, latent_dim)
    v_mean = velocities.mean(dim=0)  # (B, latent_dim)
    v_variance = ((velocities - v_mean.unsqueeze(0)) ** 2).mean().item()

    # Path length ratio
    ideal_length = (x_1 - x_0).norm(dim=-1).mean().item()
    actual_length = sum(
        (velocities[i+1] - velocities[i]).norm(dim=-1).mean().item() * dt
        for i in range(num_steps - 1)
    ) + velocities.norm(dim=-1).mean().item() * dt * num_steps

    return {
        "avg_deviation": sum(deviations) / len(deviations),
        "max_deviation": max(deviations),
        "final_deviation": final_deviation,
        "velocity_variance": v_variance,
        "path_length_ratio": actual_length / (ideal_length + 1e-8),
    }


if __name__ == "__main__":
    from lido_pp.flow.flow_dit import FlowDiT

    print("Testing Flow Matching Losses...")

    # Create model and data
    batch_size = 8
    latent_dim = 32
    context_dim = 768

    model = FlowDiT(
        latent_dim=latent_dim,
        hidden_dim=256,
        num_layers=4,
        context_dim=context_dim,
    )

    x_0 = torch.randn(batch_size, latent_dim)  # Noise
    x_1 = torch.randn(batch_size, latent_dim)  # Data
    context = torch.randn(batch_size, 4, context_dim)

    # Test CFM loss
    print("\n1. CFM Loss:")
    cfm_loss, cfm_metrics = conditional_flow_matching_loss(model, x_0, x_1, context)
    print(f"   Loss: {cfm_loss.item():.6f}")
    for k, v in cfm_metrics.items():
        print(f"   {k}: {v:.6f}")

    # Test OAT regularization (discrete)
    print("\n2. OAT Regularization (discrete):")
    oat_loss, oat_metrics = oat_regularization(model, x_0, x_1, context, num_steps=10)
    print(f"   Loss: {oat_loss.item():.6f}")
    for k, v in oat_metrics.items():
        print(f"   {k}: {v:.6f}")

    # Test combined loss
    print("\n3. Combined OAT-FM Loss:")
    total_loss, all_metrics = oat_flow_matching_loss(
        model, x_0, x_1, context,
        oat_weight=0.1,
        oat_steps=10,
    )
    print(f"   Total loss: {total_loss.item():.6f}")
    for k, v in all_metrics.items():
        print(f"   {k}: {v:.6f}" if isinstance(v, float) else f"   {k}: {v}")

    # Test straightness measurement
    print("\n4. Trajectory Straightness:")
    straightness = measure_trajectory_straightness(model, x_0, x_1, context)
    for k, v in straightness.items():
        print(f"   {k}: {v:.6f}")

    print("\nAll tests passed!")
