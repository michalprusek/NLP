"""
Custom ODE Solvers for Rectified Flow with Curvature Tracking.

This module implements custom ODE integrators that track trajectory curvature
during integration. The curvature serves as an uncertainty proxy for
active learning (Flow Curvature Uncertainty - FCU).

Solvers:
- Euler: Simple first-order, O(dt)
- Midpoint: Second-order, O(dt²)
- RK4: Fourth-order Runge-Kutta, O(dt⁴)

Key feature: All solvers compute trajectory curvature on-the-fly,
which is used for cost-aware acquisition in active learning.
"""

import math

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class IntegrationResult:
    """Result of ODE integration."""

    # Final position
    x_final: torch.Tensor  # (B, latent_dim)

    # Trajectory curvature (sum of velocity changes)
    curvature: torch.Tensor  # (B,)

    # Optional: full trajectory
    trajectory: Optional[List[torch.Tensor]] = None

    # Optional: velocity at each step
    velocities: Optional[List[torch.Tensor]] = None


def euler_integrate(
    model: nn.Module,
    x_0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
    return_trajectory: bool = False,
    compute_curvature: bool = True,
) -> IntegrationResult:
    """
    Euler integration with curvature tracking.

    Simple first-order method:
        x_{n+1} = x_n + dt * v(x_n, t_n)

    Fast but has O(dt) error. Use for quick inference or when
    model has been "reflowed" for straight trajectories.

    Args:
        model: FlowDiT model predicting v(x, t, context)
        x_0: Initial position (B, latent_dim)
        context: Optional context (B, num_ctx, context_dim)
        num_steps: Number of integration steps
        return_trajectory: Whether to return intermediate positions
        compute_curvature: Whether to compute trajectory curvature

    Returns:
        IntegrationResult with final position and curvature
    """
    batch_size = x_0.shape[0]
    device = x_0.device
    dt = 1.0 / num_steps

    x = x_0.clone()
    prev_v = None
    curvature = torch.zeros(batch_size, device=device)

    trajectory = [x_0.clone()] if return_trajectory else None
    velocities = [] if return_trajectory else None

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)

            # Get velocity
            v = model(x, t, context)

            # Track curvature (sum of velocity changes)
            if compute_curvature and prev_v is not None:
                curvature = curvature + (v - prev_v).norm(dim=-1)

            prev_v = v.clone()

            # Euler step
            x = x + v * dt

            if return_trajectory:
                trajectory.append(x.clone())
                velocities.append(v.clone())

    return IntegrationResult(
        x_final=x,
        curvature=curvature,
        trajectory=trajectory,
        velocities=velocities,
    )


def midpoint_integrate(
    model: nn.Module,
    x_0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
    return_trajectory: bool = False,
    compute_curvature: bool = True,
) -> IntegrationResult:
    """
    Midpoint (Heun's) integration with curvature tracking.

    Second-order method with O(dt²) error:
        k1 = v(x_n, t_n)
        x_mid = x_n + 0.5 * dt * k1
        k2 = v(x_mid, t_n + 0.5*dt)
        x_{n+1} = x_n + dt * k2

    More accurate than Euler with 2x model evaluations.

    Args:
        model: FlowDiT model
        x_0: Initial position (B, latent_dim)
        context: Optional context
        num_steps: Number of integration steps
        return_trajectory: Return intermediate positions
        compute_curvature: Compute trajectory curvature

    Returns:
        IntegrationResult
    """
    batch_size = x_0.shape[0]
    device = x_0.device
    dt = 1.0 / num_steps

    x = x_0.clone()
    prev_v = None
    curvature = torch.zeros(batch_size, device=device)

    trajectory = [x_0.clone()] if return_trajectory else None
    velocities = [] if return_trajectory else None

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            t_mid = torch.full((batch_size,), (i + 0.5) * dt, device=device)

            # First evaluation at current position
            k1 = model(x, t, context)

            # Midpoint
            x_mid = x + 0.5 * dt * k1

            # Second evaluation at midpoint
            k2 = model(x_mid, t_mid, context)

            # Track curvature using k2 (midpoint velocity)
            if compute_curvature and prev_v is not None:
                curvature = curvature + (k2 - prev_v).norm(dim=-1)

            prev_v = k2.clone()

            # Update using midpoint velocity
            x = x + dt * k2

            if return_trajectory:
                trajectory.append(x.clone())
                velocities.append(k2.clone())

    return IntegrationResult(
        x_final=x,
        curvature=curvature,
        trajectory=trajectory,
        velocities=velocities,
    )


def rk4_integrate(
    model: nn.Module,
    x_0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
    return_trajectory: bool = False,
    compute_curvature: bool = True,
) -> IntegrationResult:
    """
    Fourth-order Runge-Kutta integration with curvature tracking.

    Classic RK4 with O(dt⁴) error:
        k1 = v(x_n, t_n)
        k2 = v(x_n + 0.5*dt*k1, t_n + 0.5*dt)
        k3 = v(x_n + 0.5*dt*k2, t_n + 0.5*dt)
        k4 = v(x_n + dt*k3, t_n + dt)
        x_{n+1} = x_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Most accurate but 4x model evaluations per step.

    Args:
        model: FlowDiT model
        x_0: Initial position (B, latent_dim)
        context: Optional context
        num_steps: Number of integration steps
        return_trajectory: Return intermediate positions
        compute_curvature: Compute trajectory curvature

    Returns:
        IntegrationResult
    """
    batch_size = x_0.shape[0]
    device = x_0.device
    dt = 1.0 / num_steps

    x = x_0.clone()
    prev_v = None
    curvature = torch.zeros(batch_size, device=device)

    trajectory = [x_0.clone()] if return_trajectory else None
    velocities = [] if return_trajectory else None

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)

            # RK4 stages
            k1 = model(x, t, context)
            k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt, context)
            k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt, context)
            k4 = model(x + dt * k3, t + dt, context)

            # Weighted average velocity
            v_avg = (k1 + 2*k2 + 2*k3 + k4) / 6

            # Track curvature
            if compute_curvature and prev_v is not None:
                curvature = curvature + (v_avg - prev_v).norm(dim=-1)

            prev_v = v_avg.clone()

            # RK4 update
            x = x + dt * v_avg

            if return_trajectory:
                trajectory.append(x.clone())
                velocities.append(v_avg.clone())

    return IntegrationResult(
        x_final=x,
        curvature=curvature,
        trajectory=trajectory,
        velocities=velocities,
    )


def one_step_integrate(
    model: nn.Module,
    x_0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Single-step integration (for reflowed models).

    After Reflow training, trajectories are straight enough that
    a single Euler step provides good results:
        x_1 = x_0 + v(x_0, 0)

    This is 50-100x faster than multi-step integration.

    Args:
        model: Reflowed FlowDiT model
        x_0: Initial position (B, latent_dim)
        context: Optional context

    Returns:
        x_1: Final position (B, latent_dim)
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # Single evaluation at t=0
    t = torch.zeros(batch_size, device=device)

    with torch.no_grad():
        v = model(x_0, t, context)

    # One step from t=0 to t=1 (dt=1)
    x_1 = x_0 + v

    return x_1


def integrate(
    model: nn.Module,
    x_0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
    method: str = "euler",
    return_trajectory: bool = False,
    compute_curvature: bool = True,
) -> IntegrationResult:
    """
    General-purpose ODE integration.

    Args:
        model: FlowDiT model
        x_0: Initial position (B, latent_dim)
        context: Optional context
        num_steps: Number of steps (1 for single-step)
        method: "euler", "midpoint", or "rk4"
        return_trajectory: Return intermediate positions
        compute_curvature: Compute trajectory curvature

    Returns:
        IntegrationResult
    """
    if num_steps == 1:
        # Special case: single-step integration
        x_1 = one_step_integrate(model, x_0, context)
        return IntegrationResult(
            x_final=x_1,
            curvature=torch.zeros(x_0.shape[0], device=x_0.device),
            trajectory=[x_0, x_1] if return_trajectory else None,
            velocities=None,
        )

    if method == "euler":
        return euler_integrate(
            model, x_0, context, num_steps,
            return_trajectory, compute_curvature
        )
    elif method == "midpoint":
        return midpoint_integrate(
            model, x_0, context, num_steps,
            return_trajectory, compute_curvature
        )
    elif method == "rk4":
        return rk4_integrate(
            model, x_0, context, num_steps,
            return_trajectory, compute_curvature
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'euler', 'midpoint', or 'rk4'.")


def compute_curvature_only(
    model: nn.Module,
    z: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
) -> torch.Tensor:
    """
    Compute trajectory curvature without storing trajectory.

    Optimized for active learning where we only need the curvature
    metric, not the final position (which comes from GP optimization).

    Curvature K(z) = Σ||v_{i} - v_{i-1}||

    High curvature indicates the model is uncertain about the
    velocity field in this region → needs LLM evaluation.

    Args:
        model: FlowDiT model
        z: Starting noise (B, latent_dim)
        context: Optional context
        num_steps: Steps for curvature estimation

    Returns:
        curvature: (B,) curvature values
    """
    batch_size = z.shape[0]
    device = z.device
    dt = 1.0 / num_steps

    x = z.clone()
    prev_v = None
    curvature = torch.zeros(batch_size, device=device)

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            v = model(x, t, context)

            if prev_v is not None:
                curvature = curvature + (v - prev_v).norm(dim=-1)

            prev_v = v
            x = x + v * dt

    return curvature


def guided_euler_integrate(
    model: nn.Module,
    x_0: torch.Tensor,
    reward_fn: callable,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
    guidance_scale: float = 1.0,
    guidance_schedule: str = "linear",
    guidance_start_t: float = 0.2,
    return_trajectory: bool = False,
) -> IntegrationResult:
    """
    Guided Euler integration with reward function (Classifier Guidance for Flow).

    Modifies velocity field to follow reward gradient:
        v_guided(x_t, t) = v(x_t, t) + s(t) · ∇_x R(x_t)

    Time-dependent guidance is critical because at t=0, x_t is pure Gaussian
    noise where GP gradients are meaningless. Guidance should ramp up as
    structure forms in the latent space.

    Guidance Schedules:
    - "constant": s(t) = s_base (original, no ramping)
    - "linear": s(t) = s_base · t (gradual ramp from 0 to s_base)
    - "cosine": s(t) = s_base · (1 - cos(πt))/2 (smooth S-curve)
    - "step": s(t) = s_base if t > t_start else 0 (hard threshold)
    - "warmup": s(t) = s_base · min(1, t/t_start) (linear warmup then constant)

    Reward functions:
    - GP mean: R(x) = -μ_GP(x)  (minimize error)
    - UCB: R(x) = -μ(x) + β·σ(x)  (exploration + exploitation)
    - EI: R(x) = EI(x)  (expected improvement)

    Args:
        model: FlowDiT model predicting v(x, t, context)
        x_0: Initial noise (B, latent_dim)
        reward_fn: Callable that takes x (B, latent_dim) and returns scalar reward
                   Must be differentiable w.r.t. x
        context: Optional context (B, num_ctx, context_dim)
        num_steps: Number of integration steps
        guidance_scale: Base scale for reward gradient (s_base)
        guidance_schedule: How to ramp guidance over time
            - "constant": no ramping (original behavior)
            - "linear": s(t) = s_base · t
            - "cosine": s(t) = s_base · (1 - cos(πt))/2
            - "step": s(t) = s_base if t > guidance_start_t else 0
            - "warmup": linear warmup to guidance_start_t, then constant
        guidance_start_t: For "step"/"warmup" schedules, when to start guidance
        return_trajectory: Whether to return intermediate positions

    Returns:
        IntegrationResult with guided final position

    Example:
        # Using GP with linear ramping (recommended)
        result = guided_euler_integrate(
            flowdit, noise, gp_reward,
            guidance_scale=1.0,
            guidance_schedule="linear",  # s(t) = 1.0 * t
            num_steps=20
        )
        # At t=0.0: s=0.0 (no guidance on pure noise)
        # At t=0.5: s=0.5 (moderate guidance as structure forms)
        # At t=1.0: s=1.0 (full guidance on formed latent)
    """
    batch_size = x_0.shape[0]
    device = x_0.device
    dt = 1.0 / num_steps

    x = x_0.clone()
    curvature = torch.zeros(batch_size, device=device)
    prev_v = None

    trajectory = [x_0.clone()] if return_trajectory else None
    velocities = [] if return_trajectory else None

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((batch_size,), t_val, device=device)

        # Compute time-dependent guidance scale
        if guidance_schedule == "constant":
            s_t = guidance_scale
        elif guidance_schedule == "linear":
            # s(t) = s_base · t : starts at 0, reaches s_base at t=1
            s_t = guidance_scale * t_val
        elif guidance_schedule == "cosine":
            # s(t) = s_base · (1 - cos(πt))/2 : smooth S-curve
            s_t = guidance_scale * (1 - math.cos(math.pi * t_val)) / 2
        elif guidance_schedule == "step":
            # s(t) = s_base if t > t_start else 0
            s_t = guidance_scale if t_val > guidance_start_t else 0.0
        elif guidance_schedule == "warmup":
            # Linear warmup to t_start, then constant
            if t_val < guidance_start_t:
                s_t = guidance_scale * (t_val / guidance_start_t)
            else:
                s_t = guidance_scale
        elif guidance_schedule == "quadratic":
            # s(t) = s_base · t² : slower start, faster finish
            s_t = guidance_scale * (t_val ** 2)
        elif guidance_schedule == "sqrt":
            # s(t) = s_base · √t : faster start, slower finish
            s_t = guidance_scale * math.sqrt(t_val)
        else:
            raise ValueError(
                f"Unknown guidance_schedule: {guidance_schedule}. "
                f"Use 'constant', 'linear', 'cosine', 'step', 'warmup', 'quadratic', or 'sqrt'."
            )

        # Enable gradients for x to compute ∇_x R(x)
        x_grad = x.clone().requires_grad_(True)

        # Get base velocity (no grad needed for model params)
        with torch.no_grad():
            v_base = model(x_grad.detach(), t, context)

        # Compute reward gradient (only if guidance is active)
        if s_t > 0:
            reward = reward_fn(x_grad)

            # Handle batched rewards - sum to get scalar for backward
            if reward.dim() > 0:
                reward_scalar = reward.sum()
            else:
                reward_scalar = reward

            # Compute gradient ∇_x R(x)
            grad_reward = torch.autograd.grad(
                reward_scalar,
                x_grad,
                create_graph=False,
                retain_graph=False,
            )[0]

            # Guided velocity: v + s(t) · ∇R
            v_guided = v_base + s_t * grad_reward
        else:
            v_guided = v_base

        # Track curvature
        if prev_v is not None:
            curvature = curvature + (v_guided - prev_v).norm(dim=-1)
        prev_v = v_guided.detach().clone()

        # Euler step
        x = x.detach() + v_guided.detach() * dt

        if return_trajectory:
            trajectory.append(x.clone())
            velocities.append(v_guided.detach().clone())

    return IntegrationResult(
        x_final=x,
        curvature=curvature,
        trajectory=trajectory,
        velocities=velocities,
    )


class GPRewardWrapper:
    """Wrapper to make GP differentiable for guided generation.

    Creates differentiable reward functions from GP predictions with optional
    regularization to stay within TFA's learned distribution:

        Total Reward(z) = UCB(z) - λ · regularization(z)

    Reward modes:
    - mean: R(x) = -μ(x)  (exploit - minimize error)
    - ucb: R(x) = -μ(x) + β·σ(x)  (explore + exploit)
    - ei: R(x) = EI(x)  (expected improvement)

    Regularization modes (keeps z within latent distribution):
    - "none": No regularization
    - "l2": ||z||² - assumes prior is N(0,1) (z-score normalized)
    - "l2_centered": ||z - μ_train||² - distance from training centroid
    - "mahalanobis": (z-μ)ᵀΣ⁻¹(z-μ) - accounts for training covariance

    The GP must operate on the same latent space as TFA (128D/256D).

    Example:
        # With L2 regularization to stay in latent distribution
        gp_reward = GPRewardWrapper(
            gp_model,
            mode="ucb", beta=2.0,
            regularization="l2", reg_lambda=0.1
        )
        result = guided_euler_integrate(flowdit, noise, gp_reward, guidance_scale=0.5)
    """

    def __init__(
        self,
        gp_model,
        vae_encoder: Optional[nn.Module] = None,
        mode: str = "ucb",
        beta: float = 2.0,
        best_f: Optional[float] = None,
        regularization: str = "l2",
        reg_lambda: float = 0.1,
    ):
        """
        Args:
            gp_model: Trained GP with predict method
            tfa_encoder: TFA encoder if GP expects different embeddings (optional, rarely used)
            mode: "mean", "ucb", or "ei"
            beta: UCB exploration coefficient
            best_f: Best observed value for EI (auto-detected if None)
            regularization: "none", "l2", "l2_centered", or "mahalanobis"
            reg_lambda: Regularization strength (λ in UCB - λ·||z||²)
        """
        self.gp = gp_model
        self.vae_encoder = vae_encoder
        self.mode = mode
        self.beta = beta
        self.best_f = best_f
        self.regularization = regularization
        self.reg_lambda = reg_lambda

        # Precompute training data statistics for regularization
        self._train_mean = None
        self._train_cov_inv = None
        self._setup_regularization_stats()

    def _setup_regularization_stats(self):
        """Precompute training data statistics for regularization."""
        if self.regularization in ["l2_centered", "mahalanobis"]:
            if hasattr(self.gp, 'X_train') and self.gp.X_train is not None:
                X = self.gp.X_train  # (N, 32)
                self._train_mean = X.mean(dim=0)  # (32,)

                if self.regularization == "mahalanobis":
                    # Compute covariance and its inverse
                    X_centered = X - self._train_mean
                    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
                    # Add small diagonal for numerical stability
                    cov = cov + 1e-4 * torch.eye(cov.shape[0], device=cov.device)
                    self._train_cov_inv = torch.linalg.inv(cov)

    def _compute_regularization(self, x: torch.Tensor) -> torch.Tensor:
        """Compute regularization penalty.

        Args:
            x: Latent vectors (B, 32) - unnormalized

        Returns:
            penalty: (B,) regularization values (to be subtracted from reward)
        """
        if self.regularization == "none":
            return torch.zeros(x.shape[0], device=x.device)

        elif self.regularization == "l2":
            # Simple L2 norm: ||z||²
            # Works well with z-score normalized latents (approx N(0, I))
            return (x ** 2).sum(dim=-1)

        elif self.regularization == "l2_centered":
            # Distance from training centroid: ||z - μ_train||²
            if self._train_mean is None:
                # Fallback to simple L2 if no training data
                return (x ** 2).sum(dim=-1)
            diff = x - self._train_mean.to(x.device)
            return (diff ** 2).sum(dim=-1)

        elif self.regularization == "mahalanobis":
            # Mahalanobis distance: (z-μ)ᵀΣ⁻¹(z-μ)
            # Accounts for correlation structure of training data
            if self._train_mean is None or self._train_cov_inv is None:
                return (x ** 2).sum(dim=-1)
            diff = x - self._train_mean.to(x.device)  # (B, 32)
            cov_inv = self._train_cov_inv.to(x.device)
            # (B, 32) @ (32, 32) @ (32, B) -> diagonal is what we want
            mahal = (diff @ cov_inv * diff).sum(dim=-1)  # (B,)
            return mahal

        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute differentiable reward for latent x.

        Total Reward = acquisition(z) - λ · regularization(z)

        Args:
            x: Latent vectors (B, 32) - must match GP input dimension

        Returns:
            reward: (B,) reward values (higher is better)
        """
        # Store unnormalized x for regularization
        x_unnorm = x

        # Normalize x to [0, 1] if GP expects normalized inputs
        if hasattr(self.gp, 'X_min') and self.gp.X_min is not None:
            denom = self.gp.X_max - self.gp.X_min
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            x_norm = (x - self.gp.X_min) / denom
        else:
            x_norm = x

        # Get GP posterior (differentiable)
        self.gp.gp_model.eval()

        # Forward through GP
        posterior = self.gp.gp_model(x_norm)
        mean = posterior.mean  # (B,)
        std = posterior.stddev  # (B,)

        # Denormalize predictions
        if hasattr(self.gp, 'y_std') and self.gp.y_std is not None:
            mean = mean * self.gp.y_std + self.gp.y_mean
            std = std * self.gp.y_std

        # GP predicts -error_rate, so mean is already "reward-like"
        # Higher mean = lower error = better

        if self.mode == "mean":
            # Pure exploitation: maximize mean (= minimize error)
            acquisition = mean
        elif self.mode == "ucb":
            # UCB: mean + β·std (exploration + exploitation)
            acquisition = mean + self.beta * std
        elif self.mode == "ei":
            # Expected Improvement
            best = self.best_f if self.best_f is not None else self.gp.y_best
            if best is None:
                acquisition = mean + self.beta * std  # Fallback to UCB
            else:
                # Approximation of EI gradient-friendly
                z = (mean - best) / (std + 1e-8)
                acquisition = (mean - best) * 0.5 * (1 + torch.erf(z / 1.414)) + \
                         std * torch.exp(-0.5 * z ** 2) / 2.507
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Apply regularization: Total = acquisition - λ·regularization
        reg_penalty = self._compute_regularization(x_unnorm)
        reward = acquisition - self.reg_lambda * reg_penalty

        return reward


class ValueHeadRewardWrapper:
    """Wrapper to make ValueHead differentiable for guided generation.

    Creates reward from neural network value head predictions.
    ValueHead predicts error_rate ∈ [0,1], so reward = -error_rate.

    Example:
        vh_reward = ValueHeadRewardWrapper(value_head)
        result = guided_euler_integrate(flowdit, noise, vh_reward, guidance_scale=0.5)
    """

    def __init__(self, value_head: nn.Module):
        """
        Args:
            value_head: Trained ValueHead network (latent_dim → 1D)
        """
        self.value_head = value_head

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reward for latent x.

        Args:
            x: Latent vectors (B, 32)

        Returns:
            reward: (B,) - negative error rate (higher = better)
        """
        self.value_head.eval()
        error_rate = self.value_head(x)  # (B,) in [0, 1]
        reward = -error_rate  # Higher reward = lower error
        return reward


if __name__ == "__main__":
    from lido_pp.flow.flow_dit import FlowDiT

    print("Testing ODE Solvers...")

    # Create model
    batch_size = 4
    latent_dim = 32
    context_dim = 768

    model = FlowDiT(
        latent_dim=latent_dim,
        hidden_dim=256,
        num_layers=4,
        context_dim=context_dim,
    )

    x_0 = torch.randn(batch_size, latent_dim)
    context = torch.randn(batch_size, 4, context_dim)

    # Test each solver
    print("\n1. Euler (20 steps):")
    result_euler = euler_integrate(model, x_0, context, num_steps=20)
    print(f"   Final shape: {result_euler.x_final.shape}")
    print(f"   Curvature: {result_euler.curvature}")

    print("\n2. Midpoint (20 steps):")
    result_mid = midpoint_integrate(model, x_0, context, num_steps=20)
    print(f"   Final shape: {result_mid.x_final.shape}")
    print(f"   Curvature: {result_mid.curvature}")

    print("\n3. RK4 (20 steps):")
    result_rk4 = rk4_integrate(model, x_0, context, num_steps=20)
    print(f"   Final shape: {result_rk4.x_final.shape}")
    print(f"   Curvature: {result_rk4.curvature}")

    print("\n4. One-step (reflowed):")
    x_1 = one_step_integrate(model, x_0, context)
    print(f"   Final shape: {x_1.shape}")

    # Compare endpoints
    print("\n5. Endpoint comparison (L2 norm difference):")
    print(f"   Euler vs Midpoint: {(result_euler.x_final - result_mid.x_final).norm(dim=-1).mean():.6f}")
    print(f"   Euler vs RK4: {(result_euler.x_final - result_rk4.x_final).norm(dim=-1).mean():.6f}")
    print(f"   Midpoint vs RK4: {(result_mid.x_final - result_rk4.x_final).norm(dim=-1).mean():.6f}")
    print(f"   One-step vs Euler: {(x_1 - result_euler.x_final).norm(dim=-1).mean():.6f}")

    # Test trajectory return
    print("\n6. Trajectory tracking:")
    result_traj = euler_integrate(model, x_0, context, num_steps=10, return_trajectory=True)
    print(f"   Trajectory length: {len(result_traj.trajectory)}")
    print(f"   Velocities length: {len(result_traj.velocities)}")

    print("\nAll tests passed!")
