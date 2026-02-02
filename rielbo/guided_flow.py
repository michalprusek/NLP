"""UCB-guided flow sampling with time-dependent guidance schedules.

Implements guided ODE sampling that steers samples toward high-scoring regions
using GP surrogate gradients. Supports multiple guidance schedules:

1. CFG-Zero* (step): Zero guidance for first 4%, then constant
2. Linear ramp: Smooth linear increase after zero-init
3. Cosine ramp: Smooth cosine increase after zero-init
4. Uncertainty-adaptive: Scales guidance by 1/(1 + β·σ(z)) - more guidance where GP is confident

Guided ODE: dz/dt = v(z, t) + λ(t, z) * grad_UCB(z)

Supports both legacy SonarGPSurrogate and new BaseGPSurrogate (including
RiemannianGP with geodesic kernels for better performance on SONAR embeddings).
"""

import math
from enum import Enum
from typing import Optional, Union

import torch


class GuidanceSchedule(Enum):
    """Time-dependent guidance schedule types."""
    STEP = "step"           # CFG-Zero*: 0 then constant (original)
    LINEAR = "linear"       # Linear ramp after zero-init
    COSINE = "cosine"       # Cosine ramp after zero-init (smoother)
    ADAPTIVE = "adaptive"   # Uncertainty-weighted (more guidance where GP is confident)
    RIEMANNIAN = "riemannian"  # Full Riemannian: spherical projection + cutoff + flow-relative scaling

from rielbo.flow_model import FlowMatchingModel
from rielbo.gp_surrogate import SonarGPSurrogate

# Import new GP surrogates if available
import logging as _logging
_gp_logger = _logging.getLogger(__name__)

try:
    from study.gp_ablation.surrogates.base import BaseGPSurrogate
    from study.gp_ablation.surrogates.riemannian_gp import RiemannianGP
    NEW_GP_AVAILABLE = True
except ImportError as _import_err:
    NEW_GP_AVAILABLE = False
    BaseGPSurrogate = None
    RiemannianGP = None
    _gp_logger.warning(
        f"Could not import new GP surrogates: {_import_err}. "
        "Riemannian GP features will not be available. "
        "Falling back to legacy SonarGPSurrogate interface."
    )


class GuidedFlowSampler:
    """Samples from flow model with UCB guidance from GP surrogate.

    Supports two GP interfaces:
    1. Legacy SonarGPSurrogate with ucb_gradient() method
    2. New BaseGPSurrogate with acquisition_gradient() method (includes RiemannianGP)

    For best results with SONAR embeddings, use RiemannianGP with arccosine or
    geodesic_matern52 kernel - these respect the hyperspherical geometry.
    """

    def __init__(
        self,
        flow_model: FlowMatchingModel,
        gp_surrogate: Union["SonarGPSurrogate", "BaseGPSurrogate"],
        alpha: float = 1.0,
        guidance_strength: float = 1.0,
        zero_init_fraction: float = 0.04,
        norm_stats: Optional[dict] = None,
        guidance_schedule: Union[str, GuidanceSchedule] = GuidanceSchedule.RIEMANNIAN,
        uncertainty_beta: float = 2.0,
        guidance_cutoff: float = 0.8,
        spherical_projection: bool = True,
        flow_relative_scaling: bool = True,
    ):
        """
        Args:
            flow_model: Trained FlowMatchingModel for velocity computation
            gp_surrogate: GP surrogate for UCB gradient computation.
                          Supports SonarGPSurrogate or any BaseGPSurrogate
                          (e.g., RiemannianGP with geodesic kernels).
            alpha: UCB exploration weight (default 1.0)
            guidance_strength: Maximum guidance strength lambda (default 1.0)
            zero_init_fraction: Fraction of steps with zero guidance (default 0.04)
            norm_stats: Normalization statistics {'mean': [1024], 'std': [1024]}
            guidance_schedule: Schedule type - "step", "linear", "cosine", "adaptive", or "riemannian"
            uncertainty_beta: For adaptive schedule, controls exploitation vs exploration.
                             Higher = more exploitation (less guidance in uncertain regions)
            guidance_cutoff: Stop guidance after this fraction of ODE (default 0.8).
                            Last 20% is "smoothing" phase where flow refines without GP interference.
            spherical_projection: Project gradient onto tangent plane of sphere (Riemannian gradient).
                                 Prevents gradient from changing vector norm, only direction.
            flow_relative_scaling: Scale gradient magnitude relative to flow velocity.
                                  Prevents gradient from overwhelming the flow field.
        """
        self.flow_model = flow_model
        self.gp = gp_surrogate
        self.alpha = alpha
        self.guidance_strength = guidance_strength
        self.zero_init_fraction = zero_init_fraction
        self.uncertainty_beta = uncertainty_beta
        self.guidance_cutoff = guidance_cutoff
        self.spherical_projection = spherical_projection
        self.flow_relative_scaling = flow_relative_scaling

        # Parse guidance schedule
        if isinstance(guidance_schedule, str):
            self.guidance_schedule = GuidanceSchedule(guidance_schedule.lower())
        else:
            self.guidance_schedule = guidance_schedule

        # For RIEMANNIAN schedule, enable all advanced features
        if self.guidance_schedule == GuidanceSchedule.RIEMANNIAN:
            self.spherical_projection = True
            self.flow_relative_scaling = True

        # Detect GP interface type
        self._use_new_interface = self._is_new_gp_interface(gp_surrogate)

        if norm_stats is not None:
            self.norm_stats = norm_stats
        elif hasattr(flow_model, 'norm_stats') and flow_model.norm_stats is not None:
            self.norm_stats = flow_model.norm_stats
        else:
            self.norm_stats = None

    def _is_new_gp_interface(self, gp_surrogate) -> bool:
        """Check if GP uses new BaseGPSurrogate interface."""
        return (
            NEW_GP_AVAILABLE
            and BaseGPSurrogate is not None
            and isinstance(gp_surrogate, BaseGPSurrogate)
        )

    def _get_guidance_lambda(self, step: int, total_steps: int) -> float:
        """Get base guidance strength using configured schedule."""
        zero_init_steps = max(1, int(self.zero_init_fraction * total_steps))

        if step < zero_init_steps:
            return 0.0

        remaining_steps = total_steps - zero_init_steps
        if remaining_steps <= 0:
            return self.guidance_strength

        progress = (step - zero_init_steps) / remaining_steps
        schedule = self.guidance_schedule

        if schedule == GuidanceSchedule.STEP:
            return self.guidance_strength

        if schedule == GuidanceSchedule.LINEAR:
            return self.guidance_strength * progress

        if schedule in (GuidanceSchedule.COSINE, GuidanceSchedule.ADAPTIVE, GuidanceSchedule.RIEMANNIAN):
            cosine_factor = (1 - math.cos(math.pi * progress)) / 2
            return self.guidance_strength * cosine_factor

        return self.guidance_strength

    def _get_uncertainty_factor(self, z_sonar: torch.Tensor) -> torch.Tensor:
        """Compute exploitation factor based on GP uncertainty.

        Returns factor in [0, 1] where high sigma leads to less guidance
        (explore freely) and low sigma leads to more guidance (exploit confidently).
        """
        if not self._is_gp_fitted():
            return torch.ones(z_sonar.shape[0], device=z_sonar.device)

        _, std = self.gp.predict(z_sonar)
        return 1.0 / (1.0 + self.uncertainty_beta * std)

    def _project_to_tangent_plane(
        self, grad: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Project gradient onto tangent plane of sphere (Riemannian gradient).

        ∇_sphere f(x) = ∇f(x) - (∇f(x) · x̂) x̂

        where x̂ = x / ||x||

        This removes the radial component of the gradient, ensuring all
        guidance force goes into changing the direction (semantics), not
        the magnitude (which the flow model controls).
        """
        # Normalize x to unit vector
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute radial component: (grad · x̂) x̂
        radial_component = (grad * x_norm).sum(dim=-1, keepdim=True) * x_norm

        # Tangent gradient = total gradient - radial component
        tangent_grad = grad - radial_component

        return tangent_grad

    def _scale_relative_to_flow(
        self, grad: torch.Tensor, flow_velocity: torch.Tensor, max_ratio: float = 0.5
    ) -> torch.Tensor:
        """Scale gradient magnitude relative to flow velocity.

        Prevents gradient from overwhelming the flow field.
        grad_scaled = grad * min(max_ratio, ||v|| / ||grad||)

        Args:
            grad: GP gradient [B, D]
            flow_velocity: Flow velocity [B, D]
            max_ratio: Maximum ratio of gradient to velocity magnitude (default 0.5)

        Returns:
            Scaled gradient that won't overpower the flow
        """
        grad_norm = grad.norm(dim=-1, keepdim=True) + 1e-8
        flow_norm = flow_velocity.norm(dim=-1, keepdim=True) + 1e-8

        # Scale factor: ensure gradient is at most max_ratio * flow_velocity
        scale = torch.clamp(max_ratio * flow_norm / grad_norm, max=1.0)

        return grad * scale

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Convert from flow space to SONAR space.

        Delegates to flow_model.denormalize() which handles both regular and
        spherical flows correctly.
        """
        return self.flow_model.denormalize(z)

    def _compute_ucb_gradient(
        self, z_sonar: torch.Tensor, scale_to_flow_space: bool = True
    ) -> torch.Tensor:
        """Compute gradient of UCB, scaled for flow space with gradient clipping.

        Supports both legacy SonarGPSurrogate.ucb_gradient() and new
        BaseGPSurrogate.acquisition_gradient() interfaces.
        """
        # Check if GP is fitted (different attribute for different GP types)
        if self._use_new_interface:
            # New interface uses _train_X to check if fitted
            if self.gp._train_X is None:
                raise RuntimeError("GP must be fitted before computing guidance")
            grad_ucb = self.gp.acquisition_gradient(z_sonar, acquisition="ucb", alpha=self.alpha)
        else:
            # Legacy interface
            if self.gp.model is None:
                raise RuntimeError("GP must be fitted before computing guidance")
            grad_ucb = self.gp.ucb_gradient(z_sonar, alpha=self.alpha)

        # Convert gradient from SONAR space to flow space using chain rule
        if scale_to_flow_space and self.norm_stats is not None:
            norm_std = self.norm_stats["std"].to(z_sonar.device)
            grad_ucb = grad_ucb / (norm_std + 1e-8)

        # Clip gradient norm to prevent explosion
        max_grad_norm = 10.0
        grad_norm = grad_ucb.norm(dim=-1, keepdim=True)
        clip_mask = grad_norm > max_grad_norm
        if clip_mask.any():
            grad_ucb = torch.where(
                clip_mask,
                grad_ucb * max_grad_norm / grad_norm,
                grad_ucb
            )

        return grad_ucb

    def _is_gp_fitted(self) -> bool:
        """Check if GP surrogate is fitted."""
        if self._use_new_interface:
            return self.gp._train_X is not None
        else:
            return self.gp.model is not None

    def _guided_velocity(
        self, z: torch.Tensor, t: torch.Tensor, step: int, num_steps: int
    ) -> torch.Tensor:
        """Compute velocity with optional UCB guidance.

        Supports multiple guidance modes:
        - STEP/LINEAR/COSINE: Basic time-dependent scheduling
        - ADAPTIVE: + uncertainty-weighted guidance
        - RIEMANNIAN: + spherical projection + flow-relative scaling + cutoff

        For RIEMANNIAN (recommended for ArcCosine kernel):
        1. Project gradient onto tangent plane of sphere (don't change norm)
        2. Scale gradient relative to flow velocity (don't overpower flow)
        3. Stop guidance at t > cutoff (let flow smooth the result)
        """
        v = self.flow_model._ode_func(t, z)

        # Get current time fraction
        t_frac = step / num_steps if num_steps > 0 else 0.0

        # Cutoff: stop guidance in final phase to let flow "smooth" the result
        if t_frac > self.guidance_cutoff:
            return v

        lambda_t = self._get_guidance_lambda(step, num_steps)
        if lambda_t > 0 and self._is_gp_fitted():
            z_sonar = self._denormalize(z)
            grad_ucb = self._compute_ucb_gradient(z_sonar)

            # 1. Spherical projection: remove radial component
            if self.spherical_projection:
                grad_ucb = self._project_to_tangent_plane(grad_ucb, z_sonar)

            # 2. Flow-relative scaling: don't overpower the flow
            if self.flow_relative_scaling:
                grad_ucb = self._scale_relative_to_flow(grad_ucb, v, max_ratio=0.5)

            # 3. Uncertainty-weighted (for ADAPTIVE and RIEMANNIAN)
            if self.guidance_schedule in (GuidanceSchedule.ADAPTIVE, GuidanceSchedule.RIEMANNIAN):
                uncertainty_factor = self._get_uncertainty_factor(z_sonar)
                # Shape: [B] -> [B, 1] for broadcasting with grad_ucb [B, D]
                lambda_effective = lambda_t * uncertainty_factor.unsqueeze(-1)
                v = v + lambda_effective * grad_ucb
            else:
                v = v + lambda_t * grad_ucb

        # 4. Spherical correction: project entire velocity to tangent plane
        # This ensures ODE integration stays on the sphere (not just gradient direction)
        if self.spherical_projection:
            z_sonar_for_proj = self._denormalize(z)
            v = self._project_to_tangent_plane(v, z_sonar_for_proj)

        return v

    def _integrate_ode(
        self,
        z: torch.Tensor,
        num_steps: int,
        method: str,
        with_guidance: bool = True,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Integrate ODE from t=0 to t=1.

        Args:
            z: Initial state [B, D] at t=0
            num_steps: Number of integration steps
            method: "euler" or "heun"
            with_guidance: Whether to apply UCB guidance
            return_trajectory: If True, return full trajectory [steps+1, B, D]

        Returns:
            Final state [B, D] or trajectory [steps+1, B, D]
        """
        device = z.device
        trajectory = [z.clone()] if return_trajectory else None
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)

            if method == "euler":
                if with_guidance:
                    v = self._guided_velocity(z, t, i, num_steps)
                else:
                    v = self.flow_model._ode_func(t, z)
                z = z + v * dt

            elif method == "heun":
                t_next = torch.tensor((i + 1) * dt, device=device)

                # Predictor
                if with_guidance:
                    v1 = self._guided_velocity(z, t, i, num_steps)
                else:
                    v1 = self.flow_model._ode_func(t, z)
                z_pred = z + v1 * dt

                # Corrector
                if with_guidance:
                    v2 = self._guided_velocity(z_pred, t_next, i + 1, num_steps)
                else:
                    v2 = self.flow_model._ode_func(t_next, z_pred)
                z = z + 0.5 * (v1 + v2) * dt

            else:
                raise ValueError(f"Unknown method: {method}. Use 'euler' or 'heun'.")

            if return_trajectory:
                trajectory.append(z.clone())

        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        return z

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: str | torch.device = "cuda",
        num_steps: int = 50,
        method: str = "heun",
        return_trajectory: bool = False,
        start_on_sphere: bool = True,
    ) -> torch.Tensor:
        """
        Generate guided samples from noise.

        Args:
            n_samples: Number of samples to generate
            device: Device for computation
            num_steps: Number of ODE integration steps
            method: "heun" or "euler"
            return_trajectory: If True, return full trajectory [steps+1, B, 1024]
            start_on_sphere: If True, normalize initial noise to unit sphere.
                            Recommended for spherical flow models (spherical-ot).

        Returns:
            Samples in SONAR space [n_samples, 1024]
        """
        device = torch.device(device) if isinstance(device, str) else device
        self.flow_model.velocity_net.eval()

        z = torch.randn(n_samples, self.flow_model.input_dim, device=device)

        # For spherical flow: start on unit sphere in NORMALIZED flow space
        # (normalized Gaussian = uniform on S^{d-1})
        # This is correct because flow training normalizes data to ~unit variance
        if start_on_sphere:
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)

        z = self._integrate_ode(z, num_steps, method, with_guidance=True,
                                return_trajectory=return_trajectory)

        # Denormalize to SONAR space using flow_model.denormalize()
        # This handles both regular flows (mean/std) and spherical flows (scale to SONAR norm)
        if return_trajectory:
            # Denormalize each timestep
            z = torch.stack([self._denormalize(z_t) for z_t in z], dim=0)
        else:
            z = self._denormalize(z)

        return z

    @torch.no_grad()
    def sample_from_best(
        self,
        best_embedding: torch.Tensor,
        n_samples: int,
        device: str | torch.device = "cuda",
        num_steps: int = 50,
        method: str = "heun",
        perturbation_std: float = 0.1,
    ) -> torch.Tensor:
        """
        Generate guided samples starting from the best embedding's noise space.

        Encodes best embedding to noise, adds perturbations, then integrates
        forward with guidance. Keeps samples on SONAR manifold.

        Args:
            best_embedding: Best embedding [1, 1024] or [1024] in SONAR space
            n_samples: Number of samples to generate
            device: Device for computation
            num_steps: Number of ODE integration steps
            method: "heun" or "euler"
            perturbation_std: Std of Gaussian perturbation in noise space

        Returns:
            Samples in SONAR space [n_samples, 1024]
        """
        device = torch.device(device) if isinstance(device, str) else device
        self.flow_model.velocity_net.eval()

        if best_embedding.dim() == 1:
            best_embedding = best_embedding.unsqueeze(0)
        best_embedding = best_embedding.to(device)

        # Encode to noise space
        z_best = self.flow_model.encode(
            best_embedding, method=method, num_steps=num_steps, normalize=True
        )

        # Add perturbations
        z = z_best.expand(n_samples, -1).clone()
        z = z + torch.randn_like(z) * perturbation_std

        # Integrate forward with guidance
        z = self._integrate_ode(z, num_steps, method, with_guidance=True)
        return self._denormalize(z)

    def sample_optimal(
        self,
        device: str | torch.device = "cuda",
        num_steps: int = 50,
        method: str = "heun",
        ucb_alpha: float = 1.96,
        n_restarts: int = 5,
        n_opt_steps: int = 100,
        lr: float = 0.1,
    ) -> tuple[torch.Tensor, dict]:
        """
        Generate optimal sample via GP-UCB optimization + flow projection.

        Finds z* = argmax UCB(z), then projects onto manifold via encode/decode.

        Args:
            device: Device for computation
            num_steps: ODE integration steps
            method: "heun" or "euler"
            ucb_alpha: UCB exploration weight
            n_restarts: GP optimization restarts
            n_opt_steps: Gradient steps per restart
            lr: Learning rate for GP optimization

        Returns:
            z_projected: Optimal embedding [1, 1024] in SONAR space
            info: Dict with z_optimal, ucb_value, l2_projection
        """
        device = torch.device(device) if isinstance(device, str) else device
        self.flow_model.velocity_net.eval()

        # GP-UCB optimization
        z_optimal, ucb_value = self.gp.optimize_ucb(
            alpha=ucb_alpha, n_restarts=n_restarts, n_steps=n_opt_steps, lr=lr
        )
        z_optimal = z_optimal.to(device)

        with torch.no_grad():
            # Encode to noise space
            z_noise = self.flow_model.encode(
                z_optimal, method=method, num_steps=num_steps, normalize=True
            )

            # Decode back (without guidance for manifold projection)
            z = self._integrate_ode(z_noise, num_steps, method, with_guidance=False)
            z_projected = self.flow_model.denormalize(z)

            l2_projection = (z_optimal - z_projected).norm().item()

        return z_projected, {
            "z_optimal": z_optimal,
            "ucb_value": ucb_value,
            "l2_projection": l2_projection,
        }

    def set_guidance_strength(self, guidance_strength: float) -> None:
        """Update guidance strength."""
        self.guidance_strength = guidance_strength

    def set_alpha(self, alpha: float) -> None:
        """Update UCB exploration weight."""
        self.alpha = alpha

    def update_gp(self, gp_surrogate: Union["SonarGPSurrogate", "BaseGPSurrogate"]) -> None:
        """Update GP surrogate (called after each BO iteration).

        Args:
            gp_surrogate: New GP surrogate (SonarGPSurrogate or BaseGPSurrogate)
        """
        self.gp = gp_surrogate
        self._use_new_interface = self._is_new_gp_interface(gp_surrogate)


def create_optimal_gp_for_guided_flow(
    input_dim: int = 1024,
    kernel: str = "arccosine",
    device: str = "cuda",
) -> "RiemannianGP":
    """Create the optimal GP surrogate for guided flow on SONAR embeddings.

    Based on ablation study results, Riemannian GP with geodesic kernels
    provides the best performance for SONAR embeddings because:
    - SONAR embeddings are normalized (lie on unit hypersphere)
    - Geodesic distance respects this geometry
    - ArcCosine kernel has best calibration (ECE=0.012)
    - Geodesic Matern-5/2 has highest ranking correlation (ρ=0.46)

    Args:
        input_dim: Embedding dimension (default 1024 for SONAR)
        kernel: Kernel type - "arccosine" (recommended) or "geodesic_matern52"
        device: Torch device

    Returns:
        RiemannianGP configured for optimal guided flow performance

    Raises:
        ImportError: If study.gp_ablation module is not available
    """
    if not NEW_GP_AVAILABLE:
        raise ImportError(
            "RiemannianGP requires study.gp_ablation module. "
            "Make sure the module is installed."
        )

    from study.gp_ablation.config import GPConfig
    from study.gp_ablation.surrogates.riemannian_gp import RiemannianGP

    config = GPConfig(
        method="riemannian",
        kernel=kernel,
        input_dim=input_dim,
        normalize_inputs=True,  # Project to unit sphere
    )

    return RiemannianGP(config, device=device)
