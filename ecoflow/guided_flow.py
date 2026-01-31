"""UCB-guided flow sampling with CFG-Zero* schedule.

Implements guided ODE sampling that steers samples toward high-scoring regions
using GP surrogate gradients. Uses CFG-Zero* schedule (zero guidance for first
4% of steps) to prevent early trajectory corruption.

Guided ODE: dz/dt = v(z, t) + lambda(t) * grad_UCB(z)
"""

from typing import Optional

import torch

from ecoflow.flow_model import FlowMatchingModel
from ecoflow.gp_surrogate import SonarGPSurrogate


class GuidedFlowSampler:
    """Samples from flow model with UCB guidance from GP surrogate."""

    def __init__(
        self,
        flow_model: FlowMatchingModel,
        gp_surrogate: SonarGPSurrogate,
        alpha: float = 1.0,
        guidance_strength: float = 1.0,
        zero_init_fraction: float = 0.04,
        norm_stats: Optional[dict] = None,
    ):
        """
        Args:
            flow_model: Trained FlowMatchingModel for velocity computation
            gp_surrogate: SonarGPSurrogate for UCB gradient computation
            alpha: UCB exploration weight (default 1.0)
            guidance_strength: Maximum guidance strength lambda (default 1.0)
            zero_init_fraction: Fraction of steps with zero guidance (default 0.04)
            norm_stats: Normalization statistics {'mean': [1024], 'std': [1024]}
        """
        self.flow_model = flow_model
        self.gp = gp_surrogate
        self.alpha = alpha
        self.guidance_strength = guidance_strength
        self.zero_init_fraction = zero_init_fraction

        if norm_stats is not None:
            self.norm_stats = norm_stats
        elif hasattr(flow_model, 'norm_stats') and flow_model.norm_stats is not None:
            self.norm_stats = flow_model.norm_stats
        else:
            self.norm_stats = None

    def _get_guidance_lambda(self, step: int, total_steps: int) -> float:
        """Get guidance strength using CFG-Zero* schedule (zero for first 4%)."""
        zero_init_steps = max(1, int(self.zero_init_fraction * total_steps))
        if step < zero_init_steps:
            return 0.0
        return self.guidance_strength

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Convert from flow space to SONAR space."""
        if self.norm_stats is None:
            return z
        mean = self.norm_stats["mean"].to(z.device)
        std = self.norm_stats["std"].to(z.device)
        return z * std + mean

    def _compute_ucb_gradient(
        self, z_sonar: torch.Tensor, scale_to_flow_space: bool = True
    ) -> torch.Tensor:
        """Compute gradient of UCB, scaled for flow space with gradient clipping."""
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

    def _guided_velocity(
        self, z: torch.Tensor, t: torch.Tensor, step: int, num_steps: int
    ) -> torch.Tensor:
        """Compute velocity with optional UCB guidance."""
        v = self.flow_model._ode_func(t, z)

        lambda_t = self._get_guidance_lambda(step, num_steps)
        if lambda_t > 0 and self.gp.model is not None:
            z_sonar = self._denormalize(z)
            grad_ucb = self._compute_ucb_gradient(z_sonar)
            v = v + lambda_t * grad_ucb

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
    ) -> torch.Tensor:
        """
        Generate guided samples from noise.

        Args:
            n_samples: Number of samples to generate
            device: Device for computation
            num_steps: Number of ODE integration steps
            method: "heun" or "euler"
            return_trajectory: If True, return full trajectory [steps+1, B, 1024]

        Returns:
            Samples in SONAR space [n_samples, 1024]
        """
        device = torch.device(device) if isinstance(device, str) else device
        self.flow_model.velocity_net.eval()

        z = torch.randn(n_samples, self.flow_model.input_dim, device=device)
        z = self._integrate_ode(z, num_steps, method, with_guidance=True,
                                return_trajectory=return_trajectory)

        # Denormalize to SONAR space
        if return_trajectory:
            if self.norm_stats is not None:
                mean = self.norm_stats["mean"].to(device)
                std = self.norm_stats["std"].to(device)
                z = z * std + mean
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

    def update_gp(self, gp_surrogate: SonarGPSurrogate) -> None:
        """Update GP surrogate (called after each BO iteration)."""
        self.gp = gp_surrogate
