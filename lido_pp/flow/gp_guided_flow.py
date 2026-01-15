"""
GP-Guided Flow Generation for FlowPO.

Novel Contribution #2: Inject GP acquisition function gradients into flow velocity.

v'(x, t) = v(x, t) + s(t) · ∇R(x)

Where:
- v(x, t): Base FlowDiT velocity field
- s(t): Time-dependent guidance schedule
- R(x): GP acquisition function (UCB, EI, or negative error)

Key insight: Standard flow matching generates samples from the learned distribution.
By injecting acquisition gradients, we bias generation toward high-reward regions
while maintaining the flow structure for smooth, decodable trajectories.

Time-dependent scheduling is critical:
- t=0 (pure noise): No guidance (nothing meaningful to optimize)
- t=1 (clean sample): Full guidance (meaningful semantic space)

Reference: Classifier-Free Guidance (Ho & Salimans, 2021), but for GP acquisition
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_acquisition_reward(
    mean: torch.Tensor,
    std: torch.Tensor,
    acquisition: Literal["ucb", "ei", "neg_error"],
    ucb_beta: float = 2.0,
    best_value: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute acquisition function reward for minimization.

    Args:
        mean: (B,) GP mean predictions (error rate)
        std: (B,) GP std predictions
        acquisition: Acquisition function type
        ucb_beta: UCB exploration parameter
        best_value: Best observed value for EI (uses mean.min() if None)

    Returns:
        (B,) reward values (higher is better)
    """
    if acquisition == "ucb":
        return -mean + ucb_beta * std

    if acquisition == "ei":
        best = best_value if best_value is not None else mean.min().item()
        improvement = best - mean
        Z = improvement / (std + 1e-8)
        normal = torch.distributions.Normal(0, 1)
        cdf = normal.cdf(Z)
        pdf = normal.log_prob(Z).exp()
        return improvement * cdf + std * pdf

    # neg_error - pure exploitation
    return -mean


@dataclass
class GuidedGenerationResult:
    """Result of GP-guided flow generation."""

    latents: torch.Tensor  # (B, latent_dim) final generated latents
    trajectory: Optional[torch.Tensor] = None  # (T, B, latent_dim) full trajectory
    acquisition_values: Optional[torch.Tensor] = None  # (B,) final acquisition values
    guidance_norms: Optional[List[float]] = None  # Per-step gradient norms

    def __post_init__(self):
        """Validate tensor shapes are consistent."""
        B = self.latents.shape[0]
        if self.acquisition_values is not None and self.acquisition_values.shape[0] != B:
            raise ValueError(
                f"acquisition_values batch size {self.acquisition_values.shape[0]} != "
                f"latents batch size {B}"
            )
        if self.trajectory is not None and self.trajectory.shape[1] != B:
            raise ValueError(
                f"trajectory batch dimension {self.trajectory.shape[1]} != "
                f"latents batch size {B}"
            )


class GPGuidedFlowGenerator(nn.Module):
    """
    GP-Guided Flow Generator - Novel contribution for FlowPO.

    Injects GP acquisition function gradients into the flow velocity field
    to bias generation toward high-reward (low-error) regions.

    The key innovation is time-dependent guidance scheduling:
    - Early in generation (t≈0): Sample is noisy, guidance unhelpful
    - Late in generation (t≈1): Sample is clean, guidance effective

    This differs from classifier-free guidance:
    - CFG uses a learned unconditional model
    - We use GP surrogate gradients for task-specific optimization
    """

    def __init__(
        self,
        flowdit: nn.Module,
        latent_dim: int = 128,
        guidance_scale: float = 1.0,
        schedule: Literal["linear", "cosine", "warmup", "sqrt", "constant"] = "linear",
        ucb_beta: float = 2.0,
    ):
        """
        Initialize GP-guided flow generator.

        Args:
            flowdit: Flow-DiT velocity field model
            latent_dim: Latent dimension (for initialization)
            guidance_scale: Base guidance strength (multiplied by schedule)
            schedule: Time-dependent guidance schedule
            ucb_beta: UCB exploration parameter (β in μ + β·σ)
        """
        super().__init__()

        # Validate schedule at construction time
        valid_schedules = {"linear", "cosine", "warmup", "sqrt", "constant"}
        if schedule not in valid_schedules:
            raise ValueError(
                f"Invalid schedule '{schedule}'. "
                f"Valid options: {', '.join(sorted(valid_schedules))}"
            )

        self.flowdit = flowdit
        self.latent_dim = latent_dim
        self.scale = guidance_scale
        self.schedule = schedule
        self.ucb_beta = ucb_beta

        # GP model set via set_gp_model()
        self.gp = None

        logger.info(
            f"GPGuidedFlowGenerator: scale={guidance_scale}, "
            f"schedule={schedule}, ucb_beta={ucb_beta}"
        )

    def set_gp_model(self, gp_model: nn.Module):
        """
        Set the GP model for acquisition gradient computation.

        Args:
            gp_model: Trained GP model with predict(z) → (mean, std) method
        """
        self.gp = gp_model

    def _get_schedule_weight(self, t: float) -> float:
        """
        Compute time-dependent guidance weight.

        Critical: No guidance at t=0 (pure noise) where gradients are meaningless.

        Args:
            t: Time in [0, 1] where 0=noise, 1=clean

        Returns:
            Weight in [0, 1] for guidance strength
        """
        schedules = {
            "linear": lambda: t,
            "cosine": lambda: (1 - math.cos(t * math.pi)) / 2,
            "sqrt": lambda: t**0.5,
            "warmup": lambda: 0.0 if t < 0.2 else (t - 0.2) / 0.8,
            "constant": lambda: 1.0,
        }

        if self.schedule not in schedules:
            raise ValueError(
                f"Unknown guidance schedule '{self.schedule}'. "
                f"Valid options: {', '.join(schedules.keys())}"
            )

        return schedules[self.schedule]()

    def _compute_acquisition_gradient(
        self,
        z: torch.Tensor,
        acquisition: Literal["ucb", "ei", "neg_error"] = "ucb",
    ) -> torch.Tensor:
        """
        Compute gradient of acquisition function.

        For prompt optimization, we want to minimize error rate.
        GP predicts error rate (NOT accuracy), so:
        - UCB: R = -mean + beta*std (minimize error + explore)
        - EI: Expected improvement for minimization
        - neg_error: R = -mean (pure exploitation)

        For high-dimensional spaces (curse of dimensionality), uses the GP's
        compute_guidance_gradient method if available, which falls back to
        nearest-neighbor direction when analytic gradient is too small.

        IMPORTANT: If your GP predicts accuracy instead of error rate,
        flip the sign: R = mean + beta*std

        Args:
            z: (B, latent_dim) current latent positions
            acquisition: Which acquisition function to use

        Returns:
            (B, latent_dim) gradient of acquisition w.r.t. z

        Raises:
            RuntimeError: If GP model is not set
        """
        if self.gp is None:
            raise RuntimeError(
                "Cannot compute acquisition gradient: GP model not set. "
                "Call set_gp_model() before using GP-guided generation."
            )

        # Use high-dimensional gradient method if available
        if hasattr(self.gp, 'compute_guidance_gradient'):
            return self.gp.compute_guidance_gradient(z, self.ucb_beta)

        # Standard analytic gradient
        z_grad = z.detach().requires_grad_(True)
        mean, std = self.gp.predict(z_grad)

        best_value = getattr(self.gp, "best_error_rate", None)
        reward = compute_acquisition_reward(
            mean, std, acquisition, self.ucb_beta, best_value
        )

        reward.sum().backward()

        if z_grad.grad is None:
            logger.warning(
                "Gradient computation returned None. "
                "This may happen with high-dimensional GP and few training points."
            )
            return torch.zeros_like(z)

        grad = z_grad.grad.detach().clone()

        # Check if gradient is meaningful
        grad_norm = grad.norm(dim=-1).mean()
        if grad_norm < 1e-6:
            logger.debug(
                f"Analytic gradient is very small ({grad_norm:.2e}). "
                "Consider using more training data for high-dimensional GP."
            )

        z_grad.grad = None
        return grad

    def guided_step(
        self,
        x: torch.Tensor,
        t: float,
        context: Optional[torch.Tensor] = None,
        acquisition: Literal["ucb", "ei", "neg_error"] = "ucb",
    ) -> Tuple[torch.Tensor, float]:
        """
        Single guided integration step.

        Computes: v'(x, t) = v(x, t) + s(t) · ∇R(x)

        Args:
            x: (B, latent_dim) current position
            t: Current time
            context: Optional conditioning context
            acquisition: Acquisition function type

        Returns:
            v_guided: (B, latent_dim) guided velocity
            grad_norm: Norm of guidance gradient (for logging)
        """
        # Base velocity from FlowDiT
        t_tensor = torch.full((x.shape[0],), t, device=x.device)

        with torch.no_grad():
            v_base = self.flowdit(x, t_tensor, context)

        # Compute guidance weight based on schedule
        s_t = self._get_schedule_weight(t) * self.scale

        if s_t > 0 and self.gp is not None:
            # Compute acquisition gradient
            grad_R = self._compute_acquisition_gradient(x, acquisition)
            grad_norm = grad_R.norm(dim=-1).mean().item()

            # Apply guided velocity
            v_guided = v_base + s_t * grad_R
        else:
            v_guided = v_base
            grad_norm = 0.0

        return v_guided, grad_norm

    def generate(
        self,
        batch_size: int,
        num_steps: int = 20,
        context: Optional[torch.Tensor] = None,
        acquisition: Literal["ucb", "ei", "neg_error"] = "ucb",
        return_trajectory: bool = False,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> GuidedGenerationResult:
        """
        Generate optimized latents via GP-guided flow.

        This is the main interface for generating new prompt latents
        that are biased toward high-reward regions.

        Args:
            batch_size: Number of latents to generate
            num_steps: ODE integration steps
            context: Optional conditioning (e.g., task description embedding)
            acquisition: Acquisition function type
            return_trajectory: Whether to return full trajectory
            initial_noise: Optional initial noise (for reproducibility)

        Returns:
            GuidedGenerationResult with latents and optional trajectory
        """
        device = next(self.flowdit.parameters()).device

        # Initialize from noise (t=0)
        if initial_noise is not None:
            x = initial_noise.to(device)
        else:
            x = torch.randn(batch_size, self.latent_dim, device=device)

        trajectory = [x.clone()] if return_trajectory else None
        guidance_norms = []

        dt = 1.0 / num_steps

        # Euler integration from t=0 to t=1
        for i in range(num_steps):
            t = i * dt

            # Guided step
            v, grad_norm = self.guided_step(x, t, context, acquisition)
            guidance_norms.append(grad_norm)

            # Euler update
            x = x + dt * v

            if return_trajectory:
                trajectory.append(x.clone())

        # Compute final acquisition values if GP is available
        acquisition_values = None
        if self.gp is not None:
            with torch.no_grad():
                mean, _ = self.gp.predict(x)
                acquisition_values = mean

        return GuidedGenerationResult(
            latents=x,
            trajectory=torch.stack(trajectory) if return_trajectory else None,
            acquisition_values=acquisition_values,
            guidance_norms=guidance_norms,
        )

    def generate_diverse(
        self,
        batch_size: int,
        num_candidates: int = 10,
        num_steps: int = 20,
        context: Optional[torch.Tensor] = None,
        acquisition: Literal["ucb", "ei", "neg_error"] = "ucb",
        diversity_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Generate diverse set of optimized latents.

        Uses DPP-style diversity: select candidates that are both
        high-quality (low error) and diverse (high pairwise distance).

        Args:
            batch_size: Number of final latents to return
            num_candidates: Number of candidates to generate (> batch_size)
            num_steps: ODE integration steps
            context: Optional conditioning
            acquisition: Acquisition function
            diversity_weight: Weight for diversity vs quality

        Returns:
            (batch_size, latent_dim) diverse high-quality latents
        """
        result = self.generate(
            batch_size=num_candidates,
            num_steps=num_steps,
            context=context,
            acquisition=acquisition,
        )

        candidates = result.latents

        if result.acquisition_values is not None:
            scores = -result.acquisition_values
        else:
            logger.warning(
                "No acquisition values available for quality-based selection. "
                "Selecting based on diversity only. Set GP model for quality-aware selection."
            )
            scores = torch.zeros(num_candidates, device=candidates.device)

        selected_indices: List[int] = []
        remaining = set(range(num_candidates))

        for _ in range(min(batch_size, num_candidates)):
            if not selected_indices:
                best_idx = scores.argmax().item()
            else:
                remaining_list = list(remaining)
                remaining_candidates = candidates[remaining_list]
                selected_latents = candidates[selected_indices]

                # Vectorized: compute min distance from each remaining to selected
                # (R, S, D) -> (R, S) -> (R,)
                dist_matrix = torch.cdist(remaining_candidates, selected_latents)
                min_distances = dist_matrix.min(dim=1).values

                combined = scores[remaining_list] + diversity_weight * min_distances
                best_idx = remaining_list[combined.argmax().item()]

            selected_indices.append(best_idx)
            remaining.discard(best_idx)

        return candidates[selected_indices]


class AcquisitionGradientGuide(nn.Module):
    """
    Lightweight module for computing acquisition gradients.

    Useful when you want to add guidance to an existing flow generator
    without the full GPGuidedFlowGenerator wrapper.
    """

    def __init__(
        self,
        gp_model: nn.Module,
        acquisition: Literal["ucb", "ei", "neg_error"] = "ucb",
        ucb_beta: float = 2.0,
    ):
        """
        Initialize acquisition gradient guide.

        Args:
            gp_model: GP model with predict method
            acquisition: Acquisition function type
            ucb_beta: UCB beta parameter
        """
        super().__init__()
        self.gp = gp_model
        self.acquisition = acquisition
        self.ucb_beta = ucb_beta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute acquisition gradient.

        Args:
            z: (B, D) latent positions

        Returns:
            (B, D) acquisition gradients
        """
        z = z.detach().requires_grad_(True)
        mean, std = self.gp.predict(z)

        best_value = getattr(self.gp, "best_error_rate", None)
        reward = compute_acquisition_reward(
            mean, std, self.acquisition, self.ucb_beta, best_value
        )

        reward.sum().backward()

        if z.grad is None:
            raise RuntimeError("Gradient computation failed: z.grad is None")

        grad = z.grad.detach().clone()
        z.grad = None
        return grad


def create_guided_generator(
    flowdit: nn.Module,
    gp_model: Optional[nn.Module] = None,
    latent_dim: int = 128,
    guidance_scale: float = 1.0,
    schedule: Literal["linear", "cosine", "warmup", "sqrt", "constant"] = "linear",
    ucb_beta: float = 2.0,
) -> GPGuidedFlowGenerator:
    """
    Factory function to create GP-guided flow generator.

    Args:
        flowdit: Flow-DiT velocity field
        gp_model: Optional trained GP model
        latent_dim: Latent dimension
        guidance_scale: Guidance strength
        schedule: Time-dependent schedule
        ucb_beta: UCB parameter

    Returns:
        Configured GPGuidedFlowGenerator
    """
    generator = GPGuidedFlowGenerator(
        flowdit=flowdit,
        latent_dim=latent_dim,
        guidance_scale=guidance_scale,
        schedule=schedule,
        ucb_beta=ucb_beta,
    )

    if gp_model is not None:
        generator.set_gp_model(gp_model)

    return generator


if __name__ == "__main__":
    print("Testing GP-Guided Flow Generation...")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create mock FlowDiT
    class MockFlowDiT(nn.Module):
        def __init__(self, latent_dim=128):
            super().__init__()
            self.latent_dim = latent_dim
            self.net = nn.Sequential(
                nn.Linear(latent_dim + 1, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim),
            )

        def forward(self, x, t, context=None):
            t_expanded = t.unsqueeze(-1) if t.dim() == 1 else t
            inp = torch.cat([x, t_expanded], dim=-1)
            return self.net(inp)

    # Create mock GP
    class MockGP(nn.Module):
        def __init__(self, latent_dim=128):
            super().__init__()
            self.net = nn.Linear(latent_dim, 2)  # mean, log_std
            self.best_error_rate = 0.3

        def predict(self, z):
            out = self.net(z)
            mean = torch.sigmoid(out[:, 0])  # Error rate in [0, 1]
            std = torch.nn.functional.softplus(out[:, 1]) * 0.1  # Small std
            return mean, std

    flowdit = MockFlowDiT().to(device)
    gp = MockGP().to(device)

    # Create guided generator
    generator = GPGuidedFlowGenerator(
        flowdit=flowdit,
        latent_dim=128,
        guidance_scale=1.0,
        schedule="linear",
        ucb_beta=2.0,
    )
    generator.set_gp_model(gp)
    generator = generator.to(device)

    # Test generation
    print("\n--- Basic Generation ---")
    result = generator.generate(
        batch_size=4,
        num_steps=20,
        return_trajectory=True,
    )
    print(f"Generated latents: {result.latents.shape}")
    print(f"Trajectory shape: {result.trajectory.shape}")
    print(f"Acquisition values: {result.acquisition_values}")
    print(f"Guidance norms: {result.guidance_norms[:5]}... (first 5)")

    # Test diverse generation
    print("\n--- Diverse Generation ---")
    diverse_latents = generator.generate_diverse(
        batch_size=3,
        num_candidates=10,
        num_steps=20,
    )
    print(f"Diverse latents: {diverse_latents.shape}")

    # Verify diversity
    pairwise_dist = torch.cdist(diverse_latents, diverse_latents)
    print(f"Pairwise distances:\n{pairwise_dist}")

    print("\n[OK] GP-Guided Flow tests passed!")
