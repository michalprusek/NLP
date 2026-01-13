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

import torch
import torch.nn as nn
from typing import Callable, Literal, Optional, Tuple, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GuidedGenerationResult:
    """Result of GP-guided flow generation."""

    latents: torch.Tensor  # (B, latent_dim) final generated latents
    trajectory: Optional[torch.Tensor] = None  # (T, B, latent_dim) full trajectory
    acquisition_values: Optional[torch.Tensor] = None  # (B,) final acquisition values
    guidance_norms: Optional[List[float]] = None  # Per-step gradient norms


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
        if self.schedule == "linear":
            # Linear ramp: 0 at t=0, 1 at t=1
            return t

        elif self.schedule == "cosine":
            # Smooth cosine ramp
            return (1 - torch.cos(torch.tensor(t) * torch.pi).item()) / 2

        elif self.schedule == "sqrt":
            # Faster initial ramp, slower at end
            return t**0.5

        elif self.schedule == "warmup":
            # No guidance until t=0.2, then linear
            return 0.0 if t < 0.2 else (t - 0.2) / 0.8

        elif self.schedule == "constant":
            # Full guidance at all times (not recommended)
            return 1.0

        else:
            logger.warning(f"Unknown schedule '{self.schedule}', using linear")
            return t

    def _compute_acquisition_gradient(
        self,
        z: torch.Tensor,
        acquisition: Literal["ucb", "ei", "neg_error"] = "ucb",
    ) -> torch.Tensor:
        """
        Compute gradient of acquisition function.

        For prompt optimization, we want to minimize error rate.
        GP predicts error rate, so:
        - UCB: R = -mean + β·std (minimize error + explore)
        - EI: Expected improvement for minimization
        - neg_error: R = -mean (pure exploitation)

        Args:
            z: (B, latent_dim) current latent positions
            acquisition: Which acquisition function to use

        Returns:
            (B, latent_dim) gradient of acquisition w.r.t. z
        """
        if self.gp is None:
            return torch.zeros_like(z)

        # Enable gradients for z
        z_grad = z.detach().requires_grad_(True)

        # Get GP predictions
        mean, std = self.gp.predict(z_grad)

        # Compute reward based on acquisition function
        if acquisition == "ucb":
            # UCB for minimization: want low mean (error) + high std (exploration)
            # R = -mean + β·std
            reward = -mean + self.ucb_beta * std

        elif acquisition == "ei":
            # Expected Improvement for minimization
            if hasattr(self.gp, "best_error_rate"):
                best = self.gp.best_error_rate
            else:
                best = mean.min().item()

            # EI = E[max(best - Y, 0)]
            improvement = best - mean
            Z = improvement / (std + 1e-8)

            # EI = improvement · Φ(Z) + std · φ(Z)
            normal = torch.distributions.Normal(0, 1)
            cdf = normal.cdf(Z)
            pdf = normal.log_prob(Z).exp()
            reward = improvement * cdf + std * pdf

        else:  # neg_error - pure exploitation
            reward = -mean

        # Backpropagate to get gradient
        reward.sum().backward()
        grad = z_grad.grad.detach()

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
        # Generate more candidates than needed
        result = self.generate(
            batch_size=num_candidates,
            num_steps=num_steps,
            context=context,
            acquisition=acquisition,
        )

        candidates = result.latents

        # Score by acquisition (lower error = better)
        if result.acquisition_values is not None:
            scores = -result.acquisition_values  # Negate so higher = better
        else:
            scores = torch.zeros(num_candidates, device=candidates.device)

        # Greedy selection with diversity
        selected_indices = []
        remaining = list(range(num_candidates))

        for _ in range(batch_size):
            if not remaining:
                break

            if not selected_indices:
                # First selection: just pick best score
                best_idx = remaining[scores[remaining].argmax()]
            else:
                # Subsequent: balance score and diversity
                best_score = float("-inf")
                best_idx = remaining[0]

                selected_latents = candidates[selected_indices]

                for idx in remaining:
                    # Quality score
                    quality = scores[idx].item()

                    # Diversity: min distance to selected
                    distances = (candidates[idx] - selected_latents).norm(dim=-1)
                    diversity = distances.min().item()

                    # Combined score
                    combined = quality + diversity_weight * diversity

                    if combined > best_score:
                        best_score = combined
                        best_idx = idx

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

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

        if self.acquisition == "ucb":
            reward = -mean + self.ucb_beta * std
        elif self.acquisition == "ei":
            best = getattr(self.gp, "best_error_rate", mean.min().item())
            improvement = best - mean
            Z = improvement / (std + 1e-8)
            normal = torch.distributions.Normal(0, 1)
            reward = improvement * normal.cdf(Z) + std * normal.log_prob(Z).exp()
        else:
            reward = -mean

        reward.sum().backward()
        return z.grad.detach()


def create_guided_generator(
    flowdit: nn.Module,
    gp_model: Optional[nn.Module] = None,
    latent_dim: int = 128,
    guidance_scale: float = 1.0,
    schedule: str = "linear",
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
            std = torch.softplus(out[:, 1]) * 0.1  # Small std
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
