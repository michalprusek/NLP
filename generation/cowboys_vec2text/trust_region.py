"""TuRBO-Style Trust Region Management.

Implements dynamic trust regions that constrain MCMC sampling to
regions where the VAE decoder is known to work well.

Key insight: VAE latent spaces have "holes" - regions that produce
garbage when decoded. Trust regions anchor optimization to regions
with known good decodings, expanding only when confident.

References:
- TuRBO: Eriksson et al., 2019 - "Scalable Global Optimization via Local BO"
- LOL-BO: Maus et al., 2022 - "Local Latent Space BO over Structured Inputs"
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TRConfig:
    """Trust Region configuration.

    Attributes:
        initial_radius: Initial L-infinity radius around anchor
        min_radius: Minimum radius before triggering restart
        max_radius: Maximum radius (prevents over-expansion)
        expand_factor: Multiply radius by this on success
        contract_factor: Multiply radius by this on failure
        success_threshold: Consecutive successes needed to expand
        failure_threshold: Consecutive failures needed to contract
        n_restarts_max: Maximum number of restarts allowed
    """

    initial_radius: float = 1.0
    min_radius: float = 0.1
    max_radius: float = 5.0
    expand_factor: float = 2.0
    contract_factor: float = 0.5
    success_threshold: int = 3
    failure_threshold: int = 5
    n_restarts_max: int = 10


@dataclass
class TRState:
    """Internal state tracking for trust region."""

    radius: float
    success_count: int = 0
    failure_count: int = 0
    n_restarts: int = 0
    best_error: float = float("inf")


class TrustRegionManager:
    """TuRBO-style trust region for latent space optimization.

    Manages an L-infinity ball around the current anchor point.
    Expands on successful evaluations, contracts on failures.
    Restarts when radius becomes too small.

    Key insight: Trust regions anchor optimization to regions
    where the VAE decoder is known to work well. This prevents
    the optimizer from wandering into "holes" in the latent space
    where the decoder produces garbage.

    The L-infinity norm is used because:
    1. It's computationally simple (max over dimensions)
    2. It creates hypercubes, which are easy to sample from
    3. It allows independent per-dimension control

    Usage:
        tr = TrustRegionManager(anchor, config)
        for each MCMC step:
            proposal = sampler.proposal(current)
            if tr.is_within_region(proposal):
                # accept/reject based on MH criterion
            else:
                # reject (outside trust region)
        after evaluation:
            tr.update(z_evaluated, error_rate, best_y)
    """

    def __init__(
        self,
        anchor: torch.Tensor,
        config: TRConfig,
        device: str = "cuda",
    ):
        """Initialize trust region.

        Args:
            anchor: Center of trust region (32D latent)
            config: Trust region configuration
            device: Computation device
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.anchor = anchor.to(self.device)
        self.config = config

        # Initialize state
        self.state = TRState(radius=config.initial_radius)

        # Track best point within current region
        self.best_z = anchor.clone()

        # History for debugging
        self.history: List[dict] = []

    @property
    def radius(self) -> float:
        """Current trust region radius."""
        return self.state.radius

    def is_within_region(self, z: torch.Tensor) -> bool:
        """Check if point is within trust region.

        Uses L-infinity norm: max_i |z_i - anchor_i| <= radius

        Args:
            z: Point to check (32,)

        Returns:
            True if within region
        """
        z = z.to(self.device)
        linf_dist = torch.max(torch.abs(z - self.anchor))
        return linf_dist.item() <= self.state.radius

    def distance_to_boundary(self, z: torch.Tensor) -> float:
        """Compute distance to trust region boundary.

        Positive = inside, Negative = outside.

        Args:
            z: Point to check (32,)

        Returns:
            Signed distance to boundary
        """
        z = z.to(self.device)
        linf_dist = torch.max(torch.abs(z - self.anchor))
        return self.state.radius - linf_dist.item()

    def project_to_region(self, z: torch.Tensor) -> torch.Tensor:
        """Project point onto trust region boundary.

        Clips each dimension to [anchor_i - radius, anchor_i + radius].

        Args:
            z: Point to project (32,)

        Returns:
            Projected point (32,)
        """
        z = z.to(self.device)
        lower = self.anchor - self.state.radius
        upper = self.anchor + self.state.radius
        return torch.clamp(z, lower, upper)

    def get_random_point_in_region(self) -> torch.Tensor:
        """Sample uniformly within trust region.

        Returns:
            Random point within L-infinity ball
        """
        # Uniform in L-infinity ball = uniform per dimension in [-radius, +radius]
        offset = (2 * torch.rand_like(self.anchor) - 1) * self.state.radius
        return self.anchor + offset

    def update(
        self,
        z_evaluated: torch.Tensor,
        error_rate: float,
        best_y: float,
        verbose: bool = False,
    ) -> bool:
        """Update trust region based on evaluation result.

        Success: error_rate < best_y (found improvement)
        Failure: error_rate >= best_y

        Args:
            z_evaluated: Evaluated point
            error_rate: Achieved error rate
            best_y: Best error rate before this evaluation
            verbose: Print status updates

        Returns:
            True if trust region was modified (expanded, contracted, or restarted)
        """
        improved = error_rate < best_y
        modified = False
        action = None

        if improved:
            self.state.success_count += 1
            self.state.failure_count = 0

            # Update best within region
            if error_rate < self.state.best_error:
                self.state.best_error = error_rate
                self.best_z = z_evaluated.clone().to(self.device)

            # Expand on consecutive successes
            if self.state.success_count >= self.config.success_threshold:
                old_radius = self.state.radius
                self.state.radius = min(
                    self.state.radius * self.config.expand_factor,
                    self.config.max_radius,
                )
                self.state.success_count = 0
                modified = True
                action = f"expanded {old_radius:.3f} -> {self.state.radius:.3f}"
        else:
            self.state.failure_count += 1
            self.state.success_count = 0

            # Contract on consecutive failures
            if self.state.failure_count >= self.config.failure_threshold:
                old_radius = self.state.radius
                self.state.radius *= self.config.contract_factor
                self.state.failure_count = 0
                modified = True

                # Check for restart
                if self.state.radius < self.config.min_radius:
                    if self.state.n_restarts < self.config.n_restarts_max:
                        self._restart()
                        action = f"restarted (radius was {old_radius:.3f})"
                    else:
                        action = f"contracted {old_radius:.3f} -> {self.state.radius:.3f} (max restarts reached)"
                else:
                    action = f"contracted {old_radius:.3f} -> {self.state.radius:.3f}"

        # Log history
        self.history.append(
            {
                "error_rate": error_rate,
                "best_y": best_y,
                "improved": improved,
                "radius": self.state.radius,
                "action": action,
            }
        )

        if verbose and action:
            print(f"    Trust region: {action}")

        return modified

    def _restart(self):
        """Restart trust region from best known point.

        Called when radius shrinks below minimum threshold.
        Resets to best_z with initial radius.
        """
        self.state.n_restarts += 1
        self.anchor = self.best_z.clone()
        self.state.radius = self.config.initial_radius
        self.state.success_count = 0
        self.state.failure_count = 0

    def set_anchor(self, new_anchor: torch.Tensor, reset_radius: bool = False):
        """Update the anchor point.

        Args:
            new_anchor: New anchor point (32,)
            reset_radius: If True, reset radius to initial value
        """
        self.anchor = new_anchor.to(self.device)
        if reset_radius:
            self.state.radius = self.config.initial_radius
            self.state.success_count = 0
            self.state.failure_count = 0

    @property
    def should_restart(self) -> bool:
        """Check if trust region should restart."""
        return (
            self.state.radius < self.config.min_radius
            and self.state.n_restarts < self.config.n_restarts_max
        )

    def get_status(self) -> dict:
        """Get current trust region status.

        Returns:
            Dictionary with current state information
        """
        return {
            "radius": self.state.radius,
            "success_count": self.state.success_count,
            "failure_count": self.state.failure_count,
            "n_restarts": self.state.n_restarts,
            "best_error": self.state.best_error,
            "anchor_norm": torch.norm(self.anchor).item(),
        }
