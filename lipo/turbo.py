"""TuRBO trust region and Potential-Aware Anchor Selection for LIPO.

Implements InvBO components:
- TrustRegionManager: TuRBO-style dynamic trust region management
- PotentialAwareAnchorSelector: Thompson Sampling-based anchor selection (Algorithm 2)

References:
- TuRBO: Eriksson et al., "Scalable Global Optimization via Local Bayesian Optimization" (NeurIPS 2019)
- InvBO: Chu et al., "Inversion-based Latent Bayesian Optimization" (NeurIPS 2024)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Tuple, Optional
import gpytorch


@dataclass
class TrustRegionState:
    """State of a single trust region.

    Attributes:
        center: Current anchor in normalized latent space [0, 1]^d
        length: Current side length (fraction of unit hypercube)
        success_count: Consecutive successes since last adjustment
        fail_count: Consecutive failures since last adjustment
        restart_count: Number of trust region restarts
    """

    center: Optional[torch.Tensor] = None
    length: float = 0.8
    success_count: int = 0
    fail_count: int = 0
    restart_count: int = 0


class TrustRegionManager:
    """Manages TuRBO-style trust region state and bounds computation.

    The trust region adapts based on optimization success:
    - Consecutive successes: expand (double L up to L_max)
    - Consecutive failures: shrink (halve L down to L_min, then restart)

    This creates adaptive exploration/exploitation balance:
    - Early: wide search with large trust regions
    - Late: focused search after multiple failures shrink the region
    """

    def __init__(
        self,
        dim: int,
        device: torch.device,
        L_init: float = 0.8,
        L_max: float = 1.6,
        L_min: float = 0.0078,  # 0.5^7
        tau_succ: int = 3,
        tau_fail: int = 40,
    ):
        """Initialize trust region manager.

        Args:
            dim: Dimensionality of latent space (16 for VAE latent, matches config.latent_dim)
            device: Torch device
            L_init: Initial side length (fraction of unit cube)
            L_max: Maximum side length
            L_min: Minimum side length (triggers restart if L < L_min)
            tau_succ: Consecutive successes needed to expand
            tau_fail: Consecutive failures needed to shrink
        """
        self.dim = dim
        self.device = device
        self.L_init = L_init
        self.L_max = L_max
        self.L_min = L_min
        self.tau_succ = tau_succ
        self.tau_fail = tau_fail

        self.state = TrustRegionState(length=L_init)

    def set_anchor(self, anchor: torch.Tensor) -> None:
        """Set trust region center.

        Args:
            anchor: Center point in normalized [0, 1]^d space
        """
        self.state.center = anchor.to(self.device)

    def get_bounds(self, global_bounds: torch.Tensor) -> torch.Tensor:
        """Compute trust region bounds centered on anchor.

        Bounds are [center - L/2, center + L/2] clamped to global bounds.

        Args:
            global_bounds: Global optimization bounds, shape (2, dim)
                          bounds[0] = lower, bounds[1] = upper

        Returns:
            Trust region bounds, shape (2, dim)
        """
        if self.state.center is None:
            # No anchor set - use global bounds (expected on first iteration)
            import warnings
            warnings.warn(
                "TrustRegionManager.get_bounds() called with no anchor set. "
                "Using global bounds. This is normal on first iteration."
            )
            return global_bounds

        center = self.state.center
        half_length = self.state.length / 2

        # Compute trust region bounds
        lower = center - half_length
        upper = center + half_length

        # Clamp to global bounds
        lower = torch.maximum(lower, global_bounds[0])
        upper = torch.minimum(upper, global_bounds[1])

        return torch.stack([lower, upper], dim=0)

    def update(self, improved: bool) -> dict:
        """Update trust region state based on iteration result.

        Implements TuRBO scheduling:
        - τ_succ consecutive successes: L = min(2*L, L_max)
        - τ_fail consecutive failures: L = L/2, restart if L < L_min

        Args:
            improved: Whether this iteration improved the best score

        Returns:
            Dict with update info (for logging)
        """
        info = {
            "improved": improved,
            "length_before": self.state.length,
            "action": "none",
        }

        if improved:
            self.state.success_count += 1
            self.state.fail_count = 0

            if self.state.success_count >= self.tau_succ:
                # Expand trust region
                self.state.length = min(2 * self.state.length, self.L_max)
                self.state.success_count = 0
                info["action"] = "expand"
        else:
            self.state.fail_count += 1
            self.state.success_count = 0

            if self.state.fail_count >= self.tau_fail:
                # Shrink trust region
                self.state.length = self.state.length / 2
                self.state.fail_count = 0

                if self.state.length < self.L_min:
                    # Restart with initial length
                    self.state.length = self.L_init
                    self.state.restart_count += 1
                    info["action"] = "restart"
                else:
                    info["action"] = "shrink"

        info["length_after"] = self.state.length
        info["success_count"] = self.state.success_count
        info["fail_count"] = self.state.fail_count
        info["restart_count"] = self.state.restart_count

        return info

    def get_state_summary(self) -> str:
        """Get human-readable state summary for logging."""
        return (
            f"L={self.state.length:.4f} "
            f"(succ={self.state.success_count}/{self.tau_succ}, "
            f"fail={self.state.fail_count}/{self.tau_fail}, "
            f"restarts={self.state.restart_count})"
        )


class PotentialAwareAnchorSelector:
    """Selects trust region anchor using Potential-Aware Selection (PAS).

    Implements Algorithm 2 from InvBO paper.

    PAS combines observed objective value with Thompson Sampling-based
    potential score to select anchors that balance exploitation (good
    observed values) with exploration (high potential regions).

    Score formula: s^i = y^i + α_scaled^i
    where α_scaled^i is the scaled Thompson Sampling potential.
    """

    def __init__(
        self,
        device: torch.device,
        n_candidates: int = 100,
    ):
        """Initialize anchor selector.

        Args:
            device: Torch device
            n_candidates: Number of candidates to sample per anchor for potential estimation
        """
        self.device = device
        self.n_candidates = n_candidates

    def _sample_candidates_in_region(
        self,
        center: torch.Tensor,
        trust_length: float,
        global_bounds: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Sample candidate points in a local trust region.

        Args:
            center: Region center, shape (dim,)
            trust_length: Side length of trust region
            global_bounds: Global bounds, shape (2, dim)
            n_samples: Number of samples

        Returns:
            Candidates, shape (n_samples, dim)
        """
        dim = center.shape[0]

        # Sample uniformly in [-0.5, 0.5]^d and scale
        samples = torch.rand(n_samples, dim, device=self.device) - 0.5
        samples = samples * trust_length + center

        # Clamp to global bounds
        samples = torch.clamp(samples, global_bounds[0], global_bounds[1])

        return samples

    def _thompson_sample(
        self,
        gp_model: nn.Module,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from GP posterior at points X using Thompson Sampling.

        Args:
            gp_model: Trained GP model
            X: Points to evaluate, shape (N, dim)

        Returns:
            Sampled function values, shape (N,)

        Raises:
            RuntimeError: If GP model produces NaN values (indicates failed training)
        """
        gp_model.eval()
        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Get posterior distribution
                posterior = gp_model(X)

                # Check for NaN in posterior (indicates failed GP training)
                if torch.isnan(posterior.mean).any():
                    raise RuntimeError(
                        "GP posterior contains NaN values. "
                        "This indicates GP training failed. "
                        "The GP model should not be used for inference."
                    )

                # Sample one function realization
                # rsample() uses reparameterization trick for gradient flow
                # but we're in no_grad context so it's just a sample
                sample = posterior.rsample(torch.Size([1]))  # (1, N)

            return sample.squeeze(0)  # (N,)
        except (RuntimeError, gpytorch.utils.errors.NotPSDError) as e:
            error_msg = str(e).lower()
            if "cholesky" in error_msg or "nan" in error_msg or "singular" in error_msg:
                raise RuntimeError(
                    f"GP prediction failed due to numerical instability. "
                    f"This indicates GP training was corrupted. "
                    f"Original error: {e}"
                ) from e
            raise

    def select_anchor(
        self,
        gp_model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_min: torch.Tensor,
        X_max: torch.Tensor,
        trust_length: float,
        global_bounds: torch.Tensor,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """Select best anchor using Potential-Aware Selection (PAS).

        Algorithm 2 from InvBO paper:
        1. For each training point, sample candidates in local region
        2. Use Thompson Sampling to estimate potential (max posterior sample)
        3. Scale potentials to match objective value range
        4. Combine: s^i = y^i + α_scaled^i
        5. Return point with highest combined score

        Args:
            gp_model: Trained GP model (InstructionDeepKernelGP)
            X_train: Training latents, shape (N, latent_dim) - unnormalized VAE latents
            y_train: Training targets, shape (N,) - negated error rates
            X_min: Min values for normalization, shape (latent_dim,)
            X_max: Max values for normalization, shape (latent_dim,)
            trust_length: Current trust region side length
            global_bounds: Global bounds in normalized space, shape (2, latent_dim)
            verbose: Print debug info

        Returns:
            (anchor_latent, anchor_idx) tuple where:
            - anchor_latent: Selected anchor in normalized [0,1]^d space, shape (latent_dim,)
            - anchor_idx: Index of selected anchor in training data

        Raises:
            ValueError: If X_train is empty
        """
        n_points = X_train.shape[0]
        if n_points == 0:
            raise ValueError("Cannot select anchor from empty training data")

        # Normalize training data to [0, 1]^d
        denom = X_max - X_min
        denom[denom == 0] = 1.0
        X_norm = (X_train - X_min) / denom

        # Step 1 & 2: Compute potential score for each training point
        potentials = torch.zeros(n_points, device=self.device)

        for i in range(n_points):
            # Sample candidates in local trust region around point i
            candidates = self._sample_candidates_in_region(
                center=X_norm[i],
                trust_length=trust_length,
                global_bounds=global_bounds,
                n_samples=self.n_candidates,
            )

            # Thompson Sampling on normalized candidates
            # GP model expects normalized [0,1] inputs (applies Kumaraswamy warping internally)
            ts_values = self._thompson_sample(gp_model, candidates)
            potentials[i] = ts_values.max()

        # Step 3: Scale potentials to match objective value range
        # α_scaled^i = (α_pot^i - A_min) / (A_max - A_min) × (Y_max - Y_min)
        A_min = potentials.min()
        A_max = potentials.max()
        Y_min = y_train.min()
        Y_max = y_train.max()

        # Handle edge case where all potentials are equal
        A_range = A_max - A_min
        if A_range < 1e-8:
            scaled_potentials = torch.zeros_like(potentials)
        else:
            Y_range = Y_max - Y_min
            scaled_potentials = (potentials - A_min) / A_range * Y_range

        # Step 4: Combine observed value + scaled potential
        # s^i = y^i + α_scaled^i
        scores = y_train + scaled_potentials

        # Step 5: Select anchor with highest score
        anchor_idx = scores.argmax().item()
        anchor_latent = X_norm[anchor_idx]

        if verbose:
            print(f"  PAS: selected anchor {anchor_idx} with score {scores[anchor_idx]:.4f}")
            print(f"    y={y_train[anchor_idx]:.4f}, potential={scaled_potentials[anchor_idx]:.4f}")
            print(f"    Potential range: [{A_min:.4f}, {A_max:.4f}]")
            print(f"    Objective range: [{Y_min:.4f}, {Y_max:.4f}]")

        return anchor_latent, anchor_idx


def create_turbo_manager(config, device: torch.device) -> TrustRegionManager:
    """Factory function to create TrustRegionManager from config.

    Args:
        config: Config dataclass with turbo_* parameters
        device: Torch device

    Returns:
        Configured TrustRegionManager
    """
    return TrustRegionManager(
        dim=config.latent_dim,  # 16D VAE latent (config.latent_dim)
        device=device,
        L_init=config.turbo_L_init,
        L_max=config.turbo_L_max,
        L_min=config.turbo_L_min,
        tau_succ=config.turbo_tau_succ,
        tau_fail=config.turbo_tau_fail,
    )


def create_pas_selector(config, device: torch.device) -> PotentialAwareAnchorSelector:
    """Factory function to create PotentialAwareAnchorSelector from config.

    Args:
        config: Config dataclass with pas_* parameters
        device: Torch device

    Returns:
        Configured PotentialAwareAnchorSelector
    """
    return PotentialAwareAnchorSelector(
        device=device,
        n_candidates=config.pas_n_candidates,
    )
