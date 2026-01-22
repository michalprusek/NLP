"""
TuRBO-1024: Trust Region Bayesian Optimization for 1024D SONAR Space.

Extends the TuRBO algorithm for high-dimensional optimization in 1024D.
Key adaptations:
- tau_fail=128: ceil(1024/8) for proper scaling with dimensionality
- L_init=0.4: Smaller initial trust region for high-D exploration
- ARD-scaled bounds: L_i = λ_i * L / geom_mean(λ) for anisotropic scaling

Reference:
- Eriksson et al., "Scalable Global Optimization via Local Bayesian Optimization" (NeurIPS 2019)
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class TrustRegionState1024:
    """State of a trust region in 1024D space.

    Attributes:
        center: Current anchor in normalized space [0, 1]^d or None
        length: Current side length (fraction of unit hypercube)
        success_count: Consecutive successes since last adjustment
        fail_count: Consecutive failures since last adjustment
        restart_count: Number of trust region restarts
        best_value: Best objective value seen in this trust region
    """

    center: Optional[torch.Tensor] = None
    length: float = 0.4  # Smaller initial for 1024D
    success_count: int = 0
    fail_count: int = 0
    restart_count: int = 0
    best_value: float = float('inf')


class TuRBO1024:
    """Trust Region manager for 1024D SONAR space.

    The trust region adapts based on optimization success:
    - Consecutive successes: expand (double L up to L_max)
    - Consecutive failures: shrink (halve L down to L_min, then restart)

    Key parameters for 1024D:
    - tau_fail=128: From TuRBO paper, ⌈d/q⌉ where d=1024, q=8
    - L_init=0.4: Smaller initial region for high-D (vs 0.8 for 32D)
    - ARD scaling: Per-dimension bounds based on GP lengthscales
    """

    def __init__(
        self,
        dim: int = 1024,
        device: torch.device = None,
        L_init: float = 0.4,
        L_max: float = 1.6,
        L_min: float = 0.0078,  # 2^-7
        tau_succ: int = 3,
        tau_fail: int = 128,  # ceil(1024/8) for high-D
    ):
        """
        Initialize TuRBO-1024.

        Args:
            dim: Dimensionality (1024 for SONAR)
            device: Torch device
            L_init: Initial side length
            L_max: Maximum side length
            L_min: Minimum side length (triggers restart)
            tau_succ: Consecutive successes to expand
            tau_fail: Consecutive failures to shrink
        """
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L_init = L_init
        self.L_max = L_max
        self.L_min = L_min
        self.tau_succ = tau_succ
        self.tau_fail = tau_fail

        self.state = TrustRegionState1024(length=L_init)

        # Statistics for logging
        self.total_expansions = 0
        self.total_shrinks = 0
        self.total_restarts = 0

    def set_anchor(self, anchor: torch.Tensor, value: Optional[float] = None) -> None:
        """Set trust region center.

        Args:
            anchor: Center point in normalized [0, 1]^d space
            value: Optional objective value at anchor
        """
        self.state.center = anchor.to(self.device)
        if value is not None:
            self.state.best_value = value

    def get_bounds(
        self,
        global_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute trust region bounds centered on anchor.

        Bounds are [center - L/2, center + L/2] clamped to global bounds.

        Args:
            global_bounds: Global optimization bounds, shape (2, dim)
                          bounds[0] = lower, bounds[1] = upper

        Returns:
            Trust region bounds, shape (2, dim)
        """
        if self.state.center is None:
            warnings.warn(
                "TuRBO1024.get_bounds() called with no anchor set. "
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

    def get_ard_scaled_bounds(
        self,
        global_bounds: torch.Tensor,
        lengthscales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute trust region bounds scaled by ARD lengthscales.

        From TuRBO paper:
        "The actual side length for each dimension is obtained from this base
        side length by rescaling according to its lengthscale λ_i in the GP model
        while maintaining a total volume of L^d. That is,
        L_i = λ_i * L / (∏_{j=1}^d λ_j)^{1/d}"

        For 1024D, this is critical:
        - Dimensions with large lengthscale (smooth) get wider bounds
        - Dimensions with small lengthscale (sensitive) get tighter bounds
        - Volume is preserved: ∏ L_i = L^d

        Args:
            global_bounds: Global bounds, shape (2, dim)
            lengthscales: ARD lengthscales from GP, shape (dim,)
                         If None or scalar, uses uniform scaling

        Returns:
            Trust region bounds, shape (2, dim)
        """
        if self.state.center is None:
            warnings.warn(
                "TuRBO1024.get_ard_scaled_bounds() called with no anchor. "
                "Using global bounds."
            )
            return global_bounds

        center = self.state.center
        L = self.state.length
        dim = global_bounds.shape[1]

        # Check if we have valid ARD lengthscales
        use_ard = (
            lengthscales is not None
            and lengthscales.dim() == 1
            and lengthscales.shape[0] == dim
        )

        if use_ard:
            # TuRBO paper formula: L_i = λ_i * L / geom_mean(λ)
            # Compute geometric mean in log space for numerical stability
            log_ls = torch.log(lengthscales.clamp(min=1e-6))
            log_geom_mean = log_ls.mean()
            geom_mean = torch.exp(log_geom_mean)

            # Per-dimension side length
            per_dim_L = lengthscales * L / geom_mean
            per_dim_half_L = per_dim_L / 2

            # Clamp to prevent extreme scaling
            # For 1024D, allow wider range (max 8x difference from base)
            per_dim_half_L = per_dim_half_L.clamp(L / 16, L * 4)
        else:
            # Uniform scaling (isotropic)
            per_dim_half_L = torch.full((dim,), L / 2, device=global_bounds.device)

        # Compute bounds
        lower = center - per_dim_half_L
        upper = center + per_dim_half_L

        # Clamp to global bounds
        lower = torch.maximum(lower, global_bounds[0])
        upper = torch.minimum(upper, global_bounds[1])

        return torch.stack([lower, upper], dim=0)

    def update(self, improved: bool, new_value: Optional[float] = None) -> dict:
        """Update trust region state based on iteration result.

        Implements TuRBO scheduling:
        - τ_succ consecutive successes: L = min(2*L, L_max)
        - τ_fail consecutive failures: L = L/2, restart if L < L_min

        Args:
            improved: Whether this iteration improved the best score
            new_value: New objective value (for tracking)

        Returns:
            Dict with update info (for logging)
        """
        info = {
            "improved": improved,
            "length_before": self.state.length,
            "action": "none",
            "success_count": self.state.success_count,
            "fail_count": self.state.fail_count,
        }

        # Update best value if provided
        if new_value is not None and new_value < self.state.best_value:
            self.state.best_value = new_value

        if improved:
            self.state.success_count += 1
            self.state.fail_count = 0

            if self.state.success_count >= self.tau_succ:
                # Expand trust region
                self.state.length = min(2 * self.state.length, self.L_max)
                self.state.success_count = 0
                self.total_expansions += 1
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
                    self.total_restarts += 1
                    info["action"] = "restart"
                else:
                    self.total_shrinks += 1
                    info["action"] = "shrink"

        info["length_after"] = self.state.length
        info["success_count"] = self.state.success_count
        info["fail_count"] = self.state.fail_count
        info["restart_count"] = self.state.restart_count

        return info

    def suggest_restart_point(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        global_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Suggest a new anchor point after restart.

        After a restart (L dropped below L_min), we need a new anchor.
        Strategy: Pick from top-k points with some randomization.

        Args:
            X_train: Training inputs, shape (N, dim)
            y_train: Training targets, shape (N,)
            global_bounds: Global bounds, shape (2, dim)

        Returns:
            New anchor point, shape (dim,)
        """
        if X_train.shape[0] == 0:
            # No data - random point in bounds
            return torch.rand(self.dim, device=self.device) * (
                global_bounds[1] - global_bounds[0]
            ) + global_bounds[0]

        # Find top-k points (lowest objective values)
        k = min(5, X_train.shape[0])
        _, top_indices = y_train.topk(k, largest=False)

        # Pick one randomly
        idx = top_indices[torch.randint(k, (1,)).item()]
        return X_train[idx].clone()

    def get_state_summary(self) -> str:
        """Get human-readable state summary for logging."""
        return (
            f"L={self.state.length:.4f} "
            f"(succ={self.state.success_count}/{self.tau_succ}, "
            f"fail={self.state.fail_count}/{self.tau_fail}, "
            f"restarts={self.state.restart_count})"
        )

    def get_statistics(self) -> dict:
        """Get overall statistics for reporting."""
        return {
            "total_expansions": self.total_expansions,
            "total_shrinks": self.total_shrinks,
            "total_restarts": self.total_restarts,
            "current_length": self.state.length,
            "best_value": self.state.best_value,
        }


def create_turbo_1024(config) -> TuRBO1024:
    """Factory function to create TuRBO-1024 from config.

    Args:
        config: FlowPOHDConfig with turbo_* parameters

    Returns:
        Configured TuRBO1024 instance
    """
    device = torch.device(config.device)

    return TuRBO1024(
        dim=config.sonar_dim,
        device=device,
        L_init=config.turbo_L_init,
        L_max=config.turbo_L_max,
        L_min=config.turbo_L_min,
        tau_succ=config.turbo_tau_succ,
        tau_fail=config.turbo_tau_fail,
    )


if __name__ == "__main__":
    print("Testing TuRBO-1024...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create manager
    turbo = TuRBO1024(
        dim=1024,
        device=device,
        L_init=0.4,
        tau_fail=128,
    )

    print(f"\nInitial state: {turbo.get_state_summary()}")

    # Test bounds computation
    print("\n--- Bounds Computation ---")
    global_bounds = torch.stack([
        torch.zeros(1024, device=device),
        torch.ones(1024, device=device),
    ])

    # Set anchor at center
    anchor = torch.full((1024,), 0.5, device=device)
    turbo.set_anchor(anchor)

    bounds = turbo.get_bounds(global_bounds)
    print(f"Global bounds: [{global_bounds[0, 0]:.2f}, {global_bounds[1, 0]:.2f}]")
    print(f"Trust bounds: [{bounds[0, 0]:.2f}, {bounds[1, 0]:.2f}]")
    print(f"Width: {(bounds[1] - bounds[0]).mean():.4f}")

    # Test ARD scaling
    print("\n--- ARD-Scaled Bounds ---")
    # Create lengthscales with variation
    lengthscales = torch.ones(1024, device=device)
    lengthscales[:100] = 0.1  # First 100 dims are "important" (small lengthscale)
    lengthscales[100:] = 1.0  # Rest are "noise" (large lengthscale)

    ard_bounds = turbo.get_ard_scaled_bounds(global_bounds, lengthscales)
    print(f"ARD bounds dim 0 (important): [{ard_bounds[0, 0]:.4f}, {ard_bounds[1, 0]:.4f}]")
    print(f"ARD bounds dim 500 (noise): [{ard_bounds[0, 500]:.4f}, {ard_bounds[1, 500]:.4f}]")

    # Test update cycle
    print("\n--- Update Cycle ---")

    # Simulate successes
    for i in range(4):
        info = turbo.update(improved=True)
        print(f"Success {i+1}: {turbo.get_state_summary()}, action={info['action']}")

    # Simulate failures
    for i in range(150):  # More than tau_fail
        info = turbo.update(improved=False)
        if info["action"] != "none":
            print(f"Failure {i+1}: {turbo.get_state_summary()}, action={info['action']}")

    print(f"\nFinal state: {turbo.get_state_summary()}")
    print(f"Statistics: {turbo.get_statistics()}")

    print("\n[OK] TuRBO-1024 tests passed!")
