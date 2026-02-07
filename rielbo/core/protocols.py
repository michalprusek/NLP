"""Protocol definitions (interfaces) for composable BO components.

Each protocol defines the minimal interface a component must satisfy.
Protocol-based composition avoids deep inheritance hierarchies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from botorch.models import SingleTaskGP


@runtime_checkable
class SurrogateModel(Protocol):
    """Protocol for GP surrogate models."""

    gp: SingleTaskGP | None

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Fit the surrogate to training data."""
        ...

    @property
    def last_mll_value(self) -> float | None:
        """Log marginal likelihood from the last fit."""
        ...


@runtime_checkable
class TrustRegionStrategy(Protocol):
    """Protocol for trust region adaptation strategies."""

    @property
    def radius(self) -> float:
        """Current trust region radius."""
        ...

    def update(self, improved: bool, gp_std: float | None = None) -> None:
        """Update trust region based on latest observation."""
        ...

    @property
    def needs_restart(self) -> bool:
        """Whether the TR has collapsed and needs a subspace restart."""
        ...

    def reset(self) -> None:
        """Reset TR state (e.g. after a subspace restart)."""
        ...

    @property
    def needs_rotation(self) -> bool:
        """Whether sustained GP collapse requires a subspace rotation."""
        ...


@runtime_checkable
class ProjectionStrategy(Protocol):
    """Protocol for subspace projection strategies."""

    @property
    def A(self) -> torch.Tensor:
        """Current projection matrix [D, d]."""
        ...

    @property
    def subspace_dim(self) -> int:
        """Current subspace dimensionality."""
        ...

    def project(self, u: torch.Tensor) -> torch.Tensor:
        """Project from ambient space to subspace: S^(D-1) -> S^(d-1)."""
        ...

    def lift(self, v: torch.Tensor) -> torch.Tensor:
        """Lift from subspace to ambient space: S^(d-1) -> S^(D-1)."""
        ...

    def reinitialize(self, seed: int, train_U: torch.Tensor | None = None) -> None:
        """Create a fresh projection (e.g. new random QR)."""
        ...


@runtime_checkable
class AcquisitionStrategy(Protocol):
    """Protocol for acquisition function selection."""

    def select(
        self,
        gp: SingleTaskGP,
        candidates: torch.Tensor,
        train_Y: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Select the best candidate from the set.

        Returns:
            (selected_point, diagnostics_dict)
        """
        ...


@runtime_checkable
class CandidateGenerator(Protocol):
    """Protocol for candidate generation strategies."""

    def generate(
        self,
        center: torch.Tensor,
        n_candidates: int,
        radius: float,
    ) -> torch.Tensor:
        """Generate candidate points around center.

        Args:
            center: Best point in the subspace [1, d]
            n_candidates: Number of candidates to generate
            radius: Trust region radius

        Returns:
            Candidate tensor [n_candidates, d]
        """
        ...
