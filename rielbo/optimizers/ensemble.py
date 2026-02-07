"""EnsembleBO: Multi-scale subspace ensemble optimization.

Re-exports from the existing ensemble_bo module. The Ensemble pattern
(K independent GPs with cross-selection) doesn't cleanly decompose
into BaseOptimizer composition, so it remains its own class.

The SubspaceMember + SphericalEnsembleBO implementation already uses
components internally (QR projection, geodesic TR, adaptive TR).
"""

from __future__ import annotations

from rielbo.ensemble_bo import (
    EnsembleConfig,
    SphericalEnsembleBO,
    SubspaceMember,
)

# Alias for consistency with the optimizers package
EnsembleBO = SphericalEnsembleBO

__all__ = [
    "EnsembleBO",
    "EnsembleConfig",
    "SphericalEnsembleBO",
    "SubspaceMember",
]
