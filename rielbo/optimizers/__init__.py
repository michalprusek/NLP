"""Thin optimizer wrappers for common configurations."""

from rielbo.optimizers.subspace import SubspaceBO
from rielbo.optimizers.turbo import TuRBO
from rielbo.optimizers.vanilla import VanillaBO
from rielbo.optimizers.ensemble import EnsembleBO

__all__ = [
    "SubspaceBO",
    "TuRBO",
    "VanillaBO",
    "EnsembleBO",
]
