"""Norm reconstruction: Mean norm and probabilistic variants.

Wraps rielbo.norm_distribution for use as a composable component.
"""

from __future__ import annotations

import torch

from rielbo.core.config import NormReconstructionConfig
from rielbo.norm_distribution import NormDistribution, ProbabilisticReconstructor


class MeanNormReconstructor:
    """Reconstruct embedding as direction * mean_norm."""

    def __init__(self, mean_norm: float = 1.0):
        self.mean_norm = mean_norm

    def reconstruct(
        self,
        u_opt: torch.Tensor,
        gp=None,
        project_fn=None,
    ) -> tuple[torch.Tensor, dict]:
        x_opt = u_opt * self.mean_norm
        return x_opt, {"embedding_norm": self.mean_norm}


class ProbabilisticNormReconstructor:
    """Reconstruct embedding with probabilistic norm sampling."""

    def __init__(
        self,
        config: NormReconstructionConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device
        self.norm_dist: NormDistribution | None = None
        self.reconstructor: ProbabilisticReconstructor | None = None
        self.mean_norm: float = 0.0

    def fit(self, norms: torch.Tensor) -> None:
        """Fit norm distribution from observed norms."""
        self.mean_norm = norms.mean().item()
        self.norm_dist = NormDistribution(
            method=self.config.prob_method,
            device=self.device,
        )
        self.norm_dist.fit(norms)
        self.reconstructor = ProbabilisticReconstructor(
            self.norm_dist,
            n_candidates=self.config.n_candidates,
            selection="gp_mean",
            device=self.device,
        )

    def reconstruct(
        self,
        u_opt: torch.Tensor,
        gp=None,
        project_fn=None,
    ) -> tuple[torch.Tensor, dict]:
        if self.reconstructor is not None and gp is not None:
            return self.reconstructor.reconstruct(
                u_opt, gp=gp, project_fn=project_fn,
            )
        else:
            x_opt = u_opt * self.mean_norm
            return x_opt, {"embedding_norm": self.mean_norm}


def create_norm_reconstructor(
    config: NormReconstructionConfig,
    device: str = "cuda",
) -> MeanNormReconstructor | ProbabilisticNormReconstructor:
    """Factory for norm reconstruction strategies."""
    if config.method == "probabilistic":
        return ProbabilisticNormReconstructor(config, device=device)
    else:
        return MeanNormReconstructor()
