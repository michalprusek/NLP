"""Candidate generation strategies: Geodesic disk and Sobol box.

Extracted from V2's _generate_candidates() and V1/VanillaBO/TuRBO candidate gen.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.quasirandom import SobolEngine

from rielbo.core.config import CandidateGenConfig
from rielbo.spherical_transforms import GeodesicTrustRegion


class GeodesicGenerator:
    """Generate candidates within a geodesic disk on S^(d-1).

    Uses the exponential map for proper Riemannian sampling.
    """

    def __init__(
        self,
        geodesic_max_angle: float = 0.5,
        global_fraction: float = 0.2,
        device: str = "cuda",
    ):
        self.geodesic_tr = GeodesicTrustRegion(
            max_angle=geodesic_max_angle,
            global_fraction=global_fraction,
            device=device,
        )
        self.device = device

    def generate(
        self,
        center: torch.Tensor,
        n_candidates: int,
        radius: float,
    ) -> torch.Tensor:
        """Sample candidates within geodesic disk of given radius.

        Args:
            center: [1, d] center point on S^(d-1)
            n_candidates: Number of candidates
            radius: Geodesic radius in radians

        Returns:
            [n_candidates, d] candidates on S^(d-1)
        """
        return self.geodesic_tr.sample(
            center=center,
            n_samples=n_candidates,
            adaptive_radius=radius,
        )


class SobolBoxGenerator:
    """Generate candidates in a Sobol box trust region.

    Samples quasi-random points in a hypercube around the center,
    then normalizes to the sphere (for spherical methods) or
    clips to bounds (for Euclidean methods).
    """

    def __init__(self, dim: int, device: str = "cuda"):
        self.dim = dim
        self.device = device

    def generate(
        self,
        center: torch.Tensor,
        n_candidates: int,
        radius: float,
        normalize_to_sphere: bool = True,
        clip_bounds: tuple[float, float] | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """Generate Sobol candidates in box trust region.

        Args:
            center: [1, d] center point
            n_candidates: Number of candidates
            radius: Half-length of the box
            normalize_to_sphere: If True, normalize output to S^(d-1)
            clip_bounds: If set, clip candidates to (lo, hi)
            seed: Sobol seed

        Returns:
            [n_candidates, d] candidates
        """
        half_length = radius / 2
        tr_lb = center - half_length
        tr_ub = center + half_length

        if clip_bounds is not None:
            lo, hi = clip_bounds
            tr_lb = tr_lb.clamp(lo, hi)
            tr_ub = tr_ub.clamp(lo, hi)

        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(n_candidates).to(
            dtype=torch.float32, device=self.device,
        )

        v_cand = tr_lb + (tr_ub - tr_lb) * pert

        if normalize_to_sphere:
            v_cand = F.normalize(v_cand, p=2, dim=-1)

        return v_cand

    def generate_pca(
        self,
        center: torch.Tensor,
        v_std: torch.Tensor,
        n_candidates: int,
        tr_scale: float,
    ) -> torch.Tensor:
        """Generate Sobol candidates for PCA mode (Euclidean box, no sphere norm).

        Args:
            center: [1, d] center in PCA space
            v_std: [1, d] per-dim std of training data in PCA space
            n_candidates: Number of candidates
            tr_scale: Trust region scale factor

        Returns:
            [n_candidates, d] candidates in PCA space
        """
        half_length = v_std * tr_scale * 2
        tr_lb = center - half_length
        tr_ub = center + half_length

        sobol = SobolEngine(self.dim, scramble=True)
        pert = sobol.draw(n_candidates).to(
            dtype=torch.float32, device=self.device,
        )

        return tr_lb + (tr_ub - tr_lb) * pert


def create_candidate_generator(
    config: CandidateGenConfig,
    dim: int,
    device: str = "cuda",
    geodesic_max_angle: float = 0.5,
    geodesic_global_fraction: float = 0.2,
) -> GeodesicGenerator | SobolBoxGenerator:
    """Factory for candidate generators."""
    if config.strategy == "geodesic":
        return GeodesicGenerator(
            geodesic_max_angle=geodesic_max_angle,
            global_fraction=geodesic_global_fraction,
            device=device,
        )
    elif config.strategy == "sobol_box":
        return SobolBoxGenerator(dim=dim, device=device)
    else:
        raise ValueError(f"Unknown candidate gen strategy: {config.strategy}")
