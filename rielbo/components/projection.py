"""Subspace projection strategies: Random QR, PCA, Identity, LASS.

Extracted from V2's _init_projection(), project_to_subspace(),
lift_to_original(), and _select_best_projection().
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from rielbo.core.config import ProjectionConfig, KernelConfig
from rielbo.spherical_transforms import SphericalWhitening

logger = logging.getLogger(__name__)


class QRProjection:
    """Random orthonormal QR projection S^(D-1) -> S^(d-1)."""

    def __init__(
        self,
        config: ProjectionConfig,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.config = config
        self.device = device
        self._input_dim = config.input_dim
        self._subspace_dim = config.subspace_dim

        # Whitening
        self._whitening: SphericalWhitening | None = None
        if config.whitening:
            self._whitening = SphericalWhitening(device=device)

        # Initialize projection matrix
        self._init_projection(seed)

    def _init_projection(self, seed: int) -> None:
        torch.manual_seed(seed)
        A_raw = torch.randn(self._input_dim, self._subspace_dim, device=self.device)
        self._A, _ = torch.linalg.qr(A_raw)

    @property
    def A(self) -> torch.Tensor:
        return self._A

    @property
    def subspace_dim(self) -> int:
        return self._subspace_dim

    def fit_whitening(self, train_U: torch.Tensor) -> None:
        """Fit spherical whitening transform from training data."""
        if self._whitening is not None:
            self._whitening.fit(train_U)
            logger.info("Spherical whitening fitted")

    def project(self, u: torch.Tensor) -> torch.Tensor:
        """S^(D-1) -> S^(d-1)."""
        if self._whitening is not None and self._whitening.H is not None:
            u = self._whitening.transform(u)
        v = u @ self._A
        return F.normalize(v, p=2, dim=-1)

    def lift(self, v: torch.Tensor) -> torch.Tensor:
        """S^(d-1) -> S^(D-1)."""
        u = v @ self._A.T
        u = F.normalize(u, p=2, dim=-1)
        if self._whitening is not None and self._whitening.H is not None:
            u = self._whitening.inverse_transform(u)
        return u

    def reinitialize(self, seed: int, train_U: torch.Tensor | None = None) -> None:
        """Create fresh random QR projection."""
        self._init_projection(seed)
        if self._whitening is not None and train_U is not None:
            self._whitening.fit(train_U)

    def set_projection(self, A: torch.Tensor) -> None:
        """Directly set the projection matrix (used by LASS)."""
        self._A = A


class PCAProjection:
    """PCA-based projection for high-D codecs (e.g. SONAR 1024D)."""

    def __init__(
        self,
        config: ProjectionConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device
        self._input_dim = config.input_dim
        self._subspace_dim = config.subspace_dim
        self._A: torch.Tensor | None = None
        self._pca_mean: torch.Tensor | None = None

    @property
    def A(self) -> torch.Tensor:
        return self._A

    @property
    def subspace_dim(self) -> int:
        return self._subspace_dim

    def fit(self, train_U: torch.Tensor) -> None:
        """Compute PCA projection from training directions."""
        self._pca_mean = train_U.mean(dim=0, keepdim=True).to(self.device)
        centered = train_U - self._pca_mean
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        d = min(self._subspace_dim, Vt.shape[0])
        self._A = Vt[:d].T.contiguous().to(self.device)
        var_explained = (S[:d] ** 2).sum() / (S ** 2).sum()
        logger.info(
            f"PCA projection: {self._input_dim}D -> {d}D, "
            f"variance explained: {var_explained:.3f}"
        )

    def project(self, u: torch.Tensor) -> torch.Tensor:
        """Centered PCA coordinates in R^d."""
        return (u - self._pca_mean) @ self._A

    def lift(self, v: torch.Tensor) -> torch.Tensor:
        """PCA coordinates -> unit sphere (approximately)."""
        return v @ self._A.T + self._pca_mean

    def reinitialize(self, seed: int, train_U: torch.Tensor | None = None) -> None:
        if train_U is not None:
            self.fit(train_U)


class IdentityProjection:
    """No-op projection for full-dimensional methods (TuRBO, VanillaBO)."""

    def __init__(self, config: ProjectionConfig, device: str = "cuda"):
        self._dim = config.input_dim
        self.device = device

    @property
    def A(self) -> torch.Tensor:
        return torch.eye(self._dim, device=self.device)

    @property
    def subspace_dim(self) -> int:
        return self._dim

    def project(self, u: torch.Tensor) -> torch.Tensor:
        return u

    def lift(self, v: torch.Tensor) -> torch.Tensor:
        return v

    def reinitialize(self, seed: int, train_U: torch.Tensor | None = None) -> None:
        pass


class LASSSelector:
    """Look-Ahead Subspace Selection: evaluate K random projections, pick best.

    Criterion: GP log marginal likelihood (how well the ArcCosine GP
    explains score variation in this subspace).
    """

    def __init__(
        self,
        config: ProjectionConfig,
        kernel_config: KernelConfig,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.config = config
        self.kernel_config = kernel_config
        self.device = device
        self.seed = seed

    def select_best_projection(
        self,
        projection: QRProjection,
        train_U: torch.Tensor,
        train_Y: torch.Tensor,
    ) -> None:
        """Evaluate K candidate projections and set the best one on `projection`."""
        from rielbo.components.surrogate import SphericalGPSurrogate

        K = self.config.lass_n_candidates
        best_score = float("-inf")
        best_A = None
        best_k = -1
        all_scores = []

        logger.info(f"LASS: evaluating {K} candidate projections (criterion=log_ml)...")

        surrogate = SphericalGPSurrogate(
            self.kernel_config, self.config.subspace_dim, self.device, verbose=False,
        )

        for k in range(K):
            torch.manual_seed(self.seed + k * 137)
            A_raw = torch.randn(
                self.config.input_dim, self.config.subspace_dim, device=self.device,
            )
            A_k, _ = torch.linalg.qr(A_raw)

            # Project through this candidate
            v_k = F.normalize(train_U @ A_k, p=2, dim=-1)
            log_ml = surrogate.fit_quick(v_k, train_Y)
            all_scores.append(log_ml)

            if log_ml > best_score:
                best_score = log_ml
                best_A = A_k.clone()
                best_k = k

        if best_A is not None:
            projection.set_projection(best_A)
            valid_scores = [s for s in all_scores if s > float("-inf")]
            logger.info(
                f"LASS: selected projection {best_k}/{K} "
                f"with log_ml = {best_score:.4f} "
                f"(range: [{min(valid_scores):.4f}, {max(valid_scores):.4f}])"
            )
        else:
            logger.warning("LASS: all candidates failed, keeping default projection")


def create_projection(
    config: ProjectionConfig,
    device: str = "cuda",
    seed: int = 42,
) -> QRProjection | PCAProjection | IdentityProjection:
    """Factory for projection strategies."""
    ptype = config.projection_type

    if ptype == "random":
        return QRProjection(config, device=device, seed=seed)
    elif ptype == "pca":
        return PCAProjection(config, device=device)
    elif ptype == "identity":
        return IdentityProjection(config, device=device)
    else:
        raise ValueError(f"Unknown projection type: {ptype}")
