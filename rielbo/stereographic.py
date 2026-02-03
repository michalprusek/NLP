"""Stereographic Projection for RieLBO.

Deterministic bijection R^D ↔ S^D that encodes magnitude in geometry.

Mathematical Foundation:
- Lift (R^D → S^D): For x ∈ R^D, with R = mean(||x||):
    x̂ = x/R
    u_i = 2x̂_i / (1 + ||x̂||²)     for i=1..D
    u_{D+1} = (||x̂||² - 1) / (||x̂||² + 1)

- Project (S^D → R^D): For u ∈ S^D:
    x̂ = u_{1:D} / (1 - u_{D+1})
    x = x̂ * R

Advantages over NormPredictor:
- No learned parameters (deterministic)
- Exact magnitude recovery (bijection)
- Unified Riemannian geometry on S^D

Usage:
    x = torch.randn(100, 256)  # Embeddings
    stereo = StereographicTransform.from_embeddings(x)
    u = stereo.lift(x)          # [100, 257] on unit sphere
    x_rec = stereo.project(u)   # [100, 256] exact recovery
    assert torch.allclose(x, x_rec, atol=1e-5)
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StereographicTransform:
    """Stereographic projection bijection between R^D and S^D.

    Encodes D-dimensional Euclidean vectors as points on the D+1 dimensional
    unit sphere, preserving magnitude information in the geometry.

    The north pole (0, 0, ..., 1) is the projection point. Points with
    larger magnitude map closer to the north pole; smaller magnitude
    maps closer to the equator.
    """

    def __init__(self, input_dim: int, radius_scaling: float):
        """Initialize stereographic transform.

        Args:
            input_dim: Dimension D of input vectors
            radius_scaling: R = mean(||x||) for scaling before projection
        """
        self.input_dim = input_dim        # D (e.g., 256)
        self.output_dim = input_dim + 1   # D+1 (e.g., 257)
        self.radius_scaling = radius_scaling

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        """Lift R^D to S^D using stereographic projection.

        Maps Euclidean vectors to the unit sphere in D+1 dimensions.
        The magnitude information is encoded in the last coordinate.

        Args:
            x: Embeddings [B, D]

        Returns:
            Unit vectors [B, D+1] on the sphere
        """
        # Scale by mean norm
        x_hat = x / self.radius_scaling

        # Compute ||x̂||²
        norm_sq = (x_hat ** 2).sum(dim=-1, keepdim=True)

        # Stereographic formulas:
        # u_{1:D} = 2x̂ / (1 + ||x̂||²)
        # u_{D+1} = (||x̂||² - 1) / (||x̂||² + 1)
        u_coords = 2 * x_hat / (1 + norm_sq)
        u_last = (norm_sq - 1) / (norm_sq + 1)

        # Concatenate to form (D+1)-dimensional vector
        u = torch.cat([u_coords, u_last], dim=-1)

        # Numerical safety: re-normalize to unit sphere
        # (should already be unit, but floating point precision)
        u = F.normalize(u, p=2, dim=-1)

        return u

    def project(self, u: torch.Tensor) -> torch.Tensor:
        """Project S^D to R^D using inverse stereographic projection.

        Maps unit sphere points back to Euclidean vectors with
        exact magnitude recovery.

        Args:
            u: Unit vectors [B, D+1] on the sphere

        Returns:
            Embeddings [B, D] with correct magnitude
        """
        # Split into D coordinates and last coordinate
        u_coords = u[..., :-1]  # [B, D]
        u_last = u[..., -1:]    # [B, 1]

        # Inverse stereographic:
        # x̂ = u_{1:D} / (1 - u_{D+1})
        # Avoid division by zero at north pole
        denom = (1 - u_last).clamp(min=1e-6)
        x_hat = u_coords / denom

        # Scale back by mean norm
        x = x_hat * self.radius_scaling

        return x

    @classmethod
    def from_embeddings(cls, embeddings: torch.Tensor) -> "StereographicTransform":
        """Create transform from a sample of embeddings.

        Computes the mean norm R from the embeddings to use as scaling.

        Args:
            embeddings: Sample embeddings [N, D]

        Returns:
            StereographicTransform configured for this data
        """
        mean_norm = embeddings.norm(dim=-1).mean().item()
        input_dim = embeddings.shape[-1]
        logger.info(f"StereographicTransform: D={input_dim}, R={mean_norm:.4f}")
        return cls(input_dim, mean_norm)

    def save(self, path: str) -> None:
        """Save transform parameters to file."""
        torch.save({
            "input_dim": self.input_dim,
            "radius_scaling": self.radius_scaling,
        }, path)
        logger.info(f"Saved StereographicTransform to {path}")

    @classmethod
    def load(cls, path: str) -> "StereographicTransform":
        """Load transform from file."""
        data = torch.load(path, map_location="cpu")
        transform = cls(data["input_dim"], data["radius_scaling"])
        logger.info(
            f"Loaded StereographicTransform: D={transform.input_dim}, "
            f"R={transform.radius_scaling:.4f}"
        )
        return transform

    def __repr__(self) -> str:
        return (
            f"StereographicTransform(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, radius_scaling={self.radius_scaling:.4f})"
        )


def test_roundtrip(n_samples: int = 100, dim: int = 256) -> None:
    """Test that lift and project are inverses."""
    # Random embeddings with varied magnitudes
    x = torch.randn(n_samples, dim) * torch.rand(n_samples, 1) * 10

    # Create transform
    stereo = StereographicTransform.from_embeddings(x)

    # Lift to sphere
    u = stereo.lift(x)

    # Verify on unit sphere
    norms = u.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(n_samples), atol=1e-5), \
        f"Not on unit sphere: norms range [{norms.min():.6f}, {norms.max():.6f}]"

    # Project back
    x_rec = stereo.project(u)

    # Verify roundtrip
    max_error = (x - x_rec).abs().max().item()
    assert torch.allclose(x, x_rec, atol=1e-4), \
        f"Roundtrip failed: max error = {max_error:.6f}"

    logger.info(f"✓ Roundtrip test passed (n={n_samples}, dim={dim})")
    logger.info(f"  Max reconstruction error: {max_error:.2e}")
    logger.info(f"  u norms: [{norms.min():.6f}, {norms.max():.6f}]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing StereographicTransform...")

    # Test basic roundtrip
    test_roundtrip(n_samples=100, dim=256)

    # Test with different dimensions
    test_roundtrip(n_samples=50, dim=64)
    test_roundtrip(n_samples=50, dim=1024)

    # Test edge cases
    print("\nTesting edge cases...")

    # Near-zero vectors
    x_small = torch.randn(10, 256) * 0.01
    stereo = StereographicTransform.from_embeddings(x_small)
    u = stereo.lift(x_small)
    x_rec = stereo.project(u)
    print(f"  Small vectors: max error = {(x_small - x_rec).abs().max():.2e}")

    # Large vectors
    x_large = torch.randn(10, 256) * 100
    stereo = StereographicTransform.from_embeddings(x_large)
    u = stereo.lift(x_large)
    x_rec = stereo.project(u)
    print(f"  Large vectors: max error = {(x_large - x_rec).abs().max():.2e}")

    print("\n✓ All tests passed!")
