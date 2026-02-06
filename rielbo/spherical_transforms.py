"""Spherical transformations for Riemannian optimization.

Implements:
- SphericalWhitening: Rotate sphere so mean direction → north pole
- GeodesicTrustRegion: Sample within geodesic disk on sphere

These transforms properly respect the Riemannian geometry of the sphere,
unlike naive Euclidean operations.
"""

import torch
import torch.nn.functional as F


class SphericalWhitening:
    """Rotate sphere so mean direction becomes the north pole [1, 0, ..., 0].

    This centers the data on the sphere, improving projection efficiency
    when using random subspace projections.

    Uses Householder reflection: H = I - 2vv^T where v = (μ - e₁)/||μ - e₁||

    The Householder matrix H is orthogonal (H^T = H, H·H = I) and maps
    μ → e₁ = [1, 0, ..., 0] (the north pole).

    Why this matters:
    - VAE/SONAR embeddings typically cluster in one "cone" of the sphere
    - Random projections around the origin miss most of the data
    - Centering at the north pole maximizes information from projections
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.H = None  # Householder matrix
        self.mu = None  # Mean direction

    def fit(self, X: torch.Tensor) -> "SphericalWhitening":
        """Fit the whitening transform from data.

        Args:
            X: Tensor of shape [N, D], will be normalized to unit sphere

        Returns:
            self for chaining
        """
        # Normalize inputs to unit sphere
        X_norm = F.normalize(X, p=2, dim=-1)

        # Compute mean direction (Fréchet mean on sphere ≈ normalized arithmetic mean)
        self.mu = F.normalize(X_norm.mean(dim=0), p=2, dim=0)

        # Target: north pole e1 = [1, 0, ..., 0]
        d = X.shape[-1]
        e1 = torch.zeros(d, device=self.device, dtype=X.dtype)
        e1[0] = 1.0

        # Householder vector: v = (μ - e₁) / ||μ - e₁||
        # Special case: if μ ≈ e₁, use identity
        diff = self.mu - e1
        diff_norm = diff.norm()

        if diff_norm < 1e-6:
            # Already at north pole, use identity
            self.H = torch.eye(d, device=self.device, dtype=X.dtype)
        else:
            v = diff / diff_norm
            # Householder matrix: H = I - 2vv^T
            self.H = torch.eye(d, device=self.device, dtype=X.dtype) - 2 * torch.outer(v, v)

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply whitening transform.

        Args:
            X: Tensor of shape [..., D]

        Returns:
            Rotated tensor of same shape, normalized to unit sphere
        """
        if self.H is None:
            raise RuntimeError("Must call fit() before transform()")

        # Normalize input
        X_norm = F.normalize(X, p=2, dim=-1)

        # Apply Householder rotation
        X_rot = X_norm @ self.H.T

        # Re-normalize (should be unit already, but for numerical safety)
        return F.normalize(X_rot, p=2, dim=-1)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Inverse whitening transform.

        Args:
            X: Tensor of shape [..., D] (whitened)

        Returns:
            Original-space tensor of same shape
        """
        if self.H is None:
            raise RuntimeError("Must call fit() before inverse_transform()")

        # Normalize input
        X_norm = F.normalize(X, p=2, dim=-1)

        # Householder is self-inverse: H^{-1} = H
        X_orig = X_norm @ self.H.T

        return F.normalize(X_orig, p=2, dim=-1)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class GeodesicTrustRegion:
    """Sample candidates within a geodesic disk on the sphere.

    Instead of Euclidean box bounds (which make no sense on a sphere),
    this samples within a geodesic disk of radius ρ around a center point.

    Method:
    1. Generate random tangent vectors (orthogonal to center)
    2. Sample angles uniformly in [0, max_angle]
    3. Use exponential map: x(θ) = cos(θ)·center + sin(θ)·tangent

    The geodesic disk {x ∈ S^d : d_g(x, center) ≤ ρ} is the set of all points
    within geodesic distance ρ from the center.
    """

    def __init__(
        self,
        max_angle: float = 0.5,  # Maximum geodesic angle in radians
        global_fraction: float = 0.2,  # Fraction of global exploration samples
        device: str = "cuda",
    ):
        """
        Args:
            max_angle: Maximum angle from center in radians (π/2 = hemisphere)
            global_fraction: Fraction of samples for global exploration
            device: Torch device
        """
        self.max_angle = max_angle
        self.global_fraction = global_fraction
        self.device = device

    def sample(
        self,
        center: torch.Tensor,
        n_samples: int,
        adaptive_radius: float | None = None,
    ) -> torch.Tensor:
        """Sample candidates within geodesic trust region.

        Args:
            center: Center point [1, D] or [D], must be on unit sphere
            n_samples: Number of candidates to generate
            adaptive_radius: Override max_angle if provided

        Returns:
            Candidates tensor [n_samples, D] on unit sphere
        """
        center = F.normalize(center.reshape(1, -1), p=2, dim=-1)
        d = center.shape[-1]

        max_angle = adaptive_radius if adaptive_radius is not None else self.max_angle

        # Split between local and global
        n_local = int(n_samples * (1 - self.global_fraction))
        n_global = n_samples - n_local

        candidates = []

        if n_local > 0:
            # Generate random tangent vectors (orthogonal to center)
            tangent = torch.randn(n_local, d, device=self.device, dtype=center.dtype)

            # Project out the component along center
            proj = (tangent * center).sum(dim=-1, keepdim=True) * center
            tangent = tangent - proj

            # Normalize tangent vectors
            tangent = F.normalize(tangent, p=2, dim=-1)

            # Sample angles uniformly in [0, max_angle]
            angles = torch.rand(n_local, 1, device=self.device, dtype=center.dtype) * max_angle

            # Exponential map: x(θ) = cos(θ)·center + sin(θ)·tangent
            local_samples = torch.cos(angles) * center + torch.sin(angles) * tangent
            local_samples = F.normalize(local_samples, p=2, dim=-1)
            candidates.append(local_samples)

        if n_global > 0:
            # Global exploration: random points on sphere
            global_samples = torch.randn(n_global, d, device=self.device, dtype=center.dtype)
            global_samples = F.normalize(global_samples, p=2, dim=-1)
            candidates.append(global_samples)

        return torch.cat(candidates, dim=0)

    def sample_concentrated(
        self,
        center: torch.Tensor,
        n_samples: int,
        concentration: float = 2.0,
    ) -> torch.Tensor:
        """Sample with von Mises-Fisher-like concentration around center.

        Uses rejection sampling to concentrate samples near center,
        with higher concentration meaning tighter clustering.

        Args:
            center: Center point [1, D] or [D]
            n_samples: Number of candidates
            concentration: Higher = tighter around center

        Returns:
            Candidates tensor [n_samples, D]
        """
        center = F.normalize(center.reshape(1, -1), p=2, dim=-1)
        d = center.shape[-1]

        # Use exponential-weighted angles
        u = torch.rand(n_samples, 1, device=self.device, dtype=center.dtype)
        angles = -torch.log(1 - u * (1 - torch.exp(torch.tensor(-concentration * self.max_angle)))) / concentration

        # Generate random tangent vectors
        tangent = torch.randn(n_samples, d, device=self.device, dtype=center.dtype)
        proj = (tangent * center).sum(dim=-1, keepdim=True) * center
        tangent = F.normalize(tangent - proj, p=2, dim=-1)

        # Exponential map
        samples = torch.cos(angles) * center + torch.sin(angles) * tangent
        return F.normalize(samples, p=2, dim=-1)


def geodesic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute geodesic distance between points on the sphere.

    d_g(x, y) = arccos(x · y)

    Args:
        x: Tensor [..., D]
        y: Tensor [..., D]

    Returns:
        Geodesic distances in radians
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    cos_sim = (x_norm * y_norm).sum(dim=-1)
    cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)

    return torch.arccos(cos_sim)


def exponential_map(base: torch.Tensor, tangent: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    """Exponential map on the sphere.

    Maps a tangent vector v at base point p to a point on the sphere:
        exp_p(t·v) = cos(t·||v||)·p + sin(t·||v||)·v/||v||

    Args:
        base: Base point on sphere [..., D]
        tangent: Tangent vector at base [..., D] (should be orthogonal to base)
        t: Step size

    Returns:
        Point on sphere [..., D]
    """
    base = F.normalize(base, p=2, dim=-1)
    v_norm = tangent.norm(dim=-1, keepdim=True) + 1e-8
    v_unit = tangent / v_norm

    theta = t * v_norm
    result = torch.cos(theta) * base + torch.sin(theta) * v_unit
    return F.normalize(result, p=2, dim=-1)


def logarithmic_map(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Logarithmic map on the sphere.

    Inverse of exponential map: returns tangent vector at base pointing to target.

    Args:
        base: Base point on sphere [..., D]
        target: Target point on sphere [..., D]

    Returns:
        Tangent vector at base [..., D]
    """
    base = F.normalize(base, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)

    # Project target onto tangent space at base
    cos_theta = (base * target).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.arccos(cos_theta)

    # Direction in tangent space
    proj = cos_theta * base
    direction = target - proj
    direction_norm = direction.norm(dim=-1, keepdim=True) + 1e-8
    direction = direction / direction_norm

    return theta * direction
