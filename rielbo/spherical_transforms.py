"""Spherical transformations for Riemannian optimization.

- SphericalWhitening: Householder rotation so mean direction -> north pole
- GeodesicTrustRegion: Area-uniform sampling within geodesic disk on sphere
- geodesic_distance, exponential_map, logarithmic_map: Riemannian primitives
"""

import torch
import torch.nn.functional as F


class SphericalWhitening:
    """Rotate sphere so mean direction becomes the north pole [1, 0, ..., 0].

    Uses Householder reflection: H = I - 2vv^T where v = (mu - e1)/||mu - e1||.
    Since H is orthogonal and self-inverse (H^T = H, H*H = I), the same
    matrix-vector product implements both transform and inverse_transform.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.H = None
        self.mu = None

    def fit(self, X: torch.Tensor) -> "SphericalWhitening":
        X_norm = F.normalize(X, p=2, dim=-1)
        self.mu = F.normalize(X_norm.mean(dim=0), p=2, dim=0)

        d = X.shape[-1]
        e1 = torch.zeros(d, device=self.device, dtype=X.dtype)
        e1[0] = 1.0

        diff = self.mu - e1
        diff_norm = diff.norm()

        if diff_norm < 1e-6:
            self.H = torch.eye(d, device=self.device, dtype=X.dtype)
        else:
            v = diff / diff_norm
            self.H = torch.eye(d, device=self.device, dtype=X.dtype) - 2 * torch.outer(v, v)

        return self

    def _apply_householder(self, X: torch.Tensor) -> torch.Tensor:
        """Apply Householder rotation (self-inverse) and re-normalize."""
        if self.H is None:
            raise RuntimeError("Must call fit() before transform()")
        return F.normalize(F.normalize(X, p=2, dim=-1) @ self.H.T, p=2, dim=-1)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return self._apply_householder(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self._apply_householder(X)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)


class GeodesicTrustRegion:
    """Sample candidates within a geodesic disk on the sphere.

    Generates points within geodesic distance rho of a center point using
    the exponential map: x(theta) = cos(theta)*center + sin(theta)*tangent.
    Default cap_sampling="uniform_angle" is center-biased (~12,800x on S^15).
    Use cap_sampling="area_uniform" for sin^(d-2)(theta)-weighted sampling.
    """

    def __init__(
        self,
        max_angle: float = 0.5,
        global_fraction: float = 0.2,
        device: str = "cuda",
        cap_sampling: str = "uniform_angle",
    ):
        """
        Args:
            max_angle: Maximum angle from center in radians (π/2 = hemisphere)
            global_fraction: Fraction of samples for global exploration
            device: Torch device
            cap_sampling: "uniform_angle" (default, center-biased, better for BO)
                or "area_uniform" (unbiased, correct area measure on S^d)
        """
        self.max_angle = max_angle
        self.global_fraction = global_fraction
        self.device = device
        self.cap_sampling = cap_sampling

    def _sample_angles(
        self, n: int, max_angle: float, ambient_dim: int, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample angles within [0, max_angle].

        Two modes:
        - uniform_angle: Uniform in angle θ ∈ [0, max_angle]. Center-biased
          on S^d (area ∝ sin^(d-2)(θ)), but better for BO since it concentrates
          candidates near the trust region center (current best).
        - area_uniform: Uniform in area on S^d via rejection sampling.
          Mathematically "correct" but wastes candidates on cap periphery.
        """
        if self.cap_sampling == "area_uniform":
            exponent = ambient_dim - 2
            if exponent <= 0:
                return torch.rand(n, 1, device=self.device, dtype=dtype) * max_angle
            sin_max = torch.sin(torch.tensor(max_angle, device=self.device, dtype=dtype))
            accepted = []
            while len(accepted) < n:
                batch_size = n * 4
                theta = torch.rand(batch_size, device=self.device, dtype=dtype) * max_angle
                accept_prob = (torch.sin(theta) / sin_max) ** exponent
                mask = torch.rand(batch_size, device=self.device, dtype=dtype) < accept_prob
                accepted.append(theta[mask])
            return torch.cat(accepted)[:n].unsqueeze(1)
        else:
            # Default: uniform in angle (center-biased, better for optimization)
            return torch.rand(n, 1, device=self.device, dtype=dtype) * max_angle

    def sample(
        self,
        center: torch.Tensor,
        n_samples: int,
        adaptive_radius: float | None = None,
    ) -> torch.Tensor:
        center = F.normalize(center.reshape(1, -1), p=2, dim=-1)
        d = center.shape[-1]
        max_angle = adaptive_radius if adaptive_radius is not None else self.max_angle

        n_local = int(n_samples * (1 - self.global_fraction))
        n_global = n_samples - n_local

        candidates = []

        if n_local > 0:
            # Random tangent vectors orthogonal to center
            tangent = torch.randn(n_local, d, device=self.device, dtype=center.dtype)
            tangent = tangent - (tangent * center).sum(dim=-1, keepdim=True) * center
            tangent = F.normalize(tangent, p=2, dim=-1)

            angles = self._sample_angles(n_local, max_angle, d, center.dtype)
            local_samples = torch.cos(angles) * center + torch.sin(angles) * tangent
            candidates.append(F.normalize(local_samples, p=2, dim=-1))

        if n_global > 0:
            global_samples = torch.randn(n_global, d, device=self.device, dtype=center.dtype)
            candidates.append(F.normalize(global_samples, p=2, dim=-1))

        return torch.cat(candidates, dim=0)

    def sample_concentrated(
        self,
        center: torch.Tensor,
        n_samples: int,
        concentration: float = 2.0,
    ) -> torch.Tensor:
        """Sample with von Mises-Fisher-like concentration around center."""
        center = F.normalize(center.reshape(1, -1), p=2, dim=-1)
        d = center.shape[-1]

        u = torch.rand(n_samples, 1, device=self.device, dtype=center.dtype)
        angles = -torch.log(1 - u * (1 - torch.exp(torch.tensor(-concentration * self.max_angle)))) / concentration

        tangent = torch.randn(n_samples, d, device=self.device, dtype=center.dtype)
        tangent = tangent - (tangent * center).sum(dim=-1, keepdim=True) * center
        tangent = F.normalize(tangent, p=2, dim=-1)

        samples = torch.cos(angles) * center + torch.sin(angles) * tangent
        return F.normalize(samples, p=2, dim=-1)


def geodesic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute geodesic distance d_g(x, y) = arccos(x . y) in radians."""
    cos_sim = (F.normalize(x, p=2, dim=-1) * F.normalize(y, p=2, dim=-1)).sum(dim=-1)
    return torch.arccos(cos_sim.clamp(-1 + 1e-6, 1 - 1e-6))


def exponential_map(base: torch.Tensor, tangent: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    """Exponential map: exp_p(t*v) = cos(t*||v||)*p + sin(t*||v||)*v/||v||."""
    base = F.normalize(base, p=2, dim=-1)
    v_norm = tangent.norm(dim=-1, keepdim=True) + 1e-8
    v_unit = tangent / v_norm
    theta = t * v_norm
    return F.normalize(torch.cos(theta) * base + torch.sin(theta) * v_unit, p=2, dim=-1)


def logarithmic_map(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Logarithmic map: inverse of exp_map, returns tangent vector at base."""
    base = F.normalize(base, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)

    cos_theta = (base * target).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.arccos(cos_theta)

    direction = target - cos_theta * base
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

    return theta * direction
