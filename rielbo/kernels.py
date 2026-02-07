"""Advanced kernels for spherical Gaussian Processes.

Implements:
- ArcCosine Order 0: k(x,y) = 1 - arccos(x·y)/pi (rough, no lengthscale)
- ArcCosine Order 2: Smoother GP (like Matern-5/2, no lengthscale)
- GeodesicMatern: Matern kernel using geodesic distance with learnable lengthscale
- Product Sphere: S^3 x S^3 x S^3 x S^3 instead of S^15

References:
- Cho & Saul (2009): Kernel Methods for Deep Learning
- Davidson et al. (2019): Hyperspherical Variational Auto-Encoders
- Borovitskiy et al. (2020): Matern GPs on Riemannian Manifolds
"""

import math

import gpytorch
import torch
import torch.nn.functional as F


def _cosine_similarity(x1, x2, diag: bool) -> torch.Tensor:
    """Compute clamped cosine similarity between normalized inputs."""
    x1_norm = F.normalize(x1, p=2, dim=-1)
    x2_norm = F.normalize(x2, p=2, dim=-1)

    if diag:
        cos_sim = (x1_norm * x2_norm).sum(dim=-1)
    else:
        cos_sim = x1_norm @ x2_norm.transpose(-2, -1)

    return cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)


class ArcCosineKernel(gpytorch.kernels.Kernel):
    """ArcCosine kernel order 0 for unit sphere data.

    k(x, y) = 1 - arccos(x·y) / pi

    Equivalent to Matern-1/2 (very rough, like Brownian motion paths).
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        cos_sim = _cosine_similarity(x1, x2, diag)
        return 1.0 - torch.arccos(cos_sim) / math.pi


class ArcCosineKernelOrder2(gpytorch.kernels.Kernel):
    """ArcCosine kernel order 2 for smoother GP regression.

    For normalized inputs x, y on the unit sphere:
        theta = arccos(x·y)
        J2(theta) = 3*sin(theta)*cos(theta) + (pi-theta)*(1 + 2*cos^2(theta))
        k(x,y) = J2(theta) / (3*pi)

    Reference: Cho & Saul (2009), Kernel Methods for Deep Learning
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        cos_sim = _cosine_similarity(x1, x2, diag)
        theta = torch.arccos(cos_sim)
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        j2 = 3 * sin_t * cos_t + (math.pi - theta) * (1 + 2 * cos_t**2)
        return j2 / (3 * math.pi)


class GeodesicMaternKernel(gpytorch.kernels.Kernel):
    """Matern kernel using geodesic distance on the unit sphere.

    Isotropic mode (default):
        k(x,y) = Matern_nu(arccos(x·y) / l)

    ARD mode (ard_num_dims > 0):
        x_s = normalize(x / l), y_s = normalize(y / l)
        k(x,y) = Matern_nu(arccos(x_s · y_s))
        Per-dimension lengthscales deform S^d into an ellipsoid.

    Supported nu: 0.5 (rough, always PD), 1.5, 2.5 (smooth).

    WARNING: For nu>0.5, plugging geodesic distance into the Euclidean Matern
    formula does NOT guarantee PD (Feragen et al. 2015). Negative eigenvalues
    observed for nu=1.5 at lengthscale>=2.8, nu=2.5 at lengthscale>=2.2 on S^15.
    For proper Matern on spheres, use GeometricKernels (Borovitskiy et al. 2020).
    """

    has_lengthscale = True

    def __init__(self, nu: float = 1.5, ard_num_dims: int | None = None, **kwargs):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"nu must be 0.5, 1.5, or 2.5, got {nu}")
        self.nu = nu
        self._ard = ard_num_dims is not None

    def _matern(self, dist: torch.Tensor) -> torch.Tensor:
        if self.nu == 0.5:
            return torch.exp(-dist)
        elif self.nu == 1.5:
            scaled = math.sqrt(3) * dist
            return (1.0 + scaled) * torch.exp(-scaled)
        else:  # 2.5
            scaled = math.sqrt(5) * dist
            return (1.0 + scaled + scaled.pow(2) / 3.0) * torch.exp(-scaled)

    def forward(self, x1, x2, diag=False, **params):
        if self._ard:
            # ARD: scale per-dim, re-normalize, then geodesic distance
            x1_scaled = F.normalize(F.normalize(x1, p=2, dim=-1) / self.lengthscale, p=2, dim=-1)
            x2_scaled = F.normalize(F.normalize(x2, p=2, dim=-1) / self.lengthscale, p=2, dim=-1)
            cos_sim = _cosine_similarity(x1_scaled, x2_scaled, diag)
            dist = torch.arccos(cos_sim)
        else:
            # Isotropic: geodesic distance scaled by single lengthscale
            cos_sim = _cosine_similarity(x1, x2, diag)
            dist = torch.arccos(cos_sim) / self.lengthscale

        return self._matern(dist)


class ProductSphereKernel(gpytorch.kernels.Kernel):
    """Product of ArcCosine kernels on smaller spheres.

    Instead of one S^(d-1), uses S^(k-1) x ... (n_spheres times)
    where k = d / n_spheres. Example: d=16, n_spheres=4 -> S^3 x S^3 x S^3 x S^3.

    Reference: Davidson et al. (2019), Hyperspherical VAE
    """

    has_lengthscale = False

    def __init__(self, n_spheres: int = 4, order: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.n_spheres = n_spheres
        if order not in (0, 2):
            raise ValueError(f"Order must be 0 or 2, got {order}")
        self.order = order

    def _arc_kernel(self, cos_sim: torch.Tensor) -> torch.Tensor:
        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        if self.order == 0:
            return 1.0 - torch.arccos(cos_sim) / math.pi
        # order == 2
        theta = torch.arccos(cos_sim)
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        j2 = 3 * sin_t * cos_t + (math.pi - theta) * (1 + 2 * cos_t**2)
        return j2 / (3 * math.pi)

    def forward(self, x1, x2, diag=False, **params):
        dim = x1.shape[-1]
        if dim % self.n_spheres != 0:
            raise ValueError(
                f"Input dim {dim} must be divisible by n_spheres {self.n_spheres}"
            )

        dim_per_sphere = dim // self.n_spheres

        x1_split = F.normalize(
            x1.reshape(*x1.shape[:-1], self.n_spheres, dim_per_sphere), p=2, dim=-1
        )
        x2_split = F.normalize(
            x2.reshape(*x2.shape[:-1], self.n_spheres, dim_per_sphere), p=2, dim=-1
        )

        if diag:
            cos_sims = (x1_split * x2_split).sum(dim=-1)
            return self._arc_kernel(cos_sims).prod(dim=-1)

        # Full covariance: product over spheres
        n1 = x1_split.shape[0]
        n2 = x2_split.shape[0]
        result = torch.ones(n1, n2, device=x1.device, dtype=x1.dtype)
        for i in range(self.n_spheres):
            cos_sim = x1_split[:, i, :] @ x2_split[:, i, :].T
            result = result * self._arc_kernel(cos_sim)
        return result


_ARCCOSINE_KERNELS = {0: ArcCosineKernel, 2: ArcCosineKernelOrder2}
_NU_MAP = {0: 0.5, 1: 1.5, 2: 2.5}


def create_kernel(
    kernel_type: str = "arccosine",
    kernel_order: int = 0,
    n_spheres: int = 4,
    use_scale: bool = True,
    ard_num_dims: int | None = None,
) -> gpytorch.kernels.Kernel:
    """Factory function for creating spherical kernels."""
    if kernel_type == "arccosine":
        if kernel_order not in _ARCCOSINE_KERNELS:
            raise ValueError(f"kernel_order must be 0 or 2, got {kernel_order}")
        base = _ARCCOSINE_KERNELS[kernel_order]()
    elif kernel_type == "geodesic_matern":
        nu = _NU_MAP.get(kernel_order, 1.5)
        base = GeodesicMaternKernel(nu=nu, ard_num_dims=ard_num_dims)
    elif kernel_type == "matern":
        base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
    elif kernel_type == "product":
        base = ProductSphereKernel(n_spheres=n_spheres, order=kernel_order)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    if use_scale:
        return gpytorch.kernels.ScaleKernel(base)
    return base
