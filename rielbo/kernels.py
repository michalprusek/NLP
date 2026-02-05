"""Advanced kernels for spherical Gaussian Processes.

Implements:
- ArcCosine Order 0: k(x,y) = 1 - arccos(x·y)/π (basic, rough)
- ArcCosine Order 2: Smoother GP (like Matérn-5/2 vs Matérn-1/2)
- Product Sphere: S^3 x S^3 x S^3 x S^3 instead of S^15

References:
- Cho & Saul (2009): Kernel Methods for Deep Learning
- Davidson et al. (2019): Hyperspherical Variational Auto-Encoders
"""

import math

import gpytorch
import torch
import torch.nn.functional as F


class ArcCosineKernel(gpytorch.kernels.Kernel):
    """ArcCosine kernel order 0 for unit sphere data.

    k(x, y) = 1 - arccos(x·y) / π

    This is the basic geodesic distance kernel, equivalent to Matérn-1/2
    (very rough, like Brownian motion paths).
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        x1_norm = F.normalize(x1, p=2, dim=-1)
        x2_norm = F.normalize(x2, p=2, dim=-1)

        if diag:
            cos_sim = (x1_norm * x2_norm).sum(dim=-1)
        else:
            cos_sim = x1_norm @ x2_norm.transpose(-2, -1)

        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        return 1.0 - torch.arccos(cos_sim) / math.pi


class ArcCosineKernelOrder2(gpytorch.kernels.Kernel):
    """ArcCosine kernel order 2 for smoother GP regression.

    For normalized inputs x, y on the unit sphere:
        θ = arccos(x·y)
        J₂(θ) = 3·sin(θ)·cos(θ) + (π-θ)·(1 + 2·cos²(θ))
        k(x,y) = J₂(θ) / J₂(0)

    J₂(0) = 3π, so the kernel is normalized to k(x,x) = 1.

    Order 2 produces smoother GP posteriors, similar to Matérn-5/2.
    This is better for chemical properties which are typically smooth.

    Reference: Cho & Saul (2009), Kernel Methods for Deep Learning
    """

    has_lengthscale = False

    def _J2(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute J₂(θ) = 3·sin(θ)·cos(θ) + (π-θ)·(1 + 2·cos²(θ))"""
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        return 3 * sin_t * cos_t + (math.pi - theta) * (1 + 2 * cos_t**2)

    def forward(self, x1, x2, diag=False, **params):
        x1_norm = F.normalize(x1, p=2, dim=-1)
        x2_norm = F.normalize(x2, p=2, dim=-1)

        if diag:
            cos_sim = (x1_norm * x2_norm).sum(dim=-1)
        else:
            cos_sim = x1_norm @ x2_norm.transpose(-2, -1)

        # Clamp for numerical stability
        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.arccos(cos_sim)

        # Compute J2 and normalize by J2(0) = 3π
        j2_theta = self._J2(theta)
        j2_0 = 3 * math.pi

        return j2_theta / j2_0


class ProductSphereKernel(gpytorch.kernels.Kernel):
    """Product of ArcCosine kernels on smaller spheres.

    Instead of one S^(d-1), uses S^(k-1) × S^(k-1) × ... (n_spheres times)
    where k = d / n_spheres.

    Example: d=16 with n_spheres=4 → S^3 × S^3 × S^3 × S^3

    The kernel is the product of individual ArcCosine kernels:
        k(x,y) = ∏_{i=1}^{n_spheres} k_arc(x_i, y_i)

    This avoids the "curse of dimensionality" on high-dimensional spheres
    by decomposing into lower-dimensional subspaces.

    Reference: Davidson et al. (2019), Hyperspherical VAE
    """

    has_lengthscale = False

    def __init__(self, n_spheres: int = 4, order: int = 0, **kwargs):
        """
        Args:
            n_spheres: Number of spheres in the product
            order: ArcCosine kernel order (0 or 2)
        """
        super().__init__(**kwargs)
        self.n_spheres = n_spheres
        self.order = order

    def _arc_kernel(self, cos_sim: torch.Tensor) -> torch.Tensor:
        """Compute ArcCosine kernel value from cosine similarity."""
        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)

        if self.order == 0:
            return 1.0 - torch.arccos(cos_sim) / math.pi
        elif self.order == 2:
            theta = torch.arccos(cos_sim)
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            j2 = 3 * sin_t * cos_t + (math.pi - theta) * (1 + 2 * cos_t**2)
            return j2 / (3 * math.pi)
        else:
            raise ValueError(f"Order must be 0 or 2, got {self.order}")

    def forward(self, x1, x2, diag=False, **params):
        d = x1.shape[-1]
        if d % self.n_spheres != 0:
            raise ValueError(
                f"Input dim {d} must be divisible by n_spheres {self.n_spheres}"
            )

        k = d // self.n_spheres  # Dimension per sphere

        # Split and normalize each component
        x1_split = x1.reshape(*x1.shape[:-1], self.n_spheres, k)
        x2_split = x2.reshape(*x2.shape[:-1], self.n_spheres, k)

        x1_norm = F.normalize(x1_split, p=2, dim=-1)
        x2_norm = F.normalize(x2_split, p=2, dim=-1)

        if diag:
            # Diagonal case: [batch, n_spheres]
            cos_sims = (x1_norm * x2_norm).sum(dim=-1)
            k_vals = self._arc_kernel(cos_sims)
            return k_vals.prod(dim=-1)
        else:
            # Full covariance: compute product over spheres
            # x1: [n1, n_spheres, k], x2: [n2, n_spheres, k]
            n1, n_spheres, k = x1_norm.shape
            n2 = x2_norm.shape[0]

            # Initialize product kernel
            result = torch.ones(n1, n2, device=x1.device, dtype=x1.dtype)

            for i in range(self.n_spheres):
                # Cosine similarity for sphere i
                cos_sim = x1_norm[:, i, :] @ x2_norm[:, i, :].T  # [n1, n2]
                k_i = self._arc_kernel(cos_sim)
                result = result * k_i

            return result


def create_kernel(
    kernel_type: str = "arccosine",
    kernel_order: int = 0,
    n_spheres: int = 4,
    use_scale: bool = True,
) -> gpytorch.kernels.Kernel:
    """Factory function for creating spherical kernels.

    Args:
        kernel_type: "arccosine", "matern", or "product"
        kernel_order: 0 or 2 for ArcCosine kernels
        n_spheres: Number of spheres for ProductSphereKernel
        use_scale: Whether to wrap in ScaleKernel

    Returns:
        GPyTorch kernel instance
    """
    if kernel_type == "arccosine":
        if kernel_order == 0:
            base = ArcCosineKernel()
        elif kernel_order == 2:
            base = ArcCosineKernelOrder2()
        else:
            raise ValueError(f"kernel_order must be 0 or 2, got {kernel_order}")
    elif kernel_type == "matern":
        base = gpytorch.kernels.MaternKernel(nu=2.5)
    elif kernel_type == "product":
        base = ProductSphereKernel(n_spheres=n_spheres, order=kernel_order)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    if use_scale:
        return gpytorch.kernels.ScaleKernel(base)
    return base
