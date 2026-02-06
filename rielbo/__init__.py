"""RieLBO: Riemannian Latent Bayesian Optimization via subspace projection."""

from rielbo.subspace_bo import SphericalSubspaceBO
from rielbo.kernels import ArcCosineKernel, create_kernel
from rielbo.gp_diagnostics import GPDiagnostics

__all__ = [
    "SphericalSubspaceBO",
    "ArcCosineKernel",
    "create_kernel",
    "GPDiagnostics",
]
