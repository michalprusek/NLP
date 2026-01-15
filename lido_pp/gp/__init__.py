"""
High-Dimensional Gaussian Process for FlowPO.

Optimized for 256D latent space with limited training data (~20 points).
Addresses curse of dimensionality through:
- Isotropic kernel (single lengthscale)
- SAAS prior (Sparse Axis-Aligned Subspaces)
- Adaptive switching based on data size
"""

from lido_pp.gp.high_dim_gp import (
    IsotropicHighDimGP,
    SaasHighDimGP,
    AdaptiveHighDimGP,
    TrustRegion,
    compute_ucb_beta,
)

__all__ = [
    "IsotropicHighDimGP",
    "SaasHighDimGP",
    "AdaptiveHighDimGP",
    "TrustRegion",
    "compute_ucb_beta",
]
