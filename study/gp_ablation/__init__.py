"""GP Ablation Study for High-Dimensional Sample-Efficient Bayesian Optimization.

This module implements a systematic ablation study for Gaussian Process surrogates
in 1024D SONAR embedding space with ~100 training samples.

Key Goal: Find the best GP configuration for sample-efficient BO with n < 100
observations in 1024D space.

Available Methods:
- Tier 1 (Established): standard_msr, turbo, saasbo, baxus, heteroscedastic
- Tier 2 (Advanced): riemannian, gebo, lamcts
- Tier 3 (Hybrid): turbo_grad, flow_bo, turbo_geodesic, baxus_flow
- Tier 4 (Novel): latent_bo, bayesian_flow_bo, velocity_acq

References:
- MSR: Hvarfner et al. (2024) "Vanilla Bayesian Optimization in High Dimensions"
- TuRBO: Eriksson et al. (2019) "Scalable Global Optimization via Local BO"
- SAASBO: Eriksson & Jankowiak (2021) "High-Dimensional BO with Sparse Priors"
- BAxUS: Papenmeier et al. (2022) "Adaptive BO in Nested Subspaces"
"""

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates import create_surrogate

__all__ = ["GPConfig", "create_surrogate"]
