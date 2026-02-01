"""GP surrogate implementations for the ablation study.

Factory function for creating surrogates based on method name.
"""

from typing import TYPE_CHECKING

import torch

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate

if TYPE_CHECKING:
    from study.gp_ablation.surrogates.standard_gp import StandardMSRGP
    from study.gp_ablation.surrogates.turbo_gp import TuRBOGP
    from study.gp_ablation.surrogates.saas_gp import SAASGP
    from study.gp_ablation.surrogates.sparse_gp import SparseGP
    from study.gp_ablation.surrogates.projection_gp import BAxUSGP


# Registry of available GP methods
GP_METHOD_REGISTRY = {
    # Tier 1: Core Methods
    "standard_msr": "StandardMSRGP",
    "turbo": "TuRBOGP",
    "saasbo": "SAASGP",
    "baxus": "BAxUSGP",
    "heteroscedastic": "HeteroscedasticGP",
    # Tier 2: Advanced Methods
    "riemannian": "RiemannianGP",
    "gebo": "GradientEnhancedGP",
    "lamcts": "LaMCTSOptimizer",
    # Tier 3: Hybrid Methods
    "turbo_grad": "TuRBOGradientGP",
    "flow_bo": "FlowGuidedGP",
    "turbo_geodesic": "GeodesicTuRBOGP",
    "baxus_flow": "BAxUSFlowGP",
    # Tier 4: Novel Methods
    "latent_bo": "LatentSpaceBO",
    "bayesian_flow_bo": "BayesianFlowBO",
    "velocity_acq": "VelocityGuidedGP",
    "curriculum_bo": "CurriculumBO",
}


def create_surrogate(
    config: GPConfig,
    device: torch.device | str = "cuda",
) -> BaseGPSurrogate:
    """Factory function to create a GP surrogate.

    Args:
        config: GPConfig with method name and hyperparameters.
        device: Device for computation.

    Returns:
        Initialized GP surrogate (not yet fitted).

    Raises:
        ValueError: If method is unknown.
    """
    method = config.method.lower()

    # Tier 1: Core Methods
    if method in ("standard_msr", "msr", "standard"):
        from study.gp_ablation.surrogates.standard_gp import StandardMSRGP
        return StandardMSRGP(config, device)

    elif method == "turbo":
        from study.gp_ablation.surrogates.turbo_gp import TuRBOGP
        return TuRBOGP(config, device)

    elif method == "saasbo":
        from study.gp_ablation.surrogates.saas_gp import SAASGP
        return SAASGP(config, device)

    elif method == "baxus":
        from study.gp_ablation.surrogates.projection_gp import BAxUSGP
        return BAxUSGP(config, device)

    elif method == "heteroscedastic":
        from study.gp_ablation.surrogates.standard_gp import HeteroscedasticGP
        return HeteroscedasticGP(config, device)

    # Tier 2: Advanced Methods
    elif method == "riemannian":
        from study.gp_ablation.surrogates.riemannian_gp import RiemannianGP
        return RiemannianGP(config, device)

    elif method == "gebo":
        from study.gp_ablation.surrogates.gebo_gp import GradientEnhancedGP
        return GradientEnhancedGP(config, device)

    elif method == "lamcts":
        from study.gp_ablation.surrogates.lamcts import LaMCTSOptimizer
        return LaMCTSOptimizer(config, device)

    # Tier 3: Hybrid Methods
    elif method == "turbo_grad":
        from study.gp_ablation.surrogates.turbo_gp import TuRBOGradientGP
        return TuRBOGradientGP(config, device)

    elif method == "flow_bo":
        from study.gp_ablation.surrogates.flow_gp import FlowGuidedGP
        return FlowGuidedGP(config, device)

    elif method == "turbo_geodesic":
        from study.gp_ablation.surrogates.riemannian_gp import GeodesicTuRBOGP
        return GeodesicTuRBOGP(config, device)

    elif method == "baxus_flow":
        from study.gp_ablation.surrogates.flow_gp import BAxUSFlowGP
        return BAxUSFlowGP(config, device)

    # Tier 4: Novel Methods
    elif method == "latent_bo":
        from study.gp_ablation.surrogates.latent_bo import LatentSpaceBO
        return LatentSpaceBO(config, device)

    elif method == "bayesian_flow_bo":
        from study.gp_ablation.surrogates.flow_gp import BayesianFlowBO
        return BayesianFlowBO(config, device)

    elif method == "velocity_acq":
        from study.gp_ablation.surrogates.flow_gp import VelocityGuidedGP
        return VelocityGuidedGP(config, device)

    elif method == "curriculum_bo":
        from study.gp_ablation.surrogates.curriculum_bo import CurriculumBO
        return CurriculumBO(config, device)

    else:
        available = ", ".join(sorted(GP_METHOD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown GP method: {method}. Available methods: {available}"
        )


def list_methods() -> list[str]:
    """List all available GP methods."""
    return sorted(GP_METHOD_REGISTRY.keys())


__all__ = [
    "BaseGPSurrogate",
    "create_surrogate",
    "list_methods",
    "GP_METHOD_REGISTRY",
]
