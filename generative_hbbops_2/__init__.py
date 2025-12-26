"""
HyLO2: Latent Space Optimization for Prompt Engineering.

This module extends generative_hbbops with a LatentProjector that enables
optimization directly in the 10D latent space instead of 768D embedding space.

Key innovations:
1. LatentProjector: 10D -> 768D projection trained jointly with GP
2. LatentSpaceOptimizer: Gradient-based optimization in 10D latent space
3. Joint training: GP MLL loss + instruction reconstruction loss

Architecture:
    TRAINING:
    instruction(768) + exemplar(768) -> FeatureExtractor -> latent(10)
                                                              |
                                        LatentProjector -> reconstructed(768)
    Loss = -MLL + lambda * MSE(original_inst, reconstructed_inst)

    OPTIMIZATION:
    Find optimal 10D latent -> LatentProjector -> 768D -> Vec2Text -> text
"""

from .config import HyLO2Config
from .encoder import GTREncoder
from .gp_model import (
    FeatureExtractor,
    LatentProjector,
    HyLOGP,
    GPParams2,
    GPTrainer2
)
from .optimizer import (
    LatentOptimizationResult,
    LatentSpaceOptimizer
)
from .inverter import Vec2TextInverter, NearestNeighborInverter
from .hylo2 import HyLO2

__all__ = [
    # Config
    "HyLO2Config",
    # Encoder
    "GTREncoder",
    # GP Model
    "FeatureExtractor",
    "LatentProjector",
    "HyLOGP",
    "GPParams2",
    "GPTrainer2",
    # Optimizer
    "LatentOptimizationResult",
    "LatentSpaceOptimizer",
    # Inverter
    "Vec2TextInverter",
    "NearestNeighborInverter",
    # Main
    "HyLO2",
]
