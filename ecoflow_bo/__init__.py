"""
EcoFlow-BO: Embedding-Conditioned Flow for Bayesian Optimization

A probabilistic framework for prompt optimization in latent space using:
- Matryoshka VAE Encoder: 768D GTR → z_core (16D) + z_detail (32D) = 48D residual latent
- DiT Velocity Network: Diffusion Transformer for conditional flow matching
- Rectified Flow Decoder: 1-step deterministic generation after reflow
- Coarse-to-Fine GP: Progressive dimension unlocking on z_core (16D)
- Detail Retriever: Nearest neighbor z_detail from training set
- Cycle Consistency: Hallucination detection before evaluation

Key Innovation: Residual Latent Architecture
- GP optimizes only z_core (16D) - tractable optimization
- z_detail (32D) retrieved from training set - high-fidelity reconstruction
- Total 48D capacity without GP curse of dimensionality
"""

from .config import EcoFlowConfig, ResidualLatentConfig
from .encoder import MatryoshkaEncoder
from .velocity_network import VelocityNetwork
from .cfm_decoder import RectifiedFlowDecoder
from .losses import (
    KLDivergenceLoss,
    InfoNCELoss,
    MatryoshkaCFMLoss,
    ResidualMatryoshkaCFMLoss,
    ResidualKLLoss,
)
from .latent_gp import LatentSpaceGP, CoarseToFineGP
from .density_acquisition import DensityAwareAcquisition
from .cycle_consistency import CycleConsistencyChecker
from .detail_retriever import (
    SimpleDetailRetriever,
    create_detail_retriever,
)
from .optimizer import EcoFlowBO, EcoFlowBOWithVec2Text

__all__ = [
    # Config
    "EcoFlowConfig",
    "ResidualLatentConfig",
    # Models
    "MatryoshkaEncoder",
    "VelocityNetwork",
    "RectifiedFlowDecoder",
    # Losses
    "KLDivergenceLoss",
    "InfoNCELoss",
    "MatryoshkaCFMLoss",
    "ResidualMatryoshkaCFMLoss",
    "ResidualKLLoss",
    # Bayesian Optimization
    "LatentSpaceGP",
    "CoarseToFineGP",
    "DensityAwareAcquisition",
    "CycleConsistencyChecker",
    # Detail Retrieval (Residual Latent)
    "SimpleDetailRetriever",
    "create_detail_retriever",
    # Optimizers
    "EcoFlowBO",
    "EcoFlowBOWithVec2Text",
]

__version__ = "0.1.0"
