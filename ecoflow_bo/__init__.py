"""
EcoFlow-BO: Embedding-Conditioned Flow for Bayesian Optimization

A probabilistic framework for prompt optimization in latent space using:
- Matryoshka VAE Encoder: 768D GTR → 16D hierarchical latent
- DiT Velocity Network: Diffusion Transformer for conditional flow matching
- Rectified Flow Decoder: 1-step deterministic generation after reflow
- Coarse-to-Fine GP: Progressive dimension unlocking for efficient BO
- Cycle Consistency: Hallucination detection before evaluation
"""

from .config import EcoFlowConfig
from .encoder import MatryoshkaEncoder
from .velocity_network import VelocityNetwork
from .cfm_decoder import RectifiedFlowDecoder
from .losses import KLDivergenceLoss, InfoNCELoss, MatryoshkaCFMLoss
from .latent_gp import LatentSpaceGP, CoarseToFineGP
from .density_acquisition import DensityAwareAcquisition
from .cycle_consistency import CycleConsistencyChecker
from .optimizer import EcoFlowBO

__all__ = [
    # Config
    "EcoFlowConfig",
    # Models
    "MatryoshkaEncoder",
    "VelocityNetwork",
    "RectifiedFlowDecoder",
    # Losses
    "KLDivergenceLoss",
    "InfoNCELoss",
    "MatryoshkaCFMLoss",
    # Bayesian Optimization
    "LatentSpaceGP",
    "CoarseToFineGP",
    "DensityAwareAcquisition",
    "CycleConsistencyChecker",
    "EcoFlowBO",
]

__version__ = "0.1.0"
