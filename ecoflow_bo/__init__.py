"""
EcoFlow-BO: Embedding-Conditioned Flow for Bayesian Optimization

A probabilistic framework for prompt optimization in latent space using:
- Matryoshka VAE Encoder: 768D GTR → 8D hierarchical latent
- Rectified Flow Decoder: 1-step deterministic generation
- Coarse-to-Fine GP: Progressive dimension unlocking for efficient BO
- Cycle Consistency: Hallucination detection before evaluation
"""

from .config import EcoFlowConfig
from .encoder import MatryoshkaEncoder
from .velocity_network import VelocityNetwork
from .cfm_decoder import RectifiedFlowDecoder
from .losses import EcoFlowLoss, MatryoshkaContrastiveLoss
from .latent_gp import LatentSpaceGP
from .density_acquisition import DensityAwareAcquisition
from .cycle_consistency import CycleConsistencyChecker
from .optimizer import EcoFlowBO

__all__ = [
    "EcoFlowConfig",
    "MatryoshkaEncoder",
    "VelocityNetwork",
    "RectifiedFlowDecoder",
    "EcoFlowLoss",
    "MatryoshkaContrastiveLoss",
    "LatentSpaceGP",
    "DensityAwareAcquisition",
    "CycleConsistencyChecker",
    "EcoFlowBO",
]

__version__ = "0.1.0"
