"""Robust VAE-HbBoPs: Instruction-only optimization with VAE.

This module implements a robust prompt optimization system that:
- Uses VAE (not AE) for smooth latent space (768D -> 32D -> 768D)
- Prioritizes cosine similarity loss for Vec2Text compatibility
- Uses simplified GP without FeatureExtractor
- Implements gradient-based EI optimization
- Includes cycle consistency checks for Vec2Text inversion
"""

from robust_vec2text.vae import InstructionVAE, VAELoss
from robust_vec2text.gp import LatentGP, GPTrainer
from robust_vec2text.training import VAETrainer
from robust_vec2text.optimizer import RobustHbBoPs, GridPrompt
from robust_vec2text.inference import RobustInference, InversionResult
from robust_vec2text.encoder import GTRPromptEncoder

__all__ = [
    # VAE
    "InstructionVAE",
    "VAELoss",
    "VAETrainer",
    # GP
    "LatentGP",
    "GPTrainer",
    # Optimizer
    "RobustHbBoPs",
    "GridPrompt",
    # Inference
    "RobustInference",
    "InversionResult",
    # Encoder
    "GTRPromptEncoder",
]
