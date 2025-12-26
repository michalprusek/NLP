"""Robust VAE-HbBoPs: Instruction-only optimization with VAE.

This module implements a robust prompt optimization system that:
- Uses VAE for smooth latent space (768D -> 32D -> 768D)
- Uses HbBoPs-style deep kernel GP on (instruction, exemplar) pairs
- Gradient-based EI optimization: VAE decode -> GP predict
- APE forward pass for data augmentation (1000+ instructions)
- Vec2Text inversion for latent -> text
"""

from robust_vec2text.vae import InstructionVAE, VAELoss
from robust_vec2text.training import VAETrainer
from robust_vec2text.optimizer import RobustHbBoPs, GridPrompt
from robust_vec2text.inference import RobustInference, InversionResult
from robust_vec2text.encoder import GTRPromptEncoder
from robust_vec2text.ape_generator import APEInstructionGenerator
from robust_vec2text.exemplar_selector import ExemplarSelector

__all__ = [
    # VAE
    "InstructionVAE",
    "VAELoss",
    "VAETrainer",
    # Optimizer
    "RobustHbBoPs",
    "GridPrompt",
    # Inference
    "RobustInference",
    "InversionResult",
    # Encoder
    "GTRPromptEncoder",
    # APE
    "APEInstructionGenerator",
    # Exemplar Selection (HbBoPs-style GP)
    "ExemplarSelector",
]
