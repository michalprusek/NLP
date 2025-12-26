"""Vec2Text-integrated HbBoPs: Prompt optimization with text inversion.

This module combines HbBoPs (Hyperband-based Bayesian Optimization for Prompt Selection)
with Vec2Text for bidirectional mapping between latent space and text.

Architecture:
    Prompt Text -> GTR Encoder -> 768D x 2 -> AE Encoder -> 10D -> GP -> EI
                                                             |
    10D optimum -> AE Decoder -> 768D x 2 -> Vec2Text -> Novel Prompt Text

Key components:
    - GTRPromptEncoder: GTR-T5-Base encoder for Vec2Text compatibility
    - PromptAutoencoder: Regularized AE for 1536D <-> 10D mapping
    - HbBoPsVec2Text: HbBoPs with GTR encoder and autoencoder
    - Vec2TextHbBoPsInference: Complete inference pipeline with inversion
"""

from vec2text_hbbops.encoder import GTRPromptEncoder
from vec2text_hbbops.autoencoder import PromptAutoencoder, AutoencoderLoss
from vec2text_hbbops.training import AutoencoderTrainer
from vec2text_hbbops.hbbops_vec2text import HbBoPsVec2Text
from vec2text_hbbops.inference import Vec2TextHbBoPsInference

__all__ = [
    "GTRPromptEncoder",
    "PromptAutoencoder",
    "AutoencoderLoss",
    "AutoencoderTrainer",
    "HbBoPsVec2Text",
    "Vec2TextHbBoPsInference",
]
