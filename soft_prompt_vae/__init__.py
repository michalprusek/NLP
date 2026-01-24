"""Soft-Prompt VAE with Llama-3.1-8B backbone.

A VAE architecture using soft prompts for text generation,
designed for NeurIPS-quality research on 2x NVIDIA L40S (96GB VRAM).
"""

from soft_prompt_vae.config import ModelConfig, TrainingConfig, DataConfig
from soft_prompt_vae.model import LlamaSoftPromptVAE

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "LlamaSoftPromptVAE",
]

__version__ = "0.1.0"
