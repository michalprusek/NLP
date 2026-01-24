"""Data pipeline for Soft-Prompt VAE training.

Includes format converters, filters, deduplication, and DataLoader factories.
"""

from soft_prompt_vae.data.formats import convert_to_instruction_response
from soft_prompt_vae.data.filters import apply_filters
from soft_prompt_vae.data.deduplication import deduplicate
from soft_prompt_vae.data.dataset import InstructionDataset, PreprocessedDataset
from soft_prompt_vae.data.collator import VAECollator
from soft_prompt_vae.data.loader import create_dataloader

__all__ = [
    "convert_to_instruction_response",
    "apply_filters",
    "deduplicate",
    "InstructionDataset",
    "PreprocessedDataset",
    "VAECollator",
    "create_dataloader",
]
