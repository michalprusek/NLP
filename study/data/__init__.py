"""Data pipeline for flow matching training.

This module provides utilities for:
- Dataset generation with SONAR embeddings
- Train/val/test split creation
- Normalization statistics
- PyTorch Dataset/DataLoader creation
- Data augmentation

Common paths and utilities are centralized here.
"""

from pathlib import Path

# Project root (NLP/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default dataset paths (relative to PROJECT_ROOT)
DATASETS_DIR = "study/datasets"
SPLITS_DIR = f"{DATASETS_DIR}/splits"

DEFAULT_STATS_PATH = f"{DATASETS_DIR}/normalization_stats.pt"
DEFAULT_VS_10K_PATH = f"{DATASETS_DIR}/vs_10k.pt"

# GSM8K dataset path
GSM8K_PATH = "datasets/gsm8k"

# Embedding dimension (SONAR basic encoder)
EMBEDDING_DIM = 1024

# Numerical stability constant
EPSILON = 1e-8


def get_split_path(size: str, split: str) -> str:
    """Get path to split file.

    Args:
        size: Dataset size ("1k", "5k", "10k")
        split: Split type ("train", "val", "test")

    Returns:
        Path string relative to PROJECT_ROOT
    """
    return f"{SPLITS_DIR}/{size}/{split}.pt"


def get_absolute_path(relative_path: str) -> Path:
    """Convert relative path to absolute path from PROJECT_ROOT.

    Args:
        relative_path: Path relative to PROJECT_ROOT

    Returns:
        Absolute Path object
    """
    return PROJECT_ROOT / relative_path


# Re-export main classes and functions
from study.data.dataset import FlowDataset, create_dataloader, load_all_splits
from study.data.normalize import (
    compute_stats,
    denormalize,
    load_stats,
    normalize,
    save_stats,
)
from study.data.augmentation import (
    AugmentationConfig,
    augment_batch,
    parse_aug_string,
)

__all__ = [
    # Constants
    "PROJECT_ROOT",
    "DATASETS_DIR",
    "SPLITS_DIR",
    "DEFAULT_STATS_PATH",
    "DEFAULT_VS_10K_PATH",
    "GSM8K_PATH",
    "EMBEDDING_DIM",
    "EPSILON",
    # Path utilities
    "get_split_path",
    "get_absolute_path",
    # Dataset
    "FlowDataset",
    "create_dataloader",
    "load_all_splits",
    # Normalization
    "compute_stats",
    "normalize",
    "denormalize",
    "load_stats",
    "save_stats",
    # Augmentation
    "AugmentationConfig",
    "augment_batch",
    "parse_aug_string",
]
