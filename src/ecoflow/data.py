"""
Data loading utilities for SONAR embeddings.

Provides dataset and dataloader for training flow matching models
on SONAR text embeddings.
"""

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class SonarEmbeddingDataset(Dataset):
    """
    Dataset for loading pre-computed SONAR embeddings.

    Expects a .pt file containing a tensor of shape [N, 1024]
    where N is the number of embeddings and 1024 is the SONAR dimension.
    """

    def __init__(self, path: str):
        """
        Initialize dataset from saved tensor file.

        Args:
            path: Path to .pt file containing embeddings tensor
        """
        self.embeddings = torch.load(path, weights_only=True)
        if not isinstance(self.embeddings, torch.Tensor):
            raise ValueError(f"Expected tensor in {path}, got {type(self.embeddings)}")
        if self.embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {self.embeddings.dim()}D")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


def get_sonar_dataloader(
    path: str,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for SONAR embeddings.

    Args:
        path: Path to .pt file containing embeddings tensor
        batch_size: Batch size for training
        num_workers: Number of data loading workers (default: 8 for L40S)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        shuffle: Shuffle data each epoch (default: True)

    Returns:
        DataLoader with drop_last=True for OT-CFM compatibility
    """
    dataset = SonarEmbeddingDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Essential for OT-CFM to have matching batch sizes
    )
