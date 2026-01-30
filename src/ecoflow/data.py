"""
Data loading utilities for SONAR embeddings.

Provides dataset and dataloader for training flow matching models
on SONAR text embeddings.

Supports normalization for stable training (SONAR has std~0.01, which causes
issues when training flow models from N(0,1) noise).
"""

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class SonarEmbeddingDataset(Dataset):
    """
    Dataset for loading pre-computed SONAR embeddings.

    Expects a .pt file containing a tensor of shape [N, 1024]
    where N is the number of embeddings and 1024 is the SONAR dimension.

    Optionally normalizes embeddings to have zero mean and unit variance
    per dimension for stable flow matching training.
    """

    def __init__(self, path: str, normalize: bool = False):
        """
        Initialize dataset from saved tensor file.

        Args:
            path: Path to .pt file containing embeddings tensor
            normalize: If True, normalize to zero mean and unit variance per dim
        """
        self.embeddings = torch.load(path, weights_only=True)
        if not isinstance(self.embeddings, torch.Tensor):
            raise ValueError(f"Expected tensor in {path}, got {type(self.embeddings)}")
        if self.embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {self.embeddings.dim()}D")

        self.normalize = normalize
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

        if normalize:
            # Compute per-dimension statistics
            self.mean = self.embeddings.mean(dim=0, keepdim=True)  # [1, 1024]
            self.std = self.embeddings.std(dim=0, keepdim=True) + 1e-8  # [1, 1024]
            # Normalize embeddings
            self.embeddings = (self.embeddings - self.mean) / self.std

    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get normalization statistics for denormalization during inference.

        Returns:
            Dict with 'mean' and 'std' tensors, or empty dict if not normalized
        """
        if self.normalize and self.mean is not None and self.std is not None:
            return {"mean": self.mean.squeeze(0), "std": self.std.squeeze(0)}
        return {}

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
    normalize: bool = False,
) -> Tuple[DataLoader, Dict[str, torch.Tensor]]:
    """
    Create a DataLoader for SONAR embeddings.

    Args:
        path: Path to .pt file containing embeddings tensor
        batch_size: Batch size for training
        num_workers: Number of data loading workers (default: 8 for L40S)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        shuffle: Shuffle data each epoch (default: True)
        normalize: If True, normalize embeddings to unit variance (default: False)

    Returns:
        Tuple of (DataLoader, normalization_stats dict)
        If normalize=False, normalization_stats is empty dict
    """
    dataset = SonarEmbeddingDataset(path, normalize=normalize)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Essential for OT-CFM to have matching batch sizes
    )
    return loader, dataset.get_normalization_stats()
