"""Data loading utilities for SONAR embeddings."""

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SonarEmbeddingDataset(Dataset):
    """Dataset for SONAR embeddings with optional normalization."""

    def __init__(self, path: str, normalize: bool = False):
        self.embeddings = torch.load(path, weights_only=True)
        if not isinstance(self.embeddings, torch.Tensor):
            raise ValueError(f"Expected tensor in {path}, got {type(self.embeddings)}")
        if self.embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {self.embeddings.dim()}D")

        self.normalize = normalize
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

        if normalize:
            self.mean = self.embeddings.mean(dim=0, keepdim=True)
            self.std = self.embeddings.std(dim=0, keepdim=True) + 1e-8
            self.embeddings = (self.embeddings - self.mean) / self.std

    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """Return mean/std for denormalization, or empty dict if not normalized."""
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
    """Create a DataLoader for SONAR embeddings with optional normalization."""
    dataset = SonarEmbeddingDataset(path, normalize=normalize)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return loader, dataset.get_normalization_stats()
