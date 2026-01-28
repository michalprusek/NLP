"""
Data loading utilities for EcoFlow-BO training.

Handles:
- Loading GTR embeddings from datasets/gtr_embeddings_full.pt
- Optional text loading from datasets/combined_texts.json
- Train/val/test splits
- DDP-aware DataLoader creation
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Optional, Tuple, List
from pathlib import Path


class EmbeddingDataset(Dataset):
    """
    Dataset of GTR embeddings for manifold learning.

    Loads all embeddings into memory at initialization and provides train/val/test splits.
    """

    def __init__(
        self,
        embeddings_path: str = "datasets/gtr_embeddings_full.pt",
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            embeddings_path: Path to .pt file with embeddings
            split: "train", "val", or "test"
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed for reproducible splits
        """
        self.embeddings_path = Path(embeddings_path)
        self.split = split

        # Load embeddings
        print(f"Loading embeddings from {embeddings_path}...")
        self.embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)

        if isinstance(self.embeddings, dict):
            # Handle dict format (e.g., {"embeddings": tensor})
            self.embeddings = self.embeddings.get(
                "embeddings", list(self.embeddings.values())[0]
            )

        self.n_total = len(self.embeddings)
        print(f"Loaded {self.n_total:,} embeddings of shape {self.embeddings.shape}")

        # Create splits
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(self.n_total, generator=generator).tolist()

        n_train = int(train_ratio * self.n_total)
        n_val = int(val_ratio * self.n_total)

        if split == "train":
            self.indices = indices[:n_train]
        elif split == "val":
            self.indices = indices[n_train : n_train + n_val]
        elif split == "test":
            self.indices = indices[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"{split.capitalize()} set: {len(self.indices):,} samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        real_idx = self.indices[idx]
        return self.embeddings[real_idx]


class EmbeddingTextDataset(EmbeddingDataset):
    """
    Dataset with both embeddings and corresponding texts.

    Useful for evaluation: decode embedding → compare with original text.
    """

    def __init__(
        self,
        embeddings_path: str = "datasets/gtr_embeddings_full.pt",
        texts_path: str = "datasets/combined_texts.json",
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(
            embeddings_path=embeddings_path,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        # Load texts
        self.texts_path = Path(texts_path)
        if self.texts_path.exists():
            print(f"Loading texts from {texts_path}...")
            with open(texts_path, "r", encoding="utf-8") as f:
                self.texts = json.load(f)
            print(f"Loaded {len(self.texts):,} texts")

            # Verify alignment
            if len(self.texts) != self.n_total:
                print(f"Warning: Text count ({len(self.texts)}) != embedding count ({self.n_total})")
                self.texts = None
        else:
            print(f"No texts file at {texts_path}")
            self.texts = None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[str]]:
        real_idx = self.indices[idx]
        embedding = self.embeddings[real_idx]

        if self.texts is not None:
            text = self.texts[real_idx]
        else:
            text = None

        return embedding, text

    def get_text(self, idx: int) -> Optional[str]:
        """Get text by index in current split."""
        if self.texts is None:
            return None
        real_idx = self.indices[idx]
        return self.texts[real_idx]


def create_dataloaders(
    embeddings_path: str = "datasets/gtr_embeddings_full.pt",
    batch_size: int = 2048,
    num_workers: int = 8,
    pin_memory: bool = True,
    distributed: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        embeddings_path: Path to embeddings file
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        distributed: Whether to use DistributedSampler
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = EmbeddingDataset(
        embeddings_path=embeddings_path,
        split="train",
        seed=seed,
    )
    val_dataset = EmbeddingDataset(
        embeddings_path=embeddings_path,
        split="val",
        seed=seed,
    )
    test_dataset = EmbeddingDataset(
        embeddings_path=embeddings_path,
        split="test",
        seed=seed,
    )

    # Samplers for DDP
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Important for batch norm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def load_subset_for_optimization(
    embeddings_path: str = "datasets/gtr_embeddings_full.pt",
    n_samples: int = 1000,
    seed: int = 42,
) -> torch.Tensor:
    """
    Load a small subset of embeddings for optimization initialization.

    Args:
        embeddings_path: Path to embeddings
        n_samples: Number of samples to load
        seed: Random seed

    Returns:
        embeddings: [n_samples, 768]
    """
    dataset = EmbeddingDataset(
        embeddings_path=embeddings_path,
        split="train",
        seed=seed,
    )

    # Random subset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n_samples]

    embeddings = torch.stack([dataset[i.item()] for i in indices])
    return embeddings


class ReflowDataset(Dataset):
    """
    Dataset of (x_0, z, x_1) tuples for reflow training.

    Created by generating trajectories through the initial flow model.
    """

    def __init__(
        self,
        x_0: torch.Tensor,
        z: torch.Tensor,
        x_1: torch.Tensor,
    ):
        """
        Args:
            x_0: Initial noise [N, data_dim]
            z: Latent conditions [N, latent_dim]
            x_1: Generated samples [N, data_dim]
        """
        assert x_0.shape[0] == z.shape[0] == x_1.shape[0]
        self.x_0 = x_0
        self.z = z
        self.x_1 = x_1

    def __len__(self) -> int:
        return self.x_0.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_0[idx], self.z[idx], self.x_1[idx]

    @classmethod
    def from_model(
        cls,
        encoder,
        decoder,
        embeddings: torch.Tensor,
        n_ode_steps: int = 50,
        batch_size: int = 512,
    ) -> "ReflowDataset":
        """
        Create reflow dataset by running the model.

        Args:
            encoder: MatryoshkaEncoder
            decoder: RectifiedFlowDecoder
            embeddings: Input embeddings [N, data_dim]
            n_ode_steps: ODE steps for accurate generation
            batch_size: Batch size for generation

        Returns:
            ReflowDataset
        """
        encoder.eval()
        decoder.velocity_net.eval()

        with torch.no_grad():
            # Encode all embeddings
            z_all = encoder.encode_deterministic(embeddings)

            # Generate reflow pairs
            x_0, z, x_1 = decoder.generate_reflow_pairs(
                z_all, n_ode_steps=n_ode_steps, batch_size=batch_size
            )

        return cls(x_0, z, x_1)
