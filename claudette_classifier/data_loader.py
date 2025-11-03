"""Data loading and preprocessing for Claudette dataset."""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler


class ClaudetteDataset(Dataset):
    """PyTorch dataset for Claudette ToS clauses."""

    def __init__(self, texts: list[str], labels: list[int]):
        """Initialize dataset.

        Args:
            texts: List of clause texts
            labels: List of binary labels (0=fair, 1=unfair)
        """
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def load_dataset(dataset_path: Path) -> Tuple[list[str], list[int]]:
    """Load Claudette dataset from JSON.

    Args:
        dataset_path: Path to tos_dataset.json

    Returns:
        Tuple of (texts, labels)
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['text'] for item in data]
    labels = [item['is_unfair'] for item in data]

    print(f"Loaded {len(texts)} clauses")
    print(f"Unfair: {sum(labels)} ({100 * sum(labels) / len(labels):.2f}%)")
    print(f"Fair: {len(labels) - sum(labels)} ({100 * (len(labels) - sum(labels)) / len(labels):.2f}%)")

    return texts, labels


def create_splits(
    texts: list[str],
    labels: list[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[ClaudetteDataset, ClaudetteDataset, ClaudetteDataset]:
    """Split dataset into train/val/test with stratification.

    Args:
        texts: List of clause texts
        labels: List of binary labels
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # First split: train vs. (val + test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=labels
    )

    # Second split: val vs. test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=(1 - val_size),
        random_state=random_seed,
        stratify=temp_labels
    )

    print(f"\nDataset splits:")
    print(f"Train: {len(train_texts)} samples ({100 * sum(train_labels) / len(train_labels):.2f}% unfair)")
    print(f"Val: {len(val_texts)} samples ({100 * sum(val_labels) / len(val_labels):.2f}% unfair)")
    print(f"Test: {len(test_texts)} samples ({100 * sum(test_labels) / len(test_labels):.2f}% unfair)")

    return (
        ClaudetteDataset(train_texts, train_labels),
        ClaudetteDataset(val_texts, val_labels),
        ClaudetteDataset(test_texts, test_labels)
    )


def get_class_weights(labels: list[int]) -> torch.Tensor:
    """Compute class weights for imbalanced dataset.

    Args:
        labels: List of binary labels

    Returns:
        Tensor of class weights [weight_fair, weight_unfair]
    """
    labels_array = np.array(labels)
    n_samples = len(labels)
    n_classes = 2

    # Count samples per class
    class_counts = np.bincount(labels_array, minlength=n_classes)

    # Compute weights: n_samples / (n_classes * class_count)
    weights = n_samples / (n_classes * class_counts)

    print(f"\nClass weights: fair={weights[0]:.3f}, unfair={weights[1]:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Create weighted random sampler for oversampling minority class.

    Args:
        labels: List of binary labels

    Returns:
        WeightedRandomSampler that oversamples unfair clauses
    """
    labels_array = np.array(labels)
    class_counts = np.bincount(labels_array, minlength=2)

    # Weight for each sample = inverse of its class frequency
    sample_weights = 1.0 / class_counts[labels_array]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

    return sampler


def create_dataloaders(
    train_dataset: ClaudetteDataset,
    val_dataset: ClaudetteDataset,
    test_dataset: ClaudetteDataset,
    batch_size: int = 32,
    use_oversampling: bool = True,
    use_distributed: bool = False,
    world_size: int = 1,
    rank: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders with optional oversampling and distributed support.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size per GPU
        use_oversampling: Whether to use weighted sampling for training (disabled with DDP)
        use_distributed: Whether to use DistributedSampler for multi-GPU training
        world_size: Number of GPUs (for distributed training)
        rank: Current process rank (for distributed training)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training loader
    if use_distributed:
        # Use DistributedSampler for DDP (oversampling not compatible with DDP)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        if rank == 0:
            print(f"Using DistributedSampler for training (world_size={world_size})")
    elif use_oversampling:
        sampler = create_weighted_sampler(train_dataset.labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        print("Using weighted random sampler for training (oversampling minority class)")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    # Validation and test loaders (no special sampling needed)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if use_distributed else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
