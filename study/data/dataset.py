"""PyTorch Dataset for flow model training.

Provides FlowDataset for loading normalized SONAR embeddings with proper
reproducibility and DataLoader creation utilities.
"""

import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from study.data import DEFAULT_STATS_PATH, EMBEDDING_DIM, get_split_path
from study.data.normalize import load_stats, normalize

logger = logging.getLogger(__name__)


class FlowDataset(Dataset):
    """
    Dataset for flow model training with optional normalization.

    Loads embeddings from split files and applies z-score normalization
    using pre-computed statistics from the training set.
    """

    def __init__(
        self,
        split_path: str,
        stats_path: str = DEFAULT_STATS_PATH,
        return_normalized: bool = True,
    ):
        """
        Load embeddings with optional normalization.

        Args:
            split_path: Path to split file (e.g., "study/datasets/splits/5k/train.pt")
            stats_path: Path to normalization stats
            return_normalized: If True, __getitem__ returns normalized embeddings
        """
        self.split_path = Path(split_path)
        self.stats_path = Path(stats_path)
        self.return_normalized = return_normalized

        # Load split data
        if not self.split_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_path}")

        logger.info(f"Loading split from {self.split_path}")
        data = torch.load(self.split_path, weights_only=False)

        self.embeddings: Tensor = data["embeddings"]  # Original, unnormalized
        self.instructions: list[str] = data["instructions"]
        self.indices: list[int] = data.get("indices", list(range(len(self.embeddings))))
        self.split_name: str = data.get("split", "unknown")
        self.size_name: str = data.get("size", "unknown")

        logger.info(f"Loaded {len(self)} embeddings from {self.split_name} split ({self.size_name})")

        # Load normalization stats if needed
        if self.return_normalized:
            if not self.stats_path.exists():
                raise FileNotFoundError(f"Stats file not found: {self.stats_path}")
            self.stats = load_stats(str(self.stats_path))

            # Pre-normalize all embeddings for efficiency
            self._normalized_embeddings = normalize(self.embeddings, self.stats)
            logger.info("Pre-normalized all embeddings")
        else:
            self.stats = None
            self._normalized_embeddings = None

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tensor:
        """
        Return (optionally normalized) embedding at index.

        Args:
            idx: Sample index

        Returns:
            Embedding tensor of shape [1024]
        """
        if self.return_normalized:
            return self._normalized_embeddings[idx]
        else:
            return self.embeddings[idx]

    def get_original(self, idx: int) -> Tuple[Tensor, str]:
        """
        Return original (unnormalized) embedding and instruction text.

        Useful for decoding back to text via SONAR decoder.

        Args:
            idx: Sample index

        Returns:
            Tuple of (original_embedding, instruction_text)
        """
        return self.embeddings[idx], self.instructions[idx]

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (always 1024 for SONAR)."""
        return EMBEDDING_DIM

    def get_all_embeddings(self, normalized: bool = True) -> Tensor:
        """
        Return all embeddings as a single tensor.

        Args:
            normalized: Whether to return normalized or original embeddings

        Returns:
            Tensor of shape [N, 1024]
        """
        if normalized:
            if self._normalized_embeddings is not None:
                return self._normalized_embeddings
            else:
                return normalize(self.embeddings, self.stats)
        return self.embeddings


def seed_worker(worker_id: int) -> None:
    """
    Seed worker for reproducibility.

    Called by DataLoader for each worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset: FlowDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create reproducible DataLoader with proper worker seeding.

    Args:
        dataset: FlowDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch

    Returns:
        Configured DataLoader
    """
    # Create generator with seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    logger.info(
        f"Created DataLoader: batch_size={batch_size}, shuffle={shuffle}, "
        f"workers={num_workers}, pin_memory={pin_memory}"
    )

    return loader


def load_all_splits(
    size: str = "5k",
    stats_path: str = DEFAULT_STATS_PATH,
    return_normalized: bool = True,
) -> Tuple[FlowDataset, FlowDataset, FlowDataset]:
    """
    Load train, val, and test splits for a given size.

    Args:
        size: Dataset size ("1k", "5k", "10k")
        stats_path: Path to normalization stats
        return_normalized: Whether to normalize embeddings

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_ds = FlowDataset(
        get_split_path(size, "train"),
        stats_path=stats_path,
        return_normalized=return_normalized,
    )
    val_ds = FlowDataset(
        get_split_path(size, "val"),
        stats_path=stats_path,
        return_normalized=return_normalized,
    )
    test_ds = FlowDataset(
        get_split_path(size, "test"),
        stats_path=stats_path,
        return_normalized=return_normalized,
    )

    return train_ds, val_ds, test_ds


def main():
    """Test data loading pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("TESTING DATA LOADING PIPELINE")
    print("=" * 60)

    # Test 1: Load all 9 splits
    print("\n[Test 1] Loading all 9 splits...")
    sizes = ["1k", "5k", "10k"]
    splits = ["train", "val", "test"]

    for size in sizes:
        for split in splits:
            path = get_split_path(size, split)
            ds = FlowDataset(path, return_normalized=True)
            print(f"  {size}/{split}: {len(ds)} samples, shape={ds[0].shape}")
            assert ds[0].shape == torch.Size([1024]), f"Shape mismatch for {size}/{split}"

    print("[PASS] All 9 splits loaded successfully")

    # Test 2: Check normalized statistics
    print("\n[Test 2] Checking normalized statistics...")
    ds = FlowDataset(get_split_path("5k", "train"), return_normalized=True)
    all_emb = ds.get_all_embeddings(normalized=True)

    mean_of_means = all_emb.mean(dim=0).mean().item()
    mean_of_stds = all_emb.std(dim=0).mean().item()

    print(f"  Mean of means: {mean_of_means:.4f} (should be ~0)")
    print(f"  Mean of stds: {mean_of_stds:.4f} (should be ~1)")

    assert abs(mean_of_means) < 0.1, f"Mean too far from 0: {mean_of_means}"
    assert abs(mean_of_stds - 1.0) < 0.3, f"Std too far from 1: {mean_of_stds}"

    print("[PASS] Normalized embeddings have correct statistics")

    # Test 3: DataLoader iteration
    print("\n[Test 3] Testing DataLoader iteration...")
    loader = create_dataloader(ds, batch_size=256, shuffle=True, seed=42)
    batch = next(iter(loader))

    print(f"  Batch shape: {batch.shape}")
    print(f"  Batch mean: {batch.mean():.4f}")
    print(f"  Batch std: {batch.std():.4f}")

    assert batch.shape == torch.Size([256, 1024]), f"Batch shape mismatch: {batch.shape}"

    print("[PASS] DataLoader iteration works")

    # Test 4: Reproducibility
    print("\n[Test 4] Testing reproducibility...")
    loader1 = create_dataloader(ds, batch_size=128, shuffle=True, seed=42, num_workers=0)
    loader2 = create_dataloader(ds, batch_size=128, shuffle=True, seed=42, num_workers=0)

    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))

    # Check first batch is identical
    match = torch.allclose(batch1, batch2)
    print(f"  Same seed produces same batch: {match}")

    assert match, "Reproducibility failed - different batches with same seed"

    print("[PASS] Reproducibility verified")

    # Test 5: get_original returns unnormalized
    print("\n[Test 5] Testing get_original() returns unnormalized...")
    emb_norm = ds[0]  # Normalized from ds
    emb_orig, text = ds.get_original(0)

    print(f"  Normalized mean: {emb_norm.mean():.4f}")
    print(f"  Original mean: {emb_orig.mean():.4f}")
    print(f"  Text (first 50 chars): {text[:50]}...")

    # Original should have different mean than normalized
    assert not torch.allclose(emb_norm, emb_orig), "Normalized and original should differ"
    assert isinstance(text, str) and len(text) > 0, "Text should be non-empty string"

    print("[PASS] get_original returns unnormalized data")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
