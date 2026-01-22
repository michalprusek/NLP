"""
Data loader for FlowPO-HD ManifoldKeeper training.

Loads instruction embeddings from SONAR and provides DataLoader
for OT-CFM training.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """Dataset of instruction SONAR embeddings.

    Loads pre-encoded SONAR embeddings for training ManifoldKeeper.
    Supports lazy encoding if embeddings don't exist.

    Attributes:
        embeddings: (N, 1024) tensor of SONAR embeddings
        texts: Optional list of original instruction texts
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        texts: Optional[List[str]] = None,
    ):
        """
        Initialize dataset.

        Args:
            embeddings: (N, 1024) SONAR embeddings
            texts: Optional original texts
        """
        self.embeddings = embeddings
        self.texts = texts
        self._validate()

    def _validate(self):
        """Validate dataset consistency."""
        if self.embeddings.dim() != 2:
            raise ValueError(f"Expected 2D embeddings, got {self.embeddings.dim()}D")

        if self.embeddings.shape[1] != 1024:
            raise ValueError(
                f"Expected SONAR dim 1024, got {self.embeddings.shape[1]}"
            )

        if self.texts is not None and len(self.texts) != len(self.embeddings):
            raise ValueError(
                f"texts ({len(self.texts)}) and embeddings ({len(self.embeddings)}) "
                "must have same length"
            )

        # Check for NaN/Inf
        if torch.isnan(self.embeddings).any():
            raise ValueError("NaN detected in embeddings")
        if torch.isinf(self.embeddings).any():
            raise ValueError("Inf detected in embeddings")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self.embeddings.shape[1]

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        norms = self.embeddings.norm(dim=-1)
        return {
            "n_samples": len(self),
            "dim": self.dim,
            "mean_norm": norms.mean().item(),
            "std_norm": norms.std().item(),
            "min_norm": norms.min().item(),
            "max_norm": norms.max().item(),
        }


def load_instructions_from_json(json_path: Union[str, Path]) -> List[str]:
    """Load instruction texts from JSON file.

    Expected format:
    - List of strings: ["instruction1", "instruction2", ...]
    - List of dicts: [{"instruction": "..."}, ...]
    - Dict with "instructions" key: {"instructions": [...]}

    Args:
        json_path: Path to JSON file

    Returns:
        List of instruction strings
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Instructions file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"Empty instructions list in {json_path}")

        if isinstance(data[0], str):
            instructions = data
        elif isinstance(data[0], dict):
            # Try common keys
            for key in ["instruction", "text", "prompt", "content"]:
                if key in data[0]:
                    instructions = [d[key] for d in data]
                    break
            else:
                raise ValueError(f"Unknown dict format in {json_path}: {data[0].keys()}")
        else:
            raise ValueError(f"Unknown item type in {json_path}: {type(data[0])}")

    elif isinstance(data, dict):
        for key in ["instructions", "prompts", "texts", "data"]:
            if key in data:
                instructions = data[key]
                break
        else:
            raise ValueError(f"Unknown dict format in {json_path}: {data.keys()}")
    else:
        raise ValueError(f"Unknown format in {json_path}: {type(data)}")

    # Filter empty strings
    instructions = [i.strip() for i in instructions if i.strip()]

    logger.info(f"Loaded {len(instructions)} instructions from {json_path}")

    return instructions


def encode_instructions_with_sonar(
    instructions: List[str],
    device: str = "cuda",
    batch_size: int = 32,
    normalize: bool = False,
    show_progress: bool = True,
) -> torch.Tensor:
    """Encode instructions using SONAR.

    Args:
        instructions: List of instruction texts
        device: Torch device
        batch_size: Encoding batch size
        normalize: Whether to L2 normalize embeddings (default False for decoder)
        show_progress: Show progress bar

    Returns:
        (N, 1024) tensor of SONAR embeddings
    """
    # Import here to avoid loading SONAR at module import
    from lido_pp.backbone.sonar_encoder import SONAREncoder

    logger.info(f"Encoding {len(instructions)} instructions with SONAR...")

    encoder = SONAREncoder(device=device, normalize=normalize)

    embeddings = encoder.encode_batch(
        instructions,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    logger.info(f"Encoded to shape {embeddings.shape}")
    logger.info(f"Mean norm: {embeddings.norm(dim=-1).mean():.4f}")

    return embeddings


def load_or_encode_dataset(
    instructions_path: Union[str, Path],
    embeddings_path: Union[str, Path],
    device: str = "cuda",
    normalize: bool = False,
    force_encode: bool = False,
) -> InstructionDataset:
    """Load pre-encoded embeddings or encode from instructions.

    If embeddings_path exists and force_encode=False, loads embeddings.
    Otherwise, loads instructions, encodes with SONAR, and saves embeddings.

    Args:
        instructions_path: Path to instructions JSON
        embeddings_path: Path to save/load embeddings
        device: Torch device
        normalize: L2 normalize embeddings
        force_encode: Force re-encoding even if embeddings exist

    Returns:
        InstructionDataset with loaded embeddings
    """
    instructions_path = Path(instructions_path)
    embeddings_path = Path(embeddings_path)

    # Try loading existing embeddings
    if embeddings_path.exists() and not force_encode:
        logger.info(f"Loading pre-encoded embeddings from {embeddings_path}")
        try:
            data = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        except Exception as e:
            # Fallback for embeddings with metadata (texts list)
            logger.warning(f"weights_only=True failed, retrying: {e}")
            data = torch.load(embeddings_path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            embeddings = data["embeddings"]
            texts = data.get("texts", None)
        else:
            embeddings = data
            texts = None

        dataset = InstructionDataset(embeddings, texts)
        logger.info(f"Loaded {len(dataset)} embeddings, dim={dataset.dim}")
        return dataset

    # Need to encode
    logger.info(f"Embeddings not found at {embeddings_path}, encoding...")

    # Load instructions
    instructions = load_instructions_from_json(instructions_path)

    # Encode
    embeddings = encode_instructions_with_sonar(
        instructions,
        device=device,
        normalize=normalize,
    )

    # Save
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings.cpu(),
        "texts": instructions,
        "normalize": normalize,
    }, embeddings_path)
    logger.info(f"Saved embeddings to {embeddings_path}")

    return InstructionDataset(embeddings, instructions)


def create_dataloader(
    dataset: InstructionDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 8,  # CLAUDE.md: use num_workers=8 for 2x L40S setup
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create DataLoader for ManifoldKeeper training.

    Args:
        dataset: InstructionDataset
        batch_size: Batch size (larger is better for OT pairing)
        shuffle: Shuffle data
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_train_val_dataloaders(
    dataset: InstructionDataset,
    train_ratio: float = 0.9,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        dataset: Full dataset
        train_ratio: Fraction for training
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed for split

    Returns:
        (train_loader, val_loader)
    """
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    # Set seed for reproducible split
    generator = torch.Generator().manual_seed(seed)

    train_indices, val_indices = torch.utils.data.random_split(
        range(n_total),
        [n_train, n_val],
        generator=generator,
    )

    train_dataset = InstructionDataset(
        dataset.embeddings[train_indices.indices],
        [dataset.texts[i] for i in train_indices.indices] if dataset.texts else None,
    )
    val_dataset = InstructionDataset(
        dataset.embeddings[val_indices.indices],
        [dataset.texts[i] for i in val_indices.indices] if dataset.texts else None,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    logger.info(f"Split: {n_train} train, {n_val} val")

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing data_loader...")

    # Test with synthetic data
    print("\n--- Synthetic Dataset ---")
    embeddings = torch.randn(100, 1024) * 0.2  # Approximate SONAR norm
    texts = [f"Instruction {i}" for i in range(100)]

    dataset = InstructionDataset(embeddings, texts)
    print(f"Dataset: {len(dataset)} samples")
    print(f"Stats: {dataset.get_stats()}")

    # Test DataLoader
    loader = create_dataloader(dataset, batch_size=32)
    print(f"\nDataLoader: {len(loader)} batches")

    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")

    # Test train/val split
    print("\n--- Train/Val Split ---")
    train_loader, val_loader = create_train_val_dataloaders(
        dataset,
        train_ratio=0.8,
        batch_size=16,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    print("\n[OK] data_loader tests passed!")
