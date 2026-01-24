"""DataLoader factory for VAE training.

Creates optimized DataLoaders for both streaming and preprocessed datasets.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler

from soft_prompt_vae.data.dataset import InstructionDataset, PreprocessedDataset
from soft_prompt_vae.data.collator import VAECollator, DynamicPaddingCollator, VAEBatch
from soft_prompt_vae.data.tokenization import LlamaTokenizerWrapper
from soft_prompt_vae.config import DataConfig, ModelConfig

logger = logging.getLogger(__name__)


def create_streaming_dataloader(
    dataset_names: List[str],
    tokenizer: LlamaTokenizerWrapper,
    config: DataConfig,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    split: str = "train",
) -> DataLoader:
    """Create DataLoader for streaming (on-the-fly) processing.

    Args:
        dataset_names: List of HuggingFace dataset names
        tokenizer: Tokenizer wrapper
        config: Data configuration
        batch_size: Batch size per device
        world_size: Total number of processes
        rank: Current process rank
        split: Dataset split

    Returns:
        DataLoader for streaming data
    """
    dataset = InstructionDataset(
        dataset_names=dataset_names,
        tokenizer=tokenizer,
        config=config,
        split=split,
        world_size=world_size,
        rank=rank,
    )

    collator = VAECollator(pad_token_id=tokenizer.pad_token_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    logger.info(
        f"Created streaming DataLoader: batch_size={batch_size}, "
        f"num_workers={config.num_workers}, rank={rank}/{world_size}"
    )

    return dataloader


def create_preprocessed_dataloader(
    data_path: Path,
    tokenizer: Optional[LlamaTokenizerWrapper],
    config: DataConfig,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    dynamic_padding: bool = False,
) -> DataLoader:
    """Create DataLoader for preprocessed data.

    Args:
        data_path: Path to preprocessed data file
        tokenizer: Optional tokenizer (required for JSONL)
        config: Data configuration
        batch_size: Batch size per device
        world_size: Total number of processes
        rank: Current process rank
        shuffle: Whether to shuffle data
        dynamic_padding: Use dynamic padding for memory efficiency

    Returns:
        DataLoader for preprocessed data
    """
    dataset = PreprocessedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
    )

    # Use DistributedSampler for DDP
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling

    # Choose collator
    pad_token_id = tokenizer.pad_token_id if tokenizer else 0
    if dynamic_padding:
        collator = DynamicPaddingCollator(pad_token_id=pad_token_id)
    else:
        collator = VAECollator(pad_token_id=pad_token_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        drop_last=True,  # For consistent batch sizes in DDP
    )

    logger.info(
        f"Created preprocessed DataLoader: {len(dataset)} examples, "
        f"batch_size={batch_size}, shuffle={shuffle}, "
        f"distributed={'yes' if sampler else 'no'}"
    )

    return dataloader


def create_dataloader(
    config: DataConfig,
    tokenizer: LlamaTokenizerWrapper,
    batch_size: int,
    phase: int = 1,
    use_preprocessed: bool = True,
    world_size: int = 1,
    rank: int = 0,
    split: str = "train",
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader based on configuration.

    Args:
        config: Data configuration
        tokenizer: Tokenizer wrapper
        batch_size: Batch size per device
        phase: Training phase (1, 2, or 3)
        use_preprocessed: Use preprocessed data if available
        world_size: Total number of processes
        rank: Current process rank
        split: Dataset split
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    # Get dataset names for phase
    if phase == 1:
        dataset_names = list(config.phase_1_datasets)
    elif phase == 2:
        dataset_names = list(config.phase_2_datasets)
    elif phase == 3:
        dataset_names = list(config.phase_3_datasets)
    else:
        raise ValueError(f"Invalid phase: {phase}")

    # Check for preprocessed data
    preprocessed_path = config.processed_dir / f"phase_{phase}_{split}.pt"

    if use_preprocessed and preprocessed_path.exists():
        logger.info(f"Using preprocessed data: {preprocessed_path}")
        return create_preprocessed_dataloader(
            data_path=preprocessed_path,
            tokenizer=tokenizer,
            config=config,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            shuffle=shuffle,
            dynamic_padding=True,
        )
    else:
        logger.info(f"Using streaming data for datasets: {dataset_names}")
        return create_streaming_dataloader(
            dataset_names=dataset_names,
            tokenizer=tokenizer,
            config=config,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            split=split,
        )


def get_num_batches_estimate(
    config: DataConfig,
    batch_size: int,
    phase: int = 1,
    world_size: int = 1,
) -> int:
    """Estimate number of batches for a phase.

    Args:
        config: Data configuration
        batch_size: Batch size per device
        phase: Training phase
        world_size: Number of processes

    Returns:
        Estimated number of batches
    """
    # Rough estimates based on dataset sizes
    phase_sizes = {
        1: 100_000,  # FineTome-100k
        2: 400_000,  # OpenHermes + WizardLM
        3: 100_000,  # no_robots + Magicoder
    }

    # Assume ~90% pass filtering
    estimated_samples = int(phase_sizes.get(phase, 100_000) * 0.9)

    # Account for DDP
    samples_per_rank = estimated_samples // world_size

    return samples_per_rank // batch_size
