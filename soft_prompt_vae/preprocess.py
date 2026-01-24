"""Offline preprocessing script for instruction datasets.

Downloads, converts, filters, deduplicates, and tokenizes datasets
for efficient training.

Usage:
    uv run python -m soft_prompt_vae.preprocess --phase 1
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm

from datasets import load_dataset

from soft_prompt_vae.config import VAEConfig, ModelConfig, DataConfig
from soft_prompt_vae.data.formats import convert_to_instruction_response, InstructionResponse
from soft_prompt_vae.data.filters import create_filter_pipeline, apply_filters
from soft_prompt_vae.data.deduplication import create_deduplicator, deduplicate
from soft_prompt_vae.data.tokenization import create_tokenizer
from soft_prompt_vae.data.dataset import save_preprocessed_dataset, save_jsonl_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_phase_datasets(phase: int, config: DataConfig) -> List[str]:
    """Get dataset names for a training phase."""
    if phase == 1:
        return list(config.phase_1_datasets)
    elif phase == 2:
        return list(config.phase_2_datasets)
    elif phase == 3:
        return list(config.phase_3_datasets)
    else:
        raise ValueError(f"Invalid phase: {phase}")


def preprocess_phase(
    phase: int,
    config: VAEConfig,
    max_samples: int = None,
    skip_tokenization: bool = False,
) -> None:
    """Preprocess datasets for a training phase.

    Args:
        phase: Training phase (1, 2, or 3)
        config: VAE configuration
        max_samples: Maximum samples to process (for testing)
        skip_tokenization: Skip tokenization step (save as JSONL only)
    """
    dataset_names = get_phase_datasets(phase, config.data)
    logger.info(f"Processing phase {phase} datasets: {dataset_names}")

    # Create output directory
    output_dir = config.data.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = create_tokenizer(config.model, config.data)

    # Create filter pipeline
    filter_pipeline = create_filter_pipeline(tokenizer.tokenizer, config.data)

    # Create deduplicator
    deduplicator = create_deduplicator(config.data)

    # Collect all examples
    all_examples: List[InstructionResponse] = []

    for dataset_name in dataset_names:
        logger.info(f"Loading {dataset_name}...")

        try:
            # Load dataset
            dataset = load_dataset(
                dataset_name,
                split="train",
                            )

            # Convert format
            logger.info(f"Converting {len(dataset)} examples...")
            converted = list(convert_to_instruction_response(iter(dataset), dataset_name))
            logger.info(f"Converted {len(converted)} examples from {dataset_name}")

            all_examples.extend(converted)

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            continue

    logger.info(f"Total raw examples: {len(all_examples)}")

    # Apply filtering
    logger.info("Applying filters...")
    filtered_examples = list(
        tqdm(
            apply_filters(iter(all_examples), filter_pipeline),
            desc="Filtering",
            total=len(all_examples),
        )
    )
    logger.info(f"After filtering: {len(filtered_examples)}")
    logger.info(f"Filter stats: {filter_pipeline.get_stats()}")

    # Apply deduplication
    logger.info("Deduplicating...")
    unique_examples = list(
        tqdm(
            deduplicate(iter(filtered_examples), deduplicator),
            desc="Deduplicating",
            total=len(filtered_examples),
        )
    )
    logger.info(f"After deduplication: {len(unique_examples)}")
    logger.info(f"Dedup stats: {deduplicator.get_stats()}")

    # Limit samples if requested
    if max_samples and len(unique_examples) > max_samples:
        logger.info(f"Limiting to {max_samples} samples")
        unique_examples = unique_examples[:max_samples]

    # Save JSONL (human-readable backup)
    jsonl_path = output_dir / f"phase_{phase}_train.jsonl"
    save_jsonl_dataset(unique_examples, jsonl_path)

    if skip_tokenization:
        logger.info("Skipping tokenization (--skip-tokenization)")
        return

    # Tokenize
    logger.info("Tokenizing...")
    tokenized_examples = []
    for example in tqdm(unique_examples, desc="Tokenizing"):
        tokenized = tokenizer.tokenize_pair(example.instruction, example.response)
        tokenized_examples.append(tokenized)

    # Save tokenized data
    pt_path = output_dir / f"phase_{phase}_train.pt"
    save_preprocessed_dataset(tokenized_examples, pt_path)

    # Print summary
    logger.info("=" * 60)
    logger.info(f"Phase {phase} preprocessing complete!")
    logger.info(f"  Raw examples: {len(all_examples)}")
    logger.info(f"  After filtering: {len(filtered_examples)}")
    logger.info(f"  After dedup: {len(unique_examples)}")
    logger.info(f"  Saved to: {pt_path}")
    logger.info("=" * 60)


def compute_statistics(
    phase: int,
    config: VAEConfig,
) -> None:
    """Compute statistics for preprocessed data.

    Args:
        phase: Training phase
        config: VAE configuration
    """
    import torch

    pt_path = config.data.processed_dir / f"phase_{phase}_train.pt"

    if not pt_path.exists():
        logger.error(f"Preprocessed data not found: {pt_path}")
        return

    logger.info(f"Loading {pt_path}...")
    data = torch.load(pt_path, weights_only=False)

    # Compute statistics
    num_samples = len(data["instruction_ids"])
    instr_lengths = data["instruction_attention_mask"].sum(dim=1).float()
    resp_lengths = data["response_attention_mask"].sum(dim=1).float()

    logger.info("=" * 60)
    logger.info(f"Phase {phase} statistics:")
    logger.info(f"  Total samples: {num_samples:,}")
    logger.info(f"  Instruction length: mean={instr_lengths.mean():.1f}, "
                f"std={instr_lengths.std():.1f}, "
                f"max={instr_lengths.max():.0f}")
    logger.info(f"  Response length: mean={resp_lengths.mean():.1f}, "
                f"std={resp_lengths.std():.1f}, "
                f"max={resp_lengths.max():.0f}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess instruction datasets for VAE training"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Training phase to preprocess (default: 1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)",
    )
    parser.add_argument(
        "--skip-tokenization",
        action="store_true",
        help="Skip tokenization, save JSONL only",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute statistics for existing data",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file",
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = VAEConfig.load(Path(args.config))
    else:
        config = VAEConfig()

    if args.stats_only:
        compute_statistics(args.phase, config)
    else:
        preprocess_phase(
            phase=args.phase,
            config=config,
            max_samples=args.max_samples,
            skip_tokenization=args.skip_tokenization,
        )


if __name__ == "__main__":
    main()
