"""Create nested train/val/test splits from the VS dataset.

This script creates nested splits for 1K, 5K, and 10K dataset sizes:
- Single shuffle with fixed seed for reproducibility
- 1K indices are first 1000 of shuffled, 5K are first 5000, 10K are all 10000
- Split ratio: 80% train, 10% val, 10% test

Usage:
    uv run python study/data/create_splits.py                    # Create splits from vs_10k.pt
    uv run python study/data/create_splits.py --verify           # Verify existing splits
    uv run python study/data/create_splits.py --input path.pt    # Use custom input file
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from study.data import DEFAULT_VS_10K_PATH, PROJECT_ROOT, SPLITS_DIR


def create_splits(
    input_path: str,
    output_dir: str,
    seed: int = 42,
) -> Dict[str, Dict[str, dict]]:
    """Create nested train/val/test splits.

    Nested property: 1K train/val/test are subsets of 5K train/val/test,
    which are subsets of 10K train/val/test.

    Strategy: First split the full dataset into train/val/test pools,
    then take the first N samples from each pool for smaller sizes.

    Args:
        input_path: Path to full dataset (e.g., vs_10k.pt)
        output_dir: Directory for split files
        seed: Random seed for shuffling

    Returns:
        Dictionary of splits: {size: {split_name: data}}
    """
    print(f"Loading dataset from {input_path}")
    dataset = torch.load(input_path, weights_only=False)

    embeddings = dataset["embeddings"]
    instructions = dataset["instructions"]
    total_samples = len(instructions)

    print(f"Total samples: {total_samples}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")

    # Determine which sizes we can create based on available data
    all_sizes = [("1k", 1000), ("5k", 5000), ("10k", 10000)]
    available_sizes = [(name, size) for name, size in all_sizes if size <= total_samples]

    if not available_sizes:
        raise ValueError(f"Not enough samples ({total_samples}) to create even 1K split")

    print(f"Creating splits for sizes: {[name for name, _ in available_sizes]}")

    # Single shuffle with fixed seed
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(total_samples)

    # Split the full dataset into train/val/test pools (80/10/10)
    # These pools will be subsetted for smaller sizes
    n_train_full = int(total_samples * 0.8)
    n_val_full = int(total_samples * 0.1)
    # n_test_full = remaining

    train_pool = shuffled_indices[:n_train_full]
    val_pool = shuffled_indices[n_train_full:n_train_full + n_val_full]
    test_pool = shuffled_indices[n_train_full + n_val_full:]

    print(f"Full pools: train={len(train_pool)}, val={len(val_pool)}, test={len(test_pool)}")

    # Create output directories
    output_path = Path(output_dir)
    for size_name, _ in available_sizes:
        (output_path / size_name).mkdir(parents=True, exist_ok=True)

    all_splits = {}

    for size_name, size in available_sizes:
        print(f"\nCreating {size_name} splits...")

        # Calculate how many samples from each pool for this size
        n_train = int(size * 0.8)
        n_val = int(size * 0.1)
        n_test = size - n_train - n_val

        # Take first N from each pool (ensures nested property)
        train_indices = train_pool[:n_train]
        val_indices = val_pool[:n_val]
        test_indices = test_pool[:n_test]

        splits = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

        all_splits[size_name] = {}

        for split_name, indices in splits.items():
            # Extract data for this split
            split_embeddings = embeddings[indices]
            split_instructions = [instructions[i] for i in indices]

            split_data = {
                "embeddings": split_embeddings,
                "instructions": split_instructions,
                "indices": indices.tolist(),  # Original indices in vs_10k.pt
                "split": split_name,
                "size": size_name,
            }

            # Save split
            split_path = output_path / size_name / f"{split_name}.pt"
            torch.save(split_data, split_path)
            print(f"  Saved {split_name}: {len(indices)} samples -> {split_path}")

            all_splits[size_name][split_name] = split_data

    return all_splits


def verify_splits(output_dir: str) -> bool:
    """Verify split integrity.

    Checks:
    1. All expected split files exist
    2. Sample counts are correct
    3. Nested property holds
    4. No data leakage between train/val/test

    Returns:
        True if all checks pass
    """
    output_path = Path(output_dir)
    all_passed = True

    expected_counts = {
        "1k": {"train": 800, "val": 100, "test": 100},
        "5k": {"train": 4000, "val": 500, "test": 500},
        "10k": {"train": 8000, "val": 1000, "test": 1000},
    }

    # Collect all split data for cross-checks
    all_data = {}

    # Check available sizes
    available_sizes = []
    for size_name in ["1k", "5k", "10k"]:
        size_dir = output_path / size_name
        if size_dir.exists():
            available_sizes.append(size_name)

    if not available_sizes:
        print("ERROR: No split directories found")
        return False

    print(f"Verifying splits for sizes: {available_sizes}")

    for size_name in available_sizes:
        all_data[size_name] = {}

        for split_name in ["train", "val", "test"]:
            split_path = output_path / size_name / f"{split_name}.pt"

            # Check file exists
            if not split_path.exists():
                print(f"ERROR: Missing file {split_path}")
                all_passed = False
                continue

            # Load and verify
            data = torch.load(split_path, weights_only=False)
            all_data[size_name][split_name] = data

            n_samples = len(data["embeddings"])
            expected = expected_counts[size_name][split_name]

            # Check sample count
            if n_samples != expected:
                print(f"ERROR: {size_name}/{split_name} has {n_samples} samples, expected {expected}")
                all_passed = False
            else:
                print(f"OK: {size_name}/{split_name} has {n_samples} samples")

            # Check embedding shape and dtype
            if data["embeddings"].shape[1] != 1024:
                print(f"ERROR: {size_name}/{split_name} embeddings dim is {data['embeddings'].shape[1]}, expected 1024")
                all_passed = False

            if data["embeddings"].dtype != torch.float32:
                print(f"ERROR: {size_name}/{split_name} embeddings dtype is {data['embeddings'].dtype}, expected float32")
                all_passed = False

    # Check nested property: 1K train indices subset of 5K train subset of 10K train
    print("\nVerifying nested property...")
    for split_name in ["train", "val", "test"]:
        sizes_available = [s for s in available_sizes if s in all_data and split_name in all_data[s]]

        for i in range(len(sizes_available) - 1):
            smaller = sizes_available[i]
            larger = sizes_available[i + 1]

            smaller_indices = set(all_data[smaller][split_name]["indices"])
            larger_indices = set(all_data[larger][split_name]["indices"])

            if not smaller_indices.issubset(larger_indices):
                print(f"ERROR: {smaller}/{split_name} is NOT subset of {larger}/{split_name}")
                all_passed = False
            else:
                print(f"OK: {smaller}/{split_name} is subset of {larger}/{split_name}")

    # Check no data leakage within each size
    print("\nVerifying no data leakage...")
    for size_name in available_sizes:
        if size_name not in all_data:
            continue

        splits = all_data[size_name]
        if not all(s in splits for s in ["train", "val", "test"]):
            continue

        train_set = set(splits["train"]["indices"])
        val_set = set(splits["val"]["indices"])
        test_set = set(splits["test"]["indices"])

        train_val = train_set & val_set
        train_test = train_set & test_set
        val_test = val_set & test_set

        if train_val:
            print(f"ERROR: {size_name} has {len(train_val)} overlapping samples between train and val")
            all_passed = False
        if train_test:
            print(f"ERROR: {size_name} has {len(train_test)} overlapping samples between train and test")
            all_passed = False
        if val_test:
            print(f"ERROR: {size_name} has {len(val_test)} overlapping samples between val and test")
            all_passed = False

        if not train_val and not train_test and not val_test:
            print(f"OK: {size_name} has no overlapping samples between train/val/test")

    if all_passed:
        print("\nAll verification checks PASSED")
    else:
        print("\nSome verification checks FAILED")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Create nested train/val/test splits")
    parser.add_argument("--input", type=str, default=DEFAULT_VS_10K_PATH, help="Input dataset path")
    parser.add_argument("--output", type=str, default=SPLITS_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--verify", action="store_true", help="Verify existing splits instead of creating")

    args = parser.parse_args()

    # Change to project root for relative paths
    os.chdir(PROJECT_ROOT)

    if args.verify:
        success = verify_splits(args.output)
        sys.exit(0 if success else 1)
    else:
        # Check if input exists
        if not Path(args.input).exists():
            print(f"ERROR: Input file {args.input} does not exist")
            print("Run 'uv run python study/data/generate_dataset.py' first to generate the dataset")
            sys.exit(1)

        create_splits(args.input, args.output, args.seed)
        print("\nSplits created. Run with --verify to check integrity.")


if __name__ == "__main__":
    main()
