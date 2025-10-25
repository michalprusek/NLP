"""
Load ToS_wInstructions dataset and convert to HuggingFace format
"""
import os
from pathlib import Path
from datasets import Dataset, DatasetDict
from typing import Dict, List
import random

# Category mapping (same as Claudette)
CATEGORY_MAP = {
    'ltd': 0,   # Limitation of liability
    'ter': 1,   # Unilateral termination
    'ch': 2,    # Unilateral change
    'a': 3,     # Arbitration
    'cr': 4,    # Content removal
    'law': 5,   # Choice of law
    'use': 7,   # Contract by using
    'j': 8,     # Jurisdiction
    # Note: 'pinc' (Other, index 6) is not present in this dataset
}


def load_tos_dataset(dataset_path: str = "/Users/michalprusek/Downloads/ToS_wInstructions"):
    """
    Load ToS dataset from directory structure.

    Args:
        dataset_path: Path to ToS_wInstructions directory

    Returns:
        Dictionary with 'sentence', 'company', and label fields
    """
    sentences_dir = Path(dataset_path) / "Sentences"
    labels_dir = Path(dataset_path) / "Labels"

    # Get all company files
    company_files = sorted([f.stem for f in sentences_dir.glob("*.txt")])

    print(f"Found {len(company_files)} companies")

    all_data = []

    for company in company_files:
        # Load sentences
        with open(sentences_dir / f"{company}.txt", 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        # Load binary labels (combined)
        binary_labels_file = labels_dir / f"{company}.txt"
        if binary_labels_file.exists():
            with open(binary_labels_file, 'r', encoding='utf-8') as f:
                binary_labels = [int(line.strip()) for line in f if line.strip()]
        else:
            binary_labels = [-1] * len(sentences)

        # Load individual category labels
        category_labels = {}
        for cat_short in CATEGORY_MAP.keys():
            cat_dir = Path(dataset_path) / f"Labels_{cat_short.upper()}"
            cat_file = cat_dir / f"{company}.txt"

            if cat_file.exists():
                with open(cat_file, 'r', encoding='utf-8') as f:
                    category_labels[cat_short] = [int(line.strip()) for line in f if line.strip()]
            else:
                category_labels[cat_short] = [-1] * len(sentences)

        # Combine into examples
        for i, sentence in enumerate(sentences):
            if i >= len(binary_labels):
                break

            # Create boolean fields for each category (convert 1 → True, -1 → False)
            example = {
                'sentence': sentence,
                'company': company,
                'line_number': i,
                'language': 'en',
                'unfairness_level': 'unfair' if binary_labels[i] == 1 else 'fair',
                # Boolean fields for each category
                'ltd': category_labels.get('ltd', [-1] * len(sentences))[i] == 1,
                'ter': category_labels.get('ter', [-1] * len(sentences))[i] == 1,
                'ch': category_labels.get('ch', [-1] * len(sentences))[i] == 1,
                'a': category_labels.get('a', [-1] * len(sentences))[i] == 1,
                'cr': category_labels.get('cr', [-1] * len(sentences))[i] == 1,
                'law': category_labels.get('law', [-1] * len(sentences))[i] == 1,
                'pinc': False,  # Not in this dataset
                'use': category_labels.get('use', [-1] * len(sentences))[i] == 1,
                'j': category_labels.get('j', [-1] * len(sentences))[i] == 1,
            }

            all_data.append(example)

    print(f"Total examples: {len(all_data)}")

    # Count unfair examples
    unfair_count = sum(1 for ex in all_data if ex['unfairness_level'] == 'unfair')
    print(f"Unfair examples: {unfair_count} ({unfair_count/len(all_data)*100:.1f}%)")

    return all_data


def split_dataset(data: List[Dict], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test splits.

    Args:
        data: List of examples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with train/validation/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Shuffle data
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    # Calculate split indices
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} ({len(train_data)/n*100:.1f}%)")
    print(f"  Validation: {len(val_data)} ({len(val_data)/n*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/n*100:.1f}%)")

    # Convert to HuggingFace Dataset
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data),
    })

    return dataset_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and convert ToS dataset")
    parser.add_argument(
        "--input-path",
        type=str,
        default="/Users/michalprusek/Downloads/ToS_wInstructions",
        help="Path to ToS_wInstructions directory"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="datasets/tos_local",
        help="Output path for converted dataset"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)"
    )

    args = parser.parse_args()

    print("="*80)
    print("Loading ToS Dataset")
    print("="*80)

    # Load dataset
    data = load_tos_dataset(args.input_path)

    # Split into train/val/test
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    dataset_dict = split_dataset(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        seed=args.seed
    )

    # Save to disk
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    # Create metadata
    metadata = {
        "name": "ToS Local Dataset",
        "source": args.input_path,
        "description": "Terms of Service fairness classification from local files",
        "labels": {str(v): k.upper() for k, v in CATEGORY_MAP.items()},
        "splits": {
            "train": len(dataset_dict['train']),
            "validation": len(dataset_dict['validation']),
            "test": len(dataset_dict['test']),
        },
        "total_examples": len(data),
        "num_categories": len(CATEGORY_MAP),
    }

    import json
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Dataset saved successfully!")
    print(f"  Train: {len(dataset_dict['train'])} examples")
    print(f"  Validation: {len(dataset_dict['validation'])} examples")
    print(f"  Test: {len(dataset_dict['test'])} examples")
    print(f"  Total: {len(data)} examples")
