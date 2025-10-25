"""
Simple loader for ToS_wInstructions dataset without HuggingFace dependency
"""
import json
import random
from pathlib import Path

# Category mapping
CATEGORY_MAP = {
    'ltd': 0,   # Limitation of liability
    'ter': 1,   # Unilateral termination
    'ch': 2,    # Unilateral change
    'a': 3,     # Arbitration
    'cr': 4,    # Content removal
    'law': 5,   # Choice of law
    'use': 7,   # Contract by using
    'j': 8,     # Jurisdiction
}

def load_tos_dataset(dataset_path="/Users/michalprusek/Downloads/ToS_wInstructions"):
    """Load ToS dataset from directory structure."""
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

        # Load binary labels
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

            example = {
                'sentence': sentence,
                'company': company,
                'line_number': i,
                'language': 'en',
                'unfairness_level': 'unfair' if binary_labels[i] == 1 else 'fair',
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
    unfair_count = sum(1 for ex in all_data if ex['unfairness_level'] == 'unfair')
    print(f"Unfair examples: {unfair_count} ({unfair_count/len(all_data)*100:.1f}%)")

    return all_data

def split_dataset(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split dataset into train/val/test."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} ({len(train_data)/n*100:.1f}%)")
    print(f"  Validation: {len(val_data)} ({len(val_data)/n*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/n*100:.1f}%)")

    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

if __name__ == "__main__":
    print("="*80)
    print("Loading ToS Dataset")
    print("="*80)

    # Load dataset
    data = load_tos_dataset()

    # Split
    splits = split_dataset(data)

    # Save as JSON
    output_dir = Path("datasets/tos_local")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {split_name}: {len(split_data)} examples → {output_file}")

    # Metadata
    metadata = {
        "name": "ToS Local Dataset",
        "source": "/Users/michalprusek/Downloads/ToS_wInstructions",
        "labels": {
            "0": "Limitation of liability (LTD)",
            "1": "Unilateral termination (TER)",
            "2": "Unilateral change (CH)",
            "3": "Arbitration (A)",
            "4": "Content removal (CR)",
            "5": "Choice of law (LAW)",
            "7": "Contract by using (USE)",
            "8": "Jurisdiction (J)"
        },
        "splits": {k: len(v) for k, v in splits.items()},
        "total_examples": len(data)
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Dataset saved to {output_dir}!")
