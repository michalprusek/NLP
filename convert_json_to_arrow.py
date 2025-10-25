"""
Convert JSON dataset to HuggingFace Arrow format
"""
from datasets import Dataset, DatasetDict
import json
from pathlib import Path

def convert_to_arrow():
    """Convert JSON splits to Arrow format."""
    input_dir = Path("datasets/tos_local")

    # Load JSON splits
    splits = {}
    for split_name in ['train', 'validation', 'test']:
        json_file = input_dir / f"{split_name}.json"
        print(f"Loading {json_file}...")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to Dataset
        splits[split_name] = Dataset.from_list(data)
        print(f"  Loaded {len(data)} examples")

    # Create DatasetDict
    dataset_dict = DatasetDict(splits)

    # Save in Arrow format
    output_path = input_dir / "arrow"
    output_path.mkdir(exist_ok=True)

    print(f"\nSaving to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    print(f"\n✓ Dataset converted to Arrow format!")
    print(f"  Train: {len(dataset_dict['train'])} examples")
    print(f"  Validation: {len(dataset_dict['validation'])} examples")
    print(f"  Test: {len(dataset_dict['test'])} examples")

    return dataset_dict

if __name__ == "__main__":
    convert_to_arrow()
