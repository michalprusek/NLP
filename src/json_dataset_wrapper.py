"""
Simple wrapper to load JSON dataset and mimic HuggingFace Dataset interface
"""
import json
from pathlib import Path
from typing import Dict, Any, List


class JSONDataset:
    """
    Simple dataset wrapper that loads from JSON and provides
    HuggingFace Dataset-like interface.
    """

    def __init__(self, data: List[Dict[str, Any]]):
        """
        Initialize dataset from list of examples.

        Args:
            data: List of example dictionaries
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get example by index."""
        if isinstance(idx, int):
            return self.data[idx]
        elif isinstance(idx, slice):
            return [self.data[i] for i in range(*idx.indices(len(self.data)))]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def select(self, indices: List[int]) -> 'JSONDataset':
        """Create subset with specified indices."""
        subset_data = [self.data[i] for i in indices]
        return JSONDataset(subset_data)

    @classmethod
    def from_json(cls, json_path: str) -> 'JSONDataset':
        """Load dataset from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data)


class JSONDatasetDict:
    """
    Simple wrapper that mimics HuggingFace DatasetDict interface.
    """

    def __init__(self, splits: Dict[str, JSONDataset]):
        """
        Initialize dataset dictionary.

        Args:
            splits: Dictionary mapping split names to JSONDataset objects
        """
        self.splits = splits

    def __getitem__(self, split_name: str) -> JSONDataset:
        """Get dataset for specific split."""
        return self.splits[split_name]

    def __contains__(self, split_name: str) -> bool:
        """Check if split exists."""
        return split_name in self.splits

    def keys(self) -> List[str]:
        """Get list of split names."""
        return list(self.splits.keys())

    @classmethod
    def load_from_disk(cls, dataset_path: str) -> 'JSONDatasetDict':
        """
        Load dataset from directory containing JSON splits.

        Args:
            dataset_path: Path to directory with train.json, validation.json, test.json

        Returns:
            JSONDatasetDict with all splits
        """
        dataset_dir = Path(dataset_path)

        splits = {}
        for split_file in dataset_dir.glob("*.json"):
            if split_file.name == "metadata.json":
                continue

            split_name = split_file.stem
            splits[split_name] = JSONDataset.from_json(str(split_file))

            print(f"Loaded {split_name}: {len(splits[split_name])} examples from {split_file.name}")

        return cls(splits)
