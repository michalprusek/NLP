"""
Custom dataset for Vec2Text corrector fine-tuning.

The corrector takes:
- Input: hypothesis text + target embedding
- Output: corrected text (should match original instruction)
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CorrectorDataset(Dataset):
    """Dataset for Vec2Text corrector fine-tuning.

    Each example contains:
    - text: Original instruction (target for training)
    - embedding: GTR embedding (768D) of the original text
    - hypothesis: InversionModel's initial guess (input context)

    The corrector learns to transform (hypothesis, embedding) -> text
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        embedding_dim: int = 768,
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSON file with training examples
            tokenizer: T5 tokenizer for encoding text
            max_length: Maximum sequence length
            embedding_dim: Dimension of embeddings (768 for GTR)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        # Load data
        with open(data_path) as f:
            self.examples = json.load(f)

        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example.

        Returns dict with:
        - input_ids: Tokenized hypothesis
        - attention_mask: Attention mask for input
        - labels: Tokenized target text
        - frozen_embeddings: GTR embedding (768D)
        """
        example = self.examples[idx]

        # Tokenize hypothesis (input to corrector)
        hypothesis_encoding = self.tokenizer(
            example["hypothesis"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target text (what corrector should output)
        target_encoding = self.tokenizer(
            example["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Get embedding
        embedding = torch.tensor(example["embedding"], dtype=torch.float32)

        # Labels: replace padding token id with -100 for loss computation
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": hypothesis_encoding["input_ids"].squeeze(),
            "attention_mask": hypothesis_encoding["attention_mask"].squeeze(),
            "labels": labels,
            "frozen_embeddings": embedding,
        }


class SimpleSeq2SeqDataset(Dataset):
    """Simpler dataset for basic seq2seq fine-tuning.

    For cases where we just want to map hypothesis -> target text
    without explicitly passing embeddings (embedding is implicit in hypothesis).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_source_length: int = 128,
        max_target_length: int = 128,
        prefix: str = "Correct this instruction: ",
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSON file
            tokenizer: T5 tokenizer
            max_source_length: Max length for hypothesis
            max_target_length: Max length for target text
            prefix: Prefix to add to hypothesis for T5 format
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prefix = prefix

        with open(data_path) as f:
            self.examples = json.load(f)

        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training example."""
        example = self.examples[idx]

        # Source: prefixed hypothesis
        source = self.prefix + example["hypothesis"]
        # Target: original text
        target = example["text"]

        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels with -100 for padding
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }
