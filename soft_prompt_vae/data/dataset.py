"""Dataset classes for VAE training.

Implements:
- InstructionDataset: Streaming dataset for on-the-fly processing
- PreprocessedDataset: Map-style dataset for preprocessed data
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any
import json

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset

from soft_prompt_vae.data.formats import InstructionResponse, convert_to_instruction_response
from soft_prompt_vae.data.filters import create_filter_pipeline, apply_filters
from soft_prompt_vae.data.deduplication import create_deduplicator, deduplicate
from soft_prompt_vae.data.tokenization import LlamaTokenizerWrapper, TokenizedExample
from soft_prompt_vae.config import DataConfig, ModelConfig

logger = logging.getLogger(__name__)


class InstructionDataset(IterableDataset):
    """Streaming dataset for instruction-response pairs.

    Handles format conversion, filtering, and deduplication on-the-fly.
    Supports DDP with manual sharding.
    """

    def __init__(
        self,
        dataset_names: List[str],
        tokenizer: LlamaTokenizerWrapper,
        config: DataConfig,
        split: str = "train",
        world_size: int = 1,
        rank: int = 0,
    ):
        """Initialize streaming dataset.

        Args:
            dataset_names: List of HuggingFace dataset names
            tokenizer: Tokenizer wrapper
            config: Data configuration
            split: Dataset split
            world_size: Total number of processes (for DDP)
            rank: Current process rank (for DDP)
        """
        self.dataset_names = dataset_names
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.world_size = world_size
        self.rank = rank

        # Create filter pipeline
        self.filter_pipeline = create_filter_pipeline(
            tokenizer.tokenizer, config
        )

        # Create deduplicator
        self.deduplicator = create_deduplicator(config)

    def _load_and_convert(self) -> Iterator[InstructionResponse]:
        """Load datasets and convert to unified format."""
        for dataset_name in self.dataset_names:
            logger.info(f"Loading dataset: {dataset_name}")
            try:
                dataset = load_dataset(
                    dataset_name,
                    split=self.split,
                    streaming=True,
                                    )

                yield from convert_to_instruction_response(dataset, dataset_name)

            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                continue

    def _shard_data(
        self, data: Iterator[InstructionResponse]
    ) -> Iterator[InstructionResponse]:
        """Shard data for DDP."""
        for i, item in enumerate(data):
            if i % self.world_size == self.rank:
                yield item

    def __iter__(self) -> Iterator[TokenizedExample]:
        """Iterate over tokenized examples."""
        # Reset deduplicator for new epoch
        self.deduplicator.reset()

        # Pipeline: load -> convert -> filter -> deduplicate -> shard -> tokenize
        data = self._load_and_convert()
        data = apply_filters(data, self.filter_pipeline)
        data = deduplicate(data, self.deduplicator)
        data = self._shard_data(data)

        for item in data:
            yield self.tokenizer.tokenize_pair(item.instruction, item.response)


class PreprocessedDataset(Dataset):
    """Map-style dataset for preprocessed data.

    Loads preprocessed tokenized examples from disk.
    Supports DistributedSampler for DDP.
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: Optional[LlamaTokenizerWrapper] = None,
    ):
        """Initialize preprocessed dataset.

        Args:
            data_path: Path to preprocessed data file (.pt or .jsonl)
            tokenizer: Optional tokenizer for on-the-fly tokenization
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer

        # Load data
        if self.data_path.suffix == ".pt":
            self._load_tensor_data()
        elif self.data_path.suffix == ".jsonl":
            self._load_jsonl_data()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        logger.info(f"Loaded {len(self)} examples from {self.data_path}")

    def _load_tensor_data(self) -> None:
        """Load preprocessed tensor data."""
        data = torch.load(self.data_path, weights_only=False)

        self.instruction_ids = data["instruction_ids"]
        self.instruction_attention_mask = data["instruction_attention_mask"]
        self.response_ids = data["response_ids"]
        self.response_attention_mask = data["response_attention_mask"]
        self.labels = data["labels"]

    def _load_jsonl_data(self) -> None:
        """Load JSONL data (requires tokenizer for on-the-fly tokenization)."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for JSONL data")

        self.examples: List[Dict[str, str]] = []
        with open(self.data_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        """Get dataset size."""
        if hasattr(self, "instruction_ids"):
            return len(self.instruction_ids)
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedExample:
        """Get single example."""
        if hasattr(self, "instruction_ids"):
            # Return pre-tokenized tensors
            return TokenizedExample(
                instruction_ids=self.instruction_ids[idx],
                instruction_attention_mask=self.instruction_attention_mask[idx],
                response_ids=self.response_ids[idx],
                response_attention_mask=self.response_attention_mask[idx],
                labels=self.labels[idx],
            )
        else:
            # Tokenize on-the-fly
            example = self.examples[idx]
            return self.tokenizer.tokenize_pair(
                example["instruction"],
                example["response"],
            )


def save_preprocessed_dataset(
    examples: List[TokenizedExample],
    output_path: Path,
) -> None:
    """Save preprocessed dataset to disk.

    Args:
        examples: List of TokenizedExample
        output_path: Output file path (.pt)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stack all tensors
    data = {
        "instruction_ids": torch.stack([e.instruction_ids for e in examples]),
        "instruction_attention_mask": torch.stack(
            [e.instruction_attention_mask for e in examples]
        ),
        "response_ids": torch.stack([e.response_ids for e in examples]),
        "response_attention_mask": torch.stack(
            [e.response_attention_mask for e in examples]
        ),
        "labels": torch.stack([e.labels for e in examples]),
    }

    torch.save(data, output_path)
    logger.info(f"Saved {len(examples)} examples to {output_path}")


def save_jsonl_dataset(
    examples: List[InstructionResponse],
    output_path: Path,
) -> None:
    """Save instruction-response pairs to JSONL.

    Args:
        examples: List of InstructionResponse
        output_path: Output file path (.jsonl)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in examples:
            json.dump(
                {
                    "instruction": example.instruction,
                    "response": example.response,
                    "source": example.source,
                },
                f,
            )
            f.write("\n")

    logger.info(f"Saved {len(examples)} examples to {output_path}")
