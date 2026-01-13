"""
Data Preparation for LID-O++ Training.

This module handles:
1. Loading and preprocessing instruction datasets
2. Creating training/validation splits
3. Batch generation for Flow Matching training
4. Integration with GSM8K evaluation

Dataset sources:
- APE instructions (for projector pre-training)
- GSM8K (for main optimization)
- Custom instruction sets
"""

import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator, Union
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


@dataclass
class InstructionSample:
    """Single instruction sample."""
    text: str
    embedding: Optional[np.ndarray] = None
    latent: Optional[np.ndarray] = None
    error_rate: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class FlowMatchingBatch:
    """Batch for Flow Matching training."""
    # Source (noise)
    x_0: torch.Tensor  # (B, latent_dim)
    # Target (data latents)
    x_1: torch.Tensor  # (B, latent_dim)
    # Optional context
    context: Optional[torch.Tensor] = None  # (B, num_ctx, ctx_dim)
    # Original texts (for debugging)
    texts: Optional[List[str]] = None


class InstructionDataset(Dataset):
    """Dataset of instruction texts with optional embeddings."""

    def __init__(
        self,
        instructions: List[str],
        embeddings: Optional[np.ndarray] = None,
        latents: Optional[np.ndarray] = None,
        error_rates: Optional[List[float]] = None,
    ):
        """
        Args:
            instructions: List of instruction texts
            embeddings: Pre-computed embeddings (N, embed_dim)
            latents: Pre-computed latents (N, latent_dim)
            error_rates: Evaluation results (N,)
        """
        self.instructions = instructions
        self.embeddings = embeddings
        self.latents = latents
        self.error_rates = error_rates

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Dict:
        item = {"text": self.instructions[idx], "idx": idx}

        if self.embeddings is not None:
            item["embedding"] = self.embeddings[idx]

        if self.latents is not None:
            item["latent"] = self.latents[idx]

        if self.error_rates is not None:
            item["error_rate"] = self.error_rates[idx]

        return item

    @classmethod
    def from_json(cls, path: str, text_key: str = "instruction") -> "InstructionDataset":
        """Load dataset from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            if isinstance(data[0], str):
                instructions = data
            else:
                instructions = [d[text_key] for d in data]
        elif isinstance(data, dict):
            instructions = list(data.values()) if all(isinstance(v, str) for v in data.values()) else [d[text_key] for d in data.values()]
        else:
            raise ValueError(f"Unsupported JSON format in {path}")

        return cls(instructions)


class FlowMatchingDataLoader:
    """
    DataLoader for Flow Matching training.

    Generates batches of (x_0, x_1, context) where:
    - x_0: Random noise from N(0, I)
    - x_1: Latent vectors from encoder
    - context: Task/instruction context embeddings
    """

    def __init__(
        self,
        dataset: InstructionDataset,
        encoder,  # GritLMUnifiedEncoder
        batch_size: int = 64,
        latent_dim: int = 32,
        context_dim: int = 768,
        num_context_tokens: int = 4,
        shuffle: bool = True,
        device: str = "cuda",
        cache_embeddings: bool = True,
    ):
        """
        Args:
            dataset: Instruction dataset
            encoder: GritLM encoder for computing embeddings
            batch_size: Batch size
            latent_dim: Flow latent dimension (for x_0 noise)
            context_dim: Context embedding dimension
            num_context_tokens: Number of context tokens
            shuffle: Shuffle dataset
            device: Device for tensors
            cache_embeddings: Cache embeddings to avoid recomputation
        """
        self.dataset = dataset
        self.encoder = encoder
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_context_tokens = num_context_tokens
        self.shuffle = shuffle
        self.device = device
        self.cache_embeddings = cache_embeddings

        # Cache for embeddings
        self._embedding_cache: Optional[torch.Tensor] = None

        # Pre-compute embeddings if caching enabled
        if cache_embeddings and dataset.embeddings is None:
            self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Pre-compute embeddings for all instructions."""
        print(f"Pre-computing embeddings for {len(self.dataset)} instructions...")

        embeddings = self.encoder.encode_batch(
            self.dataset.instructions,
            batch_size=32,
            show_progress=True,
        )

        self._embedding_cache = torch.tensor(embeddings, dtype=torch.float32)
        self.dataset.embeddings = embeddings
        print(f"Embeddings cached: {self._embedding_cache.shape}")

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[FlowMatchingBatch]:
        """Generate batches for Flow Matching training."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self._create_batch(batch_indices)

    def _create_batch(self, indices: List[int]) -> FlowMatchingBatch:
        """Create a single batch from indices."""
        batch_size = len(indices)

        # Get embeddings (as x_1 targets)
        if self._embedding_cache is not None:
            embeddings = self._embedding_cache[indices].to(self.device)
        elif self.dataset.embeddings is not None:
            embeddings = torch.tensor(
                self.dataset.embeddings[indices],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            # Compute on-the-fly
            texts = [self.dataset.instructions[i] for i in indices]
            emb_np = self.encoder.encode_batch(texts, batch_size=len(texts), show_progress=False)
            embeddings = torch.tensor(emb_np, dtype=torch.float32, device=self.device)

        # x_1: Use embeddings as targets (768D)
        # Note: For Flow Matching in latent space, we need to project these
        # through a learned projection or use them directly as context
        x_1 = embeddings  # (B, 768)

        # x_0: Random noise (same dimension as x_1 for direct flow)
        x_0 = torch.randn(batch_size, x_1.shape[1], device=self.device)

        # Context: Reshape embeddings as context tokens
        # (B, 768) -> (B, num_ctx, ctx_dim)
        context = embeddings.unsqueeze(1).expand(-1, self.num_context_tokens, -1)

        # Get texts for debugging
        texts = [self.dataset.instructions[i] for i in indices]

        return FlowMatchingBatch(
            x_0=x_0,
            x_1=x_1,
            context=context,
            texts=texts,
        )


class LatentFlowDataLoader:
    """
    DataLoader for Flow Matching in compressed latent space.

    For efficiency, we can train the flow in a lower-dimensional space:
    - Embedding (768D) → Latent (32D) via learned projection
    - Flow operates in 32D space
    - Decode back to 768D for text generation

    This is similar to Latent Diffusion Models but for flow matching.
    """

    def __init__(
        self,
        dataset: InstructionDataset,
        encoder,  # GritLMUnifiedEncoder
        latent_projector: Optional[torch.nn.Module] = None,
        batch_size: int = 64,
        latent_dim: int = 32,
        shuffle: bool = True,
        device: str = "cuda",
    ):
        """
        Args:
            dataset: Instruction dataset
            encoder: GritLM encoder
            latent_projector: Optional projection 768D → latent_dim
            batch_size: Batch size
            latent_dim: Target latent dimension
            shuffle: Shuffle dataset
            device: Device
        """
        self.dataset = dataset
        self.encoder = encoder
        self.latent_projector = latent_projector
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.shuffle = shuffle
        self.device = device

        # Pre-compute embeddings
        self._precompute_embeddings()

        # Create simple linear projector if not provided
        if latent_projector is None:
            embed_dim = self._embedding_cache.shape[1]
            self.latent_projector = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, latent_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(latent_dim * 2, latent_dim),
            ).to(device)

    def _precompute_embeddings(self):
        """Pre-compute embeddings."""
        print(f"Pre-computing embeddings for {len(self.dataset)} instructions...")

        embeddings = self.encoder.encode_batch(
            self.dataset.instructions,
            batch_size=32,
            show_progress=True,
        )

        self._embedding_cache = torch.tensor(embeddings, dtype=torch.float32)
        print(f"Embeddings cached: {self._embedding_cache.shape}")

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[FlowMatchingBatch]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self._create_batch(batch_indices)

    def _create_batch(self, indices: List[int]) -> FlowMatchingBatch:
        batch_size = len(indices)

        # Get embeddings
        embeddings = self._embedding_cache[indices].to(self.device)

        # Project to latent space
        with torch.no_grad():
            latents = self.latent_projector(embeddings)  # (B, latent_dim)

        # x_1: Latent targets
        x_1 = latents

        # x_0: Random noise in latent space
        x_0 = torch.randn(batch_size, self.latent_dim, device=self.device)

        # Context: Original embeddings
        context = embeddings.unsqueeze(1)  # (B, 1, 768)

        texts = [self.dataset.instructions[i] for i in indices]

        return FlowMatchingBatch(x_0=x_0, x_1=x_1, context=context, texts=texts)


def load_ape_instructions(path: str = "lipo/data/ape_instructions.json") -> List[str]:
    """Load APE instructions for pre-training."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.values())
    except FileNotFoundError:
        print(f"Warning: APE instructions not found at {path}")
        return []


def load_gsm8k_dataset(
    train_path: str = "datasets/gsm8k/train.json",
    test_path: str = "datasets/gsm8k/test.json",
) -> Tuple[List[Dict], List[Dict]]:
    """Load GSM8K dataset."""
    train_data = []
    test_data = []

    try:
        with open(train_path, "r") as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: GSM8K train not found at {train_path}")

    try:
        with open(test_path, "r") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: GSM8K test not found at {test_path}")

    return train_data, test_data


def create_train_val_split(
    instructions: List[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split instructions into train/val sets."""
    random.seed(seed)
    shuffled = instructions.copy()
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_instructions = shuffled[:val_size]
    train_instructions = shuffled[val_size:]

    return train_instructions, val_instructions


if __name__ == "__main__":
    print("Testing Data Preparation...")

    # Test InstructionDataset
    print("\n1. Testing InstructionDataset...")
    instructions = [
        "Think step by step.",
        "Solve carefully.",
        "Show your work.",
        "Calculate the answer.",
    ]
    dataset = InstructionDataset(instructions)
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Sample: {dataset[0]}")

    # Test split
    print("\n2. Testing train/val split...")
    train, val = create_train_val_split(instructions * 10, val_ratio=0.2)
    print(f"   Train: {len(train)}, Val: {len(val)}")

    # Test APE loading
    print("\n3. Testing APE instructions loading...")
    ape = load_ape_instructions()
    if ape:
        print(f"   Loaded {len(ape)} APE instructions")
        print(f"   Sample: '{ape[0][:60]}...'")
    else:
        print("   APE instructions not available")

    print("\n[OK] Data preparation tests passed!")
