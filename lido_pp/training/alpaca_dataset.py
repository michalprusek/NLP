"""
Alpaca Dataset Loading for FlowPO Pre-training.

Loads the Stanford Alpaca dataset (~52k instruction-following samples)
and prepares it for TFA, Cross-Attention Decoder, and Flow-DiT training.

Dataset: tatsu-lab/alpaca on HuggingFace Hub
Format: instruction + optional input -> output
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm


def load_alpaca_dataset(
    max_samples: Optional[int] = None,
    include_output: bool = False,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Load Alpaca dataset from HuggingFace.

    Args:
        max_samples: Maximum number of samples to load (None = all ~52k)
        include_output: Whether to include the output in the text
        cache_dir: HuggingFace cache directory

    Returns:
        List of instruction strings
    """
    from datasets import load_dataset

    print(f"Loading Alpaca dataset from HuggingFace...")

    dataset = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)

    instructions = []
    for sample in tqdm(dataset, desc="Processing Alpaca"):
        # Combine instruction and input
        text = sample["instruction"]
        if sample["input"] and sample["input"].strip():
            text += f"\n\nInput: {sample['input']}"

        if include_output and sample["output"]:
            text += f"\n\nOutput: {sample['output']}"

        instructions.append(text)

        if max_samples and len(instructions) >= max_samples:
            break

    print(f"Loaded {len(instructions)} Alpaca instructions")
    return instructions


def load_ultrachat_dataset(
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Load UltraChat dataset from HuggingFace.

    Args:
        max_samples: Maximum number of samples
        cache_dir: HuggingFace cache directory

    Returns:
        List of instruction strings (first turn of each conversation)
    """
    from datasets import load_dataset

    print(f"Loading UltraChat dataset from HuggingFace...")

    # UltraChat-200k is a curated subset
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        cache_dir=cache_dir,
    )

    instructions = []
    for sample in tqdm(dataset, desc="Processing UltraChat"):
        # Get first user message from conversation
        messages = sample.get("messages", [])
        if messages and messages[0].get("role") == "user":
            instructions.append(messages[0]["content"])

        if max_samples and len(instructions) >= max_samples:
            break

    print(f"Loaded {len(instructions)} UltraChat instructions")
    return instructions


def load_openorca_dataset(
    max_samples: Optional[int] = 500000,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Load OpenOrca dataset from HuggingFace.

    OpenOrca contains diverse instruction-following data with Chain of Thought
    reasoning, which is essential for math and logical reasoning tasks.

    Args:
        max_samples: Maximum number of samples (default 500k)
        cache_dir: HuggingFace cache directory

    Returns:
        List of instruction strings
    """
    from datasets import load_dataset

    print(f"Loading OpenOrca dataset from HuggingFace (streaming)...")

    # Use streaming to avoid downloading entire dataset if only need subset
    dataset = load_dataset(
        "Open-Orca/OpenOrca",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )

    instructions = []
    for sample in tqdm(dataset, desc="Processing OpenOrca", total=max_samples):
        # OpenOrca has 'question' field for instruction
        text = sample.get("question", sample.get("instruction", ""))

        if text and len(text.strip()) > 20:
            instructions.append(text.strip())

        if max_samples and len(instructions) >= max_samples:
            break

    print(f"Loaded {len(instructions)} OpenOrca instructions")
    return instructions


def load_codealpaca_dataset(
    max_samples: Optional[int] = 200000,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Load code instruction datasets (CodeAlpaca + Glaive Code Assistant).

    Code instructions are critical for teaching the model syntax patterns
    which helps preserve code structure during latent space optimization.

    Args:
        max_samples: Maximum number of samples (default 200k)
        cache_dir: HuggingFace cache directory

    Returns:
        List of code instruction strings
    """
    from datasets import load_dataset

    instructions = []

    # 1. CodeAlpaca-20k
    print(f"Loading CodeAlpaca-20k...")
    try:
        dataset = load_dataset(
            "sahil2801/CodeAlpaca-20k",
            split="train",
            cache_dir=cache_dir,
        )
        for sample in tqdm(dataset, desc="Processing CodeAlpaca"):
            text = sample.get("instruction", "")
            if sample.get("input"):
                text += f"\n\nInput: {sample['input']}"

            if text and len(text.strip()) > 20:
                instructions.append(text.strip())

            if max_samples and len(instructions) >= max_samples:
                break
        print(f"Added {len(instructions)} from CodeAlpaca-20k")
    except Exception as e:
        print(f"Warning: Could not load CodeAlpaca-20k: {e}")

    # 2. Glaive Code Assistant (if need more samples)
    if max_samples is None or len(instructions) < max_samples:
        print(f"Loading Glaive Code Assistant...")
        try:
            remaining = (max_samples - len(instructions)) if max_samples else None
            dataset = load_dataset(
                "glaiveai/glaive-code-assistant",
                split="train",
                streaming=True,
                cache_dir=cache_dir,
            )
            count = 0
            for sample in tqdm(dataset, desc="Processing Glaive", total=remaining):
                # Glaive uses 'question' field
                text = sample.get("question", sample.get("instruction", ""))

                if text and len(text.strip()) > 20:
                    instructions.append(text.strip())
                    count += 1

                if remaining and count >= remaining:
                    break
            print(f"Added {count} from Glaive Code Assistant")
        except Exception as e:
            print(f"Warning: Could not load Glaive Code Assistant: {e}")

    print(f"Loaded {len(instructions)} total code instructions")
    return instructions


def load_sharegpt_dataset(
    max_samples: Optional[int] = 100000,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Load real human conversation dataset (LMSYS-Chat-1M or WildChat).

    These datasets contain real user prompts which are often more complex
    and diverse than synthetic instructions. Important for robustness.

    Args:
        max_samples: Maximum number of samples (default 100k)
        cache_dir: HuggingFace cache directory

    Returns:
        List of instruction strings (first human turn of each conversation)
    """
    from datasets import load_dataset

    instructions = []

    # Try LMSYS-Chat-1M first (cleaner, more reliable)
    print(f"Loading LMSYS-Chat-1M dataset (real conversations)...")
    try:
        dataset = load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )

        for sample in tqdm(dataset, desc="Processing LMSYS-Chat", total=max_samples):
            # LMSYS has 'conversation' field with list of messages
            convos = sample.get("conversation", [])

            # Get first human message
            for turn in convos:
                if turn.get("role") == "user" and turn.get("content"):
                    text = turn["content"].strip()
                    if len(text) > 20 and len(text) < 2000:  # Filter very long prompts
                        instructions.append(text)
                    break

            if max_samples and len(instructions) >= max_samples:
                break

        print(f"Loaded {len(instructions)} LMSYS-Chat instructions")
        return instructions

    except Exception as e:
        print(f"Warning: Could not load LMSYS-Chat-1M: {e}")

    # Fallback to WildChat
    print("Trying WildChat dataset...")
    try:
        dataset = load_dataset(
            "allenai/WildChat-1M",
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )

        for sample in tqdm(dataset, desc="Processing WildChat", total=max_samples):
            # WildChat has 'conversation' field
            convos = sample.get("conversation", [])

            for turn in convos:
                if turn.get("role") == "user" and turn.get("content"):
                    text = turn["content"].strip()
                    if len(text) > 20 and len(text) < 2000:
                        instructions.append(text)
                    break

            if max_samples and len(instructions) >= max_samples:
                break

        print(f"Loaded {len(instructions)} WildChat instructions")
        return instructions

    except Exception as e:
        print(f"Warning: Could not load WildChat: {e}")

    print(f"Warning: No real conversation dataset available. Returning {len(instructions)} samples.")
    return instructions


def load_universal_dataset(
    openorca_samples: int = 500000,
    ultrachat_samples: int = 500000,
    codealpaca_samples: int = 200000,
    sharegpt_samples: int = 100000,
    alpaca_samples: int = 52000,
    existing_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    deduplicate: bool = True,
) -> List[str]:
    """
    Load universal dataset combining multiple sources for production training.

    Target: 1.5-2M diverse, high-quality instructions covering:
    - Reasoning (OpenOrca)
    - Conversation (UltraChat)
    - Code (CodeAlpaca, Glaive)
    - Real prompts (ShareGPT)
    - General (Alpaca)

    Args:
        openorca_samples: Max OpenOrca samples (500k default)
        ultrachat_samples: Max UltraChat samples (500k default)
        codealpaca_samples: Max code samples (200k default)
        sharegpt_samples: Max ShareGPT samples (100k default)
        alpaca_samples: Max Alpaca samples (52k default)
        existing_path: Path to existing embeddings .pt to merge with
        cache_dir: HuggingFace cache directory
        deduplicate: Whether to deduplicate instructions (MD5 hash)

    Returns:
        List of deduplicated instruction strings
    """
    import hashlib

    instructions = []
    seen_hashes = set()

    def add_instructions(new_instructions: List[str], source: str):
        """Add instructions with optional deduplication."""
        added = 0
        for inst in new_instructions:
            if deduplicate:
                h = hashlib.md5(inst.lower().strip().encode()).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
            instructions.append(inst)
            added += 1
        print(f"  → Added {added} unique from {source} (total: {len(instructions)})")

    # Load existing instructions if provided
    if existing_path:
        print(f"Loading existing instructions from {existing_path}...")
        import torch
        data = torch.load(existing_path, weights_only=False)
        existing = data.get("instructions", [])
        if existing:
            add_instructions(existing, "existing")

    # 1. OpenOrca (reasoning, CoT)
    if openorca_samples > 0:
        orca = load_openorca_dataset(max_samples=openorca_samples, cache_dir=cache_dir)
        add_instructions(orca, "OpenOrca")

    # 2. UltraChat (conversation)
    if ultrachat_samples > 0:
        ultra = load_ultrachat_dataset(max_samples=ultrachat_samples, cache_dir=cache_dir)
        add_instructions(ultra, "UltraChat")

    # 3. Code (syntax)
    if codealpaca_samples > 0:
        code = load_codealpaca_dataset(max_samples=codealpaca_samples, cache_dir=cache_dir)
        add_instructions(code, "Code")

    # 4. ShareGPT (real prompts)
    if sharegpt_samples > 0:
        share = load_sharegpt_dataset(max_samples=sharegpt_samples, cache_dir=cache_dir)
        add_instructions(share, "ShareGPT")

    # 5. Alpaca (general)
    if alpaca_samples > 0:
        alpaca = load_alpaca_dataset(max_samples=alpaca_samples, cache_dir=cache_dir)
        add_instructions(alpaca, "Alpaca")

    print(f"\n{'='*60}")
    print(f"Universal dataset: {len(instructions)} unique instructions")
    print(f"{'='*60}")

    return instructions


class AlpacaInstructionDataset(Dataset):
    """
    PyTorch Dataset for Alpaca instructions with optional pre-computed embeddings.
    """

    def __init__(
        self,
        instructions: List[str],
        embeddings: Optional[np.ndarray] = None,
        latents: Optional[np.ndarray] = None,
    ):
        """
        Args:
            instructions: List of instruction texts
            embeddings: Pre-computed SONAR embeddings (N, 1024)
            latents: Pre-computed TFA latents (N, 128/256)
        """
        self.instructions = instructions
        self.embeddings = embeddings
        self.latents = latents

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Dict:
        item = {
            "text": self.instructions[idx],
            "idx": idx,
        }

        if self.embeddings is not None:
            item["embedding"] = torch.tensor(
                self.embeddings[idx], dtype=torch.float32
            )

        if self.latents is not None:
            item["latent"] = torch.tensor(
                self.latents[idx], dtype=torch.float32
            )

        return item

    @classmethod
    def from_precomputed(
        cls,
        embeddings_path: str,
        instructions_path: Optional[str] = None,
    ) -> "AlpacaInstructionDataset":
        """
        Load dataset from pre-computed embeddings file.

        Args:
            embeddings_path: Path to .pt file with embeddings
            instructions_path: Optional path to JSON file with instructions

        Returns:
            AlpacaInstructionDataset instance
        """
        print(f"Loading pre-computed embeddings from {embeddings_path}")
        data = torch.load(embeddings_path, weights_only=False)

        embeddings = data["embeddings"]
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()

        instructions = data.get("instructions", [])

        # Load instructions from separate file if provided
        if instructions_path and Path(instructions_path).exists():
            with open(instructions_path, "r") as f:
                instructions = json.load(f)

        # If no instructions, create placeholders
        if not instructions:
            instructions = [f"instruction_{i}" for i in range(len(embeddings))]

        return cls(instructions=instructions, embeddings=embeddings)


class EmbeddingDataset(Dataset):
    """
    Simple dataset of pre-computed embeddings for TFA training.
    """

    def __init__(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings: Tensor of shape (N, embedding_dim)
        """
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]

    @classmethod
    def from_file(cls, path: str) -> "EmbeddingDataset":
        """Load embeddings from file."""
        data = torch.load(path, weights_only=False)

        if isinstance(data, dict):
            embeddings = data["embeddings"]
        else:
            embeddings = data

        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

        return cls(embeddings)


class FlowMatchingDataset(Dataset):
    """
    Dataset for Flow Matching training with (x_0, x_1, context) pairs.

    x_0: Noise (generated on-the-fly)
    x_1: Target latent (from TFA encoder)
    context: Original embedding for conditioning
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        latents: torch.Tensor,
    ):
        """
        Args:
            embeddings: Original SONAR embeddings (N, 1024) for context
            latents: TFA-encoded latents (N, 128/256) as targets
        """
        assert len(embeddings) == len(latents), "Embeddings and latents must match"
        self.embeddings = embeddings
        self.latents = latents

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_1": self.latents[idx],  # Target
            "context": self.embeddings[idx],  # Conditioning
            # x_0 (noise) is generated in the training loop
        }

    @classmethod
    def from_files(
        cls,
        embeddings_path: str,
        latents_path: Optional[str] = None,
    ) -> "FlowMatchingDataset":
        """
        Load from pre-computed files.

        Args:
            embeddings_path: Path to embeddings .pt file
            latents_path: Path to latents .pt file (or included in embeddings file)
        """
        data = torch.load(embeddings_path, weights_only=False)

        if isinstance(data, dict):
            embeddings = data["embeddings"]
            latents = data.get("latents")
        else:
            embeddings = data
            latents = None

        if latents is None and latents_path:
            latents = torch.load(latents_path, weights_only=False)
            if isinstance(latents, dict):
                latents = latents["latents"]

        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        if isinstance(latents, np.ndarray):
            latents = torch.tensor(latents, dtype=torch.float32)

        return cls(embeddings, latents)


def save_alpaca_instructions(
    instructions: List[str],
    output_path: str,
) -> None:
    """Save instructions to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(instructions, f, indent=2)
    print(f"Saved {len(instructions)} instructions to {output_path}")


def load_combined_dataset(
    alpaca_samples: Optional[int] = None,
    ultrachat_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Load combined Alpaca + UltraChat dataset for diverse instruction training.

    Args:
        alpaca_samples: Max Alpaca samples (None = all ~52k)
        ultrachat_samples: Max UltraChat samples (None = all ~200k)
        cache_dir: HuggingFace cache directory

    Returns:
        Combined list of instruction strings
    """
    instructions = []

    # Load Alpaca
    alpaca = load_alpaca_dataset(max_samples=alpaca_samples, cache_dir=cache_dir)
    instructions.extend(alpaca)
    print(f"Added {len(alpaca)} Alpaca instructions")

    # Load UltraChat
    if ultrachat_samples is None or ultrachat_samples > 0:
        ultrachat = load_ultrachat_dataset(max_samples=ultrachat_samples, cache_dir=cache_dir)
        instructions.extend(ultrachat)
        print(f"Added {len(ultrachat)} UltraChat instructions")

    print(f"Total combined dataset: {len(instructions)} instructions")
    return instructions


def create_train_val_split(
    instructions: List[str],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split instructions into train/val sets.

    Args:
        instructions: Full list of instructions
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        (train_instructions, val_instructions)
    """
    import random

    random.seed(seed)
    shuffled = instructions.copy()
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val = shuffled[:val_size]
    train = shuffled[val_size:]

    return train, val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and inspect Alpaca dataset")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="lido_pp/data/alpaca_sample.json")
    args = parser.parse_args()

    # Test loading
    instructions = load_alpaca_dataset(max_samples=args.max_samples)

    print(f"\nSample instructions:")
    for i in range(min(3, len(instructions))):
        print(f"\n--- {i+1} ---")
        print(instructions[i][:200] + "..." if len(instructions[i]) > 200 else instructions[i])

    # Save sample
    save_alpaca_instructions(instructions, args.output)
