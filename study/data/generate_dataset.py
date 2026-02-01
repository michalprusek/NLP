"""Generate 10K verbosed sampling (VS) dataset with SONAR embeddings.

This script generates meta-cognitive instructions using the VS prompting approach:
1. Sample 3-5 GSM8K problems
2. Ask LLM to generate instructions about how to solve such problems
3. Embed instructions using SONAR encoder
4. Deduplicate by semantic similarity

Usage:
    uv run python study/data/generate_dataset.py             # Full 10K generation
    uv run python study/data/generate_dataset.py --dry-run   # Generate 100 samples for testing
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from study.data import DEFAULT_VS_10K_PATH, GSM8K_PATH, PROJECT_ROOT


def load_gsm8k_questions(gsm8k_path: str) -> List[str]:
    """Load GSM8K training questions."""
    train_path = Path(gsm8k_path) / "train.json"
    with open(train_path) as f:
        data = json.load(f)
    return [item["question"] for item in data]


def create_vs_prompt(questions: List[str]) -> str:
    """Create a verbosed sampling prompt from example questions.

    The prompt asks the LLM to generate meta-cognitive instructions
    about how to approach solving math word problems like the examples.
    """
    examples = "\n\n".join([f"Example {i+1}: {q}" for i, q in enumerate(questions)])

    return f"""Here are some math word problems:

{examples}

Based on these examples, generate 5 different high-level cognitive instructions that would help someone solve similar math problems. Each instruction should be:
- A general problem-solving tip or thinking strategy
- Applicable to various math word problems, not specific to these examples
- Focused on the reasoning process, not the specific math operations
- Written as a directive (e.g., "Consider...", "Think about...", "Check whether...")

Generate exactly 5 instructions, one per line, without numbering or bullet points."""


def parse_instructions(response: str) -> List[str]:
    """Parse LLM response into individual instructions."""
    if response is None:
        return []

    lines = response.strip().split("\n")
    instructions = []

    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Remove common prefixes (numbering, bullets)
        if line[0].isdigit() and len(line) > 1 and line[1] in ".):":
            line = line[2:].strip()
        elif line[0] in "-*":
            line = line[1:].strip()
        # Skip very short lines
        if len(line) > 20:
            instructions.append(line)

    return instructions


def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    return torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0), emb2.unsqueeze(0)
    ).item()


def is_duplicate(
    new_embedding: torch.Tensor,
    existing_embeddings: torch.Tensor,
    threshold: float = 0.95
) -> bool:
    """Check if new embedding is too similar to any existing embedding."""
    if existing_embeddings.shape[0] == 0:
        return False

    # Compute similarities in batch
    similarities = torch.nn.functional.cosine_similarity(
        new_embedding.unsqueeze(0), existing_embeddings
    )
    return (similarities > threshold).any().item()


def generate_vs_dataset(
    target_count: int,
    gsm8k_path: str,
    output_path: str,
    model: str = "qwen",
    backend: str = "vllm",
    temperature: float = 1.0,
    examples_per_batch: int = 4,
    batch_size: int = 256,
    dedup_threshold: float = 0.95,
    checkpoint_interval: int = 1000,
    seed: int = 42,
    device: str = "cuda:0",
) -> dict:
    """Generate VS dataset with SONAR embeddings.

    Args:
        target_count: Number of unique instructions to generate
        gsm8k_path: Path to GSM8K dataset
        output_path: Where to save the dataset
        model: LLM model name or alias
        backend: LLM backend (vllm, openai, etc.)
        temperature: Sampling temperature for LLM
        examples_per_batch: Number of GSM8K examples per VS prompt
        batch_size: Batch size for SONAR encoding
        dedup_threshold: Cosine similarity threshold for deduplication
        checkpoint_interval: Save checkpoint every N samples
        seed: Random seed
        device: CUDA device for SONAR

    Returns:
        Dataset dictionary with embeddings, instructions, sources, config, stats
    """
    from shared.llm_client import create_llm_client
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Generating VS dataset with {target_count} target samples")
    print(f"Output: {output_path}")
    print(f"Model: {model}, Backend: {backend}")
    print(f"Device: {device}")

    # Load GSM8K questions
    questions = load_gsm8k_questions(gsm8k_path)
    print(f"Loaded {len(questions)} GSM8K questions")

    # Initialize LLM client
    print("Initializing LLM client...")
    llm = create_llm_client(model, backend=backend)

    # Initialize SONAR encoder
    print("Initializing SONAR encoder...")
    encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(device),
    )

    # Check for existing checkpoint
    checkpoint_path = Path(output_path).with_suffix(".checkpoint.pt")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        instructions = checkpoint["instructions"]
        embeddings = checkpoint["embeddings"]
        print(f"Resuming from {len(instructions)} instructions")
    else:
        instructions = []
        embeddings = torch.empty(0, 1024, dtype=torch.float32)

    # Progress bar
    pbar = tqdm(total=target_count, initial=len(instructions), desc="Generating")

    # Generate instructions
    max_iterations = target_count * 5  # Safety limit
    iteration = 0

    while len(instructions) < target_count and iteration < max_iterations:
        iteration += 1

        # Sample random GSM8K questions
        sampled_questions = random.sample(questions, examples_per_batch)

        # Create VS prompt
        prompt = create_vs_prompt(sampled_questions)

        # Generate instructions
        response = llm.generate(prompt, temperature=temperature, max_new_tokens=512)
        new_instructions = parse_instructions(response)

        # Process each instruction
        for instr in new_instructions:
            if len(instructions) >= target_count:
                break

            # Encode instruction
            with torch.no_grad():
                emb = encoder.predict([instr], source_lang="eng_Latn")
                emb = emb.float().cpu()  # Convert to float32 and move to CPU

            # Check for duplicates
            if is_duplicate(emb[0], embeddings, dedup_threshold):
                continue

            # Add to dataset
            instructions.append(instr)
            embeddings = torch.cat([embeddings, emb], dim=0)
            pbar.update(1)

            # Checkpoint
            if len(instructions) % checkpoint_interval == 0:
                checkpoint = {
                    "instructions": instructions,
                    "embeddings": embeddings,
                }
                torch.save(checkpoint, checkpoint_path)
                pbar.set_postfix({"saved": len(instructions)})

    pbar.close()

    # Cleanup LLM
    if hasattr(llm, "cleanup"):
        llm.cleanup()

    # Build final dataset
    config = {
        "target_count": target_count,
        "gsm8k_path": gsm8k_path,
        "model": model,
        "backend": backend,
        "temperature": temperature,
        "examples_per_batch": examples_per_batch,
        "batch_size": batch_size,
        "dedup_threshold": dedup_threshold,
        "seed": seed,
        "device": device,
        "generated_at": datetime.now().isoformat(),
    }

    stats = {
        "n_text_instructions": len(instructions),
        "total": len(instructions),
        "embedding_dim": 1024,
        "iterations": iteration,
    }

    dataset = {
        "embeddings": embeddings,
        "instructions": instructions,
        "sources": {"verbalized_sampling": len(instructions)},
        "config": config,
        "stats": stats,
    }

    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    print(f"Saved dataset to {output_path}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embeddings dtype: {embeddings.dtype}")
    print(f"  Instructions: {len(instructions)}")

    # Remove checkpoint if successful
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Removed checkpoint file")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate VS dataset with SONAR embeddings")
    parser.add_argument("--target", type=int, default=10000, help="Target number of samples")
    parser.add_argument("--output", type=str, default=DEFAULT_VS_10K_PATH, help="Output path")
    parser.add_argument("--gsm8k-path", type=str, default=GSM8K_PATH, help="GSM8K dataset path")
    parser.add_argument("--model", type=str, default="qwen", help="LLM model name or alias")
    parser.add_argument("--backend", type=str, default="vllm", help="LLM backend")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--examples-per-batch", type=int, default=4, help="GSM8K examples per prompt")
    parser.add_argument("--batch-size", type=int, default=256, help="SONAR encoding batch size")
    parser.add_argument("--dedup-threshold", type=float, default=0.95, help="Deduplication threshold")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--dry-run", action="store_true", help="Generate only 100 samples for testing")

    args = parser.parse_args()

    # Adjust for dry run
    if args.dry_run:
        args.target = 100
        args.output = args.output.replace(".pt", "_dryrun.pt")
        args.checkpoint_interval = 50
        print("=== DRY RUN MODE: Generating 100 samples ===")

    # Change to project root for relative paths
    os.chdir(PROJECT_ROOT)

    generate_vs_dataset(
        target_count=args.target,
        gsm8k_path=args.gsm8k_path,
        output_path=args.output,
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        examples_per_batch=args.examples_per_batch,
        batch_size=args.batch_size,
        dedup_threshold=args.dedup_threshold,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
