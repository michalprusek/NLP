#!/usr/bin/env python
"""
Generate diverse GSM8K-conditioned instructions using Verbalized Sampling.

This script implements a hybrid approach:
1. Verbalized Sampling (VS) - Generate instructions from tail of distribution (~60%)
2. Evol-Instruct mutations - Complexity mutations of high-performing seeds (~20%)
3. SLERP augmentation - Spherical interpolation in embedding space (~20%)

The generated instructions are conditioned on GSM8K Q/A pairs stratified by difficulty,
making them task-specific for math problem solving.

References:
- Verbalized Sampling: https://arxiv.org/abs/2510.01171
- Evol-Instruct (WizardLM): https://arxiv.org/abs/2304.12244
- inversedMixup for Embedding Augmentation: https://arxiv.org/html/2601.21543

Usage:
    # Full generation (~5000 instructions)
    python scripts/generate_gsm8k_instructions_vs.py \
        --output datasets/gsm8k_instructions_vs.pt \
        --target-vs 3000 --target-evol 1000 --target-slerp 1000 \
        --model Qwen/Qwen2.5-7B-Instruct

    # Quick test
    python scripts/generate_gsm8k_instructions_vs.py \
        --output datasets/gsm8k_instructions_vs_test.pt \
        --target-vs 100 --target-evol 50 --target-slerp 50 \
        --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GSM8KExample:
    """Single GSM8K Q/A pair with difficulty estimate."""
    question: str
    answer: str
    idx: int
    difficulty: str  # "easy", "medium", "hard"
    num_steps: int   # Estimated reasoning steps


# ============================================================================
# Difficulty Estimation
# ============================================================================

def estimate_difficulty(question: str, answer: str) -> Tuple[str, int]:
    """
    Estimate difficulty of a GSM8K problem based on heuristics.

    Uses:
    - Number of arithmetic operations in solution
    - Length of reasoning chain
    - Presence of multi-step dependencies

    Returns:
        Tuple of (difficulty_level, num_steps)
    """
    # Count arithmetic operations in the solution
    ops = len(re.findall(r'[\+\-\*/]', answer))

    # Count lines in solution (proxy for reasoning steps)
    lines = len([l for l in answer.split('\n') if l.strip()])

    # Count numbers mentioned (complexity indicator)
    numbers = len(re.findall(r'\d+', answer))

    # Estimate number of steps from "<<...>>" annotations or line count
    step_annotations = len(re.findall(r'<<.*?>>', answer))
    num_steps = max(step_annotations, lines // 2, 1)

    # Classify difficulty
    complexity_score = ops + numbers * 0.5 + lines * 0.3

    if complexity_score < 10:
        difficulty = "easy"
    elif complexity_score < 25:
        difficulty = "medium"
    else:
        difficulty = "hard"

    return difficulty, num_steps


def load_gsm8k_with_difficulty(
    dataset_path: str = "datasets/gsm8k",
    split: str = "train"
) -> List[GSM8KExample]:
    """Load GSM8K dataset with difficulty annotations."""
    from datasets import load_from_disk

    logger.info(f"Loading GSM8K {split} set from {dataset_path}...")
    ds = load_from_disk(dataset_path)[split]

    examples = []
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}

    for idx in range(len(ds)):
        item = ds[idx]
        difficulty, num_steps = estimate_difficulty(item["question"], item["answer"])

        examples.append(GSM8KExample(
            question=item["question"],
            answer=item["answer"],
            idx=idx,
            difficulty=difficulty,
            num_steps=num_steps
        ))
        difficulty_counts[difficulty] += 1

    logger.info(f"Loaded {len(examples)} examples")
    logger.info(f"Difficulty distribution: {difficulty_counts}")

    return examples


def stratified_sample(
    examples: List[GSM8KExample],
    n_samples: int,
    weights: Dict[str, float] = {"easy": 0.3, "medium": 0.4, "hard": 0.3}
) -> List[GSM8KExample]:
    """Sample examples with stratification by difficulty."""
    # Group by difficulty
    by_difficulty = {"easy": [], "medium": [], "hard": []}
    for ex in examples:
        by_difficulty[ex.difficulty].append(ex)

    sampled = []
    for diff, weight in weights.items():
        n = int(n_samples * weight)
        pool = by_difficulty[diff]
        if len(pool) < n:
            # Sample with replacement if not enough
            sampled.extend(random.choices(pool, k=n))
        else:
            sampled.extend(random.sample(pool, n))

    random.shuffle(sampled)
    return sampled[:n_samples]


# ============================================================================
# Verbalized Sampling (VS)
# ============================================================================

VS_PROMPT_TEMPLATE = """<instructions>
You are an expert at creating instructions that help AI models solve math word problems.
Your goal is to generate diverse, high-quality instructions from the TAIL of the probability distribution.

Given the following GSM8K math problems and their solutions:

{examples}

Generate {n_instructions} different instructions that would help an AI solve these types of math word problems.

REQUIREMENTS:
- Each instruction should be 1-3 sentences
- Sample from the TAILS of the distribution - avoid common/obvious instructions
- Each instruction should have probability < 0.10
- Include variety: some formal, some conversational, some step-focused, some verification-focused
- Focus on mathematical reasoning, step-by-step thinking, and error checking

OUTPUT FORMAT:
Provide each instruction within a separate <response> tag with <text> and <probability> fields:
<response>
<text>Your instruction here</text>
<probability>0.05</probability>
</response>

Generate {n_instructions} responses now:
</instructions>"""


def format_example(ex: GSM8KExample, label: str) -> str:
    """Format a single GSM8K example for the prompt."""
    # Truncate long answers for context efficiency
    answer_preview = ex.answer[:500] + "..." if len(ex.answer) > 500 else ex.answer
    return f"""EXAMPLE ({label}):
Q: {ex.question}
A: {answer_preview}"""


def parse_vs_response(response: str) -> List[Tuple[str, float]]:
    """
    Parse Verbalized Sampling response format.

    Returns:
        List of (instruction_text, probability) tuples
    """
    pattern = r'<response>.*?<text>(.*?)</text>.*?<probability>([\d.]+)</probability>.*?</response>'
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        logger.warning(f"Failed to parse VS response, no matches found. Response preview: {response[:200]}")

    results = []
    for text, prob in matches:
        text = text.strip()
        try:
            prob_float = float(prob)
        except ValueError:
            prob_float = 0.05  # Default probability

        # Filter valid instructions
        if 20 < len(text) < 500 and prob_float <= 0.15:
            results.append((text, prob_float))

    return results


def generate_instructions_vs(
    llm_client,
    gsm8k_examples: List[GSM8KExample],
    target_count: int = 3000,
    examples_per_batch: int = 3,
    instructions_per_call: int = 5,
    difficulty_weights: Dict[str, float] = {"easy": 0.3, "medium": 0.4, "hard": 0.3},
    temperature: float = 1.0,
) -> List[Tuple[str, float]]:
    """
    Generate diverse instructions using Verbalized Sampling.

    Uses stratified Q/A sampling by difficulty level and prompts LLM
    to sample from the tail of the instruction distribution.

    Args:
        llm_client: LLM client for generation
        gsm8k_examples: GSM8K training examples with difficulty labels
        target_count: Target number of instructions to generate
        examples_per_batch: Number of Q/A pairs per LLM call
        instructions_per_call: Instructions requested per LLM call
        difficulty_weights: Sampling weights for difficulty levels
        temperature: LLM sampling temperature (higher = more diverse)

    Returns:
        List of (instruction, probability) tuples
    """
    logger.info(f"Generating {target_count} instructions via Verbalized Sampling...")
    logger.info(f"  Examples per batch: {examples_per_batch}")
    logger.info(f"  Instructions per call: {instructions_per_call}")
    logger.info(f"  Temperature: {temperature}")

    all_instructions = []
    seen_instructions = set()  # For deduplication
    consecutive_failures = 0

    # Calculate number of LLM calls needed
    n_calls = (target_count // instructions_per_call) + 1

    pbar = tqdm(range(n_calls), desc="VS Generation")
    for _ in pbar:
        # Stratified sample of examples
        batch_examples = stratified_sample(
            gsm8k_examples,
            examples_per_batch,
            weights=difficulty_weights
        )

        # Format examples with difficulty labels
        example_texts = []
        for i, ex in enumerate(batch_examples, 1):
            example_texts.append(format_example(ex, f"{ex.difficulty}, {ex.num_steps} steps"))

        # Build prompt
        prompt = VS_PROMPT_TEMPLATE.format(
            examples="\n\n".join(example_texts),
            n_instructions=instructions_per_call
        )

        # Generate
        try:
            response = llm_client.generate(
                prompt,
                temperature=temperature,
                max_new_tokens=2048
            )

            # Parse response
            parsed = parse_vs_response(response)

            # Deduplicate and add
            for text, prob in parsed:
                if text not in seen_instructions:
                    seen_instructions.add(text)
                    all_instructions.append((text, prob))

            pbar.set_postfix({"total": len(all_instructions)})

        except Exception as e:
            consecutive_failures += 1
            logger.error(f"VS generation failed (attempt {consecutive_failures}): {e}")
            if consecutive_failures >= 10:
                logger.error("Too many consecutive VS failures, aborting")
                break
            continue
        else:
            consecutive_failures = 0  # Reset on success

        # Early exit if we have enough
        if len(all_instructions) >= target_count:
            break

    logger.info(f"Generated {len(all_instructions)} unique instructions via VS")
    return all_instructions[:target_count]


# ============================================================================
# Evol-Instruct Mutations
# ============================================================================

EVOL_MUTATIONS = [
    "Add more specific step-by-step guidance for arithmetic operations",
    "Make it more concise and direct, focusing on the key reasoning approach",
    "Add emphasis on double-checking calculations and verifying the answer",
    "Include guidance for handling edge cases like zero, negative numbers, or fractions",
    "Add mathematical notation guidance (showing work with equations)",
    "Focus on breaking down word problems into mathematical expressions",
    "Add a verification step to check if the answer makes sense",
    "Make it more conversational and encouraging for the solver",
    "Add guidance for identifying key information vs distractors in the problem",
    "Focus on unit tracking and dimensional analysis",
]

EVOL_PROMPT_TEMPLATE = """You are an expert at improving math problem-solving instructions.

ORIGINAL INSTRUCTION (achieved >80% accuracy on math word problems):
{original}

MUTATION GOAL:
{mutation}

Generate an IMPROVED version of the instruction that incorporates the mutation goal while keeping the core reasoning guidance intact.

Requirements:
- Keep it 1-3 sentences
- Maintain the mathematical reasoning focus
- Make it clearly different from the original
- The improved version should be practical and actionable

IMPROVED INSTRUCTION:"""


def evolve_instruction(
    instruction: str,
    llm_client,
    mutations: List[str] = None,
    temperature: float = 0.8
) -> List[str]:
    """
    Generate Evol-Instruct style mutations of an instruction.

    Args:
        instruction: Base instruction to evolve
        llm_client: LLM client for generation
        mutations: List of mutation goals (uses EVOL_MUTATIONS if None)
        temperature: Sampling temperature

    Returns:
        List of evolved instructions
    """
    if mutations is None:
        mutations = EVOL_MUTATIONS

    evolved = []

    for mutation in mutations:
        prompt = EVOL_PROMPT_TEMPLATE.format(
            original=instruction,
            mutation=mutation
        )

        try:
            response = llm_client.generate(
                prompt,
                temperature=temperature,
                max_new_tokens=256
            )

            # Clean response
            evolved_inst = response.strip()
            # Remove any "IMPROVED INSTRUCTION:" prefix
            for prefix in ["IMPROVED INSTRUCTION:", "Improved instruction:", "Improved:"]:
                if evolved_inst.startswith(prefix):
                    evolved_inst = evolved_inst[len(prefix):].strip()

            # Validate length
            if 20 < len(evolved_inst) < 500:
                evolved.append(evolved_inst)

        except Exception as e:
            logger.warning(f"Evolution failed for mutation '{mutation[:30]}...': {e}")
            continue

    return evolved


def generate_evol_instructions(
    llm_client,
    seed_instructions: List[str],
    target_count: int = 1000,
    mutations_per_seed: int = 5,
    temperature: float = 0.8
) -> List[str]:
    """
    Generate Evol-Instruct mutations of seed instructions.

    Args:
        llm_client: LLM client for generation
        seed_instructions: High-quality seed instructions (>80% accuracy)
        target_count: Target number of evolved instructions
        mutations_per_seed: Number of mutations to try per seed
        temperature: LLM sampling temperature

    Returns:
        List of evolved instructions
    """
    logger.info(f"Generating {target_count} Evol-Instruct mutations...")
    logger.info(f"  Seed pool: {len(seed_instructions)} instructions")
    logger.info(f"  Mutations per seed: {mutations_per_seed}")

    all_evolved = []
    seen = set(seed_instructions)  # Don't duplicate seeds

    # Calculate how many seeds to use
    n_seeds = (target_count // mutations_per_seed) + 1
    n_seeds = min(n_seeds, len(seed_instructions))

    selected_seeds = random.sample(seed_instructions, n_seeds)

    pbar = tqdm(selected_seeds, desc="Evol-Instruct")
    for seed in pbar:
        # Sample random mutations
        mutations = random.sample(EVOL_MUTATIONS, min(mutations_per_seed, len(EVOL_MUTATIONS)))

        evolved = evolve_instruction(seed, llm_client, mutations, temperature)

        for inst in evolved:
            if inst not in seen:
                seen.add(inst)
                all_evolved.append(inst)

        pbar.set_postfix({"total": len(all_evolved)})

        if len(all_evolved) >= target_count:
            break

    logger.info(f"Generated {len(all_evolved)} evolved instructions")
    return all_evolved[:target_count]


# ============================================================================
# SLERP Embedding Augmentation
# ============================================================================

def slerp(emb1: torch.Tensor, emb2: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical Linear Interpolation between two embeddings.

    Interpolates on the hypersphere, preserving angular relationships
    better than linear interpolation for normalized embeddings.

    Args:
        emb1: First embedding (1D tensor)
        emb2: Second embedding (1D tensor)
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated embedding
    """
    # Normalize to unit sphere
    emb1_norm = emb1 / emb1.norm()
    emb2_norm = emb2 / emb2.norm()

    # Compute angle between embeddings
    dot = torch.clamp(torch.dot(emb1_norm, emb2_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    # Handle near-parallel case
    if omega.abs() < 1e-6:
        return (1 - t) * emb1 + t * emb2

    sin_omega = torch.sin(omega)

    # SLERP formula
    return (torch.sin((1 - t) * omega) * emb1_norm + torch.sin(t * omega) * emb2_norm) / sin_omega


def augment_embeddings_slerp(
    embeddings: torch.Tensor,
    n_augmented: int = 1000,
    t_range: Tuple[float, float] = (0.3, 0.7),
    seed: int = 42
) -> torch.Tensor:
    """
    Generate synthetic embeddings via SLERP interpolation.

    Creates new embeddings by interpolating between random pairs,
    using spherical interpolation to stay on the embedding manifold.

    Args:
        embeddings: Source embeddings [N, D]
        n_augmented: Number of augmented embeddings to generate
        t_range: Range of interpolation parameters (avoid extremes)
        seed: Random seed for reproducibility

    Returns:
        Augmented embeddings [n_augmented, D]
    """
    logger.info(f"Generating {n_augmented} SLERP-augmented embeddings...")
    logger.info(f"  Source pool: {len(embeddings)} embeddings")
    logger.info(f"  Interpolation range: {t_range}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    augmented = []
    n_source = len(embeddings)

    for _ in tqdm(range(n_augmented), desc="SLERP Augmentation"):
        # Sample two different embeddings
        i, j = np.random.choice(n_source, 2, replace=False)

        # Random interpolation parameter (avoid extremes)
        t = np.random.uniform(t_range[0], t_range[1])

        # SLERP interpolation
        aug = slerp(embeddings[i], embeddings[j], t)
        augmented.append(aug)

    result = torch.stack(augmented)
    logger.info(f"Generated {len(result)} augmented embeddings")

    return result


# ============================================================================
# Quality Control Pipeline
# ============================================================================

def exact_dedup(instructions: List[str]) -> List[str]:
    """Remove exact duplicate instructions."""
    seen = set()
    unique = []
    for inst in instructions:
        if inst not in seen:
            seen.add(inst)
            unique.append(inst)
    return unique


def semantic_dedup(
    embeddings: torch.Tensor,
    instructions: List[str],
    threshold: float = 0.95
) -> Tuple[torch.Tensor, List[str]]:
    """
    Remove semantically duplicate instructions using cosine similarity.

    Args:
        embeddings: Instruction embeddings [N, D]
        instructions: Instruction texts
        threshold: Cosine similarity threshold for duplicate detection

    Returns:
        Filtered (embeddings, instructions) tuple
    """
    logger.info(f"Semantic deduplication (threshold={threshold})...")

    # Normalize for cosine similarity
    emb_norm = embeddings / embeddings.norm(dim=1, keepdim=True)

    # Track which to keep (greedy selection)
    keep_mask = torch.ones(len(embeddings), dtype=torch.bool)

    for i in tqdm(range(len(embeddings)), desc="Semantic Dedup", leave=False):
        if not keep_mask[i]:
            continue

        # Check similarity to remaining candidates
        sims = (emb_norm[i] @ emb_norm[i+1:].T)
        duplicates = (sims > threshold).nonzero().squeeze(-1) + i + 1

        # Mark duplicates for removal
        keep_mask[duplicates] = False

    n_removed = (~keep_mask).sum().item()
    logger.info(f"Removed {n_removed} semantic duplicates (kept {keep_mask.sum().item()}/{len(embeddings)})")

    filtered_emb = embeddings[keep_mask]
    filtered_inst = [inst for inst, keep in zip(instructions, keep_mask) if keep]

    return filtered_emb, filtered_inst


def reference_similarity_filter(
    embeddings: torch.Tensor,
    instructions: List[str],
    reference_embeddings: torch.Tensor,
    threshold: float = 0.15
) -> Tuple[torch.Tensor, List[str]]:
    """
    Filter embeddings that are too far from reference distribution.

    Keeps only instructions that have at least threshold cosine similarity
    to some reference instruction (ensures task relevance).

    Args:
        embeddings: Instruction embeddings [N, D]
        instructions: Instruction texts
        reference_embeddings: High-quality reference embeddings [M, D]
        threshold: Minimum cosine similarity to any reference

    Returns:
        Filtered (embeddings, instructions) tuple
    """
    logger.info(f"Reference similarity filtering (threshold={threshold})...")

    # Normalize for cosine similarity
    emb_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    ref_norm = reference_embeddings / reference_embeddings.norm(dim=1, keepdim=True)

    # Max similarity to any reference point
    sims = emb_norm @ ref_norm.T
    max_sims = sims.max(dim=1).values

    # Keep only sufficiently similar embeddings
    keep_mask = max_sims > threshold

    n_removed = (~keep_mask).sum().item()
    logger.info(f"Removed {n_removed} outliers (kept {keep_mask.sum().item()}/{len(embeddings)})")

    filtered_emb = embeddings[keep_mask]
    filtered_inst = [inst for inst, keep in zip(instructions, keep_mask) if keep]

    return filtered_emb, filtered_inst


def cluster_balanced_sampling(
    embeddings: torch.Tensor,
    instructions: List[str],
    n_clusters: int = 50,
    samples_per_cluster: int = 100,
    seed: int = 42
) -> Tuple[torch.Tensor, List[str]]:
    """
    Balance dataset using K-means clustering and uniform sampling.

    Ensures diverse coverage by sampling equally from all clusters,
    preventing dominance of any single instruction style.

    Args:
        embeddings: Instruction embeddings [N, D]
        instructions: Instruction texts
        n_clusters: Number of K-means clusters
        samples_per_cluster: Maximum samples per cluster
        seed: Random seed

    Returns:
        Balanced (embeddings, instructions) tuple
    """
    logger.info(f"Cluster-balanced sampling (k={n_clusters}, per_cluster={samples_per_cluster})...")

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings.cpu().numpy())

    # Sample from each cluster
    np.random.seed(seed)
    selected_indices = []

    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    logger.info(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) <= samples_per_cluster:
            selected_indices.extend(cluster_indices.tolist())
        else:
            sampled = np.random.choice(cluster_indices, samples_per_cluster, replace=False)
            selected_indices.extend(sampled.tolist())

    # Convert to tensors
    selected_emb = embeddings[selected_indices]
    selected_inst = [instructions[i] for i in selected_indices]

    logger.info(f"Selected {len(selected_inst)} instructions from {n_clusters} clusters")

    return selected_emb, selected_inst


def quality_control_pipeline(
    instructions: List[str],
    embeddings: torch.Tensor,
    reference_embeddings: Optional[torch.Tensor] = None,
    semantic_dedup_threshold: float = 0.95,
    reference_threshold: float = 0.15,
    n_clusters: int = 50,
    samples_per_cluster: int = 100,
) -> Tuple[List[str], torch.Tensor]:
    """
    Multi-stage quality control pipeline.

    Stages:
    1. Exact deduplication
    2. Semantic deduplication (cosine > threshold)
    3. Reference similarity filter (cosine > threshold)
    4. K-means clustering + balanced sampling

    Args:
        instructions: Raw instruction texts
        embeddings: Corresponding embeddings
        reference_embeddings: High-quality reference for filtering
        semantic_dedup_threshold: Threshold for semantic dedup
        reference_threshold: Minimum similarity to references
        n_clusters: Number of clusters for balancing
        samples_per_cluster: Max samples per cluster

    Returns:
        Filtered (instructions, embeddings) tuple
    """
    logger.info("=" * 60)
    logger.info("QUALITY CONTROL PIPELINE")
    logger.info("=" * 60)

    # Stage 1: Exact deduplication (on instructions only)
    logger.info("\nStage 1: Exact deduplication")
    unique_inst = exact_dedup(instructions)
    n_exact_dupes = len(instructions) - len(unique_inst)
    logger.info(f"Removed {n_exact_dupes} exact duplicates")

    # Re-encode unique instructions if needed
    if len(unique_inst) != len(instructions):
        # Need to map back to embeddings
        inst_to_idx = {inst: i for i, inst in enumerate(instructions)}
        indices = [inst_to_idx[inst] for inst in unique_inst]
        embeddings = embeddings[indices]
        instructions = unique_inst

    # Stage 2: Semantic deduplication
    logger.info("\nStage 2: Semantic deduplication")
    embeddings, instructions = semantic_dedup(
        embeddings, instructions, threshold=semantic_dedup_threshold
    )

    # Stage 3: Reference similarity filter (if reference provided)
    if reference_embeddings is not None:
        logger.info("\nStage 3: Reference similarity filtering")
        embeddings, instructions = reference_similarity_filter(
            embeddings, instructions, reference_embeddings, threshold=reference_threshold
        )
    else:
        logger.info("\nStage 3: Skipped (no reference embeddings)")

    # Stage 4: Cluster-balanced sampling
    logger.info("\nStage 4: Cluster-balanced sampling")
    embeddings, instructions = cluster_balanced_sampling(
        embeddings, instructions,
        n_clusters=n_clusters,
        samples_per_cluster=samples_per_cluster
    )

    logger.info("=" * 60)
    logger.info(f"Final: {len(instructions)} instructions")
    logger.info("=" * 60)

    return instructions, embeddings


# ============================================================================
# SONAR Encoding
# ============================================================================

def encode_with_sonar(
    instructions: List[str],
    device: str = "cuda:0",
    batch_size: int = 64
) -> torch.Tensor:
    """
    Encode instructions using SONAR text encoder.

    Args:
        instructions: List of instruction texts
        device: CUDA device for encoding
        batch_size: Batch size for encoding

    Returns:
        Embeddings tensor [N, 1024]
    """
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    logger.info(f"Encoding {len(instructions)} instructions with SONAR...")
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch size: {batch_size}")

    encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(device),
    )

    embeddings = encoder.predict(
        instructions,
        source_lang="eng_Latn",
        batch_size=batch_size,
        progress_bar=True,
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"  Mean: {embeddings.mean().item():.6f}")
    logger.info(f"  Std: {embeddings.std().item():.6f}")
    logger.info(f"  L2 norm (mean): {embeddings.norm(dim=-1).mean().item():.4f}")

    return embeddings


# ============================================================================
# Main
# ============================================================================

def load_seed_instructions(
    path: str = "datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt",
    min_accuracy: float = 0.80
) -> Tuple[List[str], torch.Tensor]:
    """Load high-quality seed instructions and their embeddings."""
    logger.info(f"Loading seed instructions from {path}...")

    data = torch.load(path, weights_only=False)
    instructions = data["instructions"]
    embeddings = data["embeddings"]
    accuracies = data["accuracies"]

    # Filter by accuracy
    good_indices = [i for i, acc in enumerate(accuracies) if acc >= min_accuracy]
    good_instructions = [instructions[i] for i in good_indices]
    good_embeddings = embeddings[good_indices]

    logger.info(f"Loaded {len(good_instructions)} seed instructions with >={min_accuracy*100:.0f}% accuracy")

    return good_instructions, good_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse GSM8K instructions via Verbalized Sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Output
    parser.add_argument("--output", type=str, default="datasets/gsm8k_instructions_vs.pt",
                       help="Output path for embeddings and instructions")

    # Generation targets
    parser.add_argument("--target-vs", type=int, default=3000,
                       help="Target Verbalized Sampling instructions")
    parser.add_argument("--target-evol", type=int, default=1000,
                       help="Target Evol-Instruct mutations")
    parser.add_argument("--target-slerp", type=int, default=1000,
                       help="Target SLERP augmented embeddings")

    # Model settings
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="LLM model for generation")
    parser.add_argument("--backend", type=str, default="vllm",
                       help="LLM backend (vllm, openai, deepinfra)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size for vLLM")

    # VS parameters
    parser.add_argument("--vs-temperature", type=float, default=1.0,
                       help="Temperature for VS generation")
    parser.add_argument("--vs-examples-per-batch", type=int, default=3,
                       help="GSM8K examples per VS prompt")
    parser.add_argument("--vs-instructions-per-call", type=int, default=5,
                       help="Instructions to generate per LLM call")

    # Quality control
    parser.add_argument("--n-clusters", type=int, default=50,
                       help="K-means clusters for balancing")
    parser.add_argument("--samples-per-cluster", type=int, default=100,
                       help="Max samples per cluster")
    parser.add_argument("--semantic-dedup-threshold", type=float, default=0.95,
                       help="Cosine threshold for semantic dedup")
    parser.add_argument("--reference-threshold", type=float, default=0.15,
                       help="Minimum cosine similarity to reference")

    # Paths
    parser.add_argument("--gsm8k-path", type=str, default="datasets/gsm8k",
                       help="Path to GSM8K dataset")
    parser.add_argument("--seed-path", type=str,
                       default="datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt",
                       help="Path to seed instructions")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # Skip options
    parser.add_argument("--skip-vs", action="store_true", help="Skip VS generation")
    parser.add_argument("--skip-evol", action="store_true", help="Skip Evol-Instruct")
    parser.add_argument("--skip-slerp", action="store_true", help="Skip SLERP augmentation")
    parser.add_argument("--skip-qc", action="store_true", help="Skip quality control")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("=" * 60)
    logger.info("GSM8K INSTRUCTION GENERATION VIA VERBALIZED SAMPLING")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output}")
    logger.info(f"Targets: VS={args.target_vs}, Evol={args.target_evol}, SLERP={args.target_slerp}")
    logger.info(f"Model: {args.model} (backend={args.backend})")

    # Load GSM8K training data
    gsm8k_examples = load_gsm8k_with_difficulty(args.gsm8k_path, split="train")

    # Load seed instructions
    seed_instructions, seed_embeddings = load_seed_instructions(args.seed_path)

    # Initialize LLM client
    if not args.skip_vs or not args.skip_evol:
        from src.llm_client import create_llm_client
        logger.info(f"\nInitializing LLM: {args.model}")
        llm_client = create_llm_client(
            args.model,
            backend=args.backend,
            tensor_parallel_size=args.tensor_parallel_size
        )
    else:
        llm_client = None

    all_instructions = []
    all_sources = {}

    # ================================================================
    # Stage 1: Verbalized Sampling (~60%)
    # ================================================================
    if not args.skip_vs and args.target_vs > 0:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: VERBALIZED SAMPLING")
        logger.info("=" * 60)

        vs_results = generate_instructions_vs(
            llm_client,
            gsm8k_examples,
            target_count=args.target_vs,
            examples_per_batch=args.vs_examples_per_batch,
            instructions_per_call=args.vs_instructions_per_call,
            temperature=args.vs_temperature,
        )

        vs_instructions = [text for text, _ in vs_results]
        all_instructions.extend(vs_instructions)
        all_sources["verbalized_sampling"] = len(vs_instructions)

    # ================================================================
    # Stage 2: Evol-Instruct Mutations (~20%)
    # ================================================================
    if not args.skip_evol and args.target_evol > 0:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: EVOL-INSTRUCT MUTATIONS")
        logger.info("=" * 60)

        evol_instructions = generate_evol_instructions(
            llm_client,
            seed_instructions,
            target_count=args.target_evol,
            mutations_per_seed=5,
            temperature=0.8
        )

        all_instructions.extend(evol_instructions)
        all_sources["evol_instruct"] = len(evol_instructions)

    # Cleanup LLM client to free GPU memory for SONAR
    if llm_client is not None and hasattr(llm_client, 'cleanup'):
        logger.info("\nCleaning up LLM client...")
        llm_client.cleanup()

    # ================================================================
    # Stage 3: SONAR Encoding
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: SONAR ENCODING")
    logger.info("=" * 60)

    # Add seed instructions to the mix
    all_instructions.extend(seed_instructions)
    all_sources["seed"] = len(seed_instructions)

    # Encode all text instructions
    text_instructions = list(set(all_instructions))  # Quick dedup
    logger.info(f"Unique text instructions: {len(text_instructions)}")

    embeddings = encode_with_sonar(text_instructions, device=args.device)

    # ================================================================
    # Stage 4: SLERP Augmentation (~20%)
    # ================================================================
    if not args.skip_slerp and args.target_slerp > 0:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4: SLERP AUGMENTATION")
        logger.info("=" * 60)

        slerp_embeddings = augment_embeddings_slerp(
            embeddings,
            n_augmented=args.target_slerp,
            t_range=(0.3, 0.7),
            seed=args.seed
        )

        # SLERP embeddings have no text - marked as synthetic
        all_sources["slerp_augmented"] = len(slerp_embeddings)

        # Combine embeddings
        combined_embeddings = torch.cat([embeddings, slerp_embeddings], dim=0)

        # Extend instructions with None placeholders for SLERP
        combined_instructions = text_instructions + [None] * len(slerp_embeddings)
    else:
        combined_embeddings = embeddings
        combined_instructions = text_instructions

    # ================================================================
    # Stage 5: Quality Control Pipeline
    # ================================================================
    if not args.skip_qc:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 5: QUALITY CONTROL")
        logger.info("=" * 60)

        # Separate text and synthetic embeddings for QC
        text_mask = torch.tensor([inst is not None for inst in combined_instructions])

        # QC on text instructions
        text_embeddings = combined_embeddings[text_mask]
        text_only = [inst for inst in combined_instructions if inst is not None]

        final_instructions, final_embeddings = quality_control_pipeline(
            text_only,
            text_embeddings,
            reference_embeddings=seed_embeddings.to(text_embeddings.device),
            semantic_dedup_threshold=args.semantic_dedup_threshold,
            reference_threshold=args.reference_threshold,
            n_clusters=args.n_clusters,
            samples_per_cluster=args.samples_per_cluster,
        )

        # Add back SLERP embeddings (they're already on manifold by construction)
        if not args.skip_slerp and args.target_slerp > 0:
            slerp_embs = combined_embeddings[~text_mask]
            final_embeddings = torch.cat([final_embeddings, slerp_embs], dim=0)
            final_instructions = final_instructions + [None] * len(slerp_embs)

    else:
        final_embeddings = combined_embeddings
        final_instructions = combined_instructions

    # ================================================================
    # Save Results
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count text vs synthetic
    n_text = sum(1 for inst in final_instructions if inst is not None)
    n_synthetic = len(final_instructions) - n_text

    save_data = {
        "embeddings": final_embeddings.cpu(),
        "instructions": final_instructions,
        "sources": all_sources,
        "config": vars(args),
        "stats": {
            "n_text_instructions": n_text,
            "n_synthetic_embeddings": n_synthetic,
            "total": len(final_instructions),
            "embedding_dim": final_embeddings.shape[-1],
        }
    }

    torch.save(save_data, output_path)

    # Summary
    logger.info(f"\nOutput saved to: {output_path}")
    logger.info(f"Total embeddings: {len(final_embeddings)}")
    logger.info(f"  Text instructions: {n_text}")
    logger.info(f"  Synthetic (SLERP): {n_synthetic}")
    logger.info(f"Embedding shape: {final_embeddings.shape}")
    logger.info("\nSource breakdown:")
    for source, count in all_sources.items():
        logger.info(f"  {source}: {count}")

    # Diversity check
    logger.info("\nDiversity metrics:")
    emb_norm = final_embeddings / final_embeddings.norm(dim=1, keepdim=True)

    # Sample pairwise distances (full matrix too expensive)
    n_sample = min(1000, len(emb_norm))
    sample_idx = torch.randperm(len(emb_norm))[:n_sample]
    sample_emb = emb_norm[sample_idx]

    pairwise_sims = sample_emb @ sample_emb.T
    # Mask diagonal
    mask = ~torch.eye(n_sample, dtype=torch.bool)
    off_diag_sims = pairwise_sims[mask]

    mean_sim = off_diag_sims.mean().item()
    mean_dist = 1 - mean_sim

    logger.info(f"  Mean pairwise cosine distance: {mean_dist:.4f}")
    logger.info(f"  (Target: > 0.3 for good diversity)")

    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
