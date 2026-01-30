#!/usr/bin/env python
"""
Generate and evaluate 400+ new GSM8K instructions for surrogate model benchmarking.

This script:
1. Uses Claude/GPT to generate diverse math instruction prompts
2. Evaluates each instruction on 150 GSM8K questions using VLLMClient
3. Encodes all instructions with SONAR (TextToEmbeddingModelPipeline)
4. Saves dataset with embeddings + accuracies

The goal is to expand from 100 to 500+ evaluated instructions for robust
5-fold CV with narrow confidence intervals in surrogate benchmarking.

Usage:
    # Full generation (400+ new instructions)
    uv run python scripts/generate_more_instructions.py \
        --n-generate 420 \
        --eval-subset 150 \
        --output datasets/evaluated_instructions/gsm8k_500_instructions.json \
        --meta-model haiku

    # Quick test
    uv run python scripts/generate_more_instructions.py \
        --n-generate 10 \
        --eval-subset 50 \
        --output datasets/evaluated_instructions/test_gen.json

Long-running process (~20-30 hours for 400 instructions).
Run in tmux per CLAUDE.md conventions:
    tmux new-session -d -s gen_instructions \
        "CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/generate_more_instructions.py \
         --n-generate 420 --eval-subset 150 \
         --output datasets/evaluated_instructions/gsm8k_500_instructions.json \
         2>&1 | tee results/gen_instructions_$(date +%Y%m%d_%H%M%S).log; exec bash"
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Anthropic/Claude Client for Meta-Generation
# =============================================================================

class AnthropicClient:
    """Client for Claude API instruction generation."""

    MODEL_ALIASES = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-5-20251022",
        "opus": "claude-opus-4-5-20251101",
    }

    def __init__(self, model: str = "haiku"):
        from anthropic import Anthropic

        self.model = self.MODEL_ALIASES.get(model.lower(), model)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=api_key)
        logger.info(f"Initialized Anthropic client: {self.model}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        """Generate text using Claude."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# =============================================================================
# Instruction Generation Strategies
# =============================================================================

GENERATION_PROMPT = """You are an expert at creating instructions that help AI models solve math word problems.

Generate {n_instructions} unique and diverse math-solving instructions. Each instruction should:
- Be 1-3 sentences
- Guide step-by-step problem solving
- Be different in style, tone, and approach

Examples of high-quality instructions that achieved >80% accuracy on GSM8K:
{examples}

Generate {n_instructions} NEW instructions that are DIFFERENT from the examples above.
Use variety:
- Some formal, some casual
- Some step-by-step focused, some verification focused
- Some brief, some detailed
- Some emphasizing units, some emphasizing equations

Output format: One instruction per line, numbered 1-{n_instructions}. No additional text."""

PARAPHRASE_PROMPT = """You are an expert at paraphrasing while preserving meaning.

Original instruction (achieved {accuracy:.1%} accuracy on GSM8K math problems):
"{instruction}"

Generate {n_paraphrases} paraphrased versions that:
- Preserve the core guidance
- Use different wording and sentence structure
- Vary in formality and length

Output format: One paraphrase per line, numbered 1-{n_paraphrases}. No additional text."""

COMBINATION_PROMPT = """You are an expert at creating instructions for math problem solving.

Here are two high-quality instructions:
Instruction A: "{inst_a}"
Instruction B: "{inst_b}"

Create a NEW instruction that:
- Combines key elements from both
- Is 1-3 sentences
- Provides clear math problem-solving guidance

Output only the combined instruction, no explanation."""

TONE_VARIANTS = [
    "formal academic",
    "casual friendly",
    "step-by-step procedural",
    "verification-focused",
    "concise minimal",
    "detailed thorough",
    "encouraging supportive",
    "precise mathematical",
]

TONE_PROMPT = """You are an expert at creating math problem-solving instructions.

Generate a {tone} instruction for solving math word problems.
The instruction should be 1-3 sentences and help an AI solve GSM8K-style problems.

Output only the instruction, no explanation."""


def load_existing_instructions(
    path: str = "datasets/evaluated_instructions/gsm8k_100_instructions.json",
) -> Tuple[List[str], List[float]]:
    """Load existing evaluated instructions."""
    with open(path) as f:
        data = json.load(f)

    instructions = []
    accuracies = []
    for result in data["results"]:
        instructions.append(result["instruction"])
        accuracies.append(result["accuracy"])

    logger.info(f"Loaded {len(instructions)} existing instructions")
    return instructions, accuracies


def get_top_instructions(
    instructions: List[str],
    accuracies: List[float],
    threshold: float = 0.80,
    top_k: int = 10,
) -> List[str]:
    """Get top-performing instructions above threshold."""
    scored = [(inst, acc) for inst, acc in zip(instructions, accuracies) if acc >= threshold]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [inst for inst, _ in scored[:top_k]]


def generate_batch_instructions(
    client: AnthropicClient,
    seed_instructions: List[str],
    n_instructions: int = 20,
) -> List[str]:
    """Generate a batch of new instructions based on seed examples."""
    examples = "\n".join(f"- {inst}" for inst in random.sample(seed_instructions, min(5, len(seed_instructions))))

    prompt = GENERATION_PROMPT.format(
        n_instructions=n_instructions,
        examples=examples,
    )

    response = client.generate(prompt, temperature=1.0)

    # Parse numbered responses
    instructions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove number prefix like "1.", "1)", "1:"
        for prefix in [".", ")", ":", "-"]:
            if len(line) > 2 and line[0].isdigit() and prefix in line[:4]:
                line = line.split(prefix, 1)[-1].strip()
                break
        # Remove leading dash or asterisk
        if line.startswith("-") or line.startswith("*"):
            line = line[1:].strip()
        # Validate length
        if 30 < len(line) < 500:
            instructions.append(line)

    return instructions


def generate_paraphrases(
    client: AnthropicClient,
    instruction: str,
    accuracy: float,
    n_paraphrases: int = 3,
) -> List[str]:
    """Generate paraphrases of a high-quality instruction."""
    prompt = PARAPHRASE_PROMPT.format(
        instruction=instruction,
        accuracy=accuracy,
        n_paraphrases=n_paraphrases,
    )

    response = client.generate(prompt, temperature=0.9)

    paraphrases = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix in [".", ")", ":", "-"]:
            if len(line) > 2 and line[0].isdigit() and prefix in line[:4]:
                line = line.split(prefix, 1)[-1].strip()
                break
        if line.startswith("-") or line.startswith("*"):
            line = line[1:].strip()
        if 30 < len(line) < 500:
            paraphrases.append(line)

    return paraphrases


def generate_combination(
    client: AnthropicClient,
    inst_a: str,
    inst_b: str,
) -> Optional[str]:
    """Generate a combined instruction from two sources."""
    prompt = COMBINATION_PROMPT.format(inst_a=inst_a, inst_b=inst_b)
    response = client.generate(prompt, temperature=0.8)
    instruction = response.strip()
    if 30 < len(instruction) < 500:
        return instruction
    return None


def generate_tone_variant(
    client: AnthropicClient,
    tone: str,
) -> Optional[str]:
    """Generate an instruction with specific tone."""
    prompt = TONE_PROMPT.format(tone=tone)
    response = client.generate(prompt, temperature=0.9)
    instruction = response.strip()
    if 30 < len(instruction) < 500:
        return instruction
    return None


def generate_diverse_instructions(
    client: AnthropicClient,
    seed_instructions: List[str],
    seed_accuracies: List[float],
    n_target: int = 400,
) -> List[str]:
    """Generate diverse instructions using multiple strategies."""
    logger.info(f"Generating {n_target} diverse instructions...")

    all_instructions = set()
    top_seeds = get_top_instructions(seed_instructions, seed_accuracies, threshold=0.80, top_k=15)

    # Strategy distribution (approximate)
    n_batch = int(n_target * 0.5)  # 50% batch generation
    n_paraphrase = int(n_target * 0.25)  # 25% paraphrasing
    n_combination = int(n_target * 0.15)  # 15% combinations
    n_tone = int(n_target * 0.10)  # 10% tone variants

    # Strategy 1: Batch generation
    logger.info(f"Strategy 1: Generating {n_batch} via batch prompts...")
    n_batches = (n_batch // 20) + 1
    for i in tqdm(range(n_batches), desc="Batch generation"):
        try:
            batch = generate_batch_instructions(client, top_seeds, n_instructions=20)
            for inst in batch:
                if inst not in all_instructions:
                    all_instructions.add(inst)
        except Exception as e:
            logger.warning(f"Batch generation error: {e}")
            continue
        if len(all_instructions) >= n_batch:
            break
    logger.info(f"  Generated {len(all_instructions)} unique instructions")

    # Strategy 2: Paraphrasing top performers
    logger.info(f"Strategy 2: Generating {n_paraphrase} via paraphrasing...")
    paraphrase_count = 0
    for inst, acc in tqdm(
        list(zip(seed_instructions, seed_accuracies)),
        desc="Paraphrasing",
    ):
        if acc < 0.80:
            continue
        try:
            paraphrases = generate_paraphrases(client, inst, acc, n_paraphrases=3)
            for p in paraphrases:
                if p not in all_instructions:
                    all_instructions.add(p)
                    paraphrase_count += 1
        except Exception as e:
            logger.debug(f"Paraphrase error: {e}")
            continue
        if paraphrase_count >= n_paraphrase:
            break
    logger.info(f"  Added {paraphrase_count} paraphrases")

    # Strategy 3: Combining instructions
    logger.info(f"Strategy 3: Generating {n_combination} via combinations...")
    combination_count = 0
    for _ in tqdm(range(n_combination * 2), desc="Combinations"):
        try:
            inst_a, inst_b = random.sample(top_seeds, 2)
            combined = generate_combination(client, inst_a, inst_b)
            if combined and combined not in all_instructions:
                all_instructions.add(combined)
                combination_count += 1
        except Exception as e:
            logger.debug(f"Combination error: {e}")
            continue
        if combination_count >= n_combination:
            break
    logger.info(f"  Added {combination_count} combinations")

    # Strategy 4: Tone variants
    logger.info(f"Strategy 4: Generating {n_tone} tone variants...")
    tone_count = 0
    for tone in TONE_VARIANTS * ((n_tone // len(TONE_VARIANTS)) + 1):
        try:
            variant = generate_tone_variant(client, tone)
            if variant and variant not in all_instructions:
                all_instructions.add(variant)
                tone_count += 1
        except Exception as e:
            logger.debug(f"Tone variant error: {e}")
            continue
        if tone_count >= n_tone:
            break
    logger.info(f"  Added {tone_count} tone variants")

    result = list(all_instructions)
    logger.info(f"Total unique instructions generated: {len(result)}")
    return result


# =============================================================================
# GSM8K Evaluation
# =============================================================================

def evaluate_instruction_gsm8k(
    instruction: str,
    llm_client,
    evaluator,
    n_examples: int = 150,
    batch_size: int = 32,
    eval_indices: Optional[List[int]] = None,
) -> Dict:
    """
    Evaluate a single instruction on GSM8K using VLLMClient.

    Uses Q_end format (instruction after question) per OPRO paper.
    """
    # Use fixed indices for reproducibility
    if eval_indices is None:
        eval_indices = list(range(n_examples))

    # Format prompts with Q_end format
    prompts = []
    for idx in eval_indices:
        example = evaluator.dataset[idx]
        question = example["question"]
        prompt = f"Q: {question}\n{instruction}\nA:"
        prompts.append(prompt)

    # Batch evaluation with vLLM
    start_time = time.time()
    outputs = llm_client.generate_batch(prompts, max_new_tokens=512, temperature=0.0)
    eval_time = time.time() - start_time

    # Score outputs
    from src.gsm8k_evaluator import extract_answer, extract_ground_truth, compare_answers

    correct = 0
    for idx, output in zip(eval_indices, outputs):
        example = evaluator.dataset[idx]
        gt = extract_ground_truth(example["answer"])
        pred = extract_answer(output)
        if compare_answers(pred, gt):
            correct += 1

    accuracy = correct / len(eval_indices)

    return {
        "instruction": instruction,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(eval_indices),
        "eval_time": eval_time,
    }


def evaluate_instructions_batch(
    instructions: List[str],
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    n_examples: int = 150,
    tensor_parallel_size: int = 1,
    checkpoint_path: Optional[str] = None,
    checkpoint_freq: int = 10,
) -> List[Dict]:
    """
    Evaluate multiple instructions on GSM8K.

    Supports checkpointing for long-running evaluations.
    """
    from src.llm_client import create_llm_client, VLLMClient
    from src.gsm8k_evaluator import GSM8KEvaluator

    # Load checkpoint if exists
    results = []
    start_idx = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        results = checkpoint.get("results", [])
        start_idx = len(results)
        logger.info(f"Resuming from checkpoint at index {start_idx}")

    if start_idx >= len(instructions):
        logger.info("All instructions already evaluated")
        return results

    # Initialize vLLM client for fast evaluation
    logger.info(f"Initializing VLLMClient: {model}")
    client = create_llm_client(
        model,
        backend="vllm",
        tensor_parallel_size=tensor_parallel_size,
    )

    # Initialize evaluator
    logger.info("Initializing GSM8K evaluator...")
    evaluator = GSM8KEvaluator(dataset_path="datasets/gsm8k", split="test")

    # Fixed evaluation indices for reproducibility
    random.seed(42)
    eval_indices = random.sample(range(len(evaluator)), min(n_examples, len(evaluator)))
    logger.info(f"Using {len(eval_indices)} fixed evaluation examples")

    # Evaluate remaining instructions
    for i, instruction in enumerate(tqdm(
        instructions[start_idx:],
        desc="Evaluating",
        initial=start_idx,
        total=len(instructions),
    )):
        idx = start_idx + i
        try:
            result = evaluate_instruction_gsm8k(
                instruction,
                client,
                evaluator,
                n_examples=n_examples,
                eval_indices=eval_indices,
            )
            result["idx"] = idx
            results.append(result)

            logger.info(
                f"[{idx+1}/{len(instructions)}] Accuracy: {result['accuracy']:.4f} "
                f"({result['correct']}/{result['total']}) - {result['eval_time']:.1f}s"
            )
            # Log full instruction (per CLAUDE.md - never truncate)
            logger.info(f"Instruction:\n{instruction}")

        except Exception as e:
            logger.error(f"Evaluation error at index {idx}: {e}")
            results.append({
                "idx": idx,
                "instruction": instruction,
                "accuracy": 0.0,
                "correct": 0,
                "total": n_examples,
                "error": str(e),
            })

        # Checkpoint periodically
        if checkpoint_path and (idx + 1) % checkpoint_freq == 0:
            save_checkpoint(results, checkpoint_path)
            logger.info(f"Saved checkpoint at index {idx+1}")

    # Cleanup vLLM client
    if hasattr(client, "cleanup"):
        client.cleanup()

    return results


def save_checkpoint(results: List[Dict], path: str):
    """Save evaluation checkpoint."""
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "n_evaluated": len(results),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)


# =============================================================================
# SONAR Encoding
# =============================================================================

def encode_with_sonar(
    instructions: List[str],
    device: str = "cuda:0",
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode instructions using SONAR TextToEmbeddingModelPipeline."""
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    logger.info(f"Initializing SONAR encoder on {device}...")
    encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(device),
    )

    logger.info(f"Encoding {len(instructions)} instructions...")
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate 400+ GSM8K instructions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Generation parameters
    parser.add_argument(
        "--n-generate",
        type=int,
        default=400,
        help="Number of new instructions to generate (default: 400)",
    )
    parser.add_argument(
        "--meta-model",
        type=str,
        default="haiku",
        help="Model for instruction generation: haiku, sonnet, opus (default: haiku)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--eval-subset",
        type=int,
        default=150,
        help="Number of GSM8K questions per instruction (default: 150)",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for GSM8K evaluation (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM (default: 2 for 2x L40S)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for SONAR encoding (default: 32)",
    )

    # Output paths
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/evaluated_instructions/gsm8k_500_instructions.json",
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--embeddings-output",
        type=str,
        default=None,
        help="Output path for embeddings .pt file (default: derived from --output)",
    )

    # Checkpoint/resume
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Load existing instructions from --output and only generate/evaluate new ones",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Save checkpoint every N evaluations (default: 10)",
    )

    # Skip options
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip instruction generation (use existing + new from file)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip GSM8K evaluation (encode only)",
    )
    parser.add_argument(
        "--skip-encoding",
        action="store_true",
        help="Skip SONAR encoding (JSON output only)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for SONAR encoding (default: cuda:0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Derive embeddings output path
    if args.embeddings_output is None:
        output_path = Path(args.output)
        args.embeddings_output = str(
            output_path.parent / output_path.stem.replace("instructions", "with_embeddings")
        ) + ".pt"

    # Create output directories
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INSTRUCTION GENERATION AND EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Target new instructions: {args.n_generate}")
    logger.info(f"Meta-model: {args.meta_model}")
    logger.info(f"Eval model: {args.eval_model}")
    logger.info(f"Eval subset: {args.eval_subset} questions per instruction")
    logger.info(f"Output JSON: {args.output}")
    logger.info(f"Output embeddings: {args.embeddings_output}")
    logger.info("=" * 60)

    # Load existing instructions
    existing_instructions = []
    existing_accuracies = []

    existing_path = "datasets/evaluated_instructions/gsm8k_100_instructions.json"
    if Path(existing_path).exists():
        existing_instructions, existing_accuracies = load_existing_instructions(existing_path)

    # Load checkpoint if skip_existing
    if args.skip_existing and Path(args.output).exists():
        logger.info(f"Loading existing results from {args.output}")
        with open(args.output) as f:
            data = json.load(f)
        for result in data.get("results", []):
            existing_instructions.append(result["instruction"])
            existing_accuracies.append(result["accuracy"])

    # ========================================
    # Step 1: Generate new instructions
    # ========================================
    new_instructions = []

    if not args.skip_generation:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: GENERATING NEW INSTRUCTIONS")
        logger.info("=" * 60)

        meta_client = AnthropicClient(model=args.meta_model)

        new_instructions = generate_diverse_instructions(
            meta_client,
            existing_instructions,
            existing_accuracies,
            n_target=args.n_generate,
        )

        # Filter out duplicates with existing
        existing_set = set(existing_instructions)
        new_instructions = [inst for inst in new_instructions if inst not in existing_set]
        logger.info(f"New unique instructions after dedup: {len(new_instructions)}")
    else:
        logger.info("Skipping instruction generation (--skip-generation)")

    # ========================================
    # Step 2: Evaluate new instructions
    # ========================================
    all_results = []

    if not args.skip_evaluation and new_instructions:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: EVALUATING INSTRUCTIONS ON GSM8K")
        logger.info("=" * 60)

        checkpoint_path = args.output.replace(".json", "_checkpoint.json")

        new_results = evaluate_instructions_batch(
            new_instructions,
            model=args.eval_model,
            n_examples=args.eval_subset,
            tensor_parallel_size=args.tensor_parallel_size,
            checkpoint_path=checkpoint_path,
            checkpoint_freq=args.checkpoint_freq,
        )

        all_results = new_results
    elif args.skip_evaluation:
        logger.info("Skipping evaluation (--skip-evaluation)")
        # Just create placeholder results
        for i, inst in enumerate(new_instructions):
            all_results.append({
                "idx": i,
                "instruction": inst,
                "accuracy": 0.0,
                "correct": 0,
                "total": args.eval_subset,
            })

    # ========================================
    # Step 3: Merge with existing and save JSON
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: MERGING AND SAVING RESULTS")
    logger.info("=" * 60)

    # Combine existing + new results
    combined_results = []

    # Add existing with original indices
    for i, (inst, acc) in enumerate(zip(existing_instructions, existing_accuracies)):
        combined_results.append({
            "idx": i,
            "instruction": inst,
            "accuracy": acc,
            "correct": int(acc * args.eval_subset),
            "total": args.eval_subset,
            "source": "existing",
        })

    # Add new results
    for result in all_results:
        result["source"] = "generated"
        result["idx"] = len(combined_results)
        combined_results.append(result)

    # Sort by accuracy (descending)
    combined_results.sort(key=lambda x: x["accuracy"], reverse=True)

    # Calculate statistics
    accuracies = [r["accuracy"] for r in combined_results]
    stats = {
        "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
        "max_accuracy": max(accuracies) if accuracies else 0,
        "min_accuracy": min(accuracies) if accuracies else 0,
        "std": (sum((a - sum(accuracies)/len(accuracies))**2 for a in accuracies) / len(accuracies))**0.5 if accuracies else 0,
    }

    # Save JSON output
    output_data = {
        "model": args.eval_model,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_instructions": len(combined_results),
        "eval_subset_size": args.eval_subset,
        "seed": args.seed,
        "statistics": stats,
        "results": combined_results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved {len(combined_results)} instructions to {args.output}")
    logger.info(f"  Mean accuracy: {stats['mean_accuracy']:.4f}")
    logger.info(f"  Range: {stats['min_accuracy']:.4f} - {stats['max_accuracy']:.4f}")

    # ========================================
    # Step 4: SONAR Encoding
    # ========================================
    if not args.skip_encoding:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: ENCODING WITH SONAR")
        logger.info("=" * 60)

        all_instructions = [r["instruction"] for r in combined_results]
        all_accuracies = [r["accuracy"] for r in combined_results]

        embeddings = encode_with_sonar(
            all_instructions,
            device=args.device,
            batch_size=args.batch_size,
        )

        # Save embeddings
        embeddings_data = {
            "embeddings": embeddings.cpu(),
            "accuracies": torch.tensor(all_accuracies),
            "instructions": all_instructions,
            "model": args.eval_model,
            "timestamp": datetime.now().isoformat(),
            "n_instructions": len(all_instructions),
            "embedding_dim": embeddings.shape[1],
            "eval_subset_size": args.eval_subset,
        }

        torch.save(embeddings_data, args.embeddings_output)
        logger.info(f"Saved embeddings to {args.embeddings_output}")
        logger.info(f"  Shape: {embeddings.shape}")
    else:
        logger.info("Skipping SONAR encoding (--skip-encoding)")

    # ========================================
    # Final Summary
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total instructions: {len(combined_results)}")
    logger.info(f"  Existing: {len(existing_instructions)}")
    logger.info(f"  New generated: {len(new_instructions)}")
    logger.info(f"  New evaluated: {len(all_results)}")
    logger.info(f"Mean accuracy: {stats['mean_accuracy']:.4f}")
    logger.info(f"Accuracy range: {stats['min_accuracy']:.4f} - {stats['max_accuracy']:.4f}")
    logger.info(f"JSON output: {args.output}")
    if not args.skip_encoding:
        logger.info(f"Embeddings output: {args.embeddings_output}")
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
