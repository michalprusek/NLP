#!/usr/bin/env python3
"""
Encode instructions to SONAR embeddings and evaluate on full GSM8K test set.

This script:
1. Takes instruction texts and encodes them with SONAR
2. Evaluates each instruction on the full GSM8K test set (1319 examples)
3. Saves embeddings + scores for surrogate model benchmarking

Usage:
    # Encode existing evaluated instructions
    uv run python scripts/encode_and_evaluate_instructions.py \
        --input datasets/evaluated_instructions/gsm8k_100_instructions.json \
        --output datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt

    # Generate and evaluate new random instructions
    uv run python scripts/encode_and_evaluate_instructions.py \
        --generate 400 \
        --output datasets/evaluated_instructions/gsm8k_500_with_embeddings.pt \
        --model Qwen/Qwen2.5-7B-Instruct

Author: EcoFlow Team
Date: 2026-01-30
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_sonar_encoder(device: str = "cuda"):
    """Load SONAR text encoder."""
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    logger.info("Loading SONAR encoder...")
    encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(device),
    )
    logger.info("SONAR encoder loaded")
    return encoder


def encode_instructions(
    instructions: list[str],
    encoder,
    batch_size: int = 32,
) -> Tensor:
    """Encode instructions to SONAR embeddings."""

    all_embeddings = []

    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        logger.info(f"Encoding batch {i//batch_size + 1}/{(len(instructions)-1)//batch_size + 1}")

        embeddings = encoder.predict(batch, source_lang="eng_Latn")
        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)


def evaluate_instruction_on_gsm8k(
    instruction: str,
    evaluator,
    n_examples: Optional[int] = None,
) -> dict:
    """Evaluate a single instruction on GSM8K."""

    # Use full test set if n_examples not specified
    test_indices = list(range(len(evaluator.test_data)))
    if n_examples:
        test_indices = test_indices[:n_examples]

    correct = 0
    total = len(test_indices)

    for idx in test_indices:
        example = evaluator.test_data[idx]
        question = example['question']
        answer = example['answer']

        # Format with instruction (Q_end format from OPRO)
        prompt = f"Q: {question}\n{instruction}\nA:"

        try:
            response = evaluator.llm_client.generate(prompt, max_tokens=256)
            predicted = evaluator.extract_answer(response)
            expected = evaluator.extract_answer(answer)

            if evaluator.check_answer(predicted, expected):
                correct += 1
        except Exception as e:
            logger.error(f"Error evaluating question {idx}: {e}")
            # Count as failed evaluation, not wrong answer
            total -= 1  # Don't count failed evaluations in accuracy

    accuracy = correct / total if total > 0 else 0

    return {
        "instruction": instruction,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def generate_random_instructions(
    n_instructions: int,
    meta_model: str = "Qwen/Qwen2.5-7B-Instruct",
    backend: str = "vllm",
) -> list[str]:
    """Generate random instruction variations using meta-LLM."""
    from src.llm_client import create_llm_client

    logger.info(f"Generating {n_instructions} instructions using {meta_model}")

    client = create_llm_client(meta_model, backend=backend)

    base_prompt = """Generate a unique instruction for solving math word problems.
The instruction should:
1. Be 1-3 sentences
2. Provide clear step-by-step guidance
3. Be different from standard instructions

Examples of good instructions:
- "Break the problem into smaller parts, identify key numbers and relationships, then calculate step by step."
- "First read carefully to understand what's asked, then set up equations based on the given information."

Generate ONE new unique instruction (just the instruction, no explanation):"""

    instructions = []
    seen = set()
    consecutive_failures = 0

    while len(instructions) < n_instructions:
        try:
            # Add variation prompt
            variation = f"\n\nVariation {len(instructions) + 1}:"
            response = client.generate(base_prompt + variation, max_tokens=200, temperature=1.0)

            # Clean up response
            instruction = response.strip()
            if instruction.startswith('"') and instruction.endswith('"'):
                instruction = instruction[1:-1]

            # Skip duplicates and too short/long
            if instruction in seen:
                continue
            if len(instruction) < 30 or len(instruction) > 500:
                continue

            seen.add(instruction)
            instructions.append(instruction)

            if len(instructions) % 50 == 0:
                logger.info(f"Generated {len(instructions)}/{n_instructions} instructions")

        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Generation error (attempt {consecutive_failures}): {e}")
            if consecutive_failures >= 10:
                logger.error("Too many consecutive failures, aborting generation")
                break
            continue
        else:
            consecutive_failures = 0  # Reset on success

    return instructions


def main():
    parser = argparse.ArgumentParser(
        description="Encode instructions to SONAR and evaluate on GSM8K"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to existing evaluated instructions JSON",
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=0,
        help="Number of new instructions to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for embeddings + scores (.pt file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for evaluation",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        help="LLM backend (vllm, openai, etc.)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for SONAR encoder",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation, only encode",
    )
    parser.add_argument(
        "--n-eval-examples",
        type=int,
        default=None,
        help="Number of GSM8K examples to evaluate (default: full test set)",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    instructions = []
    accuracies = []

    # Load existing evaluated instructions
    if args.input:
        logger.info(f"Loading existing instructions from {args.input}")
        with open(args.input) as f:
            data = json.load(f)

        for result in data['results']:
            instructions.append(result['instruction'])
            accuracies.append(result['accuracy'])

        logger.info(f"Loaded {len(instructions)} existing instructions")

    # Generate new instructions if requested
    if args.generate > 0:
        new_instructions = generate_random_instructions(
            args.generate,
            meta_model=args.model,
            backend=args.backend,
        )

        # Evaluate new instructions
        if not args.skip_eval:
            from src.gsm8k_evaluator import GSM8KEvaluator
            from src.llm_client import create_llm_client

            logger.info("Initializing evaluator...")
            client = create_llm_client(args.model, backend=args.backend)
            evaluator = GSM8KEvaluator(llm_client=client)

            logger.info(f"Evaluating {len(new_instructions)} new instructions on GSM8K...")

            for i, instruction in enumerate(new_instructions):
                logger.info(f"Evaluating instruction {i+1}/{len(new_instructions)}")

                result = evaluate_instruction_on_gsm8k(
                    instruction,
                    evaluator,
                    n_examples=args.n_eval_examples,
                )

                instructions.append(instruction)
                accuracies.append(result['accuracy'])

                logger.info(f"  Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
        else:
            # Add without evaluation
            instructions.extend(new_instructions)
            accuracies.extend([0.0] * len(new_instructions))

    if not instructions:
        logger.error("No instructions to process!")
        return

    # Encode all instructions with SONAR
    logger.info(f"Encoding {len(instructions)} instructions with SONAR...")
    encoder = load_sonar_encoder(args.device)
    embeddings = encode_instructions(instructions, encoder)

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Save everything
    output_data = {
        'embeddings': embeddings.cpu(),
        'accuracies': torch.tensor(accuracies),
        'instructions': instructions,
        'model': args.model,
        'timestamp': datetime.now().isoformat(),
        'n_instructions': len(instructions),
        'embedding_dim': embeddings.shape[1],
    }

    torch.save(output_data, args.output)
    logger.info(f"Saved to {args.output}")

    # Print statistics
    acc_tensor = torch.tensor(accuracies)
    logger.info(f"\nStatistics:")
    logger.info(f"  N instructions: {len(instructions)}")
    logger.info(f"  Embedding dim: {embeddings.shape[1]}")
    logger.info(f"  Accuracy range: {acc_tensor.min():.4f} - {acc_tensor.max():.4f}")
    logger.info(f"  Accuracy mean: {acc_tensor.mean():.4f}")
    logger.info(f"  Accuracy std: {acc_tensor.std():.4f}")


if __name__ == "__main__":
    main()
