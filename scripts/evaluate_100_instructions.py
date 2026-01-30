#!/usr/bin/env python3
"""
Evaluate 100 diverse instructions on GSM8K test set (1319 examples).

Takes 100 instructions from APE dataset and evaluates each on the full test set.
This provides a baseline for comparing OPRO/ProTeGi optimized prompts.

Results are written incrementally to JSON after each instruction evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate_100_instructions.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.llm_client import create_llm_client
from src.gsm8k_evaluator import GSM8KEvaluator


def save_incremental_results(output_file: Path, results: List[Dict], metadata: Dict):
    """Save results incrementally to JSON file after each evaluation."""
    # Calculate statistics
    if results:
        accuracies = [r["accuracy"] for r in results]
        stats = {
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "max_accuracy": max(accuracies),
            "min_accuracy": min(accuracies),
            "std": (sum((a - sum(accuracies)/len(accuracies))**2 for a in accuracies) / len(accuracies))**0.5 if len(accuracies) > 1 else 0,
        }
    else:
        stats = {}

    # Sort by accuracy for display
    results_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    output_data = {
        **metadata,
        "completed": len(results),
        "statistics": stats,
        "results": results_sorted,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def select_diverse_instructions(instructions: List[str], n: int = 100, seed: int = 42) -> List[str]:
    """Select n diverse instructions from the pool.

    Uses random sampling with seed for reproducibility.
    Could be extended to use embedding-based diversity sampling.
    """
    random.seed(seed)

    # Filter out very short or very long instructions
    valid = [inst for inst in instructions if 50 < len(inst) < 800]
    print(f"Valid instructions (50-800 chars): {len(valid)} / {len(instructions)}")

    # Sample n instructions
    if len(valid) < n:
        print(f"Warning: Only {len(valid)} valid instructions available")
        return valid

    selected = random.sample(valid, n)
    return selected


def evaluate_instruction(
    instruction: str,
    llm_client,
    evaluator: GSM8KEvaluator,
    max_new_tokens: int = 1024,
) -> Dict:
    """Evaluate a single instruction on the full test set."""
    # Get all test examples
    test_batch = evaluator.get_batch(0, len(evaluator))

    # Format prompts (Q_end style from OPRO paper)
    prompts = [
        f"Q: {ex['question']}\n{instruction}\nA:"
        for ex in test_batch
    ]

    # Generate answers
    outputs = llm_client.generate_batch(
        prompts,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
        use_tqdm=True,
    )

    # Evaluate
    indices = [ex['idx'] for ex in test_batch]
    results = evaluator.evaluate_batch(outputs, indices)

    return {
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 100 instructions on GSM8K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-instructions", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--instructions-path", type=str,
                        default="datasets/hbbops/ape_instructions_1000.json")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("INSTRUCTION EVALUATION ON GSM8K TEST SET")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Instructions: {args.n_instructions}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Load instructions
    print(f"\nLoading instructions from {args.instructions_path}...")
    with open(args.instructions_path, 'r', encoding='utf-8') as f:
        all_instructions = json.load(f)
    print(f"Loaded {len(all_instructions)} instructions")

    # Select diverse subset
    instructions = select_diverse_instructions(
        all_instructions,
        n=args.n_instructions,
        seed=args.seed
    )
    print(f"Selected {len(instructions)} instructions for evaluation")

    # Load model
    print(f"\nLoading model: {args.model}")
    llm_client = create_llm_client(
        model_name=args.model,
        backend="vllm",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Load evaluator
    print("\nLoading GSM8K test set...")
    evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split="test",
    )
    print(f"Test set: {len(evaluator)} examples")

    # Evaluate each instruction
    print(f"\nEvaluating {len(instructions)} instructions on {len(evaluator)} examples each...")
    print("=" * 70)

    # Setup output file for incremental saving
    output_file = output_dir / f"instruction_eval_{timestamp}.json"
    metadata = {
        "model": args.model,
        "timestamp": timestamp,
        "n_instructions": len(instructions),
        "test_set_size": len(evaluator),
        "seed": args.seed,
    }
    print(f"Results will be saved incrementally to: {output_file}")

    results = []

    for i, instruction in enumerate(instructions):
        print(f"\n[{i+1}/{len(instructions)}] Evaluating instruction:")
        print(f"  {instruction}")

        eval_result = evaluate_instruction(
            instruction=instruction,
            llm_client=llm_client,
            evaluator=evaluator,
            max_new_tokens=args.max_new_tokens,
        )

        result = {
            "idx": i,
            "instruction": instruction,
            "accuracy": eval_result["accuracy"],
            "correct": eval_result["correct"],
            "total": eval_result["total"],
        }
        results.append(result)

        print(f"  Accuracy: {eval_result['accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})")

        # Save incrementally after each evaluation
        save_incremental_results(output_file, results, metadata)
        print(f"  ✓ Saved to {output_file.name} ({len(results)}/{len(instructions)} completed)")

    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    accuracies = [r["accuracy"] for r in results]
    print(f"\nStatistics over {len(results)} instructions:")
    print(f"  Mean accuracy:   {sum(accuracies)/len(accuracies):.2%}")
    print(f"  Max accuracy:    {max(accuracies):.2%}")
    print(f"  Min accuracy:    {min(accuracies):.2%}")
    print(f"  Std:             {(sum((a - sum(accuracies)/len(accuracies))**2 for a in accuracies) / len(accuracies))**0.5:.2%}")

    print(f"\nTop 10 instructions:")
    for i, r in enumerate(results_sorted[:10]):
        print(f"  {i+1}. {r['accuracy']:.2%} | {r['instruction'][:60]}...")

    print(f"\nBottom 5 instructions:")
    for i, r in enumerate(results_sorted[-5:]):
        print(f"  {len(results_sorted)-4+i}. {r['accuracy']:.2%} | {r['instruction'][:60]}...")

    # Final save already done incrementally
    print(f"\nFinal results saved to: {output_file}")

    # Save top instructions to text file
    top_file = output_dir / f"top_instructions_{timestamp}.txt"
    with open(top_file, 'w', encoding='utf-8') as f:
        f.write(f"# Top Instructions for GSM8K\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Evaluated on: {len(evaluator)} test examples\n")
        f.write(f"# Timestamp: {timestamp}\n\n")

        for i, r in enumerate(results_sorted[:20]):
            f.write(f"## Rank {i+1}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})\n")
            f.write(f"{r['instruction']}\n\n")

    print(f"Top instructions saved to: {top_file}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
