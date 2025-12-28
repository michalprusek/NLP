#!/usr/bin/env python3
"""Evaluate instructions on GSM8K test set.

Evaluates each instruction on all 1319 test samples using Qwen 2.5 7B Instruct.
Outputs JSONL with format: {instruction_id, error_rate, timestamp, instruction_text}

Usage:
    uv run python generation/cowboys_vec2text/evaluate_grid.py \
        --instructions datasets/cowboys/instructions_100.txt \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output datasets/cowboys/grid_100.jsonl
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm_client import create_llm_client, LLMClient
from src.gsm8k_evaluator import GSM8KEvaluator, extract_answer, compare_numbers, extract_ground_truth


def load_instructions(path: str) -> List[str]:
    """Load instructions from numbered text file.

    Expected format:
        # Comment lines (ignored)
        1. First instruction text
        2. Second instruction text
        ...

    Returns list of instruction strings (without number prefix).
    """
    instructions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Match numbered format: "N. instruction text"
            match = re.match(r"^\d+\.\s*(.+)$", line)
            if match:
                instructions.append(match.group(1))
    return instructions


def evaluate_instruction(
    instruction: str,
    test_data: List[Dict[str, Any]],
    client: LLMClient,
    max_tokens: int = 512,
) -> tuple[float, int, int]:
    """Evaluate a single instruction on the test set.

    Args:
        instruction: The instruction/system prompt to evaluate
        test_data: List of test examples with 'question' and 'answer' keys
        client: LLM client for generation
        max_tokens: Maximum tokens for generation

    Returns:
        Tuple of (error_rate, correct_count, total_count)
    """
    # Build prompts: instruction + question
    prompts = []
    for item in test_data:
        # Format: Q_end style (OPRO paper) - instruction after question
        prompt = f"Q: {item['question']}\n{instruction}\nA:"
        prompts.append(prompt)

    # Batch generate with deterministic sampling
    responses = client.generate_batch(
        prompts,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    # Evaluate responses
    correct = 0
    for response, item in zip(responses, test_data):
        predicted = extract_answer(response)
        try:
            expected = extract_ground_truth(item['answer'])
        except ValueError:
            # Skip if we can't extract ground truth
            continue

        if predicted is not None and compare_numbers(predicted, expected):
            correct += 1

    total = len(test_data)
    error_rate = 1.0 - (correct / total) if total > 0 else 1.0

    return error_rate, correct, total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate instructions on GSM8K test set"
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default="datasets/cowboys/instructions_100.txt",
        help="Path to instructions file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "deepinfra"],
        help="LLM backend",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/cowboys/grid_100.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for generation",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume from instruction index (0-based)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    args = parser.parse_args()

    # Load instructions
    instructions = load_instructions(args.instructions)
    print(f"Loaded {len(instructions)} instructions from {args.instructions}")

    if len(instructions) == 0:
        print("ERROR: No instructions found!")
        sys.exit(1)

    # Load GSM8K test set
    print("Loading GSM8K test set...")
    evaluator = GSM8KEvaluator(split="test")
    test_data = [
        {"question": evaluator.dataset[i]["question"], "answer": evaluator.dataset[i]["answer"]}
        for i in range(len(evaluator.dataset))
    ]
    print(f"Test set: {len(test_data)} samples")

    # Initialize LLM client
    print(f"\nInitializing {args.backend} client with model: {args.model}")
    client = create_llm_client(
        args.model,
        args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If resuming, read existing results to avoid duplicates
    existing_ids = set()
    if args.resume_from > 0 and output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing_ids.add(entry["instruction_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming from instruction {args.resume_from}, {len(existing_ids)} already evaluated")

    # Evaluate each instruction
    mode = "a" if args.resume_from > 0 else "w"

    print(f"\n{'='*60}")
    print(f"Starting evaluation of {len(instructions)} instructions")
    print(f"Each instruction evaluated on {len(test_data)} test samples")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    with open(output_path, mode) as f:
        for i, instruction in enumerate(instructions):
            # Skip already evaluated
            if i < args.resume_from or i in existing_ids:
                continue

            print(f"\n[{i+1}/{len(instructions)}] Evaluating instruction {i}:")
            print(f"  Instruction: {instruction}")

            error_rate, correct, total = evaluate_instruction(
                instruction,
                test_data,
                client,
                max_tokens=args.max_tokens,
            )

            # Build result
            result = {
                "instruction_id": i,
                "error_rate": error_rate,
                "timestamp": datetime.now().isoformat(),
                "instruction_text": instruction,
            }

            # Write immediately (append mode for resume support)
            f.write(json.dumps(result) + "\n")
            f.flush()

            accuracy = (1 - error_rate) * 100
            print(f"  Result: {correct}/{total} correct ({accuracy:.2f}% accuracy, {error_rate:.4f} error rate)")

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Cleanup vLLM
    if hasattr(client, 'cleanup'):
        client.cleanup()


if __name__ == "__main__":
    main()
