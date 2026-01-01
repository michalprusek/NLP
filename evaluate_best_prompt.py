#!/usr/bin/env python3
"""Evaluate best prompt from grid on GSM8K test set."""
import os
import json
from datetime import datetime

# Use GPU 1 (free)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src.llm_client import create_llm_client
from src.gsm8k_evaluator import GSM8KEvaluator


def main():
    # Best prompt from grid (error_rate: 0.0675, accuracy: 93.25%)
    best_prompt = (
        "Identify the problem, solve it, and complete the computation. "
        "The result is a mathematical expression that can be interpreted in a variety of ways. "
        "Identify all of the numbers in the problem and solve the equation for each of them. "
        "The final result is the numerical equivalent of the digits in the answer. Mathematical Problems."
    )

    print("=" * 60)
    print("BEST PROMPT FROM GRID - TEST SET EVALUATION")
    print("=" * 60)
    print(f"\nPrompt (validation error_rate: 0.0675, accuracy: 93.25%):")
    print(f"{best_prompt}\n")
    print("=" * 60)

    # Load model
    print("\nLoading Qwen model on GPU 1...")
    llm = create_llm_client(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        backend="vllm",
        tensor_parallel_size=1
    )

    # Load test set
    print("Loading GSM8K test set...")
    test_evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split="test"
    )
    print(f"Test set size: {len(test_evaluator)} examples\n")

    # Get all test examples
    test_batch = test_evaluator.get_batch(0, len(test_evaluator))
    test_questions = [ex['question'] for ex in test_batch]

    # Format prompts in Q_end style (instruction AFTER question)
    test_prompts = [f"Q: {q}\n{best_prompt}\nA:" for q in test_questions]

    print(f"Evaluating on {len(test_prompts)} test examples...")
    test_outputs = llm.generate_batch(
        test_prompts,
        temperature=0.0,
        max_new_tokens=2048
    )

    test_indices = [ex['idx'] for ex in test_batch]
    test_results = test_evaluator.evaluate_batch(test_outputs, test_indices)
    test_accuracy = test_results['accuracy']
    test_error = 1.0 - test_accuracy

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Test error rate: {test_error:.4f} ({test_error:.2%})")
    print(f"Correct: {test_results['correct']}/{test_results['total']}")
    print("=" * 60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "prompt": best_prompt,
        "validation_error_rate": 0.0675,
        "validation_accuracy": 0.9325,
        "test_accuracy": test_accuracy,
        "test_error_rate": test_error,
        "test_correct": test_results['correct'],
        "test_total": test_results['total'],
    }

    output_file = f"results/grid_best_test_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
