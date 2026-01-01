#!/usr/bin/env python3
"""Evaluate best OPRO prompt on GSM8K test set."""
import os
import json
import glob
from datetime import datetime

# Use GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src.llm_client import create_llm_client
from src.gsm8k_evaluator import GSM8KEvaluator


def main():
    # Find most recent OPRO results
    result_files = sorted(glob.glob("results/opro_*.json"))
    if not result_files:
        print("No OPRO results found. Run OPRO first.")
        return

    latest = result_files[-1]
    print(f"Loading results from: {latest}")

    with open(latest) as f:
        results = json.load(f)

    best_prompt = results.get("best_prompt", "")
    val_accuracy = results.get("validation_accuracy", 0)

    print("=" * 60)
    print("OPRO BEST PROMPT - TEST SET EVALUATION")
    print("=" * 60)
    print(f"\nValidation accuracy: {val_accuracy:.2%}")
    print(f"\nBest prompt:\n{best_prompt}\n")
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

    # Format prompts in Q_end style
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
    print(f"Validation accuracy: {val_accuracy:.4f} ({val_accuracy:.2%})")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Test error rate: {test_error:.4f} ({test_error:.2%})")
    print(f"Correct: {test_results['correct']}/{test_results['total']}")
    print("=" * 60)

    # Update results file with test accuracy
    results["test_accuracy"] = test_accuracy
    results["test_error_rate"] = test_error
    results["test_correct"] = test_results['correct']
    results["test_total"] = test_results['total']

    with open(latest, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults updated in: {latest}")


if __name__ == "__main__":
    main()
