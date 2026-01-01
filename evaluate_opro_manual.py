#!/usr/bin/env python3
"""Evaluate best OPRO prompt (from log) on GSM8K test set."""
import os
import json
from datetime import datetime

# Use GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src.llm_client import create_llm_client
from src.gsm8k_evaluator import GSM8KEvaluator


def main():
    # Best prompts from OPRO run (90.8% validation accuracy)
    # Both achieved same score, testing the first one
    best_prompt = "文本：解决这个问题的方法是什么？"
    # Alternative: "文本：解决该问题的方法如下。"

    print("=" * 60)
    print("OPRO BEST PROMPT - TEST SET EVALUATION")
    print("=" * 60)
    print(f"\nValidation accuracy: 90.8%")
    print(f"\nBest prompt:\n{best_prompt}")
    print(f"(Translation: 'Text: What is the method to solve this problem?')\n")
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
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"Grid Best (baseline):")
    print(f"  Validation: 93.25% | Test: 90.14%")
    print(f"\nOPRO Best:")
    print(f"  Validation: 90.80% | Test: {test_accuracy:.2%}")
    print(f"\nDifference: {(test_accuracy - 0.9014)*100:+.2f}%")
    print("=" * 60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "method": "OPRO",
        "prompt": best_prompt,
        "validation_accuracy": 0.908,
        "test_accuracy": test_accuracy,
        "test_error_rate": test_error,
        "test_correct": test_results['correct'],
        "test_total": test_results['total'],
        "budget_used": 132327,
        "iterations": 64,
    }

    output_file = f"results/opro_200k_test_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
