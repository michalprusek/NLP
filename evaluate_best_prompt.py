#!/usr/bin/env python3
"""Evaluate best prompt from grid on GSM8K test set.

This script evaluates the best-performing prompt from the HbBoPs grid
on the GSM8K TEST set (not validation) to get a final held-out evaluation.
The prompt was originally selected based on validation set performance.

Requires: GPU 1 to be available (CUDA_VISIBLE_DEVICES=1)
"""
import os
import sys
import json
from datetime import datetime

# Use GPU 1 (free)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src.llm_client import create_llm_client
from src.gsm8k_evaluator import GSM8KEvaluator


def main():
    # Load BOLT best result
    bolt_result_path = "bolt/results/20260110_132806/best_result.json"
    with open(bolt_result_path) as f:
        bolt_result = json.load(f)

    best_instruction = bolt_result["best_instruction"]
    best_exemplars = bolt_result["best_exemplar_texts"]
    val_accuracy = bolt_result["best_accuracy"]
    val_error = bolt_result["best_error"]

    # Build exemplar string
    exemplar_str = "\n\n".join(best_exemplars)

    print("=" * 60)
    print("BOLT BEST PROMPT - TEST SET EVALUATION")
    print("=" * 60)
    print(f"\nInstruction: {best_instruction}")
    print(f"Exemplars: {len(best_exemplars)} examples")
    print(f"Validation accuracy: {val_accuracy:.2%}")
    print("=" * 60)

    # Load model
    print("\nLoading Qwen model on GPU 1...")
    llm = create_llm_client(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        backend="vllm",
        tensor_parallel_size=1
    )

    # Load test set (same source as BOLT validation for consistency)
    print("Loading test set...")
    with open("hbbops_improved_2/data/test.json") as f:
        test_data = json.load(f)
    print(f"Test set size: {len(test_data)} examples\n")

    test_questions = [ex['question'] for ex in test_data]
    test_answers = [ex['answer'] for ex in test_data]

    # Format prompts with exemplars + instruction (Q_end style)
    test_prompts = [
        f"{exemplar_str}\n\nQ: {q}\n{best_instruction}\nA:"
        for q in test_questions
    ]

    print(f"Evaluating on {len(test_prompts)} test examples...")
    test_outputs = llm.generate_batch(
        test_prompts,
        temperature=0.0,
        max_new_tokens=2048
    )

    # Extract answers and compare
    from src.gsm8k_evaluator import extract_answer

    correct = 0
    for i, (output, answer_text) in enumerate(zip(test_outputs, test_answers)):
        # Extract predicted answer (last number)
        pred = extract_answer(output)
        # Extract gold answer (after ####)
        if "####" in answer_text:
            gold = answer_text.split("####")[-1].strip()
            gold = gold.replace(",", "")
            try:
                gold = float(gold)
            except ValueError:
                gold = None
        else:
            gold = extract_answer(answer_text)

        # Ensure both are floats for comparison
        try:
            pred_f = float(pred) if pred is not None else None
            gold_f = float(gold) if gold is not None else None
        except (ValueError, TypeError):
            pred_f = None
            gold_f = None

        if pred_f is not None and gold_f is not None and abs(pred_f - gold_f) < 1e-6:
            correct += 1

    test_accuracy = correct / len(test_data)
    test_error = 1.0 - test_accuracy

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Test error rate: {test_error:.4f} ({test_error:.2%})")
    print(f"Correct: {correct}/{len(test_data)}")
    print(f"\nComparison:")
    print(f"  Validation accuracy: {val_accuracy:.2%}")
    print(f"  Test accuracy:       {test_accuracy:.2%}")
    print(f"  Difference:          {(test_accuracy - val_accuracy)*100:+.2f}%")
    print("=" * 60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "source": "BOLT v2 (24D, beta=0.02)",
        "instruction": best_instruction,
        "num_exemplars": len(best_exemplars),
        "exemplar_ids": bolt_result["best_exemplar_ids"],
        "validation_error_rate": val_error,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "test_error_rate": test_error,
        "test_correct": correct,
        "test_total": len(test_data),
    }

    output_file = f"results/bolt_test_{timestamp}.json"
    try:
        os.makedirs("results", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    except (IOError, OSError, PermissionError) as e:
        print(f"\nERROR: Failed to save results to {output_file}: {e}")
        print("Results (for manual recovery):")
        print(json.dumps(results, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
