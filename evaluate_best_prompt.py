#!/usr/bin/env python3
"""Evaluate best prompt from BOLT on GSM8K test set.

This script evaluates the best-performing prompt from BOLT optimization
on the GSM8K TEST set (not validation) to get a final held-out evaluation.
The prompt was originally selected based on validation set performance.

Usage:
    uv run python evaluate_best_prompt.py --result-path bolt/results/YYYYMMDD_HHMMSS/best_result.json
    uv run python evaluate_best_prompt.py --find-latest  # Auto-find most recent result
"""
import argparse
import glob
import os
import sys
import json
from datetime import datetime

# Use GPU 1 (free)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src.llm_client import create_llm_client
from src.gsm8k_evaluator import GSM8KEvaluator, extract_answer


def find_latest_result() -> str:
    """Find the most recent BOLT result file."""
    pattern = "bolt/results/*/best_result.json"
    results = sorted(glob.glob(pattern))
    if not results:
        raise FileNotFoundError(
            f"No BOLT results found matching pattern: {pattern}\n"
            "Run BOLT optimization first, or specify --result-path explicitly."
        )
    return results[-1]


def load_bolt_result(path: str) -> dict:
    """Load and validate BOLT result file."""
    try:
        with open(path) as f:
            result = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: BOLT result file not found: {path}")
        print("Run BOLT optimization first, or update --result-path.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in result file {path}: {e}")
        sys.exit(1)

    # Validate required fields
    required_fields = ["best_instruction", "best_exemplar_texts", "best_accuracy", "best_error"]
    missing = [f for f in required_fields if f not in result]
    if missing:
        print(f"ERROR: Result file missing required fields: {missing}")
        print(f"Available fields: {list(result.keys())}")
        sys.exit(1)

    return result


def load_test_data(path: str) -> list:
    """Load test data with error handling."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Test data file not found: {path}")
        print("Ensure the test data exists at the expected path.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in test data file: {e}")
        sys.exit(1)

    if not data:
        print(f"ERROR: Test data file is empty: {path}")
        sys.exit(1)

    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate BOLT prompt on GSM8K test set")
    parser.add_argument(
        "--result-path",
        type=str,
        help="Path to BOLT best_result.json file"
    )
    parser.add_argument(
        "--find-latest",
        action="store_true",
        help="Auto-find the most recent BOLT result"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="datasets/gsm8k/test.json",
        help="Path to test data JSON file (default: datasets/gsm8k/test.json)"
    )
    args = parser.parse_args()

    # Determine result path
    if args.result_path:
        bolt_result_path = args.result_path
    elif args.find_latest:
        bolt_result_path = find_latest_result()
        print(f"Found latest result: {bolt_result_path}")
    else:
        # Default to a known path for backward compatibility
        bolt_result_path = "bolt/results/20260110_132806/best_result.json"
        if not os.path.exists(bolt_result_path):
            print("No --result-path specified and default not found.")
            print("Use --find-latest to auto-detect or specify --result-path explicitly.")
            sys.exit(1)

    # Load BOLT result
    bolt_result = load_bolt_result(bolt_result_path)

    best_instruction = bolt_result["best_instruction"]
    best_exemplars = bolt_result["best_exemplar_texts"]
    val_accuracy = bolt_result["best_accuracy"]
    val_error = bolt_result["best_error"]

    # Build exemplar string
    exemplar_str = "\n\n".join(best_exemplars)

    print("=" * 60)
    print("BOLT BEST PROMPT - TEST SET EVALUATION")
    print("=" * 60)
    print(f"\nResult file: {bolt_result_path}")
    print(f"Instruction: {best_instruction}")
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

    # Load test set
    print("Loading test set...")
    test_data = load_test_data(args.test_data)
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
    correct = 0
    parse_failures = 0
    for i, (output, answer_text) in enumerate(zip(test_outputs, test_answers)):
        # Extract predicted answer (last number)
        pred = extract_answer(output)
        # Extract gold answer (after ####)
        gold = None
        if "####" in answer_text:
            gold_str = answer_text.split("####")[-1].strip().replace(",", "")
            try:
                gold = float(gold_str)
            except ValueError:
                parse_failures += 1
                continue
        else:
            gold = extract_answer(answer_text)
            if gold is not None:
                try:
                    gold = float(gold)
                except (ValueError, TypeError):
                    parse_failures += 1
                    continue

        # Ensure both are floats for comparison
        try:
            pred_f = float(pred) if pred is not None else None
            gold_f = float(gold) if gold is not None else None
        except (ValueError, TypeError):
            pred_f = None
            gold_f = None

        if pred_f is not None and gold_f is not None and abs(pred_f - gold_f) < 1e-6:
            correct += 1

    valid_samples = len(test_data) - parse_failures
    test_accuracy = correct / valid_samples if valid_samples > 0 else 0.0
    test_error = 1.0 - test_accuracy

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Test error rate: {test_error:.4f} ({test_error:.2%})")
    print(f"Correct: {correct}/{valid_samples}")
    if parse_failures > 0:
        print(f"Parse failures: {parse_failures} (excluded from accuracy)")
    print(f"\nComparison:")
    print(f"  Validation accuracy: {val_accuracy:.2%}")
    print(f"  Test accuracy:       {test_accuracy:.2%}")
    print(f"  Difference:          {(test_accuracy - val_accuracy)*100:+.2f}%")
    print("=" * 60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "source": "BOLT v4 (24D, beta=0.02)",
        "result_file": bolt_result_path,
        "instruction": best_instruction,
        "num_exemplars": len(best_exemplars),
        "exemplar_ids": bolt_result.get("best_exemplar_ids", []),
        "validation_error_rate": val_error,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "test_error_rate": test_error,
        "test_correct": correct,
        "test_total": valid_samples,
        "parse_failures": parse_failures,
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
