#!/usr/bin/env python3
"""
Claudette ToS Classification Evaluation Script with Self-Consistency Sampling

This script allows you to evaluate different prompts on the Claudette dataset
with optional self-consistency sampling (majority voting over N responses).

Usage:
    # Basic evaluation (single response per clause)
    python evaluate_claudette.py --prompt "Classify this Terms of Service clause."

    # With self-consistency (5 samples per clause)
    python evaluate_claudette.py --prompt "Classify this clause." --num-samples 5

    # Using a custom model
    python evaluate_claudette.py --model meta-llama/Llama-3.1-8B-Instruct --num-samples 10

    # Using vLLM backend for faster inference
    python evaluate_claudette.py --backend vllm --num-samples 5

    # Evaluate on a subset for quick testing
    python evaluate_claudette.py --max-examples 100 --num-samples 3
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import sys

from src.llm_client import create_llm_client
from src.claudette_evaluator import ClaudetteEvaluator, extract_labels_from_output, get_ground_truth_labels
from src.metrics import format_metrics_table, format_metrics_compact


# Default prompt templates for Claudette ToS classification
# Note: Clauses are automatically appended as "\n\nQuestion: {clause}\nAnswer:"
DEFAULT_PROMPTS = {
    "basic": "Classify the following Terms of Service clause into one of 9 categories (0-8). Provide your answer as: LABEL: <number>",
    "detailed": "Analyze the Terms of Service clause and classify it into one of these categories:\n0: Limitation of liability\n1: Unilateral termination\n2: Unilateral change\n3: Arbitration\n4: Content removal\n5: Choice of law\n6: Other\n7: Contract by using\n8: Jurisdiction\n\nProvide your classification as: LABEL: <number>",
    "reasoning": "Read the Terms of Service clause carefully, consider its legal implications, then classify it into the most appropriate category (0-8). Explain your reasoning briefly, then provide your answer as: LABEL: <number>",
}


def extract_labels_from_responses(responses: List[str], verbose: bool = False) -> List[Set[int]]:
    """Extract labels from a list of responses."""
    labels_list = []
    for i, response in enumerate(responses):
        labels = extract_labels_from_output(response, verbose=verbose)
        labels_list.append(labels)
        if verbose and i < 3:
            print(f"  Response {i+1}: {response[:100]}... -> Labels: {labels}")
    return labels_list


def majority_vote_labels(labels_list: List[Set[int]]) -> Set[int]:
    """
    Select the most common label(s) from a list of label sets.
    Returns empty set if all label sets are empty.

    For single-label classification (Claudette), we typically get sets with 0 or 1 element.
    This function finds the most frequently occurring label across all samples.
    """
    # Flatten all labels
    all_labels = []
    for labels in labels_list:
        all_labels.extend(labels)

    if not all_labels:
        return set()

    # Count occurrences
    counter = Counter(all_labels)

    # Return most common label as a set (for consistency with evaluator API)
    most_common_label = counter.most_common(1)[0][0]

    return {most_common_label}


def evaluate_with_self_consistency(
    llm_client,
    evaluator,
    prompt_template: str,
    num_samples: int = 1,
    max_examples: Optional[int] = None,
    temperature: float = 0.7,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate Claudette with optional self-consistency sampling.

    Args:
        llm_client: LLM client for generation
        evaluator: Claudette evaluator instance
        prompt_template: Prompt template
        num_samples: Number of samples per clause (1 = no self-consistency)
        max_examples: Maximum number of examples to evaluate (None = all)
        temperature: Sampling temperature (higher = more diverse)
        verbose: Print detailed information

    Returns:
        Dictionary with evaluation results
    """
    dataset = evaluator.dataset

    # Limit examples if requested
    if max_examples is not None:
        num_examples = min(max_examples, len(dataset))
    else:
        num_examples = len(dataset)

    print(f"\n{'='*80}")
    print(f"Evaluation Configuration:")
    print(f"  Dataset: Claudette ({evaluator.split} split)")
    print(f"  Examples: {num_examples}")
    print(f"  Samples per clause: {num_samples}")
    print(f"  Temperature: {temperature}")
    print(f"  Self-consistency: {'Enabled' if num_samples > 1 else 'Disabled'}")
    print(f"{'='*80}\n")

    # Prepare all prompts for all examples
    print("Preparing prompts...")
    all_prompts = []
    for idx in range(num_examples):
        example = dataset[idx]
        # Use 'sentence' field from Arrow dataset format
        clause = example.get('sentence')
        if clause is None:
            clause = example.get('text')
        if clause is None or not clause.strip():
            raise ValueError(
                f"Example {idx} has no valid text content. "
                f"Available fields: {list(example.keys())}"
            )
        # Create prompt directly (like in main.py) - more robust
        prompt = f"{prompt_template}\n\nQuestion: {clause}\nAnswer:"
        all_prompts.append(prompt)

    # Generate responses: N batches of all clauses
    print(f"Generating responses ({num_samples} sample(s) per clause)...")
    all_responses = []  # Will be list of lists: [sample_idx][clause_idx]

    for sample_idx in range(num_samples):
        print(f"  Generating sample {sample_idx + 1}/{num_samples} for all {num_examples} clauses...")
        try:
            # Generate responses for ALL clauses in one batch
            batch_responses = llm_client.generate_batch(all_prompts, temperature=temperature)

            # Validate response count
            if len(batch_responses) != num_examples:
                raise ValueError(
                    f"Expected {num_examples} responses, got {len(batch_responses)} "
                    f"in sample {sample_idx + 1}"
                )

            all_responses.append(batch_responses)

        except Exception as e:
            print(f"ERROR: Failed to generate sample {sample_idx + 1}/{num_samples}")
            print(f"  Error: {e}")
            raise RuntimeError(
                f"Batch generation failed for sample {sample_idx + 1}. "
                f"Generated {len(all_responses)}/{num_samples} samples before failure."
            ) from e

    # Now process results with majority voting
    print("\nProcessing results and applying majority voting...")
    results = []
    correct = 0
    failed_extractions = 0

    for idx in range(num_examples):
        example = dataset[idx]
        # Use 'sentence' field from Arrow dataset format
        # Validate dataset fields - fail fast on missing text
        clause = example.get('sentence')
        if clause is None:
            clause = example.get('text')
        if clause is None or not clause.strip():
            raise ValueError(
                f"Example {idx} has no valid text content. "
                f"Available fields: {list(example.keys())}"
            )

        # Collect all N responses for this clause
        responses = [all_responses[sample_idx][idx] for sample_idx in range(num_samples)]

        if verbose and idx < 3:
            print(f"\n{'='*80}")
            print(f"Example {idx + 1}/{num_examples}")
            print(f"Clause: {clause[:200]}...")
            print(f"{'='*80}")
            for sample_idx, response in enumerate(responses):
                print(f"  Sample {sample_idx + 1}: {response[:100]}...")

        # Extract labels from all responses
        extracted_labels_list = extract_labels_from_responses(responses, verbose=(verbose and idx < 3))

        # Apply majority voting if using self-consistency
        if num_samples > 1:
            predicted_labels = majority_vote_labels(extracted_labels_list)
            if verbose and idx < 3:
                print(f"  Extracted labels: {extracted_labels_list}")
                print(f"  Majority vote: {predicted_labels}")
        else:
            predicted_labels = extracted_labels_list[0]

        # Extract ground truth from the dataset (use proper multi-label extraction)
        gt_labels = get_ground_truth_labels(example)

        # Check if predicted labels match ground truth (exact match)
        is_correct = (predicted_labels == gt_labels)

        if not predicted_labels:
            failed_extractions += 1

        if is_correct:
            correct += 1

        # Store detailed results
        result = {
            'idx': idx,
            'clause': clause,
            'ground_truth': list(gt_labels)[0] if gt_labels else None,
            'predicted': list(predicted_labels)[0] if predicted_labels else None,
            'correct': is_correct,
            'num_samples': num_samples,
            'responses': responses if num_samples > 1 else [responses[0]],
            'extracted_labels': [list(labels) for labels in extracted_labels_list] if num_samples > 1 else [list(extracted_labels_list[0])],
        }
        results.append(result)

        # Print progress
        if (idx + 1) % 10 == 0 or idx == 0:
            current_acc = correct / (idx + 1) * 100
            print(f"Progress: {idx + 1}/{num_examples} | Accuracy: {current_acc:.2f}% | Correct: {correct}/{idx + 1}")

    # Calculate final metrics
    accuracy = correct / num_examples * 100
    extraction_rate = (num_examples - failed_extractions) / num_examples * 100

    # Compute comprehensive multi-label metrics
    y_true_all = []
    y_pred_all = []
    for result in results:
        # Convert to sets for metrics computation
        y_true_all.append(set(result['ground_truth']) if isinstance(result['ground_truth'], list) else {result['ground_truth']})
        y_pred_all.append(set(result['predicted']) if isinstance(result['predicted'], list) else ({result['predicted']} if result['predicted'] is not None else set()))

    from src.metrics import compute_multilabel_metrics
    multilabel_metrics = compute_multilabel_metrics(
        y_true=y_true_all,
        y_pred=y_pred_all,
        num_classes=9,
    )

    summary = {
        'accuracy': accuracy,
        'correct': correct,
        'total': num_examples,
        'failed_extractions': failed_extractions,
        'extraction_rate': extraction_rate,
        'num_samples': num_samples,
        'temperature': temperature,
        'prompt_template': prompt_template,
        # Add comprehensive metrics
        'micro_f1': multilabel_metrics['micro_f1'],
        'macro_f1': multilabel_metrics['macro_f1'],
        'weighted_f1': multilabel_metrics['weighted_f1'],
        'micro_precision': multilabel_metrics['micro_precision'],
        'micro_recall': multilabel_metrics['micro_recall'],
        'macro_precision': multilabel_metrics['macro_precision'],
        'macro_recall': multilabel_metrics['macro_recall'],
        'weighted_precision': multilabel_metrics['weighted_precision'],
        'weighted_recall': multilabel_metrics['weighted_recall'],
        'hamming_loss': multilabel_metrics['hamming_loss'],
        'per_class': multilabel_metrics['per_class'],
        'confusion_matrix': multilabel_metrics['confusion_matrix'],
        'support': multilabel_metrics['support'],
    }

    return {
        'summary': summary,
        'details': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Claudette ToS classification with optional self-consistency sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model configuration
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--backend', type=str, default='vllm', choices=['transformers', 'vllm'],
                        help='Backend to use for inference')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu, mps)')

    # Prompt configuration
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt prefix (clause will be appended automatically). Example: "Classify this clause."')
    parser.add_argument('--prompt-name', type=str, default='basic',
                        choices=list(DEFAULT_PROMPTS.keys()),
                        help='Use a predefined prompt template (default: basic)')

    # Self-consistency configuration
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples per clause (1 = disabled, 5-10 recommended for self-consistency)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (higher = more diverse responses)')

    # Dataset configuration
    parser.add_argument('--dataset-path', type=str, default='datasets/claudette',
                        help='Path to Claudette dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--max-examples', type=int, default=None,
                        help='Maximum number of examples to evaluate (for quick testing)')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')

    args = parser.parse_args()

    # Determine prompt template
    if args.prompt is not None:
        prompt_template = args.prompt
        print(f"Using custom prompt template")
    else:
        prompt_template = DEFAULT_PROMPTS[args.prompt_name]
        print(f"Using predefined prompt: {args.prompt_name}")

    print(f"\nPrompt template:\n{'-'*80}\n{prompt_template}\n{'-'*80}\n")
    print(f"Note: Clauses will be appended as '\\n\\nQuestion: <clause>\\nAnswer:'\n")

    # Create LLM client
    print(f"Initializing LLM client...")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")

    llm_client = create_llm_client(
        model_name=args.model,
        backend=args.backend,
        device=args.device,
        temperature=args.temperature,
    )

    # Create evaluator
    print(f"\nInitializing evaluator...")
    print(f"  Type: ClaudetteEvaluator")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Split: {args.split}")

    evaluator = ClaudetteEvaluator(
        dataset_path=args.dataset_path,
        split=args.split,
        debug=args.verbose,
    )

    # Run evaluation
    print(f"\nStarting evaluation...")
    results = evaluate_with_self_consistency(
        llm_client=llm_client,
        evaluator=evaluator,
        prompt_template=prompt_template,
        num_samples=args.num_samples,
        max_examples=args.max_examples,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    # Print summary with comprehensive metrics
    summary = results['summary']
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Subset Accuracy (Exact Match): {summary['accuracy']:.2f}% ({summary['correct']}/{summary['total']})")
    print(f"Failed extractions: {summary['failed_extractions']} ({100 - summary['extraction_rate']:.2f}%)")
    print(f"Self-consistency samples: {summary['num_samples']}")
    print(f"Temperature: {summary['temperature']}")
    print(f"\nComprehensive Metrics:")
    print(f"  Micro F1:    {summary.get('micro_f1', 0.0):.2%}")
    print(f"  Macro F1:    {summary.get('macro_f1', 0.0):.2%}")
    print(f"  Weighted F1: {summary.get('weighted_f1', 0.0):.2%}")
    print(f"  Hamming Loss: {summary.get('hamming_loss', 0.0):.4f}")
    print(f"{'='*80}\n")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"claudette_eval_{timestamp}.json"

    # Add metadata
    results['metadata'] = {
        'model': args.model,
        'backend': args.backend,
        'dataset_path': args.dataset_path,
        'split': args.split,
        'timestamp': timestamp,
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Also save a summary text file with comprehensive metrics
    summary_file = output_dir / f"claudette_eval_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Claudette ToS Classification Evaluation Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Backend: {args.backend}\n")
        f.write(f"Dataset: {args.dataset_path} ({args.split})\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Samples per clause: {summary['num_samples']}\n")
        f.write(f"  Temperature: {summary['temperature']}\n")
        f.write(f"  Self-consistency: {'Enabled' if summary['num_samples'] > 1 else 'Disabled'}\n\n")
        f.write(f"Prompt Template:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{prompt_template}\n")
        f.write(f"{'-'*80}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Subset Accuracy (Exact Match): {summary['accuracy']:.2f}%\n")
        f.write(f"  Correct: {summary['correct']}/{summary['total']}\n")
        f.write(f"  Failed extractions: {summary['failed_extractions']}\n")
        f.write(f"  Extraction rate: {summary['extraction_rate']:.2f}%\n\n")
        f.write(f"Comprehensive Metrics:\n")
        f.write(f"  Micro F1:         {summary.get('micro_f1', 0.0):.2%}\n")
        f.write(f"  Micro Precision:  {summary.get('micro_precision', 0.0):.2%}\n")
        f.write(f"  Micro Recall:     {summary.get('micro_recall', 0.0):.2%}\n\n")
        f.write(f"  Macro F1:         {summary.get('macro_f1', 0.0):.2%}\n")
        f.write(f"  Macro Precision:  {summary.get('macro_precision', 0.0):.2%}\n")
        f.write(f"  Macro Recall:     {summary.get('macro_recall', 0.0):.2%}\n\n")
        f.write(f"  Weighted F1:      {summary.get('weighted_f1', 0.0):.2%}\n")
        f.write(f"  Weighted Precision: {summary.get('weighted_precision', 0.0):.2%}\n")
        f.write(f"  Weighted Recall:  {summary.get('weighted_recall', 0.0):.2%}\n\n")
        f.write(f"  Hamming Loss:     {summary.get('hamming_loss', 0.0):.4f}\n\n")

        # Add per-class metrics
        if summary.get('per_class'):
            f.write(f"Per-Class Metrics:\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Class':<35} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}\n")
            f.write(f"{'-'*35} {'-'*12} {'-'*12} {'-'*12} {'-'*10}\n")
            for class_idx in sorted(summary['per_class'].keys()):
                m = summary['per_class'][class_idx]
                f.write(f"{class_idx}. {m['name']:<32} {m['precision']:>11.1%} {m['recall']:>11.1%} {m['f1']:>11.1%} {m['support']:>9}\n")

    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
