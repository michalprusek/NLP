#!/usr/bin/env python3
"""
GSM8K Evaluation Script with Self-Consistency Sampling

This script allows you to evaluate different prompts on the GSM8K dataset
with optional self-consistency sampling (majority voting over N responses).

Usage:
    # Basic evaluation (single response per question)
    python evaluate_gsm8k.py --prompt "Solve this math problem step by step."
    
    # With self-consistency (5 samples per question)
    python evaluate_gsm8k.py --prompt "Solve this math problem step by step." --num-samples 5
    
    # Using a custom model
    python evaluate_gsm8k.py --model meta-llama/Llama-3.1-8B-Instruct --num-samples 10
    
    # Using vLLM backend for faster inference
    python evaluate_gsm8k.py --backend vllm --num-samples 5
    
    # Evaluate on a subset for quick testing
    python evaluate_gsm8k.py --max-examples 100 --num-samples 3
    
    # Use Math-Verify evaluator (more robust)
    python evaluate_gsm8k.py --evaluator math-verify --num-samples 5
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter
import sys

from src.llm_client import create_llm_client
from src.evaluator import GSM8KEvaluator, extract_answer, extract_ground_truth
from src.math_verify_evaluator import MathVerifyEvaluator


# Default prompt templates
# Note: Questions are automatically appended as "\n\nQuestion: {question}\nAnswer:"
DEFAULT_PROMPTS = {
    "basic": "Solve this math problem step by step. Provide your final answer as: final_answer: <number>",
    "cot": "Let's solve this step by step. Think through this carefully and show your work. Put your final numerical answer after 'final_answer:'",
    "concise": "Solve this problem and provide the final answer as a number after 'final_answer:'",
    "detailed": "Instructions:\n1. Read the problem carefully\n2. Identify what is being asked\n3. Show your step-by-step solution\n4. Provide the final answer as: final_answer: <number>",
}


def extract_answers_from_responses(responses: List[str], verbose: bool = False) -> List[Optional[str]]:
    """Extract numerical answers from a list of responses."""
    answers = []
    for i, response in enumerate(responses):
        answer = extract_answer(response, verbose=verbose)
        answers.append(answer)
        if verbose and i < 3:
            print(f"  Response {i+1}: {response[:100]}... -> Answer: {answer}")
    return answers


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """
    Select the most common answer from a list of answers.
    Returns None if all answers are None.
    """
    # Filter out None values
    valid_answers = [a for a in answers if a is not None]
    
    if not valid_answers:
        return None
    
    # Count occurrences
    counter = Counter(valid_answers)
    
    # Return most common (in case of tie, Counter.most_common returns first encountered)
    most_common = counter.most_common(1)[0][0]
    
    return most_common


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
    Evaluate GSM8K with optional self-consistency sampling.
    
    Args:
        llm_client: LLM client for generation
        evaluator: GSM8K evaluator instance
        prompt_template: Prompt template with {question} placeholder
        num_samples: Number of samples per question (1 = no self-consistency)
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
    print(f"  Dataset: GSM8K ({evaluator.split} split)")
    print(f"  Examples: {num_examples}")
    print(f"  Samples per question: {num_samples}")
    print(f"  Temperature: {temperature}")
    print(f"  Self-consistency: {'Enabled' if num_samples > 1 else 'Disabled'}")
    print(f"{'='*80}\n")

    # Prepare all prompts for all examples
    print("Preparing prompts...")
    all_prompts = []
    for idx in range(num_examples):
        example = dataset[idx]
        question = example['question']
        # Create prompt directly (like in main.py) - more robust than template formatting
        prompt = f"{prompt_template}\n\nQuestion: {question}\nAnswer:"
        all_prompts.append(prompt)

    # Generate responses: N batches of all questions
    print(f"Generating responses ({num_samples} sample(s) per question)...")
    all_responses = []  # Will be list of lists: [sample_idx][question_idx]

    for sample_idx in range(num_samples):
        print(f"  Generating sample {sample_idx + 1}/{num_samples} for all {num_examples} questions...")
        # Generate responses for ALL questions in one batch
        batch_responses = llm_client.generate_batch(all_prompts, temperature=temperature)
        all_responses.append(batch_responses)

    # Now process results with majority voting
    print("\nProcessing results and applying majority voting...")
    results = []
    correct = 0
    failed_extractions = 0

    for idx in range(num_examples):
        example = dataset[idx]
        question = example['question']

        # Collect all N responses for this question
        responses = [all_responses[sample_idx][idx] for sample_idx in range(num_samples)]

        if verbose and idx < 3:
            print(f"\n{'='*80}")
            print(f"Example {idx + 1}/{num_examples}")
            print(f"Question: {question[:200]}...")
            print(f"{'='*80}")
            for sample_idx, response in enumerate(responses):
                print(f"  Sample {sample_idx + 1}: {response[:100]}...")

        # Extract answers from all responses
        extracted_answers = extract_answers_from_responses(responses, verbose=(verbose and idx < 3))

        # Apply majority voting if using self-consistency
        if num_samples > 1:
            predicted_answer = majority_vote(extracted_answers)
            if verbose and idx < 3:
                print(f"  Extracted answers: {extracted_answers}")
                print(f"  Majority vote: {predicted_answer}")
        else:
            predicted_answer = extracted_answers[0]

        # Extract ground truth from the dataset
        gt_answer = extract_ground_truth(example['answer'])

        # Check if predicted answer matches ground truth
        is_correct = (predicted_answer is not None and predicted_answer == gt_answer)

        if predicted_answer is None:
            failed_extractions += 1

        if is_correct:
            correct += 1

        # Store detailed results
        result = {
            'idx': idx,
            'question': question,
            'ground_truth': gt_answer,
            'predicted': predicted_answer,
            'correct': is_correct,
            'num_samples': num_samples,
            'responses': responses if num_samples > 1 else [responses[0]],
            'extracted_answers': extracted_answers if num_samples > 1 else [extracted_answers[0]],
        }
        results.append(result)

        # Print progress
        if (idx + 1) % 10 == 0 or idx == 0:
            current_acc = correct / (idx + 1) * 100
            print(f"Progress: {idx + 1}/{num_examples} | Accuracy: {current_acc:.2f}% | Correct: {correct}/{idx + 1}")
    
    # Calculate final metrics
    accuracy = correct / num_examples * 100
    extraction_rate = (num_examples - failed_extractions) / num_examples * 100
    
    summary = {
        'accuracy': accuracy,
        'correct': correct,
        'total': num_examples,
        'failed_extractions': failed_extractions,
        'extraction_rate': extraction_rate,
        'num_samples': num_samples,
        'temperature': temperature,
        'prompt_template': prompt_template,
    }
    
    return {
        'summary': summary,
        'details': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K with optional self-consistency sampling",
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
                        help='Custom prompt prefix (question will be appended automatically). Example: "Solve this step by step"')
    parser.add_argument('--prompt-name', type=str, default='basic',
                        choices=list(DEFAULT_PROMPTS.keys()),
                        help='Use a predefined prompt template (default: basic)')
    
    # Self-consistency configuration
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples per question (1 = disabled, 5-10 recommended for self-consistency)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (higher = more diverse responses)')
    
    # Dataset configuration
    parser.add_argument('--dataset-path', type=str, default='datasets/gsm8k',
                        help='Path to GSM8K dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--max-examples', type=int, default=None,
                        help='Maximum number of examples to evaluate (for quick testing)')
    
    # Evaluator configuration
    parser.add_argument('--evaluator', type=str, default='standard', choices=['standard', 'math-verify'],
                        help='Evaluator to use (standard = exact match, math-verify = symbolic)')
    
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
    print(f"Note: Questions will be appended as '\\n\\nQuestion: <question>\\nAnswer:'\n")
    
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
    print(f"  Type: {args.evaluator}")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Split: {args.split}")
    
    if args.evaluator == 'math-verify':
        evaluator = MathVerifyEvaluator(
            dataset_path=args.dataset_path,
            split=args.split,
            debug=args.verbose,
        )
    else:
        evaluator = GSM8KEvaluator(
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
    
    # Print summary
    summary = results['summary']
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy: {summary['accuracy']:.2f}% ({summary['correct']}/{summary['total']})")
    print(f"Failed extractions: {summary['failed_extractions']} ({100 - summary['extraction_rate']:.2f}%)")
    print(f"Self-consistency samples: {summary['num_samples']}")
    print(f"Temperature: {summary['temperature']}")
    print(f"{'='*80}\n")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"gsm8k_eval_{timestamp}.json"
    
    # Add metadata
    results['metadata'] = {
        'model': args.model,
        'backend': args.backend,
        'evaluator': args.evaluator,
        'dataset_path': args.dataset_path,
        'split': args.split,
        'timestamp': timestamp,
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Also save a summary text file
    summary_file = output_dir / f"gsm8k_eval_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"GSM8K Evaluation Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Backend: {args.backend}\n")
        f.write(f"Evaluator: {args.evaluator}\n")
        f.write(f"Dataset: {args.dataset_path} ({args.split})\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Samples per question: {summary['num_samples']}\n")
        f.write(f"  Temperature: {summary['temperature']}\n")
        f.write(f"  Self-consistency: {'Enabled' if summary['num_samples'] > 1 else 'Disabled'}\n\n")
        f.write(f"Prompt Template:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{prompt_template}\n")
        f.write(f"{'-'*80}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Accuracy: {summary['accuracy']:.2f}%\n")
        f.write(f"  Correct: {summary['correct']}/{summary['total']}\n")
        f.write(f"  Failed extractions: {summary['failed_extractions']}\n")
        f.write(f"  Extraction rate: {summary['extraction_rate']:.2f}%\n")
    
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()

