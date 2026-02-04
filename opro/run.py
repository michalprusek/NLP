#!/usr/bin/env python3
"""
OPRO: Optimization by PROmpting for GSM8K

Usage:
    python -m opro.run                           # Uses default Qwen model
    python -m opro.run --model qwen              # Alias for Qwen/Qwen2.5-7B-Instruct
    python -m opro.run --model qwen-3b           # Smaller Qwen model
    python -m opro.run --meta-model sonnet       # Use Claude Sonnet for meta-optimization
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from shared.llm_client import create_llm_client
from shared.gsm8k_evaluator import GSM8KEvaluator
from shared.incremental_saver import IncrementalPromptSaver
from opro.opro import OPROOptimizer


# Model aliases for convenience
MODEL_ALIASES = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "saul": "Equall/Saul-7B-Instruct-v1",
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20251022",
}

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def resolve_model_alias(model: str) -> str:
    """Resolve model alias to full HuggingFace/API model name."""
    return MODEL_ALIASES.get(model.lower(), model)


def main():
    parser = argparse.ArgumentParser(description="OPRO prompt optimization for GSM8K")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Task model (default: {DEFAULT_MODEL}). Aliases: {', '.join(MODEL_ALIASES.keys())}",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["transformers", "vllm", "claude"],
        help="LLM backend (default: vllm)",
    )

    parser.add_argument(
        "--meta-model",
        type=str,
        default=None,
        help="Meta-optimizer model (default: same as --model)",
    )

    parser.add_argument(
        "--meta-backend",
        type=str,
        default=None,
        help="Meta-optimizer backend (default: same as --backend)",
    )

    # Hardware
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (e.g., '0,1')",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism for vLLM",
    )

    # Optimization settings
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of optimization iterations (default: 200)",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Total LLM evaluation budget (default: unlimited). Each eval costs minibatch-size.",
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=261,
        help="Fixed evaluation set size (default: 261, ~3.5%% of GSM8K train)",
    )

    parser.add_argument(
        "--num-candidates",
        type=int,
        default=8,
        help="Candidates per iteration (default: 8)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens for task model (default: 2048)",
    )

    # Benchmarking parameters
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Max prompts to evaluate (for benchmarking). Stops after evaluating this many prompts.",
    )

    parser.add_argument(
        "--incremental-json",
        type=str,
        default=None,
        help="Path to save incremental JSON with evaluated prompts (for benchmarking).",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use for evaluation (default: train). Use 'test' for final benchmarking.",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="opro/results",
        help="Output directory (default: opro/results)",
    )

    parser.add_argument(
        "--save-eval-json",
        action="store_true",
        help="Save detailed evaluation JSONs",
    )

    parser.add_argument(
        "--verbose-meta",
        action="store_true",
        help="Print meta-model prompts and responses",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()

    # Resolve model aliases
    args.model = resolve_model_alias(args.model)

    # Set defaults
    if args.meta_model is None:
        args.meta_model = args.model
    else:
        args.meta_model = resolve_model_alias(args.meta_model)

    if args.meta_backend is None:
        args.meta_backend = args.backend

    # Set GPU visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "="*60)
    print("OPRO OPTIMIZATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Meta-model: {args.meta_model}")
    print(f"GPUs: {args.gpu_ids}")
    print(f"Iterations: {args.iterations}")
    print(f"Budget: {args.budget if args.budget else 'unlimited'}")
    print(f"Minibatch: {args.minibatch_size}")
    print(f"Candidates/iter: {args.num_candidates}")
    print(f"Split: {args.split}")
    print(f"Max prompts: {args.max_prompts if args.max_prompts else 'unlimited'}")
    if args.incremental_json:
        print(f"Incremental JSON: {args.incremental_json}")
    print("="*60 + "\n")

    # Initialize LLM client
    # Note: temperature and max_new_tokens are set at generation time in opro.py
    print(f"Loading model: {args.model}")
    llm_kwargs = {
        "model_name": args.model,
        "backend": args.backend,
    }
    if args.backend == "vllm":
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size

    llm_client = create_llm_client(**llm_kwargs)

    # Use same client for both task and meta (params set per-call in opro.py)
    if args.meta_model == args.model and args.meta_backend == args.backend:
        task_llm = llm_client
        meta_llm = llm_client
        print("Using same model for task and meta-optimization")
    else:
        task_llm = llm_client
        print(f"Loading meta model: {args.meta_model}")
        meta_llm_kwargs = {
            "model_name": args.meta_model,
            "backend": args.meta_backend,
        }
        meta_llm = create_llm_client(**meta_llm_kwargs)

    # Initialize evaluator
    print("Loading GSM8K dataset...")
    evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split=args.split,
        debug=args.debug,
    )
    print(f"{args.split.capitalize()} set: {len(evaluator)} examples\n")

    # Initialize incremental saver if requested
    incremental_saver = None
    if args.incremental_json:
        config = {
            "iterations": args.iterations,
            "budget": args.budget,
            "minibatch_size": args.minibatch_size,
            "num_candidates": args.num_candidates,
            "max_prompts": args.max_prompts,
            "split": args.split,
        }
        incremental_saver = IncrementalPromptSaver(
            output_path=args.incremental_json,
            method="opro",
            model=args.model,
            config=config,
        )
        print(f"Incremental saver initialized: {args.incremental_json}\n")

    # Initialize OPRO
    optimizer = OPROOptimizer(
        task_llm_client=task_llm,
        meta_llm_client=meta_llm,
        evaluator=evaluator,
        num_iterations=args.iterations,
        num_candidates_per_iter=args.num_candidates,
        minibatch_size=args.minibatch_size,
        task_max_tokens=args.max_new_tokens,  # For math solutions (default: 2048)
        meta_max_tokens=500,  # For prompt generation (short)
        total_budget=args.budget,  # Budget-based stopping
        max_prompts=args.max_prompts,  # Max prompts to evaluate
        incremental_saver=incremental_saver,  # Save prompts incrementally
    )

    # Set up eval output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output_dir = None
    if args.save_eval_json:
        eval_output_dir = str(output_dir / f"eval_{timestamp}")

    # Run optimization
    best_prompt, history = optimizer.optimize(
        verbose=not args.quiet,
        save_eval_json=args.save_eval_json,
        eval_output_dir=eval_output_dir,
        verbose_meta=args.verbose_meta,
    )

    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    test_evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split="test",
        debug=args.debug,
    )
    print(f"Test set: {len(test_evaluator)} examples")

    # Get all test examples
    test_batch = test_evaluator.get_batch(0, len(test_evaluator))
    test_questions = [ex['question'] for ex in test_batch]
    test_prompts = [f"Question: {q}\n\n{best_prompt}\n\nAnswer:" for q in test_questions]

    print(f"Evaluating best prompt on test set...")
    test_outputs = task_llm.generate_batch(
        test_prompts, temperature=0.0, max_new_tokens=args.max_new_tokens
    )

    test_indices = [ex['idx'] for ex in test_batch]
    test_results = test_evaluator.evaluate_batch(test_outputs, test_indices)
    test_accuracy = test_results['accuracy']
    test_error = 1.0 - test_accuracy

    print(f"\nTest accuracy: {test_accuracy:.2%}")
    print(f"Test error: {test_error:.2%}")
    print("="*60)

    # Save results
    output_file = output_dir / f"opro_{timestamp}.json"
    results = {
        "model": args.model,
        "meta_model": args.meta_model,
        "timestamp": timestamp,
        "config": {
            "iterations": args.iterations,
            "budget": args.budget,
            "minibatch_size": args.minibatch_size,
            "num_candidates": args.num_candidates,
            "max_prompts": args.max_prompts,
            "split": args.split,
        },
        "budget_used": optimizer.budget_used,
        "prompts_evaluated": optimizer.prompts_evaluated,
        "best_prompt": best_prompt,
        "validation_accuracy": max(sp.score for sp in optimizer.scored_prompts) if optimizer.scored_prompts else 0,
        "test_accuracy": test_accuracy,
        "test_error": test_error,
        "history": history,
    }
    if args.incremental_json:
        results["incremental_json"] = args.incremental_json

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save best prompt to text file
    prompt_file = output_file.with_suffix('.txt')
    with open(prompt_file, "w", encoding='utf-8') as f:
        f.write(f"# OPRO Best Prompt\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Timestamp: {timestamp}\n\n")
        f.write(best_prompt)

    print(f"\nResults saved to: {output_file}")
    print(f"Best prompt saved to: {prompt_file}")


if __name__ == "__main__":
    main()
