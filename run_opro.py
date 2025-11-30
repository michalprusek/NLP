#!/usr/bin/env python3
"""
OPRO: Optimization by PROmpting for GSM8K

Usage:
    python run_opro.py --model Qwen/Qwen2.5-7B-Instruct --backend vllm --iterations 10
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from src.llm_client import create_llm_client
from src.gsm8k_evaluator import GSM8KEvaluator
from src.opro import OPRO


def main():
    parser = argparse.ArgumentParser(description="OPRO prompt optimization for GSM8K")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Task model (e.g., Qwen/Qwen2.5-7B-Instruct)",
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

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
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

    # Set defaults
    if args.meta_model is None:
        args.meta_model = args.model
    if args.meta_backend is None:
        args.meta_backend = args.backend

    # Set GPU visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Print configuration
    print("\n" + "="*60)
    print("OPRO OPTIMIZATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Meta-model: {args.meta_model}")
    print(f"GPUs: {args.gpu_ids}")
    print(f"Iterations: {args.iterations}")
    print(f"Minibatch: {args.minibatch_size}")
    print(f"Candidates/iter: {args.num_candidates}")
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
        split="train",
        debug=args.debug,
    )
    print(f"Training set: {len(evaluator)} examples\n")

    # Initialize OPRO
    optimizer = OPRO(
        task_llm_client=task_llm,
        meta_llm_client=meta_llm,
        evaluator=evaluator,
        num_iterations=args.iterations,
        num_candidates_per_iter=args.num_candidates,
        minibatch_size=args.minibatch_size,
        task_max_tokens=args.max_new_tokens,  # For math solutions (default: 2048)
        meta_max_tokens=500,  # For prompt generation (short)
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

    # Save results
    output_file = output_dir / f"opro_{timestamp}.json"
    results = {
        "model": args.model,
        "meta_model": args.meta_model,
        "timestamp": timestamp,
        "config": {
            "iterations": args.iterations,
            "minibatch_size": args.minibatch_size,
            "num_candidates": args.num_candidates,
        },
        "best_prompt": best_prompt,
        "history": history,
    }

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
