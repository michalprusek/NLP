#!/usr/bin/env python3
"""
ProTeGi: Prompt Optimization with Textual Gradients for GSM8K

Based on "Automatic Prompt Optimization with Gradient Descent and Beam Search"
https://arxiv.org/abs/2305.03495

Usage:
    python -m protegi.run                           # Uses default Qwen model
    python -m protegi.run --model qwen              # Alias for Qwen/Qwen2.5-7B-Instruct
    python -m protegi.run --meta-model sonnet       # Use Claude Sonnet for meta-optimization
    python -m protegi.run --budget 150000           # 150k task LLM eval budget
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime

from shared.llm_client import create_llm_client
from shared.gsm8k_evaluator import GSM8KEvaluator
from protegi.protegi import ProTeGiOptimizer


# Model aliases for convenience
MODEL_ALIASES = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20251022",
}

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def resolve_model_alias(model: str) -> str:
    """Resolve model alias to full HuggingFace/API model name."""
    return MODEL_ALIASES.get(model.lower(), model)


def main():
    parser = argparse.ArgumentParser(
        description="ProTeGi prompt optimization for GSM8K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Task model. Aliases: {', '.join(MODEL_ALIASES.keys())}",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "deepinfra", "auto"],
        help="LLM backend",
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

    # ProTeGi parameters (from paper)
    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size b (paper default: 4)",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Optimization steps r (paper default: 6)",
    )

    parser.add_argument(
        "--gradients",
        type=int,
        default=4,
        help="Gradients per error group m (paper default: 4)",
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=2,
        help="Monte Carlo paraphrases per edit p (paper default: 2)",
    )

    parser.add_argument(
        "--max-successors",
        type=int,
        default=8,
        help="Max candidates per step (paper default: 8)",
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="Evaluation minibatch size (paper default: 64)",
    )

    # Budget
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Total task LLM evaluation budget (None = unlimited)",
    )

    # Generation params
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens for task model",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="protegi/results",
        help="Output directory",
    )

    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed step JSONs",
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
    print("ProTeGi: PROMPT OPTIMIZATION WITH TEXTUAL GRADIENTS")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Meta-model: {args.meta_model}")
    print(f"GPUs: {args.gpu_ids}")
    print(f"Beam size: {args.beam_size}")
    print(f"Steps: {args.steps}")
    print(f"Gradients/group: {args.gradients}")
    print(f"MC samples: {args.mc_samples}")
    print(f"Max successors: {args.max_successors}")
    print(f"Minibatch: {args.minibatch_size}")
    print(f"Budget: {args.budget if args.budget else 'unlimited'}")
    print("="*60 + "\n")

    # Initialize LLM client
    print(f"Loading model: {args.model}")
    llm_kwargs = {
        "model_name": args.model,
        "backend": args.backend,
    }
    if args.backend == "vllm":
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size

    llm_client = create_llm_client(**llm_kwargs)

    # Use same client for both task and meta if same model
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

    # Initialize ProTeGi
    optimizer = ProTeGiOptimizer(
        task_llm_client=task_llm,
        meta_llm_client=meta_llm,
        evaluator=evaluator,
        beam_size=args.beam_size,
        num_steps=args.steps,
        gradients_per_group=args.gradients,
        mc_samples=args.mc_samples,
        max_successors=args.max_successors,
        minibatch_size=args.minibatch_size,
        task_max_tokens=args.max_new_tokens,
        total_budget=args.budget,
    )

    # Set up detail output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_output_dir = None
    if args.save_details:
        detail_output_dir = str(output_dir / f"protegi_details_{timestamp}")

    # Run optimization
    best_prompt, history = optimizer.optimize(
        verbose=not args.quiet,
        save_details=args.save_details,
        output_dir=detail_output_dir,
    )

    # =========================================================================
    # TEST SET EVALUATION
    # =========================================================================
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    test_evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split="test",
        debug=args.debug,
    )
    print(f"Test set: {len(test_evaluator)} examples")

    # Get all test examples - Q_end format per CLAUDE.md: Q: {question}\n{instruction}\nA:
    test_batch = test_evaluator.get_batch(0, len(test_evaluator))
    test_questions = [ex['question'] for ex in test_batch]
    test_prompts = [f"Q: {q}\n{best_prompt}\nA:" for q in test_questions]

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

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_file = output_dir / f"protegi_{timestamp}.json"
    results = {
        "method": "protegi",
        "model": args.model,
        "meta_model": args.meta_model,
        "timestamp": timestamp,
        "config": {
            "beam_size": args.beam_size,
            "num_steps": args.steps,
            "gradients_per_group": args.gradients,
            "mc_samples": args.mc_samples,
            "max_successors": args.max_successors,
            "minibatch_size": args.minibatch_size,
        },
        "budget": {
            "task_llm_evaluations": optimizer.task_budget_used,
            "meta_gradient_calls": optimizer.meta_calls_gradient,
            "meta_edit_calls": optimizer.meta_calls_edit,
            "meta_paraphrase_calls": optimizer.meta_calls_paraphrase,
            "total_meta_calls": optimizer.total_meta_calls,
        },
        "best_prompt": best_prompt,
        "validation_accuracy": optimizer.beam[0].score if optimizer.beam else 0,
        "test_accuracy": test_accuracy,
        "test_error": test_error,
        "history": history,
    }

    # Save results with error handling
    import tempfile
    saved_to_primary = True

    try:
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save best prompt to text file
        prompt_file = output_file.with_suffix('.txt')
        with open(prompt_file, "w", encoding='utf-8') as f:
            f.write(f"# ProTeGi Best Prompt\n")
            f.write(f"# Model: {args.model}\n")
            f.write(f"# Meta-model: {args.meta_model}\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Test accuracy: {test_accuracy:.2%}\n\n")
            f.write(best_prompt)
    except (OSError, IOError) as e:
        print(f"\n[WARNING] Failed to save results to {output_file}: {e}")
        saved_to_primary = False
        # Fallback to temp directory
        fallback_dir = tempfile.gettempdir()
        try:
            output_file = Path(fallback_dir) / f"protegi_{timestamp}.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            prompt_file = output_file.with_suffix('.txt')
            with open(prompt_file, "w", encoding='utf-8') as f:
                f.write(f"# ProTeGi Best Prompt\n")
                f.write(f"# Model: {args.model}\n")
                f.write(f"# Meta-model: {args.meta_model}\n")
                f.write(f"# Timestamp: {timestamp}\n")
                f.write(f"# Test accuracy: {test_accuracy:.2%}\n\n")
                f.write(best_prompt)
            print(f"Results saved to fallback: {output_file}")
        except Exception as fallback_e:
            print(f"[ERROR] Fallback save also failed: {fallback_e}")

    if saved_to_primary:
        print(f"\nResults saved to: {output_file}")
        print(f"Best prompt saved to: {prompt_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best prompt:\n{best_prompt}")
    print(f"\nValidation accuracy: {optimizer.beam[0].score:.2%}" if optimizer.beam else "N/A")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Task LLM budget used: {optimizer.task_budget_used}")
    print(f"Meta LLM calls: {optimizer.total_meta_calls}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: ProTeGi optimization failed")
        print(f"{'='*60}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print(f"\nFor debugging, run with --debug flag")
        print(f"\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
