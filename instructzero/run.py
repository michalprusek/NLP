#!/usr/bin/env python3
"""
InstructZero: InstructZero-style Bayesian Optimization for GSM8K

Optimizes prompts in low-dimensional soft prompt space using
Bayesian optimization with Gaussian Process surrogate.

Usage:
    uv run python -m instructzero.run --max-calls 50000
    uv run python -m instructzero.run --model qwen --batch-size 4
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="InstructZero: InstructZero-style Prompt Optimization"
    )

    # Model configuration
    parser.add_argument(
        "--model", type=str, default="qwen",
        help="Task model (qwen, llama, haiku, sonnet)"
    )
    parser.add_argument(
        "--backend", type=str, default="vllm",
        choices=["vllm", "openai", "deepinfra", "anthropic"],
        help="LLM backend"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Tensor parallel size for vLLM"
    )

    # Optimization parameters
    parser.add_argument(
        "--max-calls", type=int, default=50000,
        help="Maximum LLM calls budget"
    )
    parser.add_argument(
        "--n-initial", type=int, default=10,
        help="Number of initial evaluations"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Candidates per iteration"
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=150,
        help="GSM8K examples per evaluation"
    )

    # Soft prompt parameters
    parser.add_argument(
        "--intrinsic-dim", type=int, default=10,
        help="Soft prompt dimension"
    )
    parser.add_argument(
        "--projection-dim", type=int, default=50,
        help="Projection dimension"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="LLM generation temperature"
    )

    # Algorithm variant
    parser.add_argument(
        "--use-turbo", action="store_true",
        help="Use TuRBO instead of standard BO"
    )

    # Other options
    parser.add_argument(
        "--results-dir", type=str, default="instructzero/results",
        help="Results directory"
    )
    parser.add_argument(
        "--skip-test-eval", action="store_true",
        help="Skip final test set evaluation"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print verbose progress"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    if args.quiet:
        args.verbose = False
        logging.getLogger().setLevel(logging.WARNING)

    # Print configuration
    logger.info("=" * 60)
    logger.info("InstructZero: InstructZero-style Prompt Optimization")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model} (backend: {args.backend})")
    logger.info(f"Budget: {args.max_calls} calls")
    logger.info(f"Soft prompt dim: {args.intrinsic_dim} → {args.projection_dim}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Minibatch size: {args.minibatch_size}")
    logger.info(f"Algorithm: {'TuRBO' if args.use_turbo else 'Standard BO'}")
    logger.info("=" * 60)

    # Import here to avoid slow startup for --help
    from shared.llm_client import create_llm_client
    from shared.gsm8k_evaluator import GSM8KEvaluator
    from .loop import InstructZeroLoop

    # Initialize LLM client
    logger.info(f"Initializing {args.model} LLM client...")
    llm_client = create_llm_client(
        args.model,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # Initialize evaluator
    logger.info("Loading GSM8K dataset...")
    evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split="train"
    )
    logger.info(f"Loaded {len(evaluator)} training examples")

    # Create optimization loop
    loop = InstructZeroLoop(
        llm_client=llm_client,
        evaluator=evaluator,
        intrinsic_dim=args.intrinsic_dim,
        projection_dim=args.projection_dim,
        n_initial=args.n_initial,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        use_turbo=args.use_turbo,
        temperature=args.temperature,
        results_dir=args.results_dir,
        verbose=args.verbose,
        seed=args.seed
    )

    # Run optimization
    start_time = time.time()
    results = loop.run(max_calls=args.max_calls)
    elapsed = time.time() - start_time

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best validation accuracy: {results['best_score']:.4f}")
    logger.info(f"Best instruction:\n{results['best_instruction']}")
    logger.info(f"Total LLM calls: {results['total_calls']}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("=" * 60)

    # Evaluate on test set
    if not args.skip_test_eval:
        logger.info("\nEvaluating best instruction on test set...")
        test_accuracy = loop.evaluate_on_test()
        results["test_accuracy"] = test_accuracy

        # Save final results
        import json
        results_path = Path(args.results_dir) / f"instructzero_final_{int(time.time())}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Final results saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
