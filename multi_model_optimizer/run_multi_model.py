#!/usr/bin/env python3
"""
CLI entry point for Multi-Model Universal Prompt Optimizer.

Finds a single prompt that works well across multiple frontier LLMs.

Usage:
    uv run python multi_model_optimizer/run_multi_model.py \
        --aggregation weighted_softmin \
        --softmin-temperature 0.1 \
        --budget 200000 \
        --iterations 10 \
        --output-dir multi_model_optimizer/results
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_model_optimizer.config import MultiModelConfig
from multi_model_optimizer.optimizer import MultiModelHybridOptimizer


def load_gsm8k_data(split: str = "test") -> list:
    """Load GSM8K data from datasets library."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("gsm8k", "main", split=split)

        data = []
        for item in dataset:
            # Extract answer from solution
            solution = item["answer"]
            # GSM8K format: solution text #### answer
            if "####" in solution:
                answer = solution.split("####")[-1].strip()
            else:
                answer = solution.strip()

            data.append({"question": item["question"], "answer": answer})

        return data
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        print("Falling back to local data...")
        return load_local_gsm8k(split)


def load_local_gsm8k(split: str = "test") -> list:
    """Load GSM8K from local file."""
    path = Path(__file__).parent.parent / "datasets" / "gsm8k" / f"{split}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"GSM8K data not found at: {path}")

    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append({"question": item["question"], "answer": item["answer"]})

    return data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Model Universal Prompt Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Models
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        help="Target models to optimize for",
    )

    parser.add_argument(
        "--gpu-assignment",
        type=str,
        default=None,
        help="GPU assignment as JSON, e.g., '{\"Qwen/...\": 0, \"Llama/...\": 1}'",
    )

    # Aggregation
    parser.add_argument(
        "--aggregation",
        choices=["average", "minimum", "weighted_softmin", "harmonic"],
        default="weighted_softmin",
        help="Aggregation strategy for combining model scores",
    )

    parser.add_argument(
        "--softmin-temperature",
        type=float,
        default=0.1,
        help="Temperature for weighted_softmin (lower = closer to min)",
    )

    # Optimization
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of optimization iterations",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=200000,
        help="Total LLM evaluation budget (all models combined)",
    )

    # Phase 1
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 HbBoPs and load pre-computed results",
    )

    parser.add_argument(
        "--phase1-results",
        type=str,
        default="datasets/hbbops/full_grid_combined.jsonl",
        help="Path to pre-computed Phase 1 results",
    )

    parser.add_argument(
        "--initial-instructions",
        type=str,
        default="datasets/hbbops/instructions_25.txt",
        help="Path to initial instructions file",
    )

    parser.add_argument(
        "--initial-exemplars",
        type=str,
        default="datasets/hbbops/examples_25.txt",
        help="Path to initial exemplars file",
    )

    # OPRO
    parser.add_argument(
        "--meta-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for OPRO instruction generation",
    )

    parser.add_argument(
        "--meta-model-backend",
        type=str,
        choices=["vllm", "gemini", "openai", "deepinfra"],
        default="vllm",
        help="Backend for meta-model (vllm for local, gemini/openai for API)",
    )

    parser.add_argument(
        "--meta-model-api-key",
        type=str,
        default=None,
        help="API key for meta-model (Gemini/OpenAI)",
    )

    parser.add_argument(
        "--opro-candidates",
        type=int,
        default=8,
        help="Number of instruction candidates per OPRO iteration",
    )

    # GP
    parser.add_argument(
        "--gp-top-k",
        type=int,
        default=10,
        help="Number of candidates to evaluate after GP screening",
    )

    parser.add_argument(
        "--gp-latent-dim",
        type=int,
        default=10,
        help="Latent dimension for GP feature extractor",
    )

    parser.add_argument(
        "--gp-rank",
        type=int,
        default=2,
        help="Rank of ICM task covariance matrix",
    )

    # Sequential testing
    parser.add_argument(
        "--sequential-confidence",
        type=float,
        default=0.95,
        help="Joint confidence level for sequential testing",
    )

    # Parallelization
    parser.add_argument(
        "--sequential-models",
        action="store_true",
        help="Evaluate models sequentially instead of in parallel",
    )

    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Use single GPU with model switching (memory efficient)",
    )

    parser.add_argument(
        "--single-gpu-id",
        type=int,
        default=0,
        help="GPU ID to use in single-GPU mode",
    )

    parser.add_argument(
        "--model-sequential",
        action="store_true",
        help="Model-sequential mode: process each model completely before switching (minimizes model switching, requires --single-gpu)",
    )

    parser.add_argument(
        "--final-verification-top-k",
        type=int,
        default=5,
        help="Number of top candidates to verify on all models in model-sequential mode",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="multi_model_optimizer/results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Multi-Model Universal Prompt Optimizer")
    print("=" * 60)

    # Build GPU assignment
    if args.gpu_assignment:
        gpu_assignment = json.loads(args.gpu_assignment)
    else:
        # Default: assign GPUs sequentially
        gpu_assignment = {model: i for i, model in enumerate(args.models)}

    # Create config
    config = MultiModelConfig(
        target_models=args.models,
        gpu_assignment=gpu_assignment,
        aggregation=args.aggregation,
        softmin_temperature=args.softmin_temperature,
        total_llm_budget=args.budget,
        skip_phase1_hbbops=args.skip_phase1,
        phase1_results_path=args.phase1_results,
        initial_instructions_path=args.initial_instructions,
        initial_exemplars_path=args.initial_exemplars,
        meta_model=args.meta_model,
        meta_model_backend=args.meta_model_backend,
        meta_model_api_key=args.meta_model_api_key,
        opro_candidates_per_iter=args.opro_candidates,
        gp_top_k=args.gp_top_k,
        gp_latent_dim=args.gp_latent_dim,
        gp_rank=args.gp_rank,
        sequential_confidence=args.sequential_confidence,
        parallel_models=not args.sequential_models,
        single_gpu=args.single_gpu,
        single_gpu_id=args.single_gpu_id,
        model_sequential_mode=args.model_sequential,
        final_verification_top_k=args.final_verification_top_k,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Print configuration
    print("\nConfiguration:")
    print(f"  Target models: {config.target_models}")
    if config.single_gpu:
        print(f"  Mode: Single-GPU (GPU {config.single_gpu_id}) with model switching")
    else:
        print(f"  GPU assignment: {config.gpu_assignment}")
        print(f"  Parallel models: {config.parallel_models}")
    print(f"  Meta-model: {config.meta_model} ({config.meta_model_backend})")
    print(f"  Aggregation: {config.aggregation} (T={config.softmin_temperature})")
    print(f"  Budget: {config.total_llm_budget:,}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Output: {config.output_dir}")

    # Load data
    print("\nLoading GSM8K data...")
    validation_data = load_gsm8k_data("test")
    train_data = load_gsm8k_data("train")
    print(f"  Validation: {len(validation_data)} examples")
    print(f"  Training: {len(train_data)} examples")

    # Create optimizer
    print("\nInitializing optimizer...")
    optimizer = MultiModelHybridOptimizer(
        config=config,
        validation_data=validation_data,
        train_data=train_data,
    )

    # Run optimization
    print("\nStarting optimization...")
    best_instruction, best_exemplar, best_accuracy, per_model_acc = optimizer.run(
        num_iterations=args.iterations,
        verbose=args.verbose,
    )

    # Save final results
    optimizer.save_final_results()

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nBest aggregated accuracy: {best_accuracy:.2%}")
    print("\nPer-model accuracies:")
    for model, acc in per_model_acc.items():
        print(f"  {model}: {acc:.2%}")
    print(f"\nBest instruction:\n{best_instruction}")
    print(f"\nResults saved to: {config.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
