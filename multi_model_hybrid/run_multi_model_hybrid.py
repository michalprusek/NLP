#!/usr/bin/env python3
"""
CLI entry point for Multi-Model Hybrid Optimizer.

Features per-model GP selection and batch evaluation with Hoeffding bounds.

Usage:
    uv run python multi_model_hybrid/run_multi_model_hybrid.py \\
        --models Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \\
        --aggregation weighted_softmin \\
        --top-k-per-model 10 \\
        --max-union 30 \\
        --budget 200000

Example with pre-computed Phase 1:
    uv run python multi_model_hybrid/run_multi_model_hybrid.py \\
        --skip-phase1 \\
        --phase1-results datasets/hbbops/full_grid_combined.jsonl \\
        --iterations 10
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_gsm8k_data(split: str = "test") -> list:
    """
    Load GSM8K data from HuggingFace datasets.

    Args:
        split: 'train' or 'test'

    Returns:
        List of {"question": str, "answer": str}
    """
    from datasets import load_dataset

    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("gsm8k", "main", split=split)

    data = []
    for item in dataset:
        solution = item["answer"]
        # Extract final answer after ####
        answer = solution.split("####")[-1].strip() if "####" in solution else solution.strip()
        data.append({"question": item["question"], "answer": answer})

    print(f"  Loaded {len(data)} examples")
    return data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Model Hybrid Optimizer with Per-Model GP Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default settings
  uv run python multi_model_hybrid/run_multi_model_hybrid.py

  # Skip Phase 1 (use pre-computed results)
  uv run python multi_model_hybrid/run_multi_model_hybrid.py \\
      --skip-phase1 --phase1-results datasets/hbbops/full_grid_combined.jsonl

  # Custom models and aggregation
  uv run python multi_model_hybrid/run_multi_model_hybrid.py \\
      --models Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \\
      --aggregation harmonic --iterations 15
        """
    )

    # Target models
    parser.add_argument(
        "--models", nargs="+",
        default=[
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        help="Target models for optimization (default: Qwen, Llama, Mistral 7B)"
    )

    # Per-model selection
    parser.add_argument(
        "--top-k-per-model", type=int, default=10,
        help="Top candidates to select per model (default: 10)"
    )
    parser.add_argument(
        "--max-union", type=int, default=30,
        help="Maximum unique candidates from union (default: 30)"
    )
    parser.add_argument(
        "--ucb-kappa", type=float, default=2.0,
        help="UCB exploration parameter (default: 2.0)"
    )

    # Aggregation
    parser.add_argument(
        "--aggregation",
        choices=["average", "minimum", "weighted_softmin", "harmonic"],
        default="weighted_softmin",
        help="Score aggregation strategy (default: weighted_softmin)"
    )
    parser.add_argument(
        "--softmin-temperature", type=float, default=0.1,
        help="Temperature for weighted_softmin (default: 0.1)"
    )

    # Budget and iterations
    parser.add_argument(
        "--budget", type=int, default=200000,
        help="Total LLM evaluation budget (default: 200000)"
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of optimization iterations (default: 10)"
    )

    # Phase 1 configuration
    parser.add_argument(
        "--skip-phase1", action="store_true",
        help="Skip Phase 1 Hyperband, load from file instead"
    )
    parser.add_argument(
        "--phase1-results", type=str,
        default="datasets/hbbops/full_grid_combined.jsonl",
        help="Path to pre-computed Phase 1 results (JSONL)"
    )
    parser.add_argument(
        "--initial-instructions", type=str,
        default="datasets/hbbops/instructions_25.txt",
        help="Path to initial instructions file"
    )
    parser.add_argument(
        "--initial-exemplars", type=str,
        default="datasets/hbbops/examples_25.txt",
        help="Path to initial exemplars file"
    )
    parser.add_argument(
        "--bmin", type=int, default=10,
        help="Minimum fidelity for Hyperband (default: 10)"
    )

    # APE Forward for initial instructions
    parser.add_argument(
        "--use-ape-forward-init", action="store_true", default=True,
        help="Use APE forward pass for initial instruction generation (default: True)"
    )
    parser.add_argument(
        "--no-ape-forward-init", action="store_false", dest="use_ape_forward_init",
        help="Disable APE forward pass, load instructions from file instead"
    )
    parser.add_argument(
        "--ape-num-samples", type=int, default=10,
        help="Number of examples to show LLM in APE forward prompt (default: 10)"
    )
    parser.add_argument(
        "--ape-num-candidates", type=int, default=100,
        help="Number of raw instruction candidates before clustering (default: 100)"
    )
    parser.add_argument(
        "--ape-num-final", type=int, default=25,
        help="Number of final instructions after K-means clustering (default: 25)"
    )

    # Initial exemplars (Stage 1)
    parser.add_argument(
        "--num-initial-exemplars", type=int, default=25,
        help="Number of initial exemplars for Phase 1 (default: 25)"
    )
    parser.add_argument(
        "--initial-exemplar-qa-pairs", type=int, default=2,
        help="Q/A pairs per initial exemplar (default: 2)"
    )

    # Meta-model
    parser.add_argument(
        "--meta-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for OPRO instruction generation (default: Qwen)"
    )
    parser.add_argument(
        "--opro-candidates", type=int, default=8,
        help="Number of OPRO candidates per iteration (default: 8)"
    )
    parser.add_argument(
        "--opro-keep-k", type=int, default=20,
        help="Number of top instructions for OPRO context (default: 20)"
    )

    # Exemplars
    parser.add_argument(
        "--num-exemplars", type=int, default=25,
        help="Number of dynamic exemplars per iteration (default: 25)"
    )
    parser.add_argument(
        "--exemplars-per-sample", type=int, default=5,
        help="Number of Q/A pairs per exemplar (default: 5)"
    )

    # GP configuration
    parser.add_argument(
        "--gp-latent-dim", type=int, default=10,
        help="GP latent space dimension (default: 10)"
    )
    parser.add_argument(
        "--gp-rank", type=int, default=2,
        help="ICM task covariance rank (default: 2)"
    )
    parser.add_argument(
        "--gp-epochs", type=int, default=3000,
        help="GP training epochs (default: 3000)"
    )
    parser.add_argument(
        "--gp-lr", type=float, default=0.01,
        help="GP learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--gp-patience", type=int, default=10,
        help="GP early stopping patience (default: 10)"
    )

    # Hoeffding bounds
    parser.add_argument(
        "--hoeffding-confidence", type=float, default=0.95,
        help="Hoeffding confidence level (default: 0.95)"
    )
    parser.add_argument(
        "--hoeffding-min-samples", type=int, default=10,
        help="Minimum samples before Hoeffding decision (default: 10)"
    )
    parser.add_argument(
        "--hoeffding-min-promote", type=int, default=30,
        help="Minimum samples for PROMOTE decision (default: 30)"
    )

    # Hardware
    parser.add_argument(
        "--single-gpu-id", type=int, default=0,
        help="GPU ID for single-GPU mode (default: 0)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for GP training: 'auto', 'cuda', 'cpu' (default: auto)"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="multi_model_hybrid/results",
        help="Output directory for results (default: multi_model_hybrid/results)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print verbose output (default: True)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 70)
    print("Multi-Model Hybrid Optimizer")
    print("(Per-Model GP Selection Strategy)")
    print("=" * 70)

    # Build configuration
    from multi_model_hybrid.config import (
        MultiModelHybridConfig,
        PerModelSelectionConfig,
    )

    per_model_config = PerModelSelectionConfig(
        top_k_per_model=args.top_k_per_model,
        max_union_candidates=args.max_union,
        ucb_kappa=args.ucb_kappa,
    )

    # GPU assignment (all on single GPU)
    gpu_assignment = {model: args.single_gpu_id for model in args.models}

    config = MultiModelHybridConfig(
        # Target models
        target_models=args.models,
        gpu_assignment=gpu_assignment,

        # Aggregation
        aggregation=args.aggregation,
        softmin_temperature=args.softmin_temperature,

        # Budget
        total_llm_budget=args.budget,

        # Phase 1
        skip_phase1_hbbops=args.skip_phase1,
        phase1_results_path=args.phase1_results,
        initial_instructions_path=args.initial_instructions,
        initial_exemplars_path=args.initial_exemplars,
        bmin=args.bmin,

        # APE Forward for initial instructions
        use_ape_forward_init=args.use_ape_forward_init,
        ape_num_samples=args.ape_num_samples,
        ape_num_candidates=args.ape_num_candidates,
        ape_num_final=args.ape_num_final,

        # Initial exemplars (Stage 1)
        num_initial_exemplars=args.num_initial_exemplars,
        initial_exemplar_qa_pairs=args.initial_exemplar_qa_pairs,

        # Meta-model
        meta_model=args.meta_model,
        opro_candidates_per_iter=args.opro_candidates,
        opro_keep_top_k=args.opro_keep_k,

        # Exemplars
        num_dynamic_exemplars=args.num_exemplars,
        exemplars_per_sample=args.exemplars_per_sample,

        # GP
        gp_top_k=args.top_k_per_model,
        gp_latent_dim=args.gp_latent_dim,
        gp_rank=args.gp_rank,
        gp_train_epochs=args.gp_epochs,
        gp_lr=args.gp_lr,
        gp_patience=args.gp_patience,

        # Hoeffding
        hoeffding_confidence=args.hoeffding_confidence,
        hoeffding_min_samples=args.hoeffding_min_samples,
        hoeffding_min_promote_samples=args.hoeffding_min_promote,

        # Per-model selection
        per_model_selection=per_model_config,

        # Hardware
        single_gpu=True,
        single_gpu_id=args.single_gpu_id,
        device=args.device,

        # Output
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Models: {config.target_models}")
    print(f"  Selection: Top {args.top_k_per_model} per model, union up to {args.max_union}")
    print(f"  UCB kappa: {args.ucb_kappa}")
    print(f"  Aggregation: {config.aggregation} (T={config.softmin_temperature})")
    print(f"  Budget: {config.total_llm_budget:,}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Skip Phase 1: {args.skip_phase1}")
    if not args.skip_phase1:
        print(f"  APE Forward Init: {args.use_ape_forward_init}")
        if args.use_ape_forward_init:
            print(f"    APE samples: {args.ape_num_samples}, candidates: {args.ape_num_candidates}, final: {args.ape_num_final}")
            print(f"    Initial exemplars: {args.num_initial_exemplars} × {args.initial_exemplar_qa_pairs} Q/A pairs")
    print(f"  Output: {config.output_dir}")

    # Load data
    print("\n" + "=" * 70)
    print("Loading GSM8K dataset...")
    print("=" * 70)
    validation_data = load_gsm8k_data("test")
    train_data = load_gsm8k_data("train")
    print(f"  Validation: {len(validation_data)}, Train: {len(train_data)}")

    # Create optimizer
    print("\n" + "=" * 70)
    print("Initializing optimizer...")
    print("=" * 70)

    from multi_model_hybrid.optimizer import MultiModelHybridOptimizer

    optimizer = MultiModelHybridOptimizer(config, validation_data, train_data)

    # Run optimization
    print("\n" + "=" * 70)
    print("Starting optimization...")
    print("=" * 70)

    try:
        best_inst, best_ex, best_acc, per_model = optimizer.run(
            num_iterations=args.iterations,
            verbose=args.verbose,
        )

        # Save results
        optimizer.save_final_results()

        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Best aggregated accuracy: {best_acc:.2%}")
        print("\nPer-model accuracies:")
        for model, acc in per_model.items():
            model_short = model.split("/")[-1]
            print(f"  {model_short}: {acc:.2%}")
        print(f"\nResults saved to: {config.output_dir}")

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        optimizer.save_final_results()
        print(f"Partial results saved to: {config.output_dir}")

    finally:
        optimizer.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
