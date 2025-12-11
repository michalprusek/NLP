#!/usr/bin/env python3
"""
Run Hybrid OPRO + HbBoPs optimization.

Usage:
    uv run python hybrid_opro_hbbops/run_hybrid.py \\
        --task-model Qwen/Qwen2.5-7B-Instruct \\
        --meta-model Qwen/Qwen2.5-7B-Instruct \\
        --budget 50000 \\
        --output-dir hybrid_opro_hbbops/results
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk

from hybrid_opro_hbbops.config import HybridConfig
from hybrid_opro_hbbops.hybrid_optimizer import HybridOPROHbBoPs


class TeeLogger:
    """Write to both console and file simultaneously."""

    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid OPRO + HbBoPs optimization for GSM8K"
    )

    # Models
    parser.add_argument(
        "--task-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for task evaluation",
    )
    parser.add_argument(
        "--meta-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for OPRO instruction generation",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers"],
        help="LLM backend",
    )

    # Budget
    parser.add_argument(
        "--budget",
        type=int,
        default=50000,
        help="Total LLM evaluation budget (validation instances)",
    )

    # Hyperband
    parser.add_argument(
        "--bmin",
        type=int,
        default=10,
        help="Minimum validation instances for Hyperband",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=2.0,
        help="Halving parameter for Hyperband",
    )

    # OPRO
    parser.add_argument(
        "--opro-candidates",
        type=int,
        default=8,
        help="OPRO candidates per iteration",
    )
    parser.add_argument(
        "--opro-keep-k",
        type=int,
        default=20,
        help="Keep top k prompts for OPRO context",
    )

    # GP
    parser.add_argument(
        "--gp-top-k",
        type=int,
        default=10,
        help="Top candidates to evaluate after GP screening",
    )
    parser.add_argument(
        "--num-exemplars",
        type=int,
        default=25,
        help="Number of dynamic exemplars per iteration",
    )

    # Initial data
    parser.add_argument(
        "--instructions",
        type=str,
        default="datasets/hbbops/instructions_25.txt",
        help="Path to initial instructions file",
    )
    parser.add_argument(
        "--exemplars",
        type=str,
        default="datasets/hbbops/examples_25.txt",
        help="Path to initial exemplars file",
    )

    # Skip Phase 1 HbBoPs and load from file
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 HbBoPs and load pre-computed results from file",
    )
    parser.add_argument(
        "--phase1-results",
        type=str,
        default="datasets/hbbops/full_grid_combined.jsonl",
        help="Path to pre-computed Phase 1 results (JSONL)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hybrid_opro_hbbops/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu)",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup logging
    log_file = output_dir / f"hybrid_{timestamp}.log"
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    print("=" * 70)
    print(f"Hybrid OPRO + HbBoPs Started at {datetime.now().isoformat()}")
    print(f"Log file: {log_file}")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Task model: {args.task_model}")
    print(f"  Meta model: {args.meta_model}")
    print(f"  Backend: {args.backend}")
    print(f"  Budget: {args.budget}")
    print(f"  bmin: {args.bmin}, eta: {args.eta}")
    print(f"  OPRO candidates: {args.opro_candidates}, keep_k: {args.opro_keep_k}")
    print(f"  GP top_k: {args.gp_top_k}, num_exemplars: {args.num_exemplars}")
    print(f"  Instructions: {args.instructions}")
    print(f"  Exemplars: {args.exemplars}")
    if args.skip_phase1:
        print(f"  Skip Phase 1: True (loading from {args.phase1_results})")
    print()

    # Load data
    print("Loading GSM8K data...")
    gsm8k = load_from_disk("datasets/gsm8k")

    # Use test as validation (to match HbBoPs experiments)
    validation_data = [
        {"question": ex["question"], "answer": ex["answer"]}
        for ex in gsm8k["test"]
    ]

    # Train data for dynamic exemplar sampling
    train_data = [
        {"question": ex["question"], "answer": ex["answer"]}
        for ex in gsm8k["train"]
    ]

    print(f"Validation (test): {len(validation_data)}, Train: {len(train_data)}")

    # Create config
    config = HybridConfig(
        initial_instructions_path=args.instructions,
        initial_exemplars_path=args.exemplars,
        bmin=args.bmin,
        eta=args.eta,
        skip_phase1_hbbops=args.skip_phase1,
        phase1_results_path=args.phase1_results,
        opro_candidates_per_iter=args.opro_candidates,
        opro_keep_top_k=args.opro_keep_k,
        meta_model=args.meta_model,
        num_dynamic_exemplars=args.num_exemplars,
        gp_top_k=args.gp_top_k,
        total_llm_budget=args.budget,
        task_model=args.task_model,
        task_backend=args.backend,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Run optimization
    print("\nInitializing optimizer...")
    optimizer = HybridOPROHbBoPs(config, validation_data, train_data)

    print("\nStarting optimization...")
    best_prompt, best_accuracy = optimizer.run(verbose=True)

    # Save results
    results = {
        "method": "Hybrid OPRO + HbBoPs",
        "timestamp": timestamp,
        "config": {
            "task_model": config.task_model,
            "meta_model": config.meta_model,
            "budget": config.total_llm_budget,
            "bmin": config.bmin,
            "eta": config.eta,
            "opro_candidates": config.opro_candidates_per_iter,
            "gp_top_k": config.gp_top_k,
            "num_exemplars": config.num_dynamic_exemplars,
        },
        "results": {
            "best_accuracy": best_accuracy,
            "iterations": optimizer.iteration,
            "total_evaluations": optimizer.budget_used,
            "num_unique_instructions": optimizer.num_instructions,
            "num_unique_exemplars": optimizer.num_exemplars,
        },
        "best_prompt": {
            "instruction": best_prompt.instruction if best_prompt else None,
            "exemplar": best_prompt.exemplar if best_prompt else None,
            "instruction_id": best_prompt.instruction_id if best_prompt else None,
            "exemplar_id": best_prompt.exemplar_id if best_prompt else None,
        },
    }

    output_file = output_dir / f"hybrid_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Total evaluations: {optimizer.budget_used}")
    print(f"Iterations: {optimizer.iteration}")
    print(f"Unique instructions: {optimizer.num_instructions}")
    print(f"Unique exemplars: {optimizer.num_exemplars}")
    print("=" * 70)

    tee_logger.close()


if __name__ == "__main__":
    main()
