"""CLI entry point for InvBO decoder inversion.

Usage:
    # Standard: Train and run 10 iterations (VAE enabled by default)
    uv run python -m generation.invbo_decoder.run --iterations 10

    # With optimal hyperparameters
    uv run python -m generation.invbo_decoder.run --iterations 10 \
        --vae-beta 0.01 --vae-annealing 500

    # Disable VAE (not recommended)
    uv run python -m generation.invbo_decoder.run --no-vae

NOTE: Do not use --trust-region flag - it doesn't work well in practice.
"""

import argparse
import json
import logging
import random
import sys
import warnings
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Suppress noisy logs from PyTorch internals
logging.getLogger('torch._dynamo').setLevel(logging.WARNING)
logging.getLogger('torch._inductor').setLevel(logging.WARNING)
# Suppress BoTorch optimization warnings (handled internally)
warnings.filterwarnings("ignore", message=".*Optimization failed.*")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Sets seed for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - cuDNN deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic cuDNN (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from generation.invbo_decoder.training import InvBOTrainer, TrainingConfig
from generation.invbo_decoder.inference import InvBOInference, IterationRecord
from generation.invbo_decoder.trust_region import TRConfig


class TeeOutput:
    """Write to both stdout and log file. Supports context manager protocol."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal
        self.close()
        return False  # Don't suppress exceptions


def evaluate_with_llm(
    instruction: str,
    model: str,
    validation_path: str,
    num_samples: int = 100,
    client=None,
) -> tuple:
    """Evaluate instruction with LLM (optional, for non-skip-eval mode).

    Args:
        instruction: Instruction to evaluate
        model: Model name for evaluation
        validation_path: Path to validation data
        num_samples: Number of samples for evaluation
        client: Optional pre-created LLM client (reused across iterations)

    Returns:
        Tuple of (error_rate, client)
    """
    import json

    # Lazy import to avoid loading when not needed
    from generation.invbo_decoder.evaluate import evaluate_instruction
    from src.llm_client import create_llm_client

    # Create client if not provided
    if client is None:
        client = create_llm_client(model, "vllm")

    # Load validation data with error handling
    from pathlib import Path

    validation_file = Path(validation_path)
    if not validation_file.exists():
        raise FileNotFoundError(
            f"Validation data not found: {validation_path}\n"
            f"Please ensure the file exists or specify a different path."
        )

    try:
        with open(validation_path) as f:
            validation_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in validation file: {validation_path}\n"
            f"Error at line {e.lineno}: {e.msg}"
        ) from e

    if not isinstance(validation_data, list):
        raise ValueError(
            f"Validation data must be a list, got {type(validation_data).__name__}"
        )

    # Sample if needed
    if num_samples < len(validation_data):
        import random
        random.seed(42)
        validation_data = random.sample(validation_data, num_samples)

    error_rate, correct, total = evaluate_instruction(
        instruction=instruction,
        test_data=validation_data,
        client=client,
    )
    return error_rate, client


def main():
    parser = argparse.ArgumentParser(
        description="InvBO Decoder Inversion for Instruction Optimization"
    )

    # Data paths
    parser.add_argument(
        "--instructions",
        type=str,
        default="datasets/inversion/instructions_100.txt",
        help="Path to instructions file",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="datasets/inversion/grid_100_qend.jsonl",
        help="Path to grid JSONL file",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="hbbops_improved_2/data/validation.json",
        help="Path to validation data for LLM evaluation",
    )

    # APE instructions (for VAE training)
    parser.add_argument(
        "--ape-cache",
        type=str,
        default="datasets/inversion/diverse_instructions_1000.json",
        help="Path to APE instructions cache",
    )
    parser.add_argument(
        "--skip-ape",
        action="store_true",
        help="Skip APE instructions, use only grid_100 for VAE training",
    )

    # Iteration parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Use GP prediction instead of LLM evaluation",
    )

    # Training parameters
    parser.add_argument(
        "--gp-epochs",
        type=int,
        default=5000,
        help="GP training epochs (reduced from 10000 for faster training)",
    )
    parser.add_argument(
        "--gp-lr",
        type=float,
        default=0.01,
        help="GP training learning rate",
    )
    parser.add_argument(
        "--decoder-epochs",
        type=int,
        default=1000,
        help="Decoder training epochs",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=10,
        help="Latent space dimension",
    )

    # Loss parameters
    parser.add_argument(
        "--lambda-cycle",
        type=float,
        default=1.0,
        help="Cyclic loss weight",
    )
    parser.add_argument(
        "--lambda-embedding",
        type=float,
        default=0.5,
        help="Embedding cosine loss weight",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Soft tolerance for cyclic loss",
    )

    # VAE mode parameters
    parser.add_argument(
        "--no-vae",
        action="store_true",
        help="Disable VAE mode (VAE is enabled by default)",
    )
    parser.add_argument(
        "--vae-beta",
        type=float,
        default=0.02,
        help="VAE KL regularization weight (0.02 optimal: sharp but smooth)",
    )
    parser.add_argument(
        "--vae-epochs",
        type=int,
        default=10000,
        help="VAE training epochs",
    )
    parser.add_argument(
        "--vae-annealing",
        type=int,
        default=500,
        help="KL annealing epochs (0 → beta, optimal: 500)",
    )
    parser.add_argument(
        "--vae-patience",
        type=int,
        default=500,
        help="VAE early stopping patience (higher = longer training)",
    )

    # Trust region parameters (disabled by default)
    parser.add_argument(
        "--trust-region",
        action="store_true",
        help="Enable trust region (disabled by default)",
    )
    parser.add_argument(
        "--tr-initial",
        type=float,
        default=0.5,
        help="Initial trust region radius",
    )
    parser.add_argument(
        "--tr-min",
        type=float,
        default=0.05,
        help="Minimum trust region radius",
    )
    parser.add_argument(
        "--tr-max",
        type=float,
        default=2.0,
        help="Maximum trust region radius",
    )

    # BoTorch optimization parameters
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=64,
        help="Number of L-BFGS-B restarts for BoTorch qLogEI optimization",
    )
    parser.add_argument(
        "--raw-samples",
        type=int,
        default=512,
        help="Raw samples for BoTorch acquisition optimization initialization",
    )

    # InvBO inversion parameters
    parser.add_argument(
        "--use-inversion",
        action="store_true",
        default=True,
        help="Use InvBO-style inversion loop (enabled by default)",
    )
    parser.add_argument(
        "--no-inversion",
        action="store_true",
        help="Disable InvBO-style inversion loop",
    )
    parser.add_argument(
        "--max-inversion-iters",
        type=int,
        default=10,
        help="Maximum inversion iterations",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.1,
        help="Cosine distance threshold for triggering re-inversion",
    )

    # Vec2Text parameters
    parser.add_argument(
        "--vec2text-steps",
        type=int,
        default=50,
        help="Vec2Text correction steps",
    )
    parser.add_argument(
        "--vec2text-beam",
        type=int,
        default=8,
        help="Vec2Text beam width (8 recommended for quality)",
    )
    parser.add_argument(
        "--vec2text-model",
        type=str,
        choices=["32_tokens", "512_tokens"],
        default="32_tokens",
        help="Vec2Text model: '32_tokens' (ielabgroup, with corrector) or '512_tokens' (cowboys, simpler)",
    )

    # GP retraining parameters
    parser.add_argument(
        "--retrain-interval",
        type=int,
        default=1,
        help="Retrain GP every N iterations (1 = every iteration)",
    )
    parser.add_argument(
        "--retrain-epochs",
        type=int,
        default=1000,
        help="GP retraining epochs (reduced for incremental updates)",
    )
    parser.add_argument(
        "--gp-patience",
        type=int,
        default=50,
        help="GP early stopping patience (reduced for faster convergence)",
    )

    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate EI landscape visualizations",
    )

    # LLM evaluation parameters (for non-skip-eval mode)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for LLM evaluation (only used without --skip-eval)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=1319,
        help="Samples for LLM evaluation (1319 = full GSM8K validation set)",
    )

    # Save/load
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save trained models to directory",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load pre-trained models from directory",
    )

    # Validation
    parser.add_argument(
        "--validate-gap",
        action="store_true",
        help="Validate inversion gap on random samples",
    )
    parser.add_argument(
        "--gap-samples",
        type=int,
        default=10,
        help="Number of samples for gap validation",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (set before training and each optimization step)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Set up results directory and logging
    results_dir = Path(args.save) if args.save else Path("generation/invbo_decoder/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"run_{timestamp}.log"
    tee = TeeOutput(log_path)
    sys.stdout = tee

    # Register cleanup to ensure stdout is restored even on exception
    import atexit
    def cleanup_tee():
        if sys.stdout is tee:
            sys.stdout = tee.terminal
            tee.close()
    atexit.register(cleanup_tee)

    print("=" * 70)
    print("InvBO Decoder Inversion for Instruction Optimization")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Random seed: {args.seed}")
    use_inversion = args.use_inversion and not args.no_inversion
    print(f"Key settings:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Optimization: BoTorch qLogEI")
    print(f"    Restarts: {args.n_restarts}, Raw samples: {args.raw_samples}")
    print(f"  Inversion: {'enabled' if use_inversion else 'disabled'}")
    if use_inversion:
        print(f"    Max inversion iters: {args.max_inversion_iters}, Gap threshold: {args.gap_threshold}")
    print(f"  Trust Region: {'enabled' if args.trust_region else 'disabled'}")
    print(f"  Skip-eval: {args.skip_eval}")
    print(f"  VAE: beta={args.vae_beta}, epochs={args.vae_epochs}, annealing={args.vae_annealing}")
    print(f"  Vec2Text model: {args.vec2text_model}")
    print(f"Results directory: {results_dir}")
    print(f"Log file: {log_path}")
    print("=" * 70)

    # Set APE path (skip if --skip-ape)
    diverse_instructions_path = args.ape_cache
    if args.skip_ape:
        print("\n[--skip-ape: using only grid_100 for VAE training]")
        diverse_instructions_path = None

    # Create config
    config = TrainingConfig(
        instructions_path=args.instructions,
        grid_path=args.grid,
        diverse_instructions_path=diverse_instructions_path,
        latent_dim=args.latent_dim,
        gp_epochs=args.gp_epochs,
        gp_lr=args.gp_lr,
        gp_patience=args.gp_patience,
        decoder_epochs=args.decoder_epochs,
        lambda_cycle=args.lambda_cycle,
        lambda_cosine=args.lambda_embedding,
        cycle_tolerance=args.tolerance,
        use_vae=not args.no_vae,
        vae_beta=args.vae_beta,
        vae_epochs=args.vae_epochs,
        vae_annealing_epochs=args.vae_annealing,
        vae_patience=args.vae_patience,
        device=args.device,
    )

    # Initialize trainer
    trainer = InvBOTrainer(config)

    # VAE quality metrics storage
    vae_quality_metrics = None

    if args.load:
        # Load pre-trained models
        print(f"\nLoading models from {args.load}...")
        trainer.load(args.load)
    else:
        # Train from scratch
        print("\nStarting training...")
        gp, decoder = trainer.train(verbose=True)

        # Evaluate VAE quality if using VAE
        if not args.no_vae:
            print("\n" + "=" * 60)
            print("Evaluating VAE Quality")
            print("=" * 60)
            vae_quality_metrics = trainer.evaluate_vae_quality(verbose=True)

        if args.save:
            trainer.save(args.save)

    # Trust region config
    tr_config = None
    if args.trust_region:
        tr_config = TRConfig(
            initial_radius=args.tr_initial,
            min_radius=args.tr_min,
            max_radius=args.tr_max,
        )

    # Create inference pipeline
    inference = InvBOInference(
        gp=trainer.gp,
        decoder=trainer.decoder,
        gtr=trainer.gtr,
        vec2text_steps=args.vec2text_steps,
        vec2text_beam=args.vec2text_beam,
        vec2text_model=args.vec2text_model,
        trust_region_config=tr_config,
        seed=args.seed,
    )

    # Validate inversion gap if requested
    if args.validate_gap:
        inference.validate_inversion_gap(n_samples=args.gap_samples, verbose=True)

    # Get initial best from grid
    best_latent, best_idx, best_error = inference.get_best_training_latent()

    print("\n" + "=" * 70)
    print(f"Starting InvBO Optimization ({args.iterations} iterations)")
    print("=" * 70)
    print(f"Initial best error (grid): {best_error:.4f}")

    if args.trust_region:
        inference.init_trust_region(best_latent)
        print(f"Trust region initialized: radius={inference.trust_region.radius:.4f}")

    # Iteration history
    iteration_history: List[Dict[str, Any]] = []
    best_instruction = trainer.instructions[best_idx] if best_idx < len(trainer.instructions) else ""
    eval_client = None  # Reuse LLM client across iterations

    # Main optimization loop
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration}/{args.iterations}")
        print("=" * 60)
        print(f"Current best error: {best_error:.4f}")
        if args.trust_region:
            print(f"Trust region radius: {inference.trust_region.radius:.4f}")

        # Run single iteration with BoTorch qLogEI optimization
        result, gap, log_ei = inference.run_single_iteration(
            num_restarts=args.n_restarts,
            raw_samples=args.raw_samples,
            use_inversion=use_inversion,
            max_inversion_iters=args.max_inversion_iters,
            gap_threshold=args.gap_threshold,
            verbose=True,
        )

        # Evaluate (GP prediction or LLM)
        if args.skip_eval:
            actual_error = result.predicted_error
            print(f"  GP-predicted error rate: {actual_error:.4f} (accuracy: {(1-actual_error)*100:.2f}%)")
            print(f"  [--skip-eval mode: no LLM evaluation]")
        else:
            print(f"  Evaluating with LLM ({args.model})...")
            actual_error, eval_client = evaluate_with_llm(
                result.instruction_text,
                args.model,
                args.validation,
                args.eval_samples,
                client=eval_client,
            )
            print(f"  LLM error rate: {actual_error:.4f} (accuracy: {(1-actual_error)*100:.2f}%)")

        # Check improvement
        improved = actual_error < best_error
        if improved:
            best_error = actual_error
            best_instruction = result.instruction_text
            print(f"  *** NEW BEST: {actual_error:.4f} ***")

        # Update trust region
        if args.trust_region:
            inference.trust_region.update(
                result.latent,
                actual_error,
                best_error if not improved else actual_error + 0.001,  # Trick to detect improvement
                verbose=True,
            )

        # Update GP with new observation
        if iteration % args.retrain_interval == 0:
            print(f"\n--- Updating GP ---")
            # Re-encode the generated text for aligned observation
            reencoded = trainer.gtr.encode_tensor(result.instruction_text)
            trainer.gp.add_observation_and_retrain(
                reencoded,
                actual_error,
                epochs=args.retrain_epochs,
                patience=args.gp_patience,
                verbose=True,
            )

        # Record iteration
        iter_record = {
            "iteration": iteration,
            "instruction": result.instruction_text,
            "cosine_similarity": result.cosine_similarity,
            "predicted_error": result.predicted_error,
            "actual_error": actual_error,
            "gap": gap,
            "log_ei": log_ei,
            "improved": improved,
            "best_error_so_far": best_error,
            "trust_region_radius": inference.trust_region.radius if args.trust_region else None,
            "gp_samples": trainer.gp.get_training_size(),
        }
        iteration_history.append(iter_record)

        # Visualization (optional)
        if args.visualize and iteration <= 10:  # Limit to first 10 for performance
            try:
                from generation.invbo_decoder.visualize import visualize_ei_landscape
                vis_path = results_dir / f"ei_landscape_iter_{iteration}.png"
                visualize_ei_landscape(
                    inference=inference,
                    center_latent=result.latent,
                    realized_text=result.instruction_text,
                    best_y=best_error,
                    trust_region=inference.trust_region if args.trust_region else None,
                    save_path=str(vis_path),
                )
                print(f"  Visualization saved: {vis_path}")
            except ImportError as e:
                print(f"  Visualization skipped: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nBest Instruction:")
    print(f"  {best_instruction}")
    print(f"\nMetrics:")
    print(f"  Initial best error (grid): {iteration_history[0]['best_error_so_far'] if iteration_history else best_error:.4f}")
    print(f"  Final best error: {best_error:.4f}")
    print(f"  Total improvement: {iteration_history[0]['best_error_so_far'] - best_error:.4f}" if iteration_history else "  N/A")
    print(f"  Iterations: {args.iterations}")
    print(f"  Final GP samples: {trainer.gp.get_training_size()}")
    print("=" * 70)

    # Save results to JSON
    results_json = {
        "timestamp": timestamp,
        "seed": args.seed,
        "method": "InvBO Decoder",
        "args": vars(args),
        "grid_best": {
            "instruction_id": int(best_idx),
            "error_rate": float(iteration_history[0]["best_error_so_far"]) if iteration_history else float(best_error),
        },
        "optimized": {
            "instruction": best_instruction,
            "error_rate": float(best_error),
        },
        "iteration_history": iteration_history,
        "improvement": float(iteration_history[0]["best_error_so_far"] - best_error) if iteration_history else 0.0,
        "vae_quality_metrics": vae_quality_metrics,
    }

    results_json_path = results_dir / f"result_{timestamp}.json"
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_json_path}")

    # Close log file
    sys.stdout = tee.terminal
    tee.close()
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
