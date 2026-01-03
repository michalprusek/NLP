#!/usr/bin/env python
"""CLI entry point for LIPO pipeline.

Full pipeline: APE → VAE → Hyperband → InvBO Inference

Usage:
    # Full run with 1000 instructions
    uv run python -m lipo.run --iterations 10 --ape-instructions 1000

    # Debug with 10 instructions (fast)
    uv run python -m lipo.run --iterations 1 --ape-instructions 10 --debug

    # Skip APE (use cached instructions)
    uv run python -m lipo.run --skip-ape

    # Hyperband only (no inference)
    uv run python -m lipo.run --hyperband-only
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

from lipo.hbbops_results import extract_from_hyperband, save_hbbops_results


def setup_logging(output_dir: str, timestamp: str) -> Path:
    """Setup logging to both console and file.

    Args:
        output_dir: Directory for log file
        timestamp: Timestamp string for log filename

    Returns:
        Path to the log file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / f"run_{timestamp}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    return log_file


class TeeOutput:
    """Tee stdout/stderr to both console and file logger.

    Filters out progress bars and other noisy output from file logging.
    """

    # Patterns to skip when logging to file (progress bars, etc.)
    SKIP_PATTERNS = ("Batches:", "|", "it/s]", "Fetching", "%|")

    def __init__(self, original, logger, level=logging.INFO):
        self.original = original
        self.logger = logger
        self.level = level
        self.buffer = ""

    def _should_skip(self, line: str) -> bool:
        """Check if line should be skipped (progress bars, etc.)."""
        return any(pattern in line for pattern in self.SKIP_PATTERNS)

    def write(self, text):
        self.original.write(text)
        # Buffer lines and log complete lines
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip() and not self._should_skip(line):
                self.logger.log(self.level, line)

    def flush(self):
        self.original.flush()
        if self.buffer.strip() and not self._should_skip(self.buffer):
            self.logger.log(self.level, self.buffer.strip())
            self.buffer = ""


def _validate_hyperband_ranking(trainer, grid_path: str, verbose: bool = True) -> dict:
    """Compare Hyperband ranking with grid ground truth.

    Args:
        trainer: Trained LIPOHyperbandTrainer with Hyperband results
        grid_path: Path to original grid JSONL
        verbose: Print detailed results

    Returns:
        Dict with validation metrics
    """
    # Load grid ground truth
    grid_truth = {}
    with open(grid_path, "r") as f:
        for line in f:
            d = json.loads(line)
            grid_truth[d["instruction_text"]] = d["error_rate"]

    # Get Hyperband results from trainer
    if trainer.hyperband is None:
        print("WARNING: No Hyperband results to validate")
        return {}

    # Get design data: list of (inst_id, embedding, error, fidelity)
    design_data = trainer.hyperband.design_data

    # Build Hyperband ranking (use highest fidelity evaluation for each instruction)
    hb_results = {}
    for inst_id, _, error, fidelity in design_data:
        inst_text = trainer.instructions[inst_id]
        if inst_text not in hb_results or fidelity > hb_results[inst_text][1]:
            hb_results[inst_text] = (error, fidelity)

    # Compare rankings
    common_insts = set(grid_truth.keys()) & set(hb_results.keys())

    if verbose:
        print(f"\n" + "=" * 60)
        print("HYPERBAND RANKING VALIDATION")
        print("=" * 60)
        print(f"Grid instructions: {len(grid_truth)}")
        print(f"Hyperband evaluated: {len(hb_results)}")
        print(f"Common: {len(common_insts)}")

    if len(common_insts) < 5:
        print("WARNING: Too few common instructions for meaningful comparison")
        return {"common_count": len(common_insts)}

    # Calculate Spearman correlation
    grid_errors = [grid_truth[inst] for inst in common_insts]
    hb_errors = [hb_results[inst][0] for inst in common_insts]

    correlation, p_value = spearmanr(grid_errors, hb_errors)

    # Calculate top-K overlap
    grid_sorted = sorted(common_insts, key=lambda x: grid_truth[x])
    hb_sorted = sorted(common_insts, key=lambda x: hb_results[x][0])

    top_5_overlap = len(set(grid_sorted[:5]) & set(hb_sorted[:5]))
    top_10_overlap = len(set(grid_sorted[:10]) & set(hb_sorted[:10]))

    # Calculate mean absolute error difference
    error_diffs = [abs(grid_truth[inst] - hb_results[inst][0]) for inst in common_insts]
    mean_error_diff = sum(error_diffs) / len(error_diffs)

    if verbose:
        print(f"\nSpearman correlation: {correlation:.4f} (p={p_value:.4e})")
        print(f"Top-5 overlap: {top_5_overlap}/5")
        print(f"Top-10 overlap: {top_10_overlap}/10")
        print(f"Mean |grid - HB| error diff: {mean_error_diff:.4f}")

        print(f"\nTop 5 by Grid:")
        for i, inst in enumerate(grid_sorted[:5]):
            hb_err, hb_fid = hb_results.get(inst, (None, None))
            print(f"  {i+1}. Grid={grid_truth[inst]:.4f}, HB={hb_err:.4f} (fid={hb_fid}):\n      {inst}")

        print(f"\nTop 5 by Hyperband:")
        for i, inst in enumerate(hb_sorted[:5]):
            hb_err, hb_fid = hb_results[inst]
            print(f"  {i+1}. HB={hb_err:.4f}, Grid={grid_truth[inst]:.4f}:\n      {inst}")

    return {
        "common_count": len(common_insts),
        "spearman_correlation": correlation,
        "spearman_p_value": p_value,
        "top_5_overlap": top_5_overlap,
        "top_10_overlap": top_10_overlap,
        "mean_error_diff": mean_error_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LIPO: Instruction-only Hyperband + VAE + InvBO inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # APE generation
    parser.add_argument(
        "--ape-instructions", type=int, default=1000,
        help="Number of instructions to generate with APE (default: 1000)"
    )
    parser.add_argument(
        "--ape-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for APE generation"
    )
    parser.add_argument(
        "--ape-cache", type=str, default="lipo/data/ape_instructions.json",
        help="Path to APE instructions cache"
    )
    parser.add_argument(
        "--skip-ape", action="store_true",
        help="Skip APE generation (use cached instructions)"
    )
    parser.add_argument(
        "--force-regenerate-ape", action="store_true",
        help="Force regenerate APE instructions even if cache exists"
    )
    parser.add_argument(
        "--no-ape-augment", action="store_true",
        help="Disable APE instruction augmentation (paraphrasing + noise injection)"
    )

    # VAE training
    parser.add_argument(
        "--vae-beta", type=float, default=0.003,
        help="VAE KL regularization weight (scaled for latent_dim=64)"
    )
    parser.add_argument(
        "--vae-epochs", type=int, default=10000,
        help="VAE training epochs"
    )
    parser.add_argument(
        "--vae-annealing", type=int, default=500,
        help="VAE KL annealing epochs"
    )
    parser.add_argument(
        "--vae-gamma", type=float, default=1.0,
        help="VAE cycle consistency weight (ensures z ≈ encode(decode(z)))"
    )

    # Hyperband
    parser.add_argument(
        "--bmin", type=int, default=10,
        help="Minimum fidelity (samples per evaluation)"
    )
    parser.add_argument(
        "--eta", type=float, default=2.0,
        help="Hyperband downsampling rate"
    )
    parser.add_argument(
        "--hyperband-only", action="store_true",
        help="Run only Hyperband (no InvBO inference)"
    )
    parser.add_argument(
        "--skip-hbbops", action="store_true",
        help="Skip HbBoPs run, load evaluations from --hyperband-evals-path"
    )
    parser.add_argument(
        "--hyperband-evals-path", type=str,
        default="lipo/data/hbbops_results_20260102.json",
        help="Path to JSON with hyperband_evaluations (default: lipo/data/hbbops_results_20260102.json)"
    )

    # GP training
    parser.add_argument(
        "--gp-epochs", type=int, default=10000,
        help="GP training epochs (default: 10000)"
    )
    parser.add_argument(
        "--gp-lr", type=float, default=0.01,
        help="GP learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--gp-patience", type=int, default=100,
        help="GP early stopping patience (default: 100)"
    )

    # Grid loading (skip APE and Hyperband)
    parser.add_argument(
        "--load-grid", type=str, default=None,
        help="Path to pre-evaluated grid JSONL. Skips APE and optionally Hyperband."
    )
    parser.add_argument(
        "--top-k", type=int, default=25,
        help="Number of top instructions to load from grid (default: 25)"
    )
    parser.add_argument(
        "--instructions-path", type=str, default=None,
        help="Path to instructions text file (for grids with instruction_id)"
    )
    parser.add_argument(
        "--diverse-instructions", type=str, default=None,
        help="Path to diverse instructions JSON for VAE training (uses grid if not specified)"
    )
    parser.add_argument(
        "--validate-hyperband", action="store_true",
        help="Run Hyperband on grid instructions and validate ranking"
    )

    # Inference
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of InvBO inference iterations"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip LLM evaluation during inference (use GP predictions)"
    )
    parser.add_argument(
        "--vec2text-model", type=str, default="512_tokens",
        choices=["32_tokens", "512_tokens"],
        help="Vec2Text model type (default: 512_tokens)"
    )
    parser.add_argument(
        "--num-restarts", type=int, default=64,
        help="L-BFGS-B restarts for BoTorch optimization (default: 64)"
    )
    parser.add_argument(
        "--raw-samples", type=int, default=1024,
        help="Raw samples for initialization seeding (default: 1024)"
    )
    parser.add_argument(
        "--max-inversion-iters", type=int, default=3,
        help="Maximum inversion iterations per step (default: 3)"
    )
    parser.add_argument(
        "--gap-threshold", type=float, default=0.1,
        help="Gap threshold for re-inversion (cosine distance) (default: 0.1)"
    )
    parser.add_argument(
        "--vec2text-beam", type=int, default=8,
        help="Beam width for Vec2Text generation (default: 8)"
    )
    parser.add_argument(
        "--vec2text-max-length", type=int, default=128,
        help="Maximum output tokens for Vec2Text (default: 128)"
    )

    # Evaluation
    parser.add_argument(
        "--eval-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for evaluation"
    )
    parser.add_argument(
        "--eval-backend", type=str, default="vllm",
        help="Evaluation backend (vllm, openai, deepinfra)"
    )

    # Data paths
    parser.add_argument(
        "--validation-path", type=str, default="hbbops_improved_2/data/validation.json",
        help="Path to validation data"
    )

    # Debug mode
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode with reduced epochs and verbose output"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda, cpu, mps)"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="lipo/results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # =========================================================================
    # Setup Logging
    # =========================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = setup_logging(args.output_dir, timestamp)

    # TeeOutput redirects stdout/stderr to file handler only (logger already prints to console)
    # Get only the file handler from the logger
    file_logger = logging.getLogger("file_only")
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False  # Don't propagate to root logger (avoids double console output)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            file_logger.addHandler(handler)
            break

    # Redirect stdout/stderr through TeeOutput - writes to original stream + file logger
    sys.stdout = TeeOutput(sys.__stdout__, file_logger, logging.INFO)
    sys.stderr = TeeOutput(sys.__stderr__, file_logger, logging.WARNING)

    print(f"Logging to: {log_file}")

    # =========================================================================
    # Setup
    # =========================================================================

    mode = "skip_hbbops" if args.skip_hbbops else ("grid" if args.load_grid else "standard")
    print("=" * 70)
    print("LIPO PIPELINE")
    print("=" * 70)
    if args.skip_hbbops:
        print(f"Mode: SKIP HBBOPS (load pre-evaluated)")
        print(f"Evaluations: {args.hyperband_evals_path}")
    elif args.load_grid:
        print(f"Mode: GRID LOADING (top-{args.top_k})")
        print(f"Grid: {args.load_grid}")
        if args.validate_hyperband:
            print(f"Hyperband validation: ENABLED")
    else:
        print(f"Mode: STANDARD (APE + Hyperband)")
        print(f"APE instructions: {args.ape_instructions}")
    print(f"Hyperband: bmin={args.bmin}, eta={args.eta}")
    print(f"InvBO iterations: {args.iterations}")
    print(f"Vec2Text model: {args.vec2text_model}, max_length: {args.vec2text_max_length}")
    print(f"Debug mode: {args.debug}")
    print("=" * 70)

    # Adjust for debug mode
    if args.debug:
        print("\n[DEBUG MODE] Reducing epochs and using minimal config")
        args.vae_epochs = 100
        args.vae_annealing = 50

    # Import after arg parsing for faster --help
    from lipo.config import Config
    from lipo.training import LIPOHyperbandTrainer
    from lipo.inference import LIPOHyperbandInference
    from lipo.evaluate import GSM8KEvaluator
    from lipo.instruction import InstructionOnlyPrompt

    # =========================================================================
    # Build Config from CLI args (SSOT)
    # =========================================================================

    config = Config(
        # APE Generation
        ape_num_instructions=args.ape_instructions,
        ape_model=args.ape_model,
        ape_backend=args.eval_backend,
        ape_cache_path=args.ape_cache,
        # VAE Training
        vae_beta=args.vae_beta,
        vae_gamma=args.vae_gamma,
        vae_epochs=args.vae_epochs,
        vae_annealing_epochs=args.vae_annealing,
        # Hyperband
        bmin=args.bmin,
        eta=args.eta,
        # GP Training
        gp_epochs=args.gp_epochs,
        gp_lr=args.gp_lr,
        gp_patience=args.gp_patience,
        # Inference
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        max_inversion_iters=args.max_inversion_iters,
        gap_threshold=args.gap_threshold,
        vec2text_beam=args.vec2text_beam,
        vec2text_model=args.vec2text_model,
        vec2text_max_length=args.vec2text_max_length,
        # Device/Paths
        validation_path=args.validation_path,
        device=args.device,
    )

    # =========================================================================
    # Create Trainer
    # =========================================================================

    trainer = LIPOHyperbandTrainer(config)

    # Create evaluator (might be needed for Hyperband validation)
    evaluator = GSM8KEvaluator(
        model=args.eval_model,
        backend=args.eval_backend,
    )

    # =========================================================================
    # SKIP HBBOPS MODE: Load from pre-evaluated hyperband_evaluations JSON
    # =========================================================================

    if args.skip_hbbops:
        trainer.load_validation_data(verbose=True)

        # Regenerate APE instructions if requested (before loading hyperband evaluations)
        if args.force_regenerate_ape:
            print("\n" + "=" * 60)
            print("Regenerating APE Instructions")
            print("=" * 60)
            trainer.generate_instructions(
                num_instructions=args.ape_instructions,
                force_regenerate=True,
                verbose=True,
                augment=not args.no_ape_augment,
            )

        # Load instructions and evaluations from JSON
        trainer.load_from_hyperband_evaluations(
            args.hyperband_evals_path,
            verbose=True,
        )

        # Train VAE on all diverse instructions (1000+)
        trainer.train_vae(embedding_source="diverse", verbose=True)
        vae_quality_metrics = trainer.evaluate_vae_quality(verbose=True)

        # Train GP on evaluated instructions (225)
        gp = trainer.train_gp_from_grid(verbose=True)

        # Get best instruction from FULL FIDELITY evaluations only
        # Low-fidelity evaluations have high variance and may be misleading
        best_prompt, best_error = trainer.get_full_fidelity_best()

        if best_prompt is None:
            # Fallback: use best from all evaluations (with warning)
            print("WARNING: No full-fidelity evaluations found!")
            print("  Using best from all evaluations (may be unreliable)")
            best_idx = int(trainer.grid_error_rates.index(min(trainer.grid_error_rates)))
            best_prompt = InstructionOnlyPrompt(
                instruction=trainer.instructions[best_idx],
                instruction_id=best_idx,
            )
            best_error = trainer.grid_error_rates[best_idx]

        print(f"\nSkip HbBoPs loading complete!")
        print(f"  Best prompt (from full-fidelity evaluations):\n{best_prompt.instruction}")
        print(f"  Best error: {best_error:.4f}")

    # =========================================================================
    # GRID MODE: Load from pre-evaluated grid
    # =========================================================================

    elif args.load_grid:
        trainer.load_validation_data(verbose=True)

        if args.validate_hyperband:
            # Mode: Load grid instructions, run Hyperband, compare ranking
            trainer.load_from_grid(
                args.load_grid,
                top_k=None,  # Load ALL for Hyperband
                instructions_path=args.instructions_path,
                verbose=True,
            )
            trainer.train_vae(verbose=True)
            vae_quality_metrics = trainer.evaluate_vae_quality(verbose=True)

            # Run Hyperband on grid instructions
            best_prompt, best_error = trainer.run_hyperband(
                llm_evaluator=evaluator,
                verbose=True,
            )

            # Save HbBoPs evaluation results to separate file
            if trainer.hyperband is not None:
                hbbops_results = extract_from_hyperband(trainer.hyperband)
                hbbops_output_path = f"lipo/data/hbbops_results_{timestamp}.json"
                save_hbbops_results(
                    results=hbbops_results,
                    output_path=hbbops_output_path,
                    source_log=str(log_path),
                    max_fidelity=len(trainer.validation_data) if trainer.validation_data else 1319,
                    instructions=trainer.instructions,
                )

            # Validate ranking
            validation_result = _validate_hyperband_ranking(
                trainer=trainer,
                grid_path=args.load_grid,
                verbose=True,
            )

            print(f"\nHyperband validation complete!")
            print(f"  Best prompt:\n{best_prompt.instruction}")
            print(f"  Best error: {best_error:.4f}")
            print(f"  LLM calls (Hyperband): {trainer.total_llm_calls}")

            if args.hyperband_only:
                _save_results(args, trainer, evaluator, best_prompt, best_error, None, timestamp, vae_quality_metrics)
                return

            gp = trainer.get_gp_for_inference()

        else:
            # Mode: Load top-k from grid, skip Hyperband, train GP directly
            trainer.load_from_grid(
                args.load_grid,
                top_k=args.top_k,
                instructions_path=args.instructions_path,
                verbose=True,
            )

            # Train VAE on diverse instructions or grid instructions
            if args.diverse_instructions:
                trainer.load_diverse_instructions(args.diverse_instructions, verbose=True)
                trainer.train_vae(embedding_source="diverse", verbose=True)
            else:
                trainer.train_vae(verbose=True)

            # Evaluate VAE quality
            vae_quality_metrics = trainer.evaluate_vae_quality(verbose=True)

            gp = trainer.train_gp_from_grid(verbose=True)

            # Get best from grid (full-fidelity only by default)
            best_prompt, best_error = trainer.get_best_from_grid(full_fidelity_only=True)

            print(f"\nGrid loading complete!")
            print(f"  Best prompt (from full-fidelity grid evaluations):\n{best_prompt.instruction}")
            print(f"  Best error (full-fidelity): {best_error:.4f}")

    # =========================================================================
    # STANDARD MODE: APE → VAE → Hyperband
    # =========================================================================

    else:
        trainer.load_validation_data(verbose=True)

        if args.skip_ape:
            # Load from cache only
            if not Path(args.ape_cache).exists():
                print(f"ERROR: APE cache not found at {args.ape_cache}")
                print("Run without --skip-ape to generate instructions first.")
                sys.exit(1)
            trainer.generate_instructions(
                num_instructions=args.ape_instructions,
                force_regenerate=False,
                verbose=True,
                augment=not args.no_ape_augment,
            )
        else:
            trainer.generate_instructions(
                num_instructions=args.ape_instructions,
                force_regenerate=args.force_regenerate_ape,
                verbose=True,
                augment=not args.no_ape_augment,
            )

        trainer.train_vae(verbose=True)
        vae_quality_metrics = trainer.evaluate_vae_quality(verbose=True)

        best_prompt, best_error = trainer.run_hyperband(
            llm_evaluator=evaluator,
            verbose=True,
        )

        # Save HbBoPs evaluation results to separate file
        if trainer.hyperband is not None:
            hbbops_results = extract_from_hyperband(trainer.hyperband)
            hbbops_output_path = f"lipo/data/hbbops_results_{timestamp}.json"
            save_hbbops_results(
                results=hbbops_results,
                output_path=hbbops_output_path,
                source_log=str(log_path),
                max_fidelity=len(trainer.validation_data) if trainer.validation_data else 1319,
                instructions=trainer.instructions,
            )

        print(f"\nHyperband complete!")
        print(f"  Best prompt:\n{best_prompt.instruction}")
        print(f"  Best error: {best_error:.4f}")
        print(f"  LLM calls (Hyperband): {trainer.total_llm_calls}")

        if args.hyperband_only:
            print("\n--hyperband-only specified, skipping inference.")
            _save_results(args, trainer, evaluator, best_prompt, best_error, None, timestamp, vae_quality_metrics)
            return

        gp = trainer.get_gp_for_inference()

    # =========================================================================
    # InvBO Inference
    # =========================================================================

    print("\n" + "=" * 60)
    print("Starting InvBO Inference")
    print("=" * 60)

    inference = LIPOHyperbandInference(
        gp=gp,
        vae=trainer.vae,
        config=config,
        gtr=trainer.gtr,
        evaluator=evaluator if not args.skip_eval else None,
        validation_data=trainer.validation_data if not args.skip_eval else None,
        initial_best_instruction=best_prompt.instruction,
        initial_best_error=best_error,
    )

    # === Round-Trip Diagnostic ===
    # Test VAE+Vec2Text reconstruction on top training instructions before inference
    if trainer.instructions and trainer.grid_error_rates:
        # Get top-10 instructions sorted by error rate (best first)
        sorted_indices = sorted(
            range(len(trainer.grid_error_rates)),
            key=lambda i: trainer.grid_error_rates[i]
        )[:10]
        top_instructions = [trainer.instructions[i] for i in sorted_indices]
        round_trip_results = inference.run_round_trip_diagnostic(top_instructions, verbose=True)
    else:
        round_trip_results = None
        print("WARNING: Skipping round-trip diagnostic (no training instructions available)")

    history = inference.run(
        iterations=args.iterations,
        num_restarts=config.num_restarts,
        raw_samples=config.raw_samples,
        use_inversion=config.use_inversion,
        max_inversion_iters=config.max_inversion_iters,
        gap_threshold=config.gap_threshold,
        skip_eval=args.skip_eval,
        verbose=True,
    )

    # =========================================================================
    # Results
    # =========================================================================

    # Note: evaluator.total_calls is cumulative (includes both Hyperband and Inference)
    # So we use the separate counts to avoid double counting
    total_llm_calls = trainer.total_llm_calls + trainer.ape_llm_calls + inference.total_llm_calls

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best instruction:\n{inference.best_instruction}")
    print(f"\nBest error: {inference.best_error:.4f}")
    print(f"Improvement from Hyperband: {best_error - inference.best_error:.4f}")
    print(f"\nTotal LLM calls:")
    print(f"  APE generation: {trainer.ape_llm_calls}")
    print(f"  Hyperband: {trainer.total_llm_calls}")
    print(f"  Inference: {inference.total_llm_calls}")
    print(f"  TOTAL: {total_llm_calls}")
    print("=" * 70)

    _save_results(args, trainer, evaluator, best_prompt, best_error, inference, timestamp, vae_quality_metrics, round_trip_results)


def _save_results(
    args, trainer, evaluator, best_prompt, best_error, inference, timestamp: str,
    vae_quality_metrics: dict = None, round_trip_results: dict = None
):
    """Save results to JSON file."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"result_{timestamp}.json"

    # Calculate total LLM calls (avoid double counting - don't add evaluator.total_calls)
    total_llm_calls = trainer.ape_llm_calls + trainer.total_llm_calls
    if inference:
        total_llm_calls += inference.total_llm_calls

    result = {
        "timestamp": timestamp,
        "vae_quality_metrics": vae_quality_metrics,
        "round_trip_diagnostic": {
            "mean_similarity": round_trip_results["mean_similarity"] if round_trip_results else None,
            "min_similarity": round_trip_results["min_similarity"] if round_trip_results else None,
            "max_similarity": round_trip_results["max_similarity"] if round_trip_results else None,
            "below_90": round_trip_results["below_90"] if round_trip_results else None,
            "below_95": round_trip_results["below_95"] if round_trip_results else None,
        } if round_trip_results else None,
        "config": {
            # Mode
            "mode": "grid_loading" if args.load_grid else "standard",
            "grid_path": args.load_grid,
            "top_k": args.top_k if args.load_grid else None,
            "validate_hyperband": args.validate_hyperband if hasattr(args, 'validate_hyperband') else False,

            # APE settings
            "ape_instructions": args.ape_instructions,
            "ape_model": args.ape_model,
            "skip_ape": args.skip_ape,

            # VAE hyperparameters
            "vae_beta": args.vae_beta,
            "vae_epochs": args.vae_epochs,
            "vae_annealing": args.vae_annealing,
            "vae_latent_dim": trainer.config.latent_dim,
            "vae_lr": trainer.config.vae_lr,
            "vae_patience": trainer.config.vae_patience,

            # Hyperband settings
            "bmin": args.bmin,
            "eta": args.eta,

            # GP hyperparameters
            "gp_epochs": trainer.config.gp_epochs,
            "gp_lr": trainer.config.gp_lr,
            "gp_patience": trainer.config.gp_patience,

            # Inference settings
            "iterations": args.iterations,
            "vec2text_model": args.vec2text_model,

            # Evaluation settings
            "eval_model": args.eval_model,
            "eval_backend": args.eval_backend,
            "validation_samples": len(trainer.validation_data) if trainer.validation_data else 0,

            # Debug
            "debug": args.debug,
        },
        "vae_training": trainer.vae_stats,
        "gp_training": trainer.gp_stats,
        "ape_generation": {
            "llm_calls": trainer.ape_llm_calls,
        },
        "hyperband": {
            "best_instruction": best_prompt.instruction,
            "best_error": best_error,
            "llm_calls": trainer.total_llm_calls,
        },
        "total_llm_calls": total_llm_calls,
    }

    if inference:
        result["inference"] = {
            "best_instruction": inference.best_instruction,
            "best_error": inference.best_error,
            "iterations": len(inference.iteration_history),
            "llm_calls": inference.total_llm_calls,
            "improvement": best_error - inference.best_error,
            "use_inversion": True,
            "use_botorch": True,
            "history": [
                {
                    "iteration": r.iteration,
                    "instruction": r.instruction,  # Never truncate - per CLAUDE.md
                    "predicted_error": r.predicted_error,
                    "actual_error": r.actual_error,
                    "improved": r.improved,
                    "best_so_far": r.best_error_so_far,
                    "cosine_similarity": r.cosine_similarity,
                    "log_ei": r.log_ei,
                    "gap": r.gap,
                    "inversion_iters": r.inversion_iters,
                    # Optimization Gap Test metrics
                    "z_opt_z_real_cosine": r.z_opt_z_real_cosine,
                    "z_opt_z_real_euclidean": r.z_opt_z_real_euclidean,
                    "z_opt_z_real_gp_cosine": r.z_opt_z_real_gp_cosine,
                    "predicted_error_at_z_real": r.predicted_error_at_z_real,
                }
                for r in inference.iteration_history
            ],
        }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
