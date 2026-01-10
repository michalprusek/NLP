#!/usr/bin/env python3
"""
BOLT Hyperparameter Tuning CLI

Run comprehensive hyperparameter optimization for the BOLT pipeline.

Usage:
    # Full tuning (may take days/weeks)
    uv run python -m bolt.tuning.run_tuning --output-dir bolt/tuning/results

    # Quick test (few trials per phase)
    uv run python -m bolt.tuning.run_tuning --quick --output-dir bolt/tuning/test

    # Single phase only
    uv run python -m bolt.tuning.run_tuning --phase vae --output-dir bolt/tuning/vae_only

    # Resume from checkpoint
    uv run python -m bolt.tuning.run_tuning --resume --output-dir bolt/tuning/results

    # Custom GPU assignment
    uv run python -m bolt.tuning.run_tuning --gpus 0,1 --output-dir bolt/tuning/results
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BOLT Hyperparameter Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bolt/tuning/results",
        help="Directory for tuning outputs (default: bolt/tuning/results)",
    )

    # GPU configuration
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated GPU IDs to use (default: 0,1)",
    )

    # Phases
    parser.add_argument(
        "--phase",
        type=str,
        choices=["vae", "scorer", "gp", "inference", "all"],
        default="all",
        help="Phase to tune (default: all)",
    )
    parser.add_argument(
        "--skip-phases",
        type=str,
        default="",
        help="Comma-separated phases to skip",
    )

    # Trial counts
    parser.add_argument(
        "--trials-critical",
        type=int,
        default=50,
        help="Number of trials for CRITICAL tier (default: 50)",
    )
    parser.add_argument(
        "--trials-important",
        type=int,
        default=30,
        help="Number of trials for IMPORTANT tier (default: 30)",
    )
    parser.add_argument(
        "--trials-finetune",
        type=int,
        default=20,
        help="Number of trials for FINETUNE tier (default: 20)",
    )

    # Quick mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 5 trials per phase for testing",
    )

    # Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoint",
    )

    # Sampling strategy
    parser.add_argument(
        "--sampling",
        type=str,
        choices=["random", "sobol", "grid"],
        default="sobol",
        help="Sampling strategy (default: sobol)",
    )

    # Inference settings
    parser.add_argument(
        "--inference-iterations",
        type=int,
        default=10,
        help="BO iterations for inference evaluation (default: 10)",
    )
    parser.add_argument(
        "--hyperband-fidelity",
        type=int,
        default=100,
        help="Hyperband fidelity for evaluation (default: 100)",
    )

    # Time limits
    parser.add_argument(
        "--max-time-hours",
        type=float,
        default=0,
        help="Maximum total runtime in hours (0 = unlimited)",
    )

    # Reporting
    parser.add_argument(
        "--report-interval",
        type=int,
        default=10,
        help="Generate report every N trials (default: 10)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse GPUs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    logger.info(f"Using GPUs: {gpu_ids}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "tuning_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Import after arg parsing (for faster --help)
    from .coordinator import CoordinateDescentTuner, DEFAULT_PHASE_CONFIGS, run_quick_tune
    from .hyperspace import TuningPhase, TuningTier

    # Quick mode
    if args.quick:
        logger.info("Running in QUICK mode (5 trials per phase)")
        results = run_quick_tune(
            output_dir=output_dir,
            gpu_ids=gpu_ids,
            n_trials_per_phase=5,
        )
        print_results(results)
        return

    # Configure phase configs
    phase_configs = {}
    for phase in TuningPhase:
        config = DEFAULT_PHASE_CONFIGS[phase]

        # Update trial counts
        config.n_trials_per_tier = {
            TuningTier.CRITICAL: args.trials_critical,
            TuningTier.IMPORTANT: args.trials_important,
            TuningTier.FINETUNE: args.trials_finetune,
        }

        phase_configs[phase] = config

    # Determine phases to run
    phases_to_run = None
    skip_phases = None

    if args.phase != "all":
        phases_to_run = [TuningPhase(args.phase)]

    if args.skip_phases:
        skip_phases = [TuningPhase(p.strip()) for p in args.skip_phases.split(",")]

    # Resume handling
    resume = args.resume and not args.no_resume

    # Create and run tuner
    tuner = CoordinateDescentTuner(
        output_dir=output_dir,
        gpu_ids=gpu_ids,
        phase_configs=phase_configs,
        resume=resume,
        sampling_strategy=args.sampling,
    )

    logger.info("=" * 60)
    logger.info("Starting BOLT Hyperparameter Tuning")
    logger.info(f"Output: {output_dir}")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Phases: {phases_to_run or 'all'}")
    logger.info(f"Skip: {skip_phases or 'none'}")
    logger.info(f"Resume: {resume}")
    logger.info(f"Sampling: {args.sampling}")
    logger.info("=" * 60)

    try:
        results = tuner.run(
            phases=phases_to_run,
            skip_phases=skip_phases,
        )

        # Print final summary
        print_results({p.value: r.to_dict() for p, r in results.items()})

        logger.info("=" * 60)
        logger.info("Tuning completed!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. State saved for resume.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during tuning: {e}")
        sys.exit(1)


def print_results(results: dict):
    """Print results summary."""
    print("\n" + "=" * 60)
    print("TUNING RESULTS SUMMARY")
    print("=" * 60)

    for phase_name, result in results.items():
        status = result.get("status", "unknown")
        best_obj = result.get("best_objective", 0)
        total_trials = result.get("total_trials", 0)
        successful = result.get("successful_trials", 0)
        time_hours = result.get("total_time_seconds", 0) / 3600

        checkpoint = "PASSED" if result.get("checkpoint_passed") else "FAILED"

        print(f"\n{phase_name.upper()}:")
        print(f"  Status: {status}")
        print(f"  Best Objective: {best_obj:.4f}")
        print(f"  Trials: {successful}/{total_trials} successful")
        print(f"  Time: {time_hours:.2f} hours")
        print(f"  Checkpoint: {checkpoint}")

        if not result.get("checkpoint_passed"):
            failures = result.get("checkpoint_failures", [])
            for f in failures[:3]:
                print(f"    - {f}")

    print("\n" + "=" * 60)


def status_command():
    """Check status of running tuning."""
    parser = argparse.ArgumentParser(description="Check tuning status")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Check heartbeat
    heartbeat_path = output_dir / "trials" / "heartbeat.json"
    if heartbeat_path.exists():
        with open(heartbeat_path) as f:
            heartbeat = json.load(f)

        print("Executor Status:")
        print(f"  Last heartbeat: {heartbeat.get('timestamp')}")
        print(f"  Pending: {heartbeat.get('pending')}")
        print(f"  Running: {heartbeat.get('running')}")
        print(f"  Completed: {heartbeat.get('completed')}")
        print(f"  Failed: {heartbeat.get('failed')}")
        print(f"  Uptime: {heartbeat.get('uptime_hours', 0):.1f} hours")
    else:
        print("No active tuning found.")

    # Check coordinator state
    state_path = output_dir / "coordinator_state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)

        print("\nCoordinator Status:")
        print(f"  Current phase: {state.get('current_phase')}")
        print(f"  Current tier: {state.get('current_tier')}")
        print(f"  Completed phases: {state.get('completed_phases')}")
        print(f"  Total trials: {state.get('total_trials_run')}")


if __name__ == "__main__":
    main()
