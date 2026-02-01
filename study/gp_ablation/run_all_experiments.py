#!/usr/bin/env python3
"""
Master script for running the complete GP Ablation Study.

This script orchestrates all ablation experiments for the NeurIPS paper:
- GP method comparison (standard_msr, turbo, saasbo, baxus, etc.)
- Kernel comparison (matern52, matern32, rbf)
- Acquisition comparison (log_ei, ucb, thompson)
- Projection dimension (for BAxUS)
- Multi-seed validation for ALL methods

Usage:
    # Run all experiments with multi-seed (NeurIPS requirement)
    python -m study.gp_ablation.run_all_experiments --all --seeds 3

    # Run specific ablation
    python -m study.gp_ablation.run_all_experiments --ablation method --seeds 3

    # Dry run (print commands without executing)
    python -m study.gp_ablation.run_all_experiments --all --dry-run

    # Resume from specific experiment
    python -m study.gp_ablation.run_all_experiments --all --resume standard_msr-s123

Environment:
    CUDA_VISIBLE_DEVICES: GPU to use (default: 0)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional, List, Dict, Any

from study.gp_ablation.config import GPConfig


# =============================================================================
# Configuration
# =============================================================================

# Default settings
DEFAULT_GROUP = "gp-ablation-study"
DEFAULT_DATA_PATH = "datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt"
DEFAULT_FLOW_CHECKPOINT = "study/checkpoints/mlp-otcfm-5k-none/best.pt"

# Multiple seeds for statistical rigor (NeurIPS requirement: ≥3 seeds)
DEFAULT_SEEDS = [42, 123, 456]
EXTENDED_SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 4242, 5678, 9999]  # 10 seeds

# Method tiers
TIER1_METHODS = ["standard_msr", "turbo", "saasbo", "baxus", "heteroscedastic"]
TIER2_METHODS = ["riemannian", "gebo", "lamcts"]
TIER3_METHODS = ["turbo_grad", "flow_bo", "turbo_geodesic", "baxus_flow"]
TIER4_METHODS = ["latent_bo", "bayesian_flow_bo", "velocity_acq", "curriculum_bo"]

# Kernel types
KERNEL_TYPES = ["matern52", "matern32", "rbf"]

# Acquisition functions
ACQUISITIONS = ["log_ei", "ucb", "thompson"]

# Projection dimensions (for BAxUS)
PROJECTION_DIMS = [32, 64, 128]


@dataclass
class Experiment:
    """Single experiment configuration."""
    config: GPConfig
    extra_args: List[str] = field(default_factory=list)

    @property
    def run_name(self) -> str:
        """Generate unique run name."""
        return self.config.run_name

    @property
    def results_path(self) -> Path:
        """Results file path."""
        return Path(self.config.results_dir) / f"{self.run_name}.json"

    def is_complete(self) -> bool:
        """Check if experiment already completed."""
        return self.results_path.exists()

    def get_command(self) -> List[str]:
        """Generate training command."""
        cmd = [
            "uv", "run", "python", "-m", "study.gp_ablation.train",
            "--method", self.config.method,
            "--kernel", self.config.kernel,
            "--acquisition", self.config.acquisition,
            "--seed", str(self.config.seed),
            "--data-path", self.config.data_path,
            "--results-dir", self.config.results_dir,
            "--wandb-group", self.config.wandb_group,
        ]

        # Add method-specific arguments
        if self.config.method == "baxus":
            cmd.extend(["--target-dim", str(self.config.target_dim)])
        elif self.config.method == "turbo":
            cmd.extend(["--length-init", str(self.config.length_init)])
        elif self.config.method == "saasbo":
            cmd.extend(["--nuts-samples", str(self.config.nuts_samples)])
        elif self.config.method in ("latent_bo", "flow_bo", "bayesian_flow_bo", "velocity_acq"):
            if self.config.flow_checkpoint:
                cmd.extend(["--flow-checkpoint", self.config.flow_checkpoint])

        cmd.extend(self.extra_args)
        return cmd


# =============================================================================
# Ablation Definitions
# =============================================================================

def create_experiment(
    method: str,
    seed: int = 42,
    kernel: str = "matern52",
    acquisition: str = "log_ei",
    **kwargs
) -> Experiment:
    """Create a single experiment."""
    config = GPConfig(
        method=method,
        kernel=kernel,
        acquisition=acquisition,
        seed=seed,
        wandb_group=DEFAULT_GROUP,
        data_path=DEFAULT_DATA_PATH,
        **kwargs
    )

    # Set flow checkpoint for flow-based methods
    if method in ("latent_bo", "flow_bo", "bayesian_flow_bo", "velocity_acq", "turbo_grad"):
        config.flow_checkpoint = DEFAULT_FLOW_CHECKPOINT

    return Experiment(config=config)


def get_method_ablation() -> List[Experiment]:
    """GP method comparison.

    Compare all Tier 1 methods with default settings.
    """
    return [
        create_experiment(method=method)
        for method in TIER1_METHODS
    ]


def get_kernel_ablation() -> List[Experiment]:
    """Kernel type comparison on standard_msr.

    Fixed: method=standard_msr
    Variable: kernel type
    """
    return [
        create_experiment(method="standard_msr", kernel=kernel)
        for kernel in KERNEL_TYPES
    ]


def get_acquisition_ablation() -> List[Experiment]:
    """Acquisition function comparison on standard_msr.

    Fixed: method=standard_msr
    Variable: acquisition function
    """
    return [
        create_experiment(method="standard_msr", acquisition=acq)
        for acq in ACQUISITIONS
    ]


def get_projection_ablation() -> List[Experiment]:
    """Projection dimension comparison for BAxUS.

    Fixed: method=baxus
    Variable: target_dim
    """
    return [
        create_experiment(method="baxus", target_dim=dim)
        for dim in PROJECTION_DIMS
    ]


def get_turbo_length_ablation() -> List[Experiment]:
    """TuRBO length scale comparison.

    Fixed: method=turbo
    Variable: length_init
    """
    return [
        create_experiment(method="turbo", length_init=length)
        for length in [0.4, 0.8, 1.6]
    ]


def get_saasbo_samples_ablation() -> List[Experiment]:
    """SAASBO MCMC samples comparison.

    Fixed: method=saasbo
    Variable: nuts_samples
    """
    return [
        create_experiment(method="saasbo", nuts_samples=n)
        for n in [64, 128, 256]
    ]


def get_tier2_ablation() -> List[Experiment]:
    """Advanced methods (Tier 2)."""
    experiments = []

    # Riemannian BO variants
    experiments.extend([
        create_experiment(method="riemannian", kernel="arccosine"),
        create_experiment(method="riemannian", kernel="geodesic_matern52"),
        create_experiment(method="riemannian", normalize_inputs=True),
    ])

    # GEBO variants
    experiments.extend([
        create_experiment(method="gebo", n_directions=1),
        create_experiment(method="gebo", n_directions=10),
    ])

    # LaMCTS variants
    experiments.extend([
        create_experiment(method="lamcts", max_depth=10),
        create_experiment(method="lamcts", max_depth=20),
    ])

    return experiments


def get_hybrid_ablation() -> List[Experiment]:
    """Hybrid methods (Tier 3)."""
    experiments = []

    # TuRBO + Gradient Refinement
    experiments.extend([
        create_experiment(method="turbo_grad", n_grad_steps=5, grad_lr=0.01),
        create_experiment(method="turbo_grad", n_grad_steps=10, grad_lr=0.01),
    ])

    # Flow-Guided BO
    experiments.append(
        create_experiment(method="flow_bo")
    )

    # Geodesic TuRBO
    experiments.extend([
        create_experiment(method="turbo_geodesic", kernel="geodesic_matern52"),
        create_experiment(method="turbo_geodesic", kernel="arccosine"),
    ])

    return experiments


def get_novel_ablation() -> List[Experiment]:
    """Novel methods (Tier 4)."""
    experiments = []

    # Latent Space BO (priority)
    experiments.append(
        create_experiment(method="latent_bo", invert_method="ode")
    )

    # Bayesian Flow BO
    experiments.extend([
        create_experiment(method="bayesian_flow_bo", prior_weight=0.1),
        create_experiment(method="bayesian_flow_bo", prior_weight=0.5),
    ])

    # Velocity-Guided Acquisition
    experiments.extend([
        create_experiment(method="velocity_acq", alignment_weight=0.3),
        create_experiment(method="velocity_acq", alignment_weight=0.5),
    ])

    # Curriculum BO
    experiments.extend([
        create_experiment(method="curriculum_bo", fidelity_start=0.1, fidelity_schedule="linear"),
        create_experiment(method="curriculum_bo", fidelity_start=0.2, fidelity_schedule="exponential"),
    ])

    return experiments


def deduplicate_experiments(experiments: List[Experiment]) -> List[Experiment]:
    """Remove duplicate experiments based on run_name."""
    seen: set = set()
    unique: List[Experiment] = []
    for exp in experiments:
        if exp.run_name not in seen:
            seen.add(exp.run_name)
            unique.append(exp)
    return unique


def get_all_experiments() -> List[Experiment]:
    """Get all experiments for the full study."""
    all_experiments = [
        *get_method_ablation(),
        *get_kernel_ablation(),
        *get_acquisition_ablation(),
        *get_projection_ablation(),
        *get_turbo_length_ablation(),
        *get_saasbo_samples_ablation(),
        *get_tier2_ablation(),
        *get_hybrid_ablation(),
        *get_novel_ablation(),
    ]
    return deduplicate_experiments(all_experiments)


def expand_with_seeds(experiments: List[Experiment], seeds: List[int]) -> List[Experiment]:
    """Expand a list of experiments to run with multiple seeds.

    Creates a new experiment for each seed in the list.

    Args:
        experiments: Base experiments to expand.
        seeds: List of random seeds (e.g., [42, 123, 456]).

    Returns:
        Expanded list with len(experiments) * len(seeds) entries.
    """
    expanded = []
    for exp in experiments:
        for seed in seeds:
            new_config = replace(
                exp.config,
                seed=seed,
            )
            expanded.append(Experiment(config=new_config, extra_args=list(exp.extra_args)))
    return expanded


# Ablation registry
ABLATION_REGISTRY = {
    # Single-factor ablations
    "method": get_method_ablation,
    "kernel": get_kernel_ablation,
    "acquisition": get_acquisition_ablation,
    "projection": get_projection_ablation,
    "turbo_length": get_turbo_length_ablation,
    "saasbo_samples": get_saasbo_samples_ablation,
    # Tier-based ablations
    "tier2": get_tier2_ablation,
    "hybrid": get_hybrid_ablation,
    "novel": get_novel_ablation,
    # Combined
    "all": get_all_experiments,
    # Specific method groups
    "tier1": get_method_ablation,
}


# =============================================================================
# Execution
# =============================================================================

def run_experiment(
    exp: Experiment,
    gpu: int,
    dry_run: bool = False,
    verbose: bool = True,
) -> tuple:
    """Run a single experiment.

    Returns:
        (success, duration_minutes)
    """
    if exp.is_complete():
        if verbose:
            print(f"  [SKIP] {exp.run_name} - already complete")
        return True, 0.0

    cmd = exp.get_command()
    env_prefix = f"CUDA_VISIBLE_DEVICES={gpu}"
    full_cmd = f"{env_prefix} {' '.join(cmd)}"

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Running: {exp.run_name}")
        print(f"  Command: {full_cmd}")
        print(f"{'='*70}")

    if dry_run:
        return True, 0.0

    start = time.time()
    try:
        subprocess.run(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            capture_output=False,
            check=True,
        )
        return True, (time.time() - start) / 60
    except subprocess.CalledProcessError:
        duration = (time.time() - start) / 60
        print(f"  [FAILED] {exp.run_name} after {duration:.1f} min")
        return False, duration
    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] {exp.run_name}")
        raise


def run_ablation(
    ablation: str,
    gpu: int,
    resume_from: Optional[str] = None,
    dry_run: bool = False,
    seeds: Optional[List[int]] = None,
) -> Dict:
    """Run an ablation study.

    Args:
        ablation: Name of the ablation from ABLATION_REGISTRY.
        gpu: GPU device to use.
        resume_from: Run name to resume from.
        dry_run: If True, print commands without executing.
        seeds: If provided, expand each experiment to run with each seed.

    Returns:
        Results dictionary with experiment outcomes and summary.
    """
    if ablation not in ABLATION_REGISTRY:
        raise ValueError(
            f"Unknown ablation: {ablation}. "
            f"Available: {list(ABLATION_REGISTRY.keys())}"
        )

    experiments = ABLATION_REGISTRY[ablation]()

    # Expand with multiple seeds if requested
    if seeds:
        experiments = expand_with_seeds(experiments, seeds)
        print(f"\nExpanded experiments with seeds {seeds}: {len(experiments)} total")

    if resume_from:
        print(f"\nResuming from: {resume_from}")

    results: Dict = {
        "ablation": ablation,
        "started": datetime.now().isoformat(),
        "seeds": seeds,
        "experiments": [],
    }

    skipping = resume_from is not None
    total = len(experiments)
    completed = 0
    failed = 0

    print(f"\n{'#'*70}")
    print(f"  GP ABLATION: {ablation.upper()}")
    print(f"  Experiments: {total}")
    print(f"  GPU: {gpu}")
    if seeds:
        print(f"  Seeds: {seeds}")
    print(f"{'#'*70}")

    for i, exp in enumerate(experiments, 1):
        # Handle resume
        if skipping:
            if exp.run_name == resume_from:
                skipping = False
            else:
                print(f"  [{i}/{total}] SKIP (resume): {exp.run_name}")
                continue

        print(f"\n  [{i}/{total}] {exp.run_name}")

        success, duration = run_experiment(exp, gpu, dry_run)

        results["experiments"].append({
            "name": exp.run_name,
            "method": exp.config.method,
            "seed": exp.config.seed,
            "success": success,
            "duration_min": duration,
            "results_path": str(exp.results_path) if success else None,
        })

        if success:
            completed += 1
        else:
            failed += 1

    results["completed"] = datetime.now().isoformat()
    results["summary"] = {
        "total": total,
        "completed": completed,
        "failed": failed,
        "skipped": total - completed - failed,
    }

    return results


def print_summary(results: Dict):
    """Print results summary."""
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Ablation: {results['ablation']}")
    print(f"  Total: {results['summary']['total']}")
    print(f"  Completed: {results['summary']['completed']}")
    print(f"  Failed: {results['summary']['failed']}")
    print(f"  Skipped: {results['summary']['skipped']}")
    print()

    for exp in results["experiments"]:
        status = "✓" if exp["success"] else "✗"
        duration = f"{exp['duration_min']:.1f}min" if exp["duration_min"] > 0 else "skip"
        print(f"  {status} {exp['name']} ({duration})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run GP Ablation Study experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--ablation",
        choices=list(ABLATION_REGISTRY.keys()),
        help="Which ablation to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU to use (default: 0)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="RUN_NAME",
        help="Resume from specific experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments and exit",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds for multi-seed runs (default: 3)",
    )
    parser.add_argument(
        "--no-multiseed",
        action="store_true",
        help="Run single seed only (not recommended for NeurIPS)",
    )

    args = parser.parse_args()

    # Determine seeds
    if args.no_multiseed:
        seeds = None
    else:
        seeds = DEFAULT_SEEDS[:args.seeds]

    # List mode
    if args.list:
        print("\nAvailable ablations:")
        for name, fn in ABLATION_REGISTRY.items():
            experiments = fn()
            print(f"\n  {name} ({len(experiments)} base experiments):")
            for exp in experiments[:5]:
                status = "✓" if exp.is_complete() else "○"
                print(f"    {status} {exp.run_name}")
            if len(experiments) > 5:
                print(f"    ... and {len(experiments) - 5} more")

        if seeds:
            print(f"\n  With {len(seeds)} seeds: experiments × {len(seeds)} = total runs")
        return

    # Determine ablation
    if args.all:
        ablation = "all"
    elif args.ablation:
        ablation = args.ablation
    else:
        parser.print_help()
        print("\nError: Must specify --ablation or --all")
        sys.exit(1)

    # Run
    try:
        results = run_ablation(
            ablation=ablation,
            gpu=args.gpu,
            resume_from=args.resume,
            dry_run=args.dry_run,
            seeds=seeds,
        )

        print_summary(results)

        # Save results
        if not args.dry_run:
            results_file = Path(
                f"study/results/gp_ablation_{ablation}_{datetime.now():%Y%m%d_%H%M%S}.json"
            )
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_file}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
