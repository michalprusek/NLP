#!/usr/bin/env python3
"""
Master script for running the complete Flow Matching Architecture Study.

This script orchestrates all ablation experiments for the NeurIPS paper:
- Flow method comparison (I-CFM, OT-CFM, Reflow, SI-GVP)
- Architecture comparison (MLP, DiT, U-Net)
- Dataset size scaling (1K, 5K, 10K)
- Augmentation effectiveness (none, mixup, noise, mixup+noise)

Usage:
    # Run all experiments (full study)
    python -m study.run_all_experiments --all

    # Run specific ablation
    python -m study.run_all_experiments --ablation flow
    python -m study.run_all_experiments --ablation arch
    python -m study.run_all_experiments --ablation dataset
    python -m study.run_all_experiments --ablation augmentation

    # Dry run (print commands without executing)
    python -m study.run_all_experiments --all --dry-run

    # Resume from specific experiment
    python -m study.run_all_experiments --all --resume mlp-otcfm-1k-none

Environment:
    CUDA_VISIBLE_DEVICES: GPU to use (default: 1)
    WANDB_MODE: Set to 'offline' if no auth (default: offline)
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
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

# Default training settings
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-4
DEFAULT_GROUP = "ablation-study"

# Multiple seeds for statistical rigor (NeurIPS requirement: ≥3 seeds, recommended: 5-10)
DEFAULT_SEEDS = [42, 123, 456]
EXTENDED_SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 4242, 5678, 9999]  # 10 seeds for robust CI

# Experiment matrix
FLOW_METHODS = ["icfm", "otcfm", "si-gvp"]  # reflow handled separately
ARCHITECTURES = ["mlp", "dit", "unet"]  # mamba blocked
ARCHITECTURE_SCALES = ["small"]  # Default scale for ablations
DATASET_SIZES = ["1k", "5k", "10k"]
AUGMENTATIONS = ["none", "mixup", "noise", "mixup+noise"]

# Hyperparameter search space (for sensitivity analysis)
LEARNING_RATES = [1e-5, 1e-4, 1e-3]
BATCH_SIZES = [64, 256, 1024]


@dataclass
class Experiment:
    """Single experiment configuration."""
    name: str
    arch: str
    flow: str
    dataset: str
    aug: str
    scale: str = "small"
    seed: int = 42
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    extra_args: list = field(default_factory=list)

    @property
    def run_name(self) -> str:
        """Generate unique run name."""
        base = f"{self.arch}-{self.flow}-{self.dataset}-{self.aug}"
        # Include scale in name if not default
        if self.scale != "small":
            base = f"{self.arch}-{self.scale}-{self.flow}-{self.dataset}-{self.aug}"
        # Include lr/batch_size in name if not default (for hyperparam ablations)
        if self.lr != DEFAULT_LR:
            base = f"{base}-lr{self.lr:.0e}"
        if self.batch_size != DEFAULT_BATCH_SIZE:
            base = f"{base}-bs{self.batch_size}"
        # Include seed in name if not default (for multi-seed runs)
        if self.seed != 42:
            base = f"{base}-s{self.seed}"
        return base

    @property
    def checkpoint_dir(self) -> Path:
        """Checkpoint directory path."""
        return Path(f"study/checkpoints/{self.run_name}")

    def is_complete(self) -> bool:
        """Check if experiment already completed."""
        return (self.checkpoint_dir / "best.pt").exists()

    def get_command(self, group: str = DEFAULT_GROUP) -> list[str]:
        """Generate training command."""
        cmd = [
            "uv", "run", "python", "-m", "study.flow_matching.train",
            "--arch", self.arch,
            "--flow", self.flow,
            "--dataset", self.dataset,
            "--aug", self.aug,
            "--scale", self.scale,
            "--seed", str(self.seed),
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--lr", str(self.lr),
            "--group", group,
        ]
        cmd.extend(self.extra_args)
        return cmd


# =============================================================================
# Ablation Definitions
# =============================================================================

def get_flow_ablation() -> list[Experiment]:
    """Flow method comparison: I-CFM vs OT-CFM vs SI-GVP.

    Fixed: arch=mlp, scale=small, dataset=1k, aug=none
    Variable: flow method
    """
    return [
        Experiment(name=f"flow-{flow}", arch="mlp", flow=flow, dataset="1k", aug="none")
        for flow in FLOW_METHODS
    ]


def get_arch_ablation() -> list[Experiment]:
    """Architecture comparison: MLP vs DiT vs U-Net.

    Fixed: flow=icfm, dataset=1k, aug=none
    Variable: architecture
    """
    return [
        Experiment(name=f"arch-{arch}", arch=arch, flow="icfm", dataset="1k", aug="none")
        for arch in ARCHITECTURES
    ]


def get_dataset_ablation() -> list[Experiment]:
    """Dataset size scaling: 1K vs 5K vs 10K.

    Fixed: arch=mlp, flow=icfm, aug=none
    Variable: dataset size
    """
    return [
        Experiment(name=f"dataset-{ds}", arch="mlp", flow="icfm", dataset=ds, aug="none")
        for ds in DATASET_SIZES
    ]


def get_augmentation_ablation() -> list[Experiment]:
    """Augmentation effectiveness.

    Fixed: arch=mlp, flow=icfm, dataset=1k
    Variable: augmentation strategy
    """
    return [
        Experiment(name=f"aug-{aug}", arch="mlp", flow="icfm", dataset="1k", aug=aug)
        for aug in AUGMENTATIONS
    ]


def get_scale_ablation() -> list[Experiment]:
    """Architecture scaling: Tiny vs Small vs Base.

    Fixed: arch=mlp, flow=icfm, dataset=1k, aug=none
    Variable: model scale
    """
    return [
        Experiment(name=f"scale-{scale}", arch="mlp", flow="icfm", dataset="1k", aug="none", scale=scale)
        for scale in ["tiny", "small", "base"]
    ]


def get_hyperparam_ablation() -> list[Experiment]:
    """Hyperparameter sensitivity analysis: Learning rate × Batch size.

    Grid search over lr in [1e-5, 1e-4, 1e-3] and batch_size in [64, 256, 1024].
    Fixed: arch=mlp, flow=otcfm, dataset=5k, aug=none (best configuration from other ablations)

    Total: 9 experiments (3 lr × 3 batch_size)

    This ablation is critical for:
    1. Validating that default hyperparameters are near-optimal
    2. Understanding lr/batch_size interaction in high-D latent spaces
    3. NeurIPS reproducibility requirements
    """
    return [
        Experiment(
            name=f"hp-lr{lr:.0e}-bs{bs}",
            arch="mlp",
            flow="otcfm",
            dataset="5k",
            aug="none",
            lr=lr,
            batch_size=bs,
        )
        for lr, bs in product(LEARNING_RATES, BATCH_SIZES)
    ]


def get_hyperparam_lr_ablation() -> list[Experiment]:
    """Learning rate sensitivity only.

    Faster than full hyperparam grid. Fixed batch_size=256.
    Tests lr in [1e-5, 1e-4, 1e-3] across flow methods.
    """
    return [
        Experiment(
            name=f"lr-{flow}-{lr:.0e}",
            arch="mlp",
            flow=flow,
            dataset="5k",
            aug="none",
            lr=lr,
        )
        for flow, lr in product(FLOW_METHODS, LEARNING_RATES)
    ]


def deduplicate_experiments(experiments: list[Experiment]) -> list[Experiment]:
    """Remove duplicate experiments based on run_name."""
    seen: set[str] = set()
    unique: list[Experiment] = []
    for exp in experiments:
        if exp.run_name not in seen:
            seen.add(exp.run_name)
            unique.append(exp)
    return unique


def get_all_experiments() -> list[Experiment]:
    """Get all experiments for the full study."""
    all_experiments = [
        *get_flow_ablation(),
        *get_arch_ablation(),
        *get_dataset_ablation(),
        *get_augmentation_ablation(),
        *get_scale_ablation(),
    ]
    # Remove duplicates (e.g., mlp-icfm-1k-none appears in multiple ablations)
    return deduplicate_experiments(all_experiments)


def get_grid_search() -> list[Experiment]:
    """Full grid search: Flow × Arch × Dataset = 27 combinations.

    This provides comprehensive coverage of the main hyperparameter space.
    All experiments use aug=none and scale=small to isolate core factors.
    """
    return [
        Experiment(name=f"grid-{arch}-{flow}-{ds}", arch=arch, flow=flow, dataset=ds, aug="none")
        for flow, arch, ds in product(FLOW_METHODS, ARCHITECTURES, DATASET_SIZES)
    ]


def get_grid_with_aug() -> list[Experiment]:
    """Extended grid: Core grid (27) + Augmentation variants on 5k dataset (12).

    Total: 27 + 12 = 39 experiments (some overlap removed = ~36 unique)
    """
    # Core grid + augmentation variants on all archs with best flow (otcfm) and 5k dataset
    aug_variants = [
        Experiment(name=f"aug-{arch}-{aug}", arch=arch, flow="otcfm", dataset="5k", aug=aug)
        for arch, aug in product(ARCHITECTURES, ["mixup", "noise", "mixup+noise"])
    ]
    return deduplicate_experiments(get_grid_search() + aug_variants)


def get_full_grid() -> list[Experiment]:
    """Complete grid: Flow × Arch × Dataset × Aug = 108 combinations.

    WARNING: This is computationally expensive!
    """
    return [
        Experiment(name=f"full-{arch}-{flow}-{ds}-{aug}", arch=arch, flow=flow, dataset=ds, aug=aug)
        for flow, arch, ds, aug in product(FLOW_METHODS, ARCHITECTURES, DATASET_SIZES, AUGMENTATIONS)
    ]


def expand_with_seeds(experiments: list[Experiment], seeds: list[int]) -> list[Experiment]:
    """Expand a list of experiments to run with multiple seeds.

    Creates a new experiment for each seed in the list.

    Args:
        experiments: Base experiments to expand.
        seeds: List of random seeds (e.g., [42, 123, 456]).

    Returns:
        Expanded list with len(experiments) * len(seeds) entries.
    """
    return [
        replace(exp, seed=seed, extra_args=list(exp.extra_args))
        for exp, seed in product(experiments, seeds)
    ]


def get_top10_multiseed() -> list[Experiment]:
    """Top-10 configurations with 3 seeds each for statistical validation.

    Selected configurations for NeurIPS statistical rigor:
    - Best of each flow method (3): icfm, otcfm, si-gvp on mlp/5k
    - Best of each architecture (3): mlp, dit, unet on otcfm/5k
    - Best augmentation configs (4): all augs on mlp/otcfm/5k

    Total: ~10 unique × 3 seeds = 30 runs
    """
    base_experiments = _get_top10_base_experiments()
    return expand_with_seeds(base_experiments, DEFAULT_SEEDS)


def get_top10_extended_seeds() -> list[Experiment]:
    """Top-10 configurations with 10 seeds each for robust statistical validation.

    Same configurations as get_top10_multiseed() but with 10 seeds for:
    - Tighter confidence intervals
    - More robust significance testing
    - Better power for detecting small effect sizes
    - NeurIPS gold standard (recommended for main results)

    Total: 10 configs × 10 seeds = 100 runs
    """
    base_experiments = _get_top10_base_experiments()
    return expand_with_seeds(base_experiments, EXTENDED_SEEDS)


def _get_top10_base_experiments() -> list[Experiment]:
    """Base experiments for top-10 configurations (shared by multiseed variants)."""
    return [
        # Flow method comparison (best arch/dataset)
        Experiment(name="top-flow-icfm", arch="mlp", flow="icfm", dataset="5k", aug="none"),
        Experiment(name="top-flow-otcfm", arch="mlp", flow="otcfm", dataset="5k", aug="none"),
        Experiment(name="top-flow-sigvp", arch="mlp", flow="si-gvp", dataset="5k", aug="none"),
        # Architecture comparison (best flow)
        Experiment(name="top-arch-mlp", arch="mlp", flow="otcfm", dataset="5k", aug="none"),
        Experiment(name="top-arch-dit", arch="dit", flow="otcfm", dataset="5k", aug="none"),
        Experiment(name="top-arch-unet", arch="unet", flow="otcfm", dataset="5k", aug="none"),
        # Augmentation comparison (best setup)
        Experiment(name="top-aug-none", arch="mlp", flow="otcfm", dataset="5k", aug="none"),
        Experiment(name="top-aug-mixup", arch="mlp", flow="otcfm", dataset="5k", aug="mixup"),
        Experiment(name="top-aug-noise", arch="mlp", flow="otcfm", dataset="5k", aug="noise"),
        Experiment(name="top-aug-both", arch="mlp", flow="otcfm", dataset="5k", aug="mixup+noise"),
    ]


ABLATION_REGISTRY = {
    # Single-factor ablations
    "flow": get_flow_ablation,
    "arch": get_arch_ablation,
    "dataset": get_dataset_ablation,
    "augmentation": get_augmentation_ablation,
    "scale": get_scale_ablation,
    # Hyperparameter sensitivity (NeurIPS requirement)
    "hyperparam": get_hyperparam_ablation,        # lr × batch_size = 9 experiments
    "hyperparam-lr": get_hyperparam_lr_ablation,  # lr × flow = 9 experiments
    # Combined ablations
    "all": get_all_experiments,
    "grid": get_grid_search,                      # Flow × Arch × Dataset = 27
    "grid-aug": get_grid_with_aug,                # Grid + augmentation variants = ~36
    "full-grid": get_full_grid,                   # Flow × Arch × Dataset × Aug = 108
    # Statistical validation
    "multiseed": get_top10_multiseed,             # Top-10 × 3 seeds = 30 (minimum NeurIPS)
    "multiseed-10": get_top10_extended_seeds,     # Top-10 × 10 seeds = 100 (robust NeurIPS)
}


# =============================================================================
# Execution
# =============================================================================

def run_experiment(
    exp: Experiment,
    group: str,
    gpu: int,
    dry_run: bool = False,
    verbose: bool = True,
) -> tuple[bool, float]:
    """Run a single experiment.

    Returns:
        (success, duration_minutes)
    """
    if exp.is_complete():
        if verbose:
            print(f"  [SKIP] {exp.run_name} - already complete")
        return True, 0.0

    cmd = exp.get_command(group)
    env_prefix = f"CUDA_VISIBLE_DEVICES={gpu} WANDB_MODE=offline"
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
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "offline"},
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
    group: str,
    gpu: int,
    resume_from: Optional[str] = None,
    dry_run: bool = False,
    seeds: Optional[list[int]] = None,
) -> dict:
    """Run an ablation study.

    Args:
        ablation: Name of the ablation from ABLATION_REGISTRY.
        group: Wandb group name.
        gpu: GPU device to use.
        resume_from: Run name to resume from (skip experiments until this one).
        dry_run: If True, print commands without executing.
        seeds: If provided, expand each experiment to run with each seed.

    Returns:
        Results dictionary with experiment outcomes and summary.
    """
    if ablation not in ABLATION_REGISTRY:
        raise ValueError(f"Unknown ablation: {ablation}. "
                        f"Available: {list(ABLATION_REGISTRY.keys())}")

    experiments = ABLATION_REGISTRY[ablation]()

    # Expand with multiple seeds if requested
    if seeds:
        experiments = expand_with_seeds(experiments, seeds)
        print(f"\nExpanded experiments with seeds {seeds}: {len(experiments)} total")

    if resume_from:
        print(f"\nResuming from: {resume_from}")

    results: dict = {
        "ablation": ablation,
        "started": datetime.now().isoformat(),
        "experiments": [],
    }

    skipping = resume_from is not None
    total = len(experiments)
    completed = 0
    failed = 0

    print(f"\n{'#'*70}")
    print(f"  ABLATION: {ablation.upper()}")
    print(f"  Experiments: {total}")
    print(f"  GPU: {gpu}")
    print(f"  Group: {group}")
    print(f"{'#'*70}")

    for i, exp in enumerate(experiments, 1):
        # Handle resume - skip until we reach the resume point
        if skipping:
            if exp.run_name == resume_from:
                skipping = False
            else:
                print(f"  [{i}/{total}] SKIP (resume): {exp.run_name}")
                continue

        print(f"\n  [{i}/{total}] {exp.run_name}")

        success, duration = run_experiment(exp, group, gpu, dry_run)

        results["experiments"].append({
            "name": exp.run_name,
            "success": success,
            "duration_min": duration,
            "checkpoint": str(exp.checkpoint_dir / "best.pt") if success else None,
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


def print_summary(results: dict):
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
        description="Run Flow Matching Architecture Study experiments",
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
        default=1,
        help="GPU to use (default: 1)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=DEFAULT_GROUP,
        help=f"Wandb run group (default: {DEFAULT_GROUP})",
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="RUN_NAME",
        help="Resume from specific experiment (e.g., mlp-otcfm-1k-none)",
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
        type=str,
        default=None,
        metavar="SEEDS",
        help="Comma-separated seeds to run (e.g., '42,123,456'). Expands each experiment to run with each seed.",
    )
    parser.add_argument(
        "--multiseed",
        action="store_true",
        help="Shortcut for --seeds=42,123,456 (NeurIPS minimum)",
    )
    parser.add_argument(
        "--extended-seeds",
        action="store_true",
        help="Shortcut for 10 seeds (robust NeurIPS validation)",
    )

    args = parser.parse_args()

    # Parse seeds
    if args.extended_seeds:
        args.seeds = EXTENDED_SEEDS
    elif args.multiseed:
        args.seeds = DEFAULT_SEEDS
    elif args.seeds:
        args.seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        args.seeds = None

    # Determine what to run
    if args.list:
        print("\nAvailable ablations:")
        for name, fn in ABLATION_REGISTRY.items():
            experiments = fn()
            print(f"\n  {name} ({len(experiments)} experiments):")
            for exp in experiments:
                status = "✓" if exp.is_complete() else "○"
                print(f"    {status} {exp.run_name}")
        return

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
            group=args.group,
            gpu=args.gpu,
            resume_from=args.resume,
            dry_run=args.dry_run,
            seeds=args.seeds,
        )

        print_summary(results)

        # Save results
        if not args.dry_run:
            results_file = Path(f"study/results/ablation_{ablation}_{datetime.now():%Y%m%d_%H%M%S}.json")
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_file}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
