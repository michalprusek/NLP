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
import subprocess
import sys
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

# Default training settings
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_GROUP = "ablation-study"

# Experiment matrix
FLOW_METHODS = ["icfm", "otcfm", "si-gvp"]  # reflow handled separately
ARCHITECTURES = ["mlp", "dit", "unet"]  # mamba blocked
ARCHITECTURE_SCALES = ["small"]  # Default scale for ablations
DATASET_SIZES = ["1k", "5k", "10k"]
AUGMENTATIONS = ["none", "mixup", "noise", "mixup+noise"]


@dataclass
class Experiment:
    """Single experiment configuration."""
    name: str
    arch: str
    flow: str
    dataset: str
    aug: str
    scale: str = "small"
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    extra_args: list = field(default_factory=list)

    @property
    def run_name(self) -> str:
        """Generate unique run name."""
        base = f"{self.arch}-{self.flow}-{self.dataset}-{self.aug}"
        # Include scale in name if not default
        if self.scale != "small":
            base = f"{self.arch}-{self.scale}-{self.flow}-{self.dataset}-{self.aug}"
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
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
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
    experiments = []
    for flow in FLOW_METHODS:
        experiments.append(Experiment(
            name=f"flow-{flow}",
            arch="mlp",
            flow=flow,
            dataset="1k",
            aug="none",
            scale="small",
        ))
    return experiments


def get_arch_ablation() -> list[Experiment]:
    """Architecture comparison: MLP vs DiT vs U-Net.

    Fixed: flow=icfm, dataset=1k, aug=none
    Variable: architecture
    """
    experiments = []
    for arch in ARCHITECTURES:
        experiments.append(Experiment(
            name=f"arch-{arch}",
            arch=arch,
            flow="icfm",
            dataset="1k",
            aug="none",
            scale="small",
        ))
    return experiments


def get_dataset_ablation() -> list[Experiment]:
    """Dataset size scaling: 1K vs 5K vs 10K.

    Fixed: arch=mlp, flow=icfm, aug=none
    Variable: dataset size
    """
    experiments = []
    for dataset in DATASET_SIZES:
        experiments.append(Experiment(
            name=f"dataset-{dataset}",
            arch="mlp",
            flow="icfm",
            dataset=dataset,
            aug="none",
            scale="small",
        ))
    return experiments


def get_augmentation_ablation() -> list[Experiment]:
    """Augmentation effectiveness.

    Fixed: arch=mlp, flow=icfm, dataset=1k
    Variable: augmentation strategy
    """
    experiments = []
    for aug in AUGMENTATIONS:
        experiments.append(Experiment(
            name=f"aug-{aug}",
            arch="mlp",
            flow="icfm",
            dataset="1k",
            aug=aug,
            scale="small",
        ))
    return experiments


def get_scale_ablation() -> list[Experiment]:
    """Architecture scaling: Tiny vs Small vs Base.

    Fixed: arch=mlp, flow=icfm, dataset=1k, aug=none
    Variable: model scale
    """
    experiments = []
    for scale in ["tiny", "small", "base"]:
        experiments.append(Experiment(
            name=f"scale-{scale}",
            arch="mlp",
            flow="icfm",
            dataset="1k",
            aug="none",
            scale=scale,
        ))
    return experiments


def get_all_experiments() -> list[Experiment]:
    """Get all experiments for the full study."""
    all_experiments = []
    all_experiments.extend(get_flow_ablation())
    all_experiments.extend(get_arch_ablation())
    all_experiments.extend(get_dataset_ablation())
    all_experiments.extend(get_augmentation_ablation())
    all_experiments.extend(get_scale_ablation())

    # Remove duplicates (e.g., mlp-icfm-1k-none appears in multiple ablations)
    seen = set()
    unique = []
    for exp in all_experiments:
        if exp.run_name not in seen:
            seen.add(exp.run_name)
            unique.append(exp)

    return unique


ABLATION_REGISTRY = {
    "flow": get_flow_ablation,
    "arch": get_arch_ablation,
    "dataset": get_dataset_ablation,
    "augmentation": get_augmentation_ablation,
    "scale": get_scale_ablation,
    "all": get_all_experiments,
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
        result = subprocess.run(
            cmd,
            env={
                **dict(__import__('os').environ),
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "WANDB_MODE": "offline",
            },
            capture_output=False,
            check=True,
        )
        duration = (time.time() - start) / 60
        return True, duration
    except subprocess.CalledProcessError as e:
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
) -> dict:
    """Run an ablation study.

    Returns:
        Results dictionary
    """
    if ablation not in ABLATION_REGISTRY:
        raise ValueError(f"Unknown ablation: {ablation}. "
                        f"Available: {list(ABLATION_REGISTRY.keys())}")

    experiments = ABLATION_REGISTRY[ablation]()

    # Resume logic
    skip_until = None
    if resume_from:
        skip_until = resume_from
        print(f"\nResuming from: {resume_from}")

    results = {
        "ablation": ablation,
        "started": datetime.now().isoformat(),
        "experiments": [],
    }

    skipping = skip_until is not None
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
        # Handle resume
        if skipping:
            if exp.run_name == skip_until:
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

    args = parser.parse_args()

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
