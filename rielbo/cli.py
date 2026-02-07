"""Unified CLI entry point for RieLBO optimization.

Replaces the 4 separate run scripts (run_guacamol_subspace.py,
run_guacamol_subspace_v2.py, run_guacamol_vanilla.py, run_guacamol_ensemble.py).

Usage:
    # Recommended: explore preset (SOTA)
    uv run python -m rielbo.cli --preset explore --task-id adip --seed 42

    # TuRBO baseline
    uv run python -m rielbo.cli --preset turbo --task-id adip --seed 42

    # Vanilla BO with Hvarfner priors
    uv run python -m rielbo.cli --preset vanilla --task-id adip --seed 42

    # Ensemble (multi-scale)
    uv run python -m rielbo.cli --ensemble --task-id adip --seed 42

    # Custom combination: geodesic + acqf schedule
    uv run python -m rielbo.cli --preset geodesic --acqf-schedule --task-id adip
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

from rielbo.core.config import OptimizerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RieLBO: Riemannian Latent-space Bayesian Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Task
    parser.add_argument(
        "--task-id", type=str, default="adip",
        help="GuacaMol task: adip, med2, pdop, etc.",
    )
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Preset
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=OptimizerConfig.available_presets(),
        help="Use a preset configuration",
    )

    # Ensemble
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use ensemble BO (multi-scale subspaces)",
    )
    parser.add_argument(
        "--ensemble-preset", type=str, default="default",
        choices=["default", "small", "medium", "large",
                 "conservative", "aggressive"],
        help="Ensemble configuration preset",
    )

    # BO parameters
    parser.add_argument("--subspace-dim", type=int, default=16)
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument(
        "--acqf", type=str, default="ts", choices=["ts", "ei", "ucb"],
    )
    parser.add_argument("--trust-region", type=float, default=0.8)

    # Kernel overrides
    parser.add_argument(
        "--kernel-type", type=str, default=None,
        choices=["arccosine", "geodesic_matern", "matern"],
    )
    parser.add_argument("--kernel-order", type=int, default=None, choices=[0, 2])
    parser.add_argument("--kernel-ard", action="store_true")

    # Projection overrides
    parser.add_argument("--whitening", action="store_true")
    parser.add_argument("--lass", action="store_true")
    parser.add_argument("--lass-n-candidates", type=int, default=None)
    parser.add_argument("--adaptive-dim", action="store_true")
    parser.add_argument("--adaptive-start-dim", type=int, default=8)
    parser.add_argument("--adaptive-end-dim", type=int, default=16)

    # Trust region overrides
    parser.add_argument("--geodesic-tr", action="store_true")
    parser.add_argument("--adaptive-tr", action="store_true")
    parser.add_argument("--geodesic-max-angle", type=float, default=None)
    parser.add_argument("--ur-tr", action="store_true")
    parser.add_argument("--no-ur-relative", action="store_true")
    parser.add_argument("--ur-std-high", type=float, default=None)
    parser.add_argument("--ur-std-low", type=float, default=None)
    parser.add_argument("--ur-std-collapse", type=float, default=None)
    parser.add_argument("--ur-collapse-patience", type=int, default=None)
    parser.add_argument("--ur-expand-factor", type=float, default=None)
    parser.add_argument("--ur-shrink-factor", type=float, default=None)

    # Acquisition overrides
    parser.add_argument("--acqf-schedule", action="store_true")
    parser.add_argument("--acqf-ucb-beta-high", type=float, default=None)
    parser.add_argument("--acqf-ucb-beta-low", type=float, default=None)

    # Norm overrides
    parser.add_argument("--prob-norm", action="store_true")
    parser.add_argument(
        "--norm-method", type=str, default=None,
        choices=["gaussian", "histogram", "gmm"],
    )

    return parser


def _apply_cli_overrides(config: OptimizerConfig, args) -> str:
    """Apply CLI flag overrides to config. Returns config_name."""
    if args.kernel_type is not None:
        config.kernel.kernel_type = args.kernel_type
    if args.kernel_order is not None:
        config.kernel.kernel_order = args.kernel_order
    if args.kernel_ard:
        config.kernel.kernel_ard = True

    if args.whitening:
        config.projection.whitening = True
    if args.lass:
        config.projection.lass = True
    if args.lass_n_candidates is not None:
        config.projection.lass_n_candidates = args.lass_n_candidates
    if args.adaptive_dim:
        config.projection.adaptive_dim = True
        config.projection.adaptive_start_dim = args.adaptive_start_dim
        config.projection.adaptive_end_dim = args.adaptive_end_dim

    if args.geodesic_tr:
        config.trust_region.geodesic = True
        config.candidate_gen.strategy = "geodesic"
    if args.adaptive_tr:
        config.trust_region.strategy = "adaptive"
    if args.geodesic_max_angle is not None:
        config.trust_region.geodesic_max_angle = args.geodesic_max_angle
    if args.ur_tr:
        config.trust_region.ur_tr = True
        config.trust_region.strategy = "ur"
    if args.no_ur_relative:
        config.trust_region.ur_relative = False
    if args.ur_std_high is not None:
        config.trust_region.ur_std_high = args.ur_std_high
    if args.ur_std_low is not None:
        config.trust_region.ur_std_low = args.ur_std_low
    if args.ur_std_collapse is not None:
        config.trust_region.ur_std_collapse = args.ur_std_collapse
    if args.ur_collapse_patience is not None:
        config.trust_region.ur_collapse_patience = args.ur_collapse_patience
    if args.ur_expand_factor is not None:
        config.trust_region.ur_expand_factor = args.ur_expand_factor
    if args.ur_shrink_factor is not None:
        config.trust_region.ur_shrink_factor = args.ur_shrink_factor

    if args.acqf_schedule:
        config.acquisition.schedule = True
    if args.acqf_ucb_beta_high is not None:
        config.acquisition.acqf_ucb_beta_high = args.acqf_ucb_beta_high
    if args.acqf_ucb_beta_low is not None:
        config.acquisition.acqf_ucb_beta_low = args.acqf_ucb_beta_low

    if args.prob_norm:
        config.norm_reconstruction.method = "probabilistic"
    if args.norm_method is not None:
        config.norm_reconstruction.prob_method = args.norm_method

    # Build config name
    config_name = args.preset or "custom"
    extras = []
    if args.kernel_type and args.kernel_type != "arccosine":
        extras.append(args.kernel_type)
    if args.kernel_ard:
        extras.append("ard")
    if extras:
        config_name += "_" + "_".join(extras)

    return config_name


def main():
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.ensemble:
        _run_ensemble(args)
        return

    if args.preset:
        config = OptimizerConfig.from_preset(args.preset)
    else:
        config = OptimizerConfig.from_preset("geodesic")

    config.seed = args.seed
    config.device = args.device
    config_name = _apply_cli_overrides(config, args)

    logger.info(f"Configuration: {config_name}")

    logger.info("Loading codec and oracle...")
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    codec = SELFIESVAECodec.from_pretrained(device=args.device)
    input_dim = 256
    oracle = GuacaMolOracle(task_id=args.task_id)

    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start, task_id=args.task_id,
    )

    from rielbo.core.optimizer import BaseOptimizer

    effective_dim = args.subspace_dim
    if config.projection.adaptive_dim:
        effective_dim = config.projection.adaptive_end_dim

    optimizer = BaseOptimizer(
        codec=codec,
        oracle=oracle,
        config=config,
        input_dim=input_dim,
        subspace_dim=effective_dim,
        n_candidates=args.n_candidates,
        acqf=args.acqf,
        ucb_beta=args.ucb_beta,
        trust_region=args.trust_region,
    )

    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    _save_results(optimizer, config_name, args)

    return optimizer.best_score


def _run_ensemble(args):
    from rielbo.ensemble_bo import EnsembleConfig, SphericalEnsembleBO
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    config = EnsembleConfig.from_preset(args.ensemble_preset)
    codec = SELFIESVAECodec.from_pretrained(device=args.device)
    oracle = GuacaMolOracle(task_id=args.task_id)
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start, task_id=args.task_id,
    )

    optimizer = SphericalEnsembleBO(
        codec=codec, oracle=oracle, config=config,
        device=args.device, seed=args.seed,
    )
    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    results_dir = "rielbo/results/ensemble"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir,
        f"ensemble_{args.ensemble_preset}_{args.task_id}_s{args.seed}_{timestamp}.json",
    )
    results = {
        "task_id": args.task_id,
        "config_name": f"ensemble_{args.ensemble_preset}",
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "history": optimizer.history,
        "args": vars(args),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


def _save_results(optimizer, config_name: str, args):
    """Save optimization results to JSON."""
    results_dir = "rielbo/results/guacamol_v2"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir,
        f"v2_{config_name}_{args.task_id}_s{args.seed}_{timestamp}.json",
    )

    results = {
        "task_id": args.task_id,
        "config_name": config_name,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": optimizer.data.n_observed,
        "history": optimizer.history.to_dict(),
        "args": vars(args),
        "mean_norm": optimizer.data.mean_norm,
        "final_subspace_dim": optimizer._current_dim,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Final best score: {optimizer.best_score:.4f}")


if __name__ == "__main__":
    main()
