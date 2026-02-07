"""Run Spherical Ensemble BO on GuacaMol.

Multi-scale subspaces with max-std candidate selection.
Each member operates at a different dimensionality (default: 4,8,12,16,20,24).

Usage:
    # Default (6 members, dims=[4,8,12,16,20,24])
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_ensemble \
        --task-id adip --seed 42

    # Custom dimensions
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_ensemble \
        --member-dims 4 8 16 --task-id adip --seed 42

    # Preset
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_ensemble \
        --preset small --task-id adip --seed 42
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Spherical Ensemble BO on GuacaMol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--task-id", type=str, default="adip",
        help="GuacaMol task: adip, med2, pdop, etc.",
    )
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--preset", type=str, default=None,
        choices=["default", "small", "medium", "large", "conservative", "aggressive"],
        help="Use a preset configuration",
    )

    parser.add_argument(
        "--member-dims", type=int, nargs="+", default=None,
        help="Per-member subspace dimensions (e.g. 4 8 12 16 20 24)",
    )
    parser.add_argument(
        "--retirement-interval", type=int, default=100,
        help="Replace worst member every N iterations",
    )

    parser.add_argument(
        "--no-geodesic-tr", action="store_true",
        help="Disable geodesic trust region",
    )
    parser.add_argument(
        "--no-adaptive-tr", action="store_true",
        help="Disable adaptive trust region",
    )
    parser.add_argument(
        "--geodesic-max-angle", type=float, default=0.5,
        help="Max geodesic angle in radians",
    )
    parser.add_argument(
        "--geodesic-global-fraction", type=float, default=0.2,
        help="Fraction of global exploration samples",
    )
    parser.add_argument("--n-candidates", type=int, default=2000)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from rielbo.ensemble_bo import EnsembleConfig

    if args.preset:
        config = EnsembleConfig.from_preset(args.preset)
        config_name = f"ensemble_{args.preset}"
    else:
        config = EnsembleConfig(
            member_dims=args.member_dims,  # None → default [4,8,12,16,20,24]
            retirement_interval=args.retirement_interval,
            geodesic_tr=not args.no_geodesic_tr,
            adaptive_tr=not args.no_adaptive_tr,
            geodesic_max_angle=args.geodesic_max_angle,
            geodesic_global_fraction=args.geodesic_global_fraction,
            n_candidates=args.n_candidates,
        )
        dims_tag = "-".join(str(d) for d in config.member_dims)
        config_name = f"ensemble_d{dims_tag}"

    logger.info(f"Configuration: {config_name}")
    logger.info(f"Config details: {config}")

    logger.info("Loading codec and oracle...")
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    codec = SELFIESVAECodec.from_pretrained(device=args.device)
    input_dim = 256
    oracle = GuacaMolOracle(task_id=args.task_id)

    logger.info("Loading cold start data...")
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start,
        task_id=args.task_id,
    )

    from rielbo.ensemble_bo import SphericalEnsembleBO

    optimizer = SphericalEnsembleBO(
        codec=codec,
        oracle=oracle,
        input_dim=input_dim,
        config=config,
        device=args.device,
        seed=args.seed,
    )

    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    results_dir = "rielbo/results/guacamol_ensemble"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir,
        f"{config_name}_{args.task_id}_s{args.seed}_{timestamp}.json",
    )

    results = {
        "task_id": args.task_id,
        "config_name": config_name,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "n_retirements": optimizer.n_retirements,
        "history": {
            **optimizer.history,
            "member_stds": [
                [float(s) for s in stds]
                for stds in optimizer.history["member_stds"]
            ],
        },
        "args": vars(args),
        "config": {
            "member_dims": config.member_dims,
            "n_subspaces": config.n_subspaces,
            "retirement_interval": config.retirement_interval,
            "geodesic_tr": config.geodesic_tr,
            "adaptive_tr": config.adaptive_tr,
            "geodesic_max_angle": config.geodesic_max_angle,
        },
        "member_stats": [
            {
                "member_id": m.member_id,
                "subspace_dim": m.subspace_dim,
                "n_selected": m.n_selected,
                "n_improved": m.n_improved,
                "n_restarts": m.n_restarts,
                "last_std": m.last_std,
            }
            for m in optimizer.members
        ],
        "mean_norm": optimizer.mean_norm,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Final best score: {optimizer.best_score:.4f}")

    return optimizer.best_score


if __name__ == "__main__":
    main()
