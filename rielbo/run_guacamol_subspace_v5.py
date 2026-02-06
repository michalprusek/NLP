"""Run Spherical Subspace BO v5 on GuacaMol.

v5 = Best-of-All-Worlds: geodesic TR + windowed GP + whitening + adaptive TR.

Usage:
    # Standard run
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v5 \
        --task-id adip --seed 42

    # With EI acquisition
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v5 \
        --task-id adip --acqf ei --seed 42

    # Custom TR parameters
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v5 \
        --task-id adip --tr-init 0.5 --tr-fail-tol 15 --seed 42

    # Benchmark loop
    for task in adip med2; do
      for seed in 42 43 44 45 46 47 48 49 50 51; do
        CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v5 \
          --task-id $task --seed $seed
      done
    done
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
        description="Spherical Subspace BO v5 on GuacaMol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Task config
    parser.add_argument("--task-id", type=str, default="adip",
                        help="GuacaMol task: adip, med2, pdop, etc.")
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # BO config
    parser.add_argument("--subspace-dim", type=int, default=16)
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--acqf", type=str, default="ts",
                        choices=["ts", "ei", "ucb"])
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--kernel", type=str, default="arccosine",
                        choices=["arccosine", "matern"])

    # Windowed GP
    parser.add_argument("--window-local", type=int, default=50)
    parser.add_argument("--window-random", type=int, default=30)

    # Geodesic trust region
    parser.add_argument("--geodesic-max-angle", type=float, default=0.5)
    parser.add_argument("--geodesic-global-fraction", type=float, default=0.2)

    # Adaptive trust region
    parser.add_argument("--tr-init", type=float, default=0.4)
    parser.add_argument("--tr-min", type=float, default=0.02)
    parser.add_argument("--tr-max", type=float, default=0.8)
    parser.add_argument("--tr-success-tol", type=int, default=3)
    parser.add_argument("--tr-fail-tol", type=int, default=10)
    parser.add_argument("--tr-grow-factor", type=float, default=1.5)
    parser.add_argument("--tr-shrink-factor", type=float, default=0.5)

    # Subspace restart
    parser.add_argument("--max-restarts", type=int, default=5)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load components
    logger.info("Loading codec and oracle...")
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    codec = SELFIESVAECodec.from_pretrained(device=args.device)
    oracle = GuacaMolOracle(task_id=args.task_id)

    # Load cold start
    logger.info("Loading cold start data...")
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start,
        task_id=args.task_id,
    )

    # Create optimizer
    from rielbo.subspace_bo_v5 import SphericalSubspaceBOv5

    optimizer = SphericalSubspaceBOv5(
        codec=codec,
        oracle=oracle,
        input_dim=256,
        subspace_dim=args.subspace_dim,
        device=args.device,
        n_candidates=args.n_candidates,
        ucb_beta=args.ucb_beta,
        acqf=args.acqf,
        seed=args.seed,
        kernel=args.kernel,
        window_local=args.window_local,
        window_random=args.window_random,
        geodesic_max_angle=args.geodesic_max_angle,
        geodesic_global_fraction=args.geodesic_global_fraction,
        tr_init=args.tr_init,
        tr_min=args.tr_min,
        tr_max=args.tr_max,
        tr_success_tol=args.tr_success_tol,
        tr_fail_tol=args.tr_fail_tol,
        tr_grow_factor=args.tr_grow_factor,
        tr_shrink_factor=args.tr_shrink_factor,
        max_restarts=args.max_restarts,
    )

    # Run
    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    # Save results
    results_dir = "rielbo/results/guacamol_v5"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir,
        f"v5_{args.task_id}_s{args.seed}_{timestamp}.json"
    )

    results = {
        "method": "subspace_v5",
        "task_id": args.task_id,
        "seed": args.seed,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "n_restarts": optimizer.n_restarts,
        "history": optimizer.history,
        "config": {
            "subspace_dim": args.subspace_dim,
            "n_candidates": args.n_candidates,
            "acqf": args.acqf,
            "kernel": args.kernel,
            "window_local": args.window_local,
            "window_random": args.window_random,
            "geodesic_max_angle": args.geodesic_max_angle,
            "geodesic_global_fraction": args.geodesic_global_fraction,
            "tr_init": args.tr_init,
            "tr_min": args.tr_min,
            "tr_max": args.tr_max,
            "tr_success_tol": args.tr_success_tol,
            "tr_fail_tol": args.tr_fail_tol,
            "max_restarts": args.max_restarts,
        },
        "mean_norm": optimizer.mean_norm,
        "args": vars(args),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Final best score: {optimizer.best_score:.4f}")

    return optimizer.best_score


if __name__ == "__main__":
    main()
