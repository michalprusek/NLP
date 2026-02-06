"""Run Spherical Subspace BO v4 on GuacaMol with geodesic novelty bonus.

Extends v3 with:
- Geodesic novelty bonus in acquisition function (reduces duplicates)
- Global novelty against ALL observed points

Usage:
    # Default (recommended settings)
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v4 \
        --task-id adip --n-cold-start 100 --iterations 500 --seed 42

    # Higher exploration
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v4 \
        --task-id adip --novelty-weight 0.3

    # Ablation: no novelty (equivalent to v3)
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v4 \
        --task-id adip --novelty-weight 0.0

    # Benchmark (5 seeds)
    for task in adip med2; do
      for seed in 42 43 44 45 46; do
        CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v4 \
            --task-id $task --n-cold-start 100 --iterations 500 --seed $seed
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
        description="Spherical Subspace BO v4 on GuacaMol (with geodesic novelty)",
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

    # V3 inherited
    parser.add_argument("--n-projections", type=int, default=3,
                        help="Number of QR projection matrices (round-robin)")
    parser.add_argument("--window-local", type=int, default=50,
                        help="Number of nearest neighbors in GP window")
    parser.add_argument("--window-random", type=int, default=30,
                        help="Number of random points in GP window")

    # V4 new
    parser.add_argument("--novelty-weight", type=float, default=0.1,
                        help="Weight for geodesic novelty bonus (0=no novelty, 0.1=recommended)")

    # BO config
    parser.add_argument("--subspace-dim", type=int, default=16,
                        help="Subspace dimension d (GP on S^(d-1))")
    parser.add_argument("--kernel", type=str, default="arccosine",
                        choices=["arccosine", "matern"])
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--acqf", type=str, default="ts",
                        choices=["ts", "ei", "ucb", "ets"])
    parser.add_argument("--trust-region", type=float, default=0.8)

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
    input_dim = 256
    logger.info("Using SELFIES VAE codec (embedding_dim=256)")

    oracle = GuacaMolOracle(task_id=args.task_id)

    # Load cold start
    logger.info("Loading cold start data...")
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start,
        task_id=args.task_id,
    )

    # Create optimizer
    from rielbo.subspace_bo_v4 import SphericalSubspaceBOv4

    optimizer = SphericalSubspaceBOv4(
        codec=codec,
        oracle=oracle,
        input_dim=input_dim,
        subspace_dim=args.subspace_dim,
        device=args.device,
        n_candidates=args.n_candidates,
        ucb_beta=args.ucb_beta,
        acqf=args.acqf,
        trust_region=args.trust_region,
        seed=args.seed,
        kernel=args.kernel,
        n_projections=args.n_projections,
        window_local=args.window_local,
        window_random=args.window_random,
        novelty_weight=args.novelty_weight,
    )

    # Run
    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    # Save results
    results_dir = "rielbo/results/guacamol_v4"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir,
        f"v4_{args.task_id}_s{args.seed}_nw{args.novelty_weight}_{timestamp}.json"
    )

    results = {
        "task_id": args.task_id,
        "method": "subspace_v4",
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "history": optimizer.history,
        "args": vars(args),
        "v4_config": {
            "n_projections": args.n_projections,
            "window_local": args.window_local,
            "window_random": args.window_random,
            "novelty_weight": args.novelty_weight,
        },
        "mean_norm": optimizer.mean_norm,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Final best score: {optimizer.best_score:.4f}")

    return optimizer.best_score


if __name__ == "__main__":
    main()
