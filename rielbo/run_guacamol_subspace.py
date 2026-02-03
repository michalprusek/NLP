"""Run Spherical Subspace BO on GuacaMol.

Projects S^255 → S^15 for tractable GP. Uses mean norm for magnitude.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
        --subspace-dim 16 --task-id pdop --n-cold-start 100 --iterations 500
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
    parser = argparse.ArgumentParser(description="Spherical Subspace BO on GuacaMol")

    parser.add_argument("--task-id", type=str, default="pdop")
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--subspace-dim", type=int, default=16,
                        help="Subspace dimension d (GP on S^(d-1))")
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--n-candidates", type=int, default=2000,
                        help="Number of Sobol candidates")
    parser.add_argument("--acqf", type=str, default="ts",
                        choices=["ts", "ei", "ucb", "qlogei"],
                        help="Acquisition: ts=Thompson, ei=ExpectedImprovement, ucb=UCB, qlogei=qLogEI")
    parser.add_argument("--trust-region", type=float, default=0.8,
                        help="Trust region length (TuRBO-style)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive subspace expansion")
    parser.add_argument("--initial-dim", type=int, default=4,
                        help="Initial subspace dimension for adaptive mode")
    parser.add_argument("--max-dim", type=int, default=64,
                        help="Max subspace dimension for adaptive mode")
    parser.add_argument("--points-per-dim", type=float, default=6.0,
                        help="Target points per dimension for data-driven expansion")
    parser.add_argument("--stall-expansion", action="store_true",
                        help="Use stall-based expansion (default: data-driven)")
    parser.add_argument("--no-improve-threshold", type=int, default=50,
                        help="Iterations without improvement before expanding (stall mode)")

    args = parser.parse_args()

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
    if args.adaptive:
        from rielbo.subspace_bo import AdaptiveSphericalSubspaceBO
        mode = "stall" if args.stall_expansion else "data-driven"
        logger.info(f"Adaptive ({mode}): S^{args.initial_dim-1} → S^{args.max_dim-1}, acqf={args.acqf}")
        optimizer = AdaptiveSphericalSubspaceBO(
            codec=codec,
            oracle=oracle,
            input_dim=256,
            initial_dim=args.initial_dim,
            max_dim=args.max_dim,
            points_per_dim=args.points_per_dim,
            no_improve_threshold=args.no_improve_threshold,
            use_data_driven=not args.stall_expansion,
            device=args.device,
            n_candidates=args.n_candidates,
            ucb_beta=args.ucb_beta,
            acqf=args.acqf,
            trust_region=args.trust_region,
            seed=args.seed,
        )
    else:
        from rielbo.subspace_bo import SphericalSubspaceBO
        logger.info(f"Fixed subspace: S^255 → S^{args.subspace_dim-1}, acqf={args.acqf}")
        optimizer = SphericalSubspaceBO(
            codec=codec,
            oracle=oracle,
            input_dim=256,
            subspace_dim=args.subspace_dim,
            device=args.device,
            n_candidates=args.n_candidates,
            ucb_beta=args.ucb_beta,
            acqf=args.acqf,
            trust_region=args.trust_region,
            seed=args.seed,
        )

    # Run
    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    # Save
    results_dir = "rielbo/results/guacamol"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dim_str = f"adaptive_{args.initial_dim}-{args.max_dim}" if args.adaptive else f"d{args.subspace_dim}"
    results_path = os.path.join(results_dir, f"subspace_{dim_str}_{args.task_id}_{timestamp}.json")

    results = {
        "task_id": args.task_id,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "subspace_dim": dim_str,
        "mean_norm": optimizer.mean_norm,
        "history": optimizer.history,
        "args": vars(args),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
