"""Run Spherical Subspace BO on GuacaMol.

Projects S^255 → S^15 for tractable GP using SELFIES VAE codec.

Usage:
    # Standard run
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
        --subspace-dim 16 --task-id pdop --n-cold-start 100 --iterations 500

    # With different acquisition function
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
        --subspace-dim 16 --acqf ei --task-id pdop --iterations 500
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
                        choices=["ts", "ei", "ucb"],
                        help="Acquisition: ts=Thompson, ei=ExpectedImprovement, ucb=UCB")
    parser.add_argument("--trust-region", type=float, default=0.8,
                        help="Trust region length (TuRBO-style)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel", type=str, default="arccosine",
                        choices=["arccosine", "matern", "hvarfner"],
                        help="Kernel type: arccosine (default), matern, hvarfner (RBF+LogNormal+ARD)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    from rielbo.subspace_bo import SphericalSubspaceBO
    logger.info(
        f"Fixed subspace: S^{input_dim-1} → S^{args.subspace_dim-1}, "
        f"kernel={args.kernel}, acqf={args.acqf}"
    )
    optimizer = SphericalSubspaceBO(
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
    )

    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    results_dir = "rielbo/results/guacamol"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dim_str = f"d{args.subspace_dim}"
    results_path = os.path.join(results_dir, f"subspace_{dim_str}_{args.task_id}_{timestamp}.json")

    results = {
        "task_id": args.task_id,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "subspace_dim": dim_str,
        "history": optimizer.history,
        "args": vars(args),
        "mean_norm": optimizer.mean_norm,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
