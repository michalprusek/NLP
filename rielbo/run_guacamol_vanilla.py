"""Run Vanilla BO (Hvarfner) on GuacaMol.

GP in full 256D with BoTorch's dimension-scaled LogNormal lengthscale priors.
Uses RBF + ARD + [0,1]^D normalization + TuRBO trust region.
Reference: Hvarfner et al. (ICML 2024)

Usage:
    # Standard run
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_vanilla \
        --task-id adip --n-cold-start 100 --iterations 500

    # With EI instead of TS
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_vanilla \
        --task-id adip --acqf ei --iterations 500
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
        description="Vanilla BO (Hvarfner) on GuacaMol"
    )

    parser.add_argument("--task-id", type=str, default="adip")
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--acqf", type=str, default="ts",
                        choices=["ts", "ei", "ucb"])
    parser.add_argument("--trust-region", type=float, default=0.8)
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info("Loading codec and oracle...")
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    codec = SELFIESVAECodec.from_pretrained(device=args.device)
    oracle = GuacaMolOracle(task_id=args.task_id)

    logger.info("Loading cold start data...")
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start,
        task_id=args.task_id,
    )

    from rielbo.vanilla_bo import VanillaBO

    optimizer = VanillaBO(
        codec=codec,
        oracle=oracle,
        input_dim=256,
        device=args.device,
        n_candidates=args.n_candidates,
        ucb_beta=args.ucb_beta,
        acqf=args.acqf,
        trust_region=args.trust_region,
        seed=args.seed,
    )

    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    results_dir = "rielbo/results/guacamol_vanilla"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir,
        f"vanilla_{args.task_id}_s{args.seed}_{timestamp}.json",
    )

    results = {
        "method": "vanilla_hvarfner",
        "task_id": args.task_id,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "history": optimizer.history,
        "gp_diagnostics": optimizer.diagnostic_history,
        "args": vars(args),
        "mean_norm": optimizer.mean_norm,
        "tr_restarts": optimizer._tr_restart_count,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Final best score: {optimizer.best_score}")


if __name__ == "__main__":
    main()
