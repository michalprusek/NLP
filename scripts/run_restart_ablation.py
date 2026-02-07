"""Ablation: frequent restarts vs. baseline.

Tests whether more frequent subspace restarts improve optimization.
Baseline: tr_fail_tol=10, max_restarts=5 (~50 failures to restart)
Fast:     tr_fail_tol=5,  max_restarts=10 (~25 failures to restart)
VFast:    tr_fail_tol=3,  max_restarts=15 (~15 failures to restart)
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


CONFIGS = {
    "baseline": dict(tr_fail_tol=10, max_restarts=5),
    "fast": dict(tr_fail_tol=5, max_restarts=10),
    "vfast": dict(tr_fail_tol=3, max_restarts=15),
}


def run_one(config_name: str, seed: int, task_id: str, iterations: int, device: str):
    from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Geodesic preset + override TR params
    config = V2Config.from_preset("geodesic")
    params = CONFIGS[config_name]
    config.tr_fail_tol = params["tr_fail_tol"]
    config.max_restarts = params["max_restarts"]

    codec = SELFIESVAECodec.from_pretrained(device=device)
    oracle = GuacaMolOracle(task_id=task_id)
    smiles_list, scores, _ = load_guacamol_data(n_samples=100, task_id=task_id)

    optimizer = SphericalSubspaceBOv2(
        codec=codec,
        oracle=oracle,
        input_dim=256,
        subspace_dim=16,
        config=config,
        device=device,
        n_candidates=2000,
        acqf="ts",
        trust_region=0.8,
        seed=seed,
    )

    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=iterations, log_interval=10)

    results_dir = "rielbo/results/restart_ablation"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{config_name}_{task_id}_s{seed}_{timestamp}.json")

    results = {
        "task_id": task_id,
        "config_name": config_name,
        "seed": seed,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_restarts": optimizer.n_restarts,
        "tr_fail_tol": config.tr_fail_tol,
        "max_restarts": config.max_restarts,
        "history": optimizer.history,
    }

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"[{config_name} s{seed}] best={optimizer.best_score:.4f} restarts={optimizer.n_restarts}")
    return optimizer.best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-id", type=str, default="adip")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_one(args.config, args.seed, args.task_id, args.iterations, args.device)


if __name__ == "__main__":
    main()
