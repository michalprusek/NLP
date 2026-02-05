"""Run Graph Laplacian BO on GuacaMol.

Uses spectral embedding via Graph Laplacian eigenvectors for manifold-aware BO.

Usage:
    # Standard run
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_graph \
        --task-id med2 --n-cold-start 100 --iterations 500

    # With different graph parameters
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_graph \
        --task-id adip --n-neighbors 20 --n-components 64 --iterations 500
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
    parser = argparse.ArgumentParser(description="Graph Laplacian BO on GuacaMol")

    parser.add_argument("--task-id", type=str, default="med2")
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)

    # Graph parameters
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="Number of neighbors for k-NN graph")
    parser.add_argument("--n-components", type=int, default=32,
                        help="Number of Laplacian eigenvectors (spectral dim)")
    parser.add_argument("--n-anchors", type=int, default=2000,
                        help="Number of anchor points for graph construction")
    parser.add_argument("--graph-sigma", type=str, default="auto",
                        help="Gaussian bandwidth for edge weights ('auto' or float)")

    # BO parameters
    parser.add_argument("--n-candidates", type=int, default=2000,
                        help="Number of candidates for acquisition")
    parser.add_argument("--acqf", type=str, default="ts",
                        choices=["ts", "ei"],
                        help="Acquisition: ts=Thompson, ei=ExpectedImprovement")
    parser.add_argument("--trust-region", type=float, default=0.5,
                        help="Initial trust region (expands when stuck)")
    parser.add_argument("--trust-region-max", type=float, default=1.5,
                        help="Maximum trust region")

    # Adaptive parameters
    parser.add_argument("--rebuild-interval", type=int, default=50,
                        help="Rebuild graph every N iterations")
    parser.add_argument("--stuck-threshold", type=int, default=20,
                        help="Expand trust region after N iterations without improvement")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load components
    logger.info("Loading codec and oracle...")
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle
    from shared.guacamol.codec import SELFIESVAECodec

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
    from rielbo.graph_laplacian_gp import GraphLaplacianBO

    # Parse graph_sigma (can be "auto" or a float)
    graph_sigma = args.graph_sigma
    if graph_sigma != "auto":
        try:
            graph_sigma = float(graph_sigma)
        except ValueError:
            logger.warning(f"Invalid graph_sigma '{graph_sigma}', using 'auto'")
            graph_sigma = "auto"

    logger.info(
        f"Graph Laplacian BO: k={args.n_neighbors}, "
        f"components={args.n_components}, anchors={args.n_anchors}, "
        f"sigma={graph_sigma}, acqf={args.acqf}"
    )
    optimizer = GraphLaplacianBO(
        codec=codec,
        oracle=oracle,
        input_dim=input_dim,
        device=args.device,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        n_anchors=args.n_anchors,
        graph_sigma=graph_sigma,
        n_candidates=args.n_candidates,
        acqf=args.acqf,
        trust_region=args.trust_region,
        trust_region_max=args.trust_region_max,
        rebuild_interval=args.rebuild_interval,
        stuck_threshold=args.stuck_threshold,
        seed=args.seed,
    )

    # Run
    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    # Save
    results_dir = "rielbo/results/guacamol"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir, f"graph_laplacian_{args.task_id}_{timestamp}.json"
    )

    results = {
        "task_id": args.task_id,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "method": "graph_laplacian",
        "history": optimizer.history,
        "args": vars(args),
        "mean_norm": optimizer.mean_norm,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
