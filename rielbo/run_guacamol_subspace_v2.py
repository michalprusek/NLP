"""Run Spherical Subspace BO v2 on GuacaMol with theoretical improvements.

Extends v1 with:
- ArcCosine Order 2 kernel (smoother GP)
- Spherical Whitening (center data at north pole)
- Geodesic Trust Region (proper Riemannian sampling)
- Adaptive Dimension (BAxUS-style d=8→16)
- Probabilistic Norm reconstruction
- Product Space geometry (S^3 × S^3 × S^3 × S^3)

Usage:
    # Baseline (matches v1)
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset baseline --task-id adip --seed 42

    # Full improvements
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset full --task-id adip --seed 42

    # Individual improvements
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --kernel-order 2 --whitening --geodesic-tr --task-id adip

Presets:
    baseline  - No improvements (matches v1)
    order2    - ArcCosine Order 2 kernel
    whitening - Spherical whitening only
    geodesic  - Geodesic trust region only
    adaptive  - Adaptive dimension (8→16)
    prob_norm - Probabilistic norm reconstruction
    product   - Product sphere geometry
    smooth    - Order 2 + whitening
    geometric - Geodesic TR + whitening
    full      - All improvements
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
        description="Spherical Subspace BO v2 on GuacaMol",
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

    # Preset (mutually exclusive with individual flags)
    parser.add_argument("--preset", type=str, default=None,
                        choices=["baseline", "order2", "whitening", "geodesic",
                                 "adaptive", "prob_norm", "product", "smooth",
                                 "geometric", "full"],
                        help="Use a preset configuration")

    # Individual improvements (can be combined)
    parser.add_argument("--kernel-order", type=int, default=0, choices=[0, 2],
                        help="ArcCosine kernel order (0=rough, 2=smooth)")
    parser.add_argument("--whitening", action="store_true",
                        help="Enable spherical whitening")
    parser.add_argument("--geodesic-tr", action="store_true",
                        help="Use geodesic trust region")
    parser.add_argument("--geodesic-max-angle", type=float, default=0.5,
                        help="Max geodesic angle in radians")
    parser.add_argument("--geodesic-global-fraction", type=float, default=0.2,
                        help="Fraction of global exploration samples")
    parser.add_argument("--adaptive-dim", action="store_true",
                        help="Enable adaptive dimension (8→16)")
    parser.add_argument("--adaptive-start-dim", type=int, default=8,
                        help="Starting dimension for adaptive")
    parser.add_argument("--adaptive-end-dim", type=int, default=16,
                        help="Ending dimension for adaptive")
    parser.add_argument("--adaptive-switch-frac", type=float, default=0.5,
                        help="Fraction of iterations before switching dim")
    parser.add_argument("--prob-norm", action="store_true",
                        help="Use probabilistic norm reconstruction")
    parser.add_argument("--norm-method", type=str, default="gaussian",
                        choices=["gaussian", "histogram", "gmm"],
                        help="Norm distribution method")
    parser.add_argument("--norm-n-candidates", type=int, default=5,
                        help="Norm candidates for probabilistic reconstruction")
    parser.add_argument("--product-space", action="store_true",
                        help="Use product sphere geometry")
    parser.add_argument("--n-spheres", type=int, default=4,
                        help="Number of spheres for product geometry")

    # BO config
    parser.add_argument("--subspace-dim", type=int, default=16,
                        help="Subspace dimension d (GP on S^(d-1))")
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--acqf", type=str, default="ts",
                        choices=["ts", "ei", "ucb"])
    parser.add_argument("--trust-region", type=float, default=0.8)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Build config
    from rielbo.subspace_bo_v2 import V2Config

    if args.preset:
        config = V2Config.from_preset(args.preset)
        config_name = args.preset
    else:
        config = V2Config(
            kernel_order=args.kernel_order,
            whitening=args.whitening,
            geodesic_tr=args.geodesic_tr,
            geodesic_max_angle=args.geodesic_max_angle,
            geodesic_global_fraction=args.geodesic_global_fraction,
            adaptive_dim=args.adaptive_dim,
            adaptive_start_dim=args.adaptive_start_dim,
            adaptive_end_dim=args.adaptive_end_dim,
            adaptive_switch_frac=args.adaptive_switch_frac,
            prob_norm=args.prob_norm,
            norm_method=args.norm_method,
            norm_n_candidates=args.norm_n_candidates,
            product_space=args.product_space,
            n_spheres=args.n_spheres,
        )
        # Build config name from enabled features
        features = []
        if config.kernel_order == 2:
            features.append("o2")
        if config.whitening:
            features.append("wh")
        if config.geodesic_tr:
            features.append("geo")
        if config.adaptive_dim:
            features.append("adp")
        if config.prob_norm:
            features.append("prn")
        if config.product_space:
            features.append("prd")
        config_name = "-".join(features) if features else "baseline"

    logger.info(f"Configuration: {config_name}")
    logger.info(f"Config details: {config}")

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
    from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2

    # Determine effective subspace dim (for adaptive, use end dim as base)
    effective_dim = args.adaptive_end_dim if config.adaptive_dim else args.subspace_dim

    optimizer = SphericalSubspaceBOv2(
        codec=codec,
        oracle=oracle,
        input_dim=input_dim,
        subspace_dim=effective_dim,
        config=config,
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

    # Save results
    results_dir = "rielbo/results/guacamol_v2"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir,
        f"v2_{config_name}_{args.task_id}_s{args.seed}_{timestamp}.json"
    )

    results = {
        "task_id": args.task_id,
        "config_name": config_name,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "history": optimizer.history,
        "args": vars(args),
        "config": {
            "kernel_order": config.kernel_order,
            "whitening": config.whitening,
            "geodesic_tr": config.geodesic_tr,
            "adaptive_dim": config.adaptive_dim,
            "prob_norm": config.prob_norm,
            "product_space": config.product_space,
        },
        "mean_norm": optimizer.mean_norm,
        "final_subspace_dim": optimizer._current_dim,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Final best score: {optimizer.best_score:.4f}")

    return optimizer.best_score


if __name__ == "__main__":
    main()
