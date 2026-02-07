"""Run Spherical Subspace BO v2 on GuacaMol.

Usage:
    # Recommended: explore preset (SOTA)
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset explore --task-id adip --seed 42

    # Geodesic preset (baseline)
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset geodesic --task-id adip --seed 42

    # Kernel override
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset geodesic --kernel-type geodesic_matern --kernel-ard --task-id adip
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

    parser.add_argument("--task-id", type=str, default="adip",
                        help="GuacaMol task: adip, med2, pdop, etc.")
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--preset", type=str, default=None,
                        choices=["baseline", "order2", "whitening", "geodesic",
                                 "adaptive", "prob_norm", "product", "smooth",
                                 "geometric", "full",
                                 "ur_tr", "lass", "lass_ur", "explore", "portfolio"],
                        help="Use a preset configuration")

    parser.add_argument("--kernel-type", type=str, default="arccosine",
                        choices=["arccosine", "geodesic_matern", "matern"],
                        help="Kernel type")
    parser.add_argument("--kernel-order", type=int, default=0, choices=[0, 2],
                        help="ArcCosine kernel order (0=rough, 2=smooth); for geodesic_matern: 0→ν=0.5, 2→ν=2.5")
    parser.add_argument("--kernel-ard", action="store_true",
                        help="Per-dimension lengthscales (ARD) for geodesic_matern/matern")
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
    parser.add_argument("--adaptive-tr", action="store_true",
                        help="Enable adaptive trust region (TuRBO-style grow/shrink + restart)")

    parser.add_argument("--ur-tr", action="store_true",
                        help="Enable uncertainty-responsive trust region")
    parser.add_argument("--no-ur-relative", action="store_true",
                        help="Use absolute thresholds instead of noise-relative")
    parser.add_argument("--ur-std-high", type=float, default=0.15,
                        help="GP std threshold for TR shrink (relative to noise_std)")
    parser.add_argument("--ur-std-low", type=float, default=0.05,
                        help="GP std threshold for TR expand (relative to noise_std)")
    parser.add_argument("--ur-expand-factor", type=float, default=1.5,
                        help="TR expansion factor on low GP std")
    parser.add_argument("--ur-shrink-factor", type=float, default=0.8,
                        help="TR shrink factor on high GP std")
    parser.add_argument("--ur-std-collapse", type=float, default=0.005,
                        help="GP std threshold for subspace rotation")
    parser.add_argument("--ur-collapse-patience", type=int, default=15,
                        help="Consecutive low-std iters before rotation")

    parser.add_argument("--acqf-schedule", action="store_true",
                        help="Enable acquisition schedule (UCB when GP collapses)")
    parser.add_argument("--acqf-ucb-beta-high", type=float, default=4.0,
                        help="UCB beta when GP is collapsing")
    parser.add_argument("--acqf-ucb-beta-low", type=float, default=0.5,
                        help="UCB beta when GP is informative")

    parser.add_argument("--lass", action="store_true",
                        help="Enable look-ahead subspace selection")
    parser.add_argument("--lass-n-candidates", type=int, default=50,
                        help="Number of candidate projections to evaluate")

    parser.add_argument("--multi-subspace", action="store_true",
                        help="Enable multi-subspace portfolio (TuRBO-M style)")
    parser.add_argument("--n-subspaces", type=int, default=5,
                        help="Number of subspaces in portfolio")
    parser.add_argument("--subspace-ucb-beta", type=float, default=2.0,
                        help="UCB exploration parameter for subspace bandit")
    parser.add_argument("--subspace-stale-patience", type=int, default=50,
                        help="Replace subspace after this many evals without improvement")

    parser.add_argument("--codec", type=str, default="selfies_vae",
                        choices=["selfies_vae", "smi_ted"],
                        help="Molecular codec: selfies_vae (256D, spherical) or smi_ted (768D, linear)")
    parser.add_argument("--projection", type=str, default="random",
                        choices=["random", "pca", "pca_spherical"],
                        help="Projection type: random QR, PCA (Euclidean), or pca_spherical (PCA dirs + spherical pipeline)")
    parser.add_argument("--subspace-dim", type=int, default=16,
                        help="Subspace dimension d (GP on S^(d-1))")
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--acqf", type=str, default="ts",
                        choices=["ts", "ei", "ucb"])
    parser.add_argument("--trust-region", type=float, default=0.8)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from rielbo.subspace_bo_v2 import V2Config

    if args.preset:
        config = V2Config.from_preset(args.preset)

        # CLI overrides for preset values (only applied when non-default)
        if args.kernel_type != "arccosine":
            config.kernel_type = args.kernel_type
        if args.kernel_ard:
            config.kernel_ard = True
        if args.kernel_order != 0:
            config.kernel_order = args.kernel_order
        if args.ur_std_high != 0.15:
            config.ur_std_high = args.ur_std_high
        if args.ur_std_low != 0.05:
            config.ur_std_low = args.ur_std_low
        if args.no_ur_relative:
            config.ur_relative = False
        if args.ur_expand_factor != 1.5:
            config.ur_expand_factor = args.ur_expand_factor
        if args.ur_shrink_factor != 0.8:
            config.ur_shrink_factor = args.ur_shrink_factor
        if args.lass_n_candidates != 50:
            config.lass_n_candidates = args.lass_n_candidates
        if args.ur_std_collapse != 0.005:
            config.ur_std_collapse = args.ur_std_collapse
        if args.ur_collapse_patience != 15:
            config.ur_collapse_patience = args.ur_collapse_patience
        if args.acqf_ucb_beta_high != 4.0:
            config.acqf_ucb_beta_high = args.acqf_ucb_beta_high
        if args.acqf_ucb_beta_low != 0.5:
            config.acqf_ucb_beta_low = args.acqf_ucb_beta_low
        if args.ur_tr:
            config.ur_tr = True
        if args.lass:
            config.lass = True
        if args.acqf_schedule:
            config.acqf_schedule = True
        if args.adaptive_tr:
            config.adaptive_tr = True
        if args.multi_subspace:
            config.multi_subspace = True
        if args.n_subspaces != 5:
            config.n_subspaces = args.n_subspaces
        if args.subspace_ucb_beta != 2.0:
            config.subspace_ucb_beta = args.subspace_ucb_beta
        if args.subspace_stale_patience != 50:
            config.subspace_stale_patience = args.subspace_stale_patience
        if args.projection != "random":
            config.projection_type = args.projection
        config_name = args.preset
        if args.codec != "selfies_vae":
            config_name += f"_{args.codec}"
        if config.projection_type != "random":
            config_name += f"_{config.projection_type}"
        if config.kernel_type != "arccosine" or config.kernel_ard:
            config_name += f"_{config.kernel_type}"
            if config.kernel_ard:
                config_name += "_ard"
    else:
        config = V2Config(
            kernel_type=args.kernel_type,
            kernel_order=args.kernel_order,
            kernel_ard=args.kernel_ard,
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
            adaptive_tr=args.adaptive_tr,
            ur_tr=args.ur_tr,
            ur_relative=not args.no_ur_relative,
            ur_std_high=args.ur_std_high,
            ur_std_low=args.ur_std_low,
            ur_expand_factor=args.ur_expand_factor,
            ur_shrink_factor=args.ur_shrink_factor,
            ur_std_collapse=args.ur_std_collapse,
            ur_collapse_patience=args.ur_collapse_patience,
            acqf_schedule=args.acqf_schedule,
            acqf_ucb_beta_high=args.acqf_ucb_beta_high,
            acqf_ucb_beta_low=args.acqf_ucb_beta_low,
            lass=args.lass,
            lass_n_candidates=args.lass_n_candidates,
            multi_subspace=args.multi_subspace,
            n_subspaces=args.n_subspaces,
            subspace_ucb_beta=args.subspace_ucb_beta,
            subspace_stale_patience=args.subspace_stale_patience,
            projection_type=args.projection,
        )
        features = []
        if config.kernel_type != "arccosine":
            features.append(config.kernel_type)
        if config.kernel_ard:
            features.append("ard")
        if config.kernel_order == 2:
            features.append("o2")
        if config.whitening:
            features.append("wh")
        if config.geodesic_tr:
            features.append("geo")
        if config.adaptive_tr:
            features.append("atr")
        if config.adaptive_dim:
            features.append("adp")
        if config.prob_norm:
            features.append("prn")
        if config.product_space:
            features.append("prd")
        if config.ur_tr:
            features.append("ur")
        if config.lass:
            features.append("lass")
        if config.multi_subspace:
            features.append(f"port{config.n_subspaces}")
        config_name = "-".join(features) if features else "baseline"

    logger.info(f"Configuration: {config_name}")
    logger.info(f"Config details: {config}")

    logger.info("Loading codec and oracle...")
    from shared.guacamol.codec import create_molecular_codec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    codec = create_molecular_codec(codec_type=args.codec, device=args.device)
    input_dim = codec.embedding_dim
    oracle = GuacaMolOracle(task_id=args.task_id)

    # SMI-TED has linear geometry → force PCA projection if not already set
    if args.codec == "smi_ted" and args.projection == "random":
        logger.info("SMI-TED codec detected: auto-switching projection to 'pca'")
        config.projection_type = "pca"

    logger.info("Loading cold start data...")
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start,
        task_id=args.task_id,
    )

    from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2

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

    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

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
            "kernel_type": config.kernel_type,
            "kernel_order": config.kernel_order,
            "kernel_ard": config.kernel_ard,
            "whitening": config.whitening,
            "geodesic_tr": config.geodesic_tr,
            "adaptive_tr": config.adaptive_tr,
            "adaptive_dim": config.adaptive_dim,
            "prob_norm": config.prob_norm,
            "product_space": config.product_space,
            "ur_tr": config.ur_tr,
            "ur_relative": config.ur_relative,
            "lass": config.lass,
            "acqf_schedule": config.acqf_schedule,
            "multi_subspace": config.multi_subspace,
            "n_subspaces": config.n_subspaces,
            "projection_type": config.projection_type,
            "codec": args.codec,
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
