#!/usr/bin/env python3
"""CLI entry point for GP ablation experiments.

Usage:
    # Run a single experiment
    CUDA_VISIBLE_DEVICES=0 uv run python -m study.gp_ablation.train \
        --method standard_msr --seed 42

    # With specific configuration
    CUDA_VISIBLE_DEVICES=0 uv run python -m study.gp_ablation.train \
        --method turbo --kernel matern52 --acquisition log_ei \
        --length-init 0.8 --seed 42

    # Latent Space BO with flow model
    CUDA_VISIBLE_DEVICES=0 uv run python -m study.gp_ablation.train \
        --method latent_bo \
        --flow-checkpoint study/checkpoints/mlp-otcfm-5k-none/best.pt \
        --seed 42
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates import create_surrogate, list_methods
from study.gp_ablation.evaluation.metrics import compute_all_metrics
from study.gp_ablation.evaluation.regret import compute_all_regret_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def verify_gpu() -> torch.device:
    """Verify GPU and return device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Training requires GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")

    logger.info(f"GPU Device: {gpu_name}")
    logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")

    return torch.device("cuda")


def load_data(data_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load evaluated instructions with embeddings.

    Args:
        data_path: Path to .pt file with embeddings and scores.
        device: Device to move data to.

    Returns:
        Dict with 'embeddings' [N, D] and 'scores' [N].
    """
    logger.info(f"Loading data from {data_path}")

    data = torch.load(data_path, map_location=device, weights_only=False)

    if isinstance(data, dict):
        embeddings = data.get("embeddings", data.get("X"))
        scores = data.get("accuracies", data.get("scores", data.get("Y")))
    else:
        raise ValueError(f"Unknown data format in {data_path}")

    if embeddings is None or scores is None:
        raise ValueError(f"Data must contain 'embeddings' and 'scores'. Found keys: {data.keys()}")

    embeddings = embeddings.to(device).float()
    scores = scores.to(device).float()

    if scores.dim() > 1:
        scores = scores.squeeze()

    logger.info(f"Loaded {len(embeddings)} samples, dim={embeddings.shape[-1]}")
    logger.info(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    return {"embeddings": embeddings, "scores": scores}


def run_leave_one_out_evaluation(
    surrogate,
    X: torch.Tensor,
    Y: torch.Tensor,
) -> Dict:
    """Run leave-one-out cross-validation evaluation.

    This is the primary evaluation mode: fit GP on N-1 points,
    predict held-out point, measure quality.

    Args:
        surrogate: GP surrogate with fit() and predict().
        X: All embeddings [N, D].
        Y: All scores [N].

    Returns:
        Dict with LOOCV metrics.
    """
    N = X.shape[0]
    predictions = []
    stds = []

    logger.info(f"Running {N}-fold leave-one-out cross-validation...")

    for i in range(N):
        if (i + 1) % 10 == 0:
            logger.info(f"  LOOCV fold {i + 1}/{N}")

        # Leave out point i
        mask = torch.ones(N, dtype=torch.bool, device=X.device)
        mask[i] = False

        X_train = X[mask]
        Y_train = Y[mask]

        # Fit on N-1 points
        surrogate.fit(X_train, Y_train)

        # Predict held-out point
        with torch.no_grad():
            mean, std = surrogate.predict(X[i:i+1])
            predictions.append(mean.squeeze().cpu())
            stds.append(std.squeeze().cpu())

    predictions = torch.stack(predictions)
    stds = torch.stack(stds)
    Y_cpu = Y.cpu()

    # Compute metrics
    metrics = compute_all_metrics(predictions, stds, Y_cpu)

    logger.info(f"LOOCV RMSE: {metrics['rmse']:.4f}")
    logger.info(f"LOOCV NLPD: {metrics['nlpd']:.4f}")
    logger.info(f"LOOCV Spearman: {metrics['spearman']:.4f}")
    logger.info(f"LOOCV ECE: {metrics['ece']:.4f}")

    return {f"loocv_{k}": v for k, v in metrics.items()}


def run_bo_simulation(
    surrogate,
    X: torch.Tensor,
    Y: torch.Tensor,
    n_initial: int = 10,
    n_iterations: int = 50,
) -> Dict:
    """Simulate BO loop on the evaluation set.

    Since we can't actually evaluate new prompts, we simulate BO
    by treating the dataset as a discrete candidate pool.

    Args:
        surrogate: GP surrogate.
        X: All embeddings [N, D].
        Y: All scores [N].
        n_initial: Number of random initial points.
        n_iterations: Number of BO iterations.

    Returns:
        Dict with regret metrics.
    """
    N = X.shape[0]

    # Track which points have been "evaluated"
    evaluated = torch.zeros(N, dtype=torch.bool, device=X.device)
    trajectory = []

    # Random initialization
    init_indices = torch.randperm(N)[:n_initial]
    for idx in init_indices:
        evaluated[idx] = True
        trajectory.append(Y[idx].item())

    # Initial fit
    X_train = X[init_indices]
    Y_train = Y[init_indices]
    surrogate.fit(X_train, Y_train)

    logger.info(f"BO simulation: {n_initial} initial + {n_iterations} iterations")

    # BO loop
    for t in range(n_iterations):
        # Get candidates (unevaluated points)
        candidates = X[~evaluated]
        candidate_indices = torch.where(~evaluated)[0]

        if len(candidates) == 0:
            logger.warning(f"All points evaluated at iteration {t}")
            break

        # Compute acquisition values
        best_f = max(trajectory)
        acq_values = surrogate.compute_acquisition(
            candidates,
            acquisition="log_ei",
            best_f=best_f,
        )

        # Select best candidate
        best_idx = acq_values.argmax()
        global_idx = candidate_indices[best_idx].item()

        # "Evaluate" the candidate
        evaluated[global_idx] = True
        y_new = Y[global_idx].item()
        trajectory.append(y_new)

        # Update surrogate
        surrogate.update(X[global_idx:global_idx+1], Y[global_idx:global_idx+1])

        if (t + 1) % 10 == 0:
            logger.info(f"  Iteration {t + 1}/{n_iterations}, best so far: {max(trajectory):.4f}")

    # Compute regret metrics
    Y_optimal = Y.max().item()
    regret_metrics = compute_all_regret_metrics(
        torch.tensor(trajectory),
        Y_optimal=Y_optimal,
    )

    logger.info(f"Final best: {regret_metrics['final_best']:.4f}")
    logger.info(f"Simple regret: {regret_metrics['final_simple_regret']:.4f}")
    logger.info(f"Steps to 95%: {regret_metrics.get('steps_to_95', 'N/A')}")

    return {f"bo_{k}": v for k, v in regret_metrics.items()}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GP ablation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list_methods(),
        help="GP method to use",
    )

    # Core settings
    parser.add_argument("--kernel", type=str, default="matern52")
    parser.add_argument("--acquisition", type=str, default="log_ei")
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation settings
    parser.add_argument("--n-initial", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=50)

    # Method-specific
    parser.add_argument("--target-dim", type=int, default=128)
    parser.add_argument("--length-init", type=float, default=0.8)
    parser.add_argument("--nuts-samples", type=int, default=128)
    parser.add_argument("--n-directions", type=int, default=1)
    parser.add_argument("--n-grad-steps", type=int, default=5)
    parser.add_argument("--grad-lr", type=float, default=0.01)
    parser.add_argument("--flow-checkpoint", type=str, default=None)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--alignment-weight", type=float, default=0.3)

    # Paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="study/results/gp_ablation",
    )
    parser.add_argument("--wandb-group", type=str, default="gp-ablation")

    # Evaluation modes
    parser.add_argument(
        "--skip-bo",
        action="store_true",
        help="Skip BO simulation (faster, LOOCV only)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    start_time = time.time()

    try:
        args = parse_args()

        # Set seed
        set_seed(args.seed)
        logger.info(f"Random seed: {args.seed}")

        # Verify GPU
        device = verify_gpu()

        # Create config
        config = GPConfig(
            method=args.method,
            kernel=args.kernel,
            acquisition=args.acquisition,
            seed=args.seed,
            target_dim=args.target_dim,
            length_init=args.length_init,
            nuts_samples=args.nuts_samples,
            n_directions=args.n_directions,
            n_grad_steps=args.n_grad_steps,
            grad_lr=args.grad_lr,
            flow_checkpoint=args.flow_checkpoint,
            prior_weight=args.prior_weight,
            alignment_weight=args.alignment_weight,
            n_initial=args.n_initial,
            n_iterations=args.n_iterations,
            data_path=args.data_path,
            results_dir=args.results_dir,
            wandb_group=args.wandb_group,
        )

        logger.info(f"Run name: {config.run_name}")
        logger.info(f"Config: {config.to_dict()}")

        # Load data
        data = load_data(args.data_path, device)
        X = data["embeddings"]
        Y = data["scores"]

        # Create surrogate
        logger.info(f"Creating {args.method} surrogate...")
        surrogate = create_surrogate(config, device)

        # Run evaluations
        results = {
            "config": config.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

        # LOOCV evaluation
        loocv_results = run_leave_one_out_evaluation(surrogate, X, Y)
        results.update(loocv_results)

        # BO simulation (optional)
        if not args.skip_bo:
            # Re-create surrogate for fresh state
            surrogate = create_surrogate(config, device)
            bo_results = run_bo_simulation(
                surrogate, X, Y,
                n_initial=args.n_initial,
                n_iterations=args.n_iterations,
            )
            results.update(bo_results)

        # Save results
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"{config.run_name}.json"

        with open(results_path, "w") as f:
            # Convert numpy/torch types to Python types
            def convert(obj):
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj

            json.dump(convert(results), f, indent=2)

        logger.info(f"Results saved to: {results_path}")

        # Print summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Method: {args.method}")
        print(f"Run name: {config.run_name}")
        print(f"Time: {elapsed:.1f}s")
        print("-" * 60)
        print("LOOCV Metrics:")
        print(f"  RMSE: {results['loocv_rmse']:.4f}")
        print(f"  NLPD: {results['loocv_nlpd']:.4f}")
        print(f"  Spearman: {results['loocv_spearman']:.4f}")
        print(f"  ECE: {results['loocv_ece']:.4f}")
        if not args.skip_bo:
            print("BO Metrics:")
            print(f"  Final best: {results['bo_final_best']:.4f}")
            print(f"  Simple regret: {results['bo_final_simple_regret']:.4f}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
