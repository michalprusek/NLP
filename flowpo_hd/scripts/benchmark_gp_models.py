#!/usr/bin/env python3
"""Benchmark GP Models for 1024D SONAR Space.

Compares:
1. Original SAAS (baseline)
2. VanillaGP with dimension-scaled prior (Hvarfner 2024)
3. DKL-10D (Deep Kernel Learning)
4. ImprovedSAAS with minimum lengthscale constraint

Uses LOOCV on HbBoPs data (fidelity >= 600) to evaluate:
- RMSE (prediction accuracy)
- Spearman correlation (ranking quality)
- Calibration (uncertainty quality)
- Prediction error in σ (overconfidence detection)

Usage:
    uv run python flowpo_hd/scripts/benchmark_gp_models.py
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flowpo_hd.warm_start import load_warm_start

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class LOOCVMetrics:
    """Metrics from Leave-One-Out Cross-Validation."""
    rmse: float
    mae: float
    spearman: float
    spearman_pvalue: float
    coverage_90: float  # Fraction of true values in 90% CI
    mean_sigma_error: float  # Mean |pred - true| / std (should be ~1 if calibrated)
    overconfident_ratio: float  # Fraction of predictions > 2σ off
    training_time: float
    predictions: List[Tuple[float, float, float]]  # (pred_mean, pred_std, true)


def run_loocv(
    model_class,
    X: torch.Tensor,
    y: torch.Tensor,
    model_kwargs: Dict = None,
    device: str = "cuda",
) -> LOOCVMetrics:
    """Run Leave-One-Out Cross-Validation.

    Args:
        model_class: GP model class with fit(X, y) and predict(X) methods
        X: Input embeddings (N, D)
        y: Target values (N,)
        model_kwargs: Optional kwargs for model constructor
        device: Torch device

    Returns:
        LOOCVMetrics with all evaluation metrics
    """
    N = X.shape[0]
    model_kwargs = model_kwargs or {}

    predictions = []
    total_time = 0.0

    for i in range(N):
        # Leave one out
        mask = torch.ones(N, dtype=torch.bool)
        mask[i] = False

        X_train = X[mask]
        y_train = y[mask]
        X_test = X[i:i+1]
        y_test = y[i].item()

        # Create and fit model
        model = model_class(device=device, **model_kwargs)

        start = time.time()
        success = model.fit(X_train, y_train)
        fit_time = time.time() - start
        total_time += fit_time

        if not success:
            logger.warning(f"Fold {i+1}/{N}: fit failed")
            predictions.append((0.15, 0.05, y_test))  # Default prediction
            continue

        # Predict
        with torch.no_grad():
            pred_mean, pred_std = model.predict(X_test)
            pred_mean = pred_mean.item()
            pred_std = pred_std.item()

        predictions.append((pred_mean, pred_std, y_test))

        if (i + 1) % 5 == 0:
            logger.info(f"  Fold {i+1}/{N}: pred={pred_mean:.4f}±{pred_std:.4f}, true={y_test:.4f}")

    # Compute metrics
    pred_means = np.array([p[0] for p in predictions])
    pred_stds = np.array([p[1] for p in predictions])
    true_values = np.array([p[2] for p in predictions])

    # RMSE, MAE
    errors = pred_means - true_values
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    # Spearman correlation (ranking quality)
    spearman_result = stats.spearmanr(pred_means, true_values)
    spearman = spearman_result.correlation
    spearman_pvalue = spearman_result.pvalue

    # Coverage @ 90% (should be ~90% if well-calibrated)
    z_90 = 1.645  # 90% CI = ±1.645σ
    in_ci = np.abs(errors) <= z_90 * pred_stds
    coverage_90 = np.mean(in_ci)

    # Sigma error (|error| / std, should be ~1 for calibrated GP)
    sigma_errors = np.abs(errors) / pred_stds
    mean_sigma_error = np.mean(sigma_errors)

    # Overconfident ratio (predictions > 2σ off)
    overconfident_ratio = np.mean(sigma_errors > 2.0)

    return LOOCVMetrics(
        rmse=rmse,
        mae=mae,
        spearman=spearman,
        spearman_pvalue=spearman_pvalue,
        coverage_90=coverage_90,
        mean_sigma_error=mean_sigma_error,
        overconfident_ratio=overconfident_ratio,
        training_time=total_time / N,
        predictions=predictions,
    )


def print_metrics(name: str, metrics: LOOCVMetrics):
    """Pretty print metrics."""
    print(f"\n{'=' * 60}")
    print(f" {name}")
    print(f"{'=' * 60}")
    print(f"  RMSE:              {metrics.rmse:.4f}")
    print(f"  MAE:               {metrics.mae:.4f}")
    print(f"  Spearman ρ:        {metrics.spearman:.4f} (p={metrics.spearman_pvalue:.4f})")
    print(f"  Coverage@90%:      {metrics.coverage_90:.2%} (target: 90%)")
    print(f"  Mean σ-error:      {metrics.mean_sigma_error:.2f} (target: ~1.0)")
    print(f"  Overconfident:     {metrics.overconfident_ratio:.2%} (target: <5%)")
    print(f"  Avg fit time:      {metrics.training_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GP models on HbBoPs data")
    parser.add_argument(
        "--hbbops-path",
        default="lipo/data/hbbops_results_20260102.json",
        help="Path to HbBoPs results"
    )
    parser.add_argument(
        "--min-fidelity",
        type=int,
        default=600,
        help="Minimum fidelity for data points"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vanilla", "dkl10", "improved_saas"],
        choices=["vanilla", "dkl10", "improved_saas", "original_saas", "all"],
        help="Models to benchmark"
    )
    parser.add_argument(
        "--output",
        default="flowpo_hd/results/gp_benchmark_results.json",
        help="Output JSON path"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" GP Model Benchmark for 1024D SONAR Space")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {args.hbbops_path}...")
    data = load_warm_start(
        hbbops_path=args.hbbops_path,
        min_fidelity=args.min_fidelity,
        device=args.device,
    )

    X = data.X
    y = data.y

    print(f"  Loaded {len(data)} points with fidelity >= {args.min_fidelity}")
    print(f"  X shape: {X.shape}")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Beta prior: α={data.beta_alpha:.2f}, β={data.beta_beta:.2f}")

    # Determine which models to run
    models_to_run = args.models
    if "all" in models_to_run:
        models_to_run = ["vanilla", "dkl10", "improved_saas", "original_saas"]

    results = {}

    # Run benchmarks
    if "vanilla" in models_to_run:
        print("\n" + "=" * 60)
        print(" Running VanillaGP (Hvarfner 2024)...")
        print("=" * 60)

        from flowpo_hd.improved_gp import create_vanilla_gp

        metrics = run_loocv(
            model_class=lambda device, **kw: create_vanilla_gp(device=device, **kw),
            X=X, y=y,
            model_kwargs={},
            device=args.device,
        )
        print_metrics("VanillaGP (Dimension-Scaled Prior)", metrics)
        results["vanilla_gp"] = {
            "rmse": metrics.rmse,
            "spearman": metrics.spearman,
            "coverage_90": metrics.coverage_90,
            "mean_sigma_error": metrics.mean_sigma_error,
            "overconfident_ratio": metrics.overconfident_ratio,
            "training_time": metrics.training_time,
        }

    if "dkl10" in models_to_run:
        print("\n" + "=" * 60)
        print(" Running DKL-10D...")
        print("=" * 60)

        from flowpo_hd.improved_gp import create_dkl10

        metrics = run_loocv(
            model_class=lambda device, **kw: create_dkl10(device=device, **kw),
            X=X, y=y,
            model_kwargs={},
            device=args.device,
        )
        print_metrics("DKL-10D (Deep Kernel Learning)", metrics)
        results["dkl_10d"] = {
            "rmse": metrics.rmse,
            "spearman": metrics.spearman,
            "coverage_90": metrics.coverage_90,
            "mean_sigma_error": metrics.mean_sigma_error,
            "overconfident_ratio": metrics.overconfident_ratio,
            "training_time": metrics.training_time,
        }

    if "improved_saas" in models_to_run:
        print("\n" + "=" * 60)
        print(" Running ImprovedSAAS (min_lengthscale=1.0)...")
        print("=" * 60)

        from flowpo_hd.improved_gp import create_improved_saas

        metrics = run_loocv(
            model_class=lambda device, **kw: create_improved_saas(device=device, **kw),
            X=X, y=y,
            model_kwargs={"min_lengthscale": 1.0},
            device=args.device,
        )
        print_metrics("ImprovedSAAS (min_ls=1.0)", metrics)
        results["improved_saas"] = {
            "rmse": metrics.rmse,
            "spearman": metrics.spearman,
            "coverage_90": metrics.coverage_90,
            "mean_sigma_error": metrics.mean_sigma_error,
            "overconfident_ratio": metrics.overconfident_ratio,
            "training_time": metrics.training_time,
        }

    if "original_saas" in models_to_run:
        print("\n" + "=" * 60)
        print(" Running Original SAAS (baseline)...")
        print("=" * 60)

        from flowpo_hd.saas_gp import SaasGPWithAcquisition, SaasConfig

        def create_original_saas(device, **kw):
            config = SaasConfig(warmup_steps=128, num_samples=64)
            gp = SaasGPWithAcquisition(config=config, device=device, fit_on_cpu=True)
            # Wrap to match expected interface
            class Wrapper:
                def __init__(self, gp):
                    self._gp = gp
                    self.training_time = 0.0
                def fit(self, X, y):
                    success = self._gp.fit(X, y)
                    self.training_time = self._gp.training_time
                    return success
                def predict(self, X):
                    pred = self._gp.predict(X)
                    return pred.mean, pred.std
            return Wrapper(gp)

        metrics = run_loocv(
            model_class=create_original_saas,
            X=X, y=y,
            model_kwargs={},
            device=args.device,
        )
        print_metrics("Original SAAS (baseline)", metrics)
        results["original_saas"] = {
            "rmse": metrics.rmse,
            "spearman": metrics.spearman,
            "coverage_90": metrics.coverage_90,
            "mean_sigma_error": metrics.mean_sigma_error,
            "overconfident_ratio": metrics.overconfident_ratio,
            "training_time": metrics.training_time,
        }

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)

    print("\n{:<20} {:>8} {:>10} {:>12} {:>10}".format(
        "Model", "RMSE", "Spearman", "Coverage90", "σ-error"
    ))
    print("-" * 60)

    for name, res in results.items():
        coverage_str = f"{res['coverage_90']:.0%}"
        if res['coverage_90'] < 0.85:
            coverage_str += " ⚠️"
        elif res['coverage_90'] > 0.95:
            coverage_str += " ✓"

        sigma_str = f"{res['mean_sigma_error']:.2f}"
        if res['mean_sigma_error'] > 2.0:
            sigma_str += " ⚠️ OVERCONF"
        elif res['mean_sigma_error'] < 1.5:
            sigma_str += " ✓"

        print("{:<20} {:>8.4f} {:>10.4f} {:>12} {:>10}".format(
            name, res['rmse'], res['spearman'], coverage_str, sigma_str
        ))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Recommendations
    print("\n" + "=" * 60)
    print(" RECOMMENDATIONS")
    print("=" * 60)

    if results:
        best_spearman = max(results.items(), key=lambda x: x[1]['spearman'])
        best_rmse = min(results.items(), key=lambda x: x[1]['rmse'])
        best_calibration = min(results.items(), key=lambda x: abs(x[1]['coverage_90'] - 0.9))

        print(f"  Best Spearman (ranking):  {best_spearman[0]} ({best_spearman[1]['spearman']:.4f})")
        print(f"  Best RMSE (accuracy):     {best_rmse[0]} ({best_rmse[1]['rmse']:.4f})")
        print(f"  Best Calibration:         {best_calibration[0]} ({best_calibration[1]['coverage_90']:.0%})")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
