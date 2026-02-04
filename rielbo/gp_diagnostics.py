"""GP Diagnostics for debugging BO optimization.

Provides tools to detect:
- Overfitting (train correlation ≈ 1.0, poor generalization)
- Kernel issues (lengthscales too short/long)
- Extrapolation (candidates far from training data)
- Uncertainty calibration (posterior variance quality)

Usage:
    diag = GPDiagnostics()
    metrics = diag.analyze(gp, train_X, train_Y, candidate_X)
    diag.log_summary(metrics)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from botorch.models import SingleTaskGP

logger = logging.getLogger(__name__)


@dataclass
class GPMetrics:
    """GP diagnostic metrics."""

    # Train fit quality
    train_rmse: float           # RMSE on training data
    train_correlation: float    # Pearson correlation (1.0 = perfect overfit)
    train_mae: float           # Mean absolute error

    # LOO cross-validation
    loo_rmse: Optional[float]   # Leave-one-out RMSE (None if too expensive)
    loo_correlation: Optional[float]

    # Kernel analysis
    lengthscale_mean: float     # Mean ARD lengthscale
    lengthscale_min: float      # Min lengthscale (short = overfitting)
    lengthscale_max: float      # Max lengthscale
    outputscale: float          # Signal variance
    noise: float                # Observation noise

    # Uncertainty quality
    train_std_mean: float       # Mean posterior std at training points
    train_std_ratio: float      # Ratio of mean std to Y range

    # Extrapolation detection
    candidate_distance_mean: float  # Mean distance to nearest train point
    candidate_distance_max: float   # Max distance (worst extrapolation)
    candidate_in_hull_frac: float   # Fraction within training convex hull approx

    # Data characteristics
    n_train: int
    n_dims: int
    y_range: float
    y_std: float


class GPDiagnostics:
    """Diagnostic tools for GP models in BO."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history = []

    def analyze(
        self,
        gp: SingleTaskGP,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        candidate_X: Optional[torch.Tensor] = None,
        do_loo: bool = False,
    ) -> GPMetrics:
        """Analyze GP model quality.

        Args:
            gp: Fitted GP model
            train_X: Training inputs [N, D]
            train_Y: Training targets [N] or [N, 1]
            candidate_X: Candidate points to check for extrapolation [M, D]
            do_loo: Whether to compute LOO (expensive for large N)

        Returns:
            GPMetrics with diagnostic information
        """
        device = train_X.device
        train_Y = train_Y.squeeze()
        n_train, n_dims = train_X.shape

        # === Train fit quality ===
        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(train_X)
            pred_mean = posterior.mean.squeeze()
            pred_std = posterior.variance.sqrt().squeeze()

        # RMSE and correlation
        residuals = pred_mean - train_Y
        train_rmse = residuals.pow(2).mean().sqrt().item()
        train_mae = residuals.abs().mean().item()

        # Pearson correlation
        y_centered = train_Y - train_Y.mean()
        pred_centered = pred_mean - pred_mean.mean()
        train_correlation = (
            (y_centered * pred_centered).sum() /
            (y_centered.norm() * pred_centered.norm() + 1e-8)
        ).item()

        # === LOO cross-validation ===
        loo_rmse = None
        loo_correlation = None
        if do_loo and n_train <= 200:  # Only for small datasets
            loo_rmse, loo_correlation = self._compute_loo(gp, train_X, train_Y)

        # === Kernel analysis ===
        lengthscales = self._get_lengthscales(gp)
        outputscale = self._get_outputscale(gp)
        noise = self._get_noise(gp)

        # === Uncertainty quality ===
        train_std_mean = pred_std.mean().item()
        y_range = (train_Y.max() - train_Y.min()).item()
        train_std_ratio = train_std_mean / (y_range + 1e-8)

        # === Extrapolation detection ===
        if candidate_X is not None and len(candidate_X) > 0:
            distances = self._compute_distances(candidate_X, train_X)
            candidate_distance_mean = distances.mean().item()
            candidate_distance_max = distances.max().item()

            # Approximate hull check: within range of training data
            in_hull = self._approx_in_hull(candidate_X, train_X)
            candidate_in_hull_frac = in_hull.float().mean().item()
        else:
            candidate_distance_mean = 0.0
            candidate_distance_max = 0.0
            candidate_in_hull_frac = 1.0

        metrics = GPMetrics(
            train_rmse=train_rmse,
            train_correlation=train_correlation,
            train_mae=train_mae,
            loo_rmse=loo_rmse,
            loo_correlation=loo_correlation,
            lengthscale_mean=lengthscales.mean().item() if lengthscales is not None else 0.0,
            lengthscale_min=lengthscales.min().item() if lengthscales is not None else 0.0,
            lengthscale_max=lengthscales.max().item() if lengthscales is not None else 0.0,
            outputscale=outputscale,
            noise=noise,
            train_std_mean=train_std_mean,
            train_std_ratio=train_std_ratio,
            candidate_distance_mean=candidate_distance_mean,
            candidate_distance_max=candidate_distance_max,
            candidate_in_hull_frac=candidate_in_hull_frac,
            n_train=n_train,
            n_dims=n_dims,
            y_range=y_range,
            y_std=train_Y.std().item(),
        )

        self.history.append(metrics)
        return metrics

    def _compute_loo(
        self, gp: SingleTaskGP, X: torch.Tensor, Y: torch.Tensor
    ) -> tuple[float, float]:
        """Compute leave-one-out cross-validation metrics."""
        n = len(X)
        loo_preds = []

        for i in range(n):
            # Leave out point i
            mask = torch.ones(n, dtype=torch.bool)
            mask[i] = False
            X_train = X[mask]
            Y_train = Y[mask]

            # Refit GP (simplified - uses same hyperparams)
            try:
                loo_gp = SingleTaskGP(
                    X_train.double(),
                    Y_train.double().unsqueeze(-1),
                    covar_module=gp.covar_module,
                )
                loo_gp.eval()

                with torch.no_grad():
                    pred = loo_gp.posterior(X[i:i+1]).mean.squeeze()
                loo_preds.append(pred.item())
            except Exception:
                loo_preds.append(Y.mean().item())

        loo_preds = torch.tensor(loo_preds, device=Y.device)

        # LOO RMSE
        loo_residuals = loo_preds - Y
        loo_rmse = loo_residuals.pow(2).mean().sqrt().item()

        # LOO correlation
        y_centered = Y - Y.mean()
        pred_centered = loo_preds - loo_preds.mean()
        loo_corr = (
            (y_centered * pred_centered).sum() /
            (y_centered.norm() * pred_centered.norm() + 1e-8)
        ).item()

        return loo_rmse, loo_corr

    def _get_lengthscales(self, gp: SingleTaskGP) -> Optional[torch.Tensor]:
        """Extract lengthscales from GP kernel."""
        try:
            # Navigate through ScaleKernel -> base_kernel
            kernel = gp.covar_module
            if hasattr(kernel, 'base_kernel'):
                kernel = kernel.base_kernel

            if hasattr(kernel, 'lengthscale'):
                return kernel.lengthscale.detach().squeeze()
        except Exception:
            pass
        return None

    def _get_outputscale(self, gp: SingleTaskGP) -> float:
        """Extract output scale from GP kernel."""
        try:
            if hasattr(gp.covar_module, 'outputscale'):
                return gp.covar_module.outputscale.detach().item()
        except Exception:
            pass
        return 1.0

    def _get_noise(self, gp: SingleTaskGP) -> float:
        """Extract noise level from GP likelihood."""
        try:
            return gp.likelihood.noise.detach().item()
        except Exception:
            return 0.0

    def _compute_distances(
        self, candidates: torch.Tensor, train: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance from each candidate to nearest training point."""
        # Pairwise distances [M, N]
        dists = torch.cdist(candidates, train)
        # Min distance to any training point
        min_dists, _ = dists.min(dim=1)
        return min_dists

    def _approx_in_hull(
        self, candidates: torch.Tensor, train: torch.Tensor
    ) -> torch.Tensor:
        """Approximate check if candidates are within training data range."""
        # Simple axis-aligned bounding box check
        train_min = train.min(dim=0).values
        train_max = train.max(dim=0).values

        in_range = (candidates >= train_min) & (candidates <= train_max)
        # Consider "in hull" if within range on most dimensions
        return in_range.float().mean(dim=1) > 0.8

    def log_summary(self, metrics: GPMetrics, prefix: str = "GP"):
        """Log diagnostic summary."""
        issues = []

        # Check for overfitting
        if metrics.train_correlation > 0.99:
            issues.append("OVERFIT: train_corr=1.0")

        # Check lengthscales
        if metrics.lengthscale_min < 0.01:
            issues.append(f"SHORT_LENGTHSCALE: {metrics.lengthscale_min:.4f}")

        # Check uncertainty
        if metrics.train_std_ratio < 0.01:
            issues.append("LOW_UNCERTAINTY: std→0")

        # Check extrapolation
        if metrics.candidate_in_hull_frac < 0.5:
            issues.append(f"EXTRAPOLATING: {1-metrics.candidate_in_hull_frac:.0%} outside hull")

        # Format log message
        status = "⚠️ " + ", ".join(issues) if issues else "✓"

        logger.info(
            f"{prefix} Diag: train_corr={metrics.train_correlation:.3f}, "
            f"ℓ=[{metrics.lengthscale_min:.3f},{metrics.lengthscale_max:.3f}], "
            f"noise={metrics.noise:.2e}, std_ratio={metrics.train_std_ratio:.3f} {status}"
        )

        if self.verbose and issues:
            logger.warning(f"{prefix} Issues detected: {issues}")

    def get_summary_dict(self, metrics: GPMetrics) -> dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "train_rmse": metrics.train_rmse,
            "train_correlation": metrics.train_correlation,
            "train_mae": metrics.train_mae,
            "loo_rmse": metrics.loo_rmse,
            "loo_correlation": metrics.loo_correlation,
            "lengthscale_mean": metrics.lengthscale_mean,
            "lengthscale_min": metrics.lengthscale_min,
            "lengthscale_max": metrics.lengthscale_max,
            "outputscale": metrics.outputscale,
            "noise": metrics.noise,
            "train_std_mean": metrics.train_std_mean,
            "train_std_ratio": metrics.train_std_ratio,
            "candidate_distance_mean": metrics.candidate_distance_mean,
            "candidate_distance_max": metrics.candidate_distance_max,
            "candidate_in_hull_frac": metrics.candidate_in_hull_frac,
            "n_train": metrics.n_train,
            "n_dims": metrics.n_dims,
            "y_range": metrics.y_range,
            "y_std": metrics.y_std,
        }


def diagnose_gp_step(
    gp: SingleTaskGP,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    candidate_X: Optional[torch.Tensor] = None,
    iteration: int = 0,
) -> dict:
    """Convenience function to diagnose a single GP step.

    Args:
        gp: Fitted GP model
        train_X: Training inputs
        train_Y: Training targets
        candidate_X: Candidate points (optional)
        iteration: Current iteration number

    Returns:
        Dictionary with diagnostic metrics
    """
    diag = GPDiagnostics(verbose=False)
    metrics = diag.analyze(gp, train_X, train_Y, candidate_X)
    diag.log_summary(metrics, prefix=f"[Iter {iteration}]")
    return diag.get_summary_dict(metrics)
