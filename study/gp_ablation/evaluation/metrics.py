"""GP surrogate quality metrics.

Metrics for evaluating how well the GP models the objective function:
- NLPD: Negative Log Predictive Density (calibration)
- RMSE: Root Mean Squared Error (accuracy)
- Spearman/Kendall: Rank correlation (ranking quality)
- R²: Coefficient of determination
- Calibration: Coverage at different confidence levels
"""

import math
from typing import Dict, Tuple, Optional

import torch
from scipy import stats


def compute_nlpd(
    mean: torch.Tensor,
    std: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Compute Negative Log Predictive Density (NLPD).

    Lower is better. Measures calibration of uncertainty estimates.

    NLPD = 0.5 * (log(2*pi*sigma^2) + (y - mu)^2 / sigma^2)

    Args:
        mean: Predicted mean [N].
        std: Predicted std [N].
        y_true: True values [N].

    Returns:
        NLPD values [N].
    """
    mean = mean.squeeze()
    std = std.squeeze()
    y_true = y_true.squeeze()

    var = std**2 + 1e-6
    nlpd = 0.5 * (torch.log(2 * math.pi * var) + (y_true - mean) ** 2 / var)

    return nlpd


def compute_rmse(
    mean: torch.Tensor,
    y_true: torch.Tensor,
) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Lower is better.

    Args:
        mean: Predicted mean [N].
        y_true: True values [N].

    Returns:
        RMSE value.
    """
    mean = mean.squeeze()
    y_true = y_true.squeeze()

    mse = ((mean - y_true) ** 2).mean()
    return torch.sqrt(mse).item()


def compute_spearman(
    mean: torch.Tensor,
    y_true: torch.Tensor,
) -> Tuple[float, float]:
    """Compute Spearman rank correlation.

    Higher is better. Measures ranking quality.

    Args:
        mean: Predicted mean [N].
        y_true: True values [N].

    Returns:
        Tuple of (correlation, p-value).
    """
    mean = mean.squeeze().cpu().numpy()
    y_true = y_true.squeeze().cpu().numpy()

    corr, pval = stats.spearmanr(mean, y_true)
    return float(corr), float(pval)


def compute_kendall(
    mean: torch.Tensor,
    y_true: torch.Tensor,
) -> Tuple[float, float]:
    """Compute Kendall's tau rank correlation.

    Higher is better. More robust than Spearman for small samples.

    Args:
        mean: Predicted mean [N].
        y_true: True values [N].

    Returns:
        Tuple of (tau, p-value).
    """
    mean = mean.squeeze().cpu().numpy()
    y_true = y_true.squeeze().cpu().numpy()

    tau, pval = stats.kendalltau(mean, y_true)
    return float(tau), float(pval)


def compute_r2(
    mean: torch.Tensor,
    y_true: torch.Tensor,
) -> float:
    """Compute coefficient of determination (R²).

    Higher is better. Fraction of variance explained.

    Args:
        mean: Predicted mean [N].
        y_true: True values [N].

    Returns:
        R² value.
    """
    mean = mean.squeeze()
    y_true = y_true.squeeze()

    ss_res = ((y_true - mean) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    if ss_tot < 1e-8:
        return 0.0

    return (1 - ss_res / ss_tot).item()


def compute_calibration_metrics(
    mean: torch.Tensor,
    std: torch.Tensor,
    y_true: torch.Tensor,
    confidence_levels: Optional[list] = None,
) -> Dict:
    """Compute calibration metrics.

    Measures if predicted confidence intervals are calibrated.

    Args:
        mean: Predicted mean [N].
        std: Predicted std [N].
        y_true: True values [N].
        confidence_levels: Confidence levels to check (default [0.5, 0.8, 0.9, 0.95, 0.99]).

    Returns:
        Dict with:
        - coverages: Dict of {confidence: actual_coverage}
        - ece: Expected Calibration Error
        - sharpness: Average prediction interval width (at 95%)
        - mace: Mean Absolute Calibration Error
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]

    mean = mean.squeeze()
    std = std.squeeze()
    y_true = y_true.squeeze()

    # Compute coverage at each confidence level
    coverages = {}
    calibration_errors = []

    for conf in confidence_levels:
        # z-score for this confidence level
        z = stats.norm.ppf((1 + conf) / 2)

        # Check if y_true is within [mean - z*std, mean + z*std]
        lower = mean - z * std
        upper = mean + z * std
        in_ci = (y_true >= lower) & (y_true <= upper)

        coverage = in_ci.float().mean().item()
        coverages[conf] = coverage

        calibration_errors.append(abs(coverage - conf))

    # Expected Calibration Error (average miscalibration)
    ece = sum(calibration_errors) / len(calibration_errors)

    # Mean Absolute Calibration Error (same as ECE for uniform bins)
    mace = ece

    # Sharpness: average 95% prediction interval width
    z_95 = stats.norm.ppf(0.975)
    sharpness = (2 * z_95 * std).mean().item()

    return {
        "coverages": coverages,
        "ece": ece,
        "mace": mace,
        "sharpness": sharpness,
    }


def compute_loocv_metrics(
    surrogate,
    X: torch.Tensor,
    Y: torch.Tensor,
) -> Dict:
    """Compute Leave-One-Out Cross-Validation metrics.

    Fits GP on N-1 points, predicts held-out point, repeats for all points.

    Args:
        surrogate: GP surrogate with fit() and predict() methods.
        X: Training inputs [N, D].
        Y: Training targets [N].

    Returns:
        Dict with LOOCV RMSE, NLPD, Spearman, calibration.
    """
    N = X.shape[0]
    predictions = []
    stds = []

    for i in range(N):
        # Leave out point i
        mask = torch.ones(N, dtype=torch.bool)
        mask[i] = False

        X_train = X[mask]
        Y_train = Y[mask]

        # Fit on N-1 points
        surrogate.fit(X_train, Y_train)

        # Predict held-out point
        mean, std = surrogate.predict(X[i:i+1])
        predictions.append(mean.squeeze())
        stds.append(std.squeeze())

    predictions = torch.stack(predictions)
    stds = torch.stack(stds)

    # Compute metrics
    rmse = compute_rmse(predictions, Y)
    nlpd = compute_nlpd(predictions, stds, Y).mean().item()
    spearman, _ = compute_spearman(predictions, Y)
    calibration = compute_calibration_metrics(predictions, stds, Y)

    return {
        "loocv_rmse": rmse,
        "loocv_nlpd": nlpd,
        "loocv_spearman": spearman,
        "loocv_ece": calibration["ece"],
        "loocv_sharpness": calibration["sharpness"],
    }


def compute_all_metrics(
    mean: torch.Tensor,
    std: torch.Tensor,
    y_true: torch.Tensor,
) -> Dict:
    """Compute all GP quality metrics at once.

    Args:
        mean: Predicted mean [N].
        std: Predicted std [N].
        y_true: True values [N].

    Returns:
        Dict with all metrics.
    """
    rmse = compute_rmse(mean, y_true)
    nlpd = compute_nlpd(mean, std, y_true).mean().item()
    spearman, spearman_p = compute_spearman(mean, y_true)
    kendall, kendall_p = compute_kendall(mean, y_true)
    r2 = compute_r2(mean, y_true)
    calibration = compute_calibration_metrics(mean, std, y_true)

    return {
        "rmse": rmse,
        "nlpd": nlpd,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "kendall": kendall,
        "kendall_p": kendall_p,
        "r2": r2,
        **calibration,
    }
