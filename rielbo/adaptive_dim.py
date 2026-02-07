"""Intrinsic dimensionality estimation for AdaS-BO.

Estimates subspace dimension from cold-start data using TwoNN + MLE ensemble.
The N//6 cap handles small-sample overestimation: at N=100, cap=16.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def estimate_intrinsic_dim(
    directions: np.ndarray,
    n_points: int,
    d_min: int = 8,
    d_max: int = 48,
) -> tuple[int, dict]:
    """Estimate subspace dimension from unit-norm directions.

    Uses TwoNN + MLE ensemble average, rounded to nearest multiple of 4,
    capped at [d_min, min(N//6, d_max)].
    """
    import skdim

    twonn = skdim.id.TwoNN(discard_fraction=0.1)
    twonn.fit(directions)
    d_twonn = float(twonn.dimension_)

    mle = skdim.id.MLE(K=20)
    mle.fit(directions)
    d_mle = float(mle.dimension_)

    d_raw = (d_twonn + d_mle) / 2.0

    # Round to nearest multiple of 4
    d_rounded = 4 * round(d_raw / 4)

    # Cap at [d_min, min(N//6, d_max)], rounded down to multiple of 4
    # N//6: at N=100 → 16 (matches V2 optimal), relaxes as N grows
    upper = 4 * (min(n_points // 6, d_max) // 4)
    d_estimate = int(max(d_min, min(d_rounded, upper)))

    diagnostics = {
        "d_twonn": d_twonn,
        "d_mle": d_mle,
        "d_raw": d_raw,
        "d_rounded": d_rounded,
        "n_points": n_points,
        "upper_cap": upper,
        "d_estimate": d_estimate,
    }

    logger.info(
        f"ID estimate: TwoNN={d_twonn:.1f}, MLE={d_mle:.1f}, "
        f"raw={d_raw:.1f}, rounded={d_rounded}, "
        f"cap=N//6={upper} -> d={d_estimate}"
    )

    return d_estimate, diagnostics


def assess_gp_health(
    diagnostic_history: list[dict],
    iteration: int,
    last_restart_iter: int,
    best_score_history: list[float],
) -> dict:
    """Assess GP health for restart decisions.

    Returns dict with dead_gp, persistent_overfit, stagnation flags.
    """
    result = {
        "dead_gp": False,
        "persistent_overfit": False,
        "stagnation": False,
        "details": {},
    }

    if len(diagnostic_history) >= 3:
        recent_corrs = [
            d.get("train_correlation", 0.0) for d in diagnostic_history[-3:]
        ]
        result["persistent_overfit"] = bool(all(c > 0.995 for c in recent_corrs))
        result["details"]["recent_correlations"] = [float(c) for c in recent_corrs]

        recent_std_ratios = [
            d.get("train_std_ratio", 1.0) for d in diagnostic_history[-3:]
        ]
        result["dead_gp"] = bool(all(r < 0.001 for r in recent_std_ratios))
        result["details"]["recent_std_ratios"] = [float(r) for r in recent_std_ratios]

    iters_since_restart = iteration - last_restart_iter
    result["details"]["iters_since_restart"] = int(iters_since_restart)

    if len(best_score_history) >= 100:
        improvement = float(best_score_history[-1] - best_score_history[-100])
        result["stagnation"] = bool(improvement < 0.001)
        result["details"]["improvement_last_100"] = improvement

    return result
