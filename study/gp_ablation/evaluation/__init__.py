"""Evaluation metrics for GP ablation study.

Provides metrics for:
- GP surrogate quality (NLPD, RMSE, calibration)
- Bayesian optimization performance (regret, best found)
- Statistical significance testing
"""

from study.gp_ablation.evaluation.metrics import (
    compute_nlpd,
    compute_rmse,
    compute_spearman,
    compute_kendall,
    compute_r2,
    compute_calibration_metrics,
    compute_loocv_metrics,
)
from study.gp_ablation.evaluation.regret import (
    compute_simple_regret,
    compute_cumulative_regret,
    compute_auc_regret,
    compute_steps_to_threshold,
    compute_improvement_rate,
    compute_all_regret_metrics,
)
from study.gp_ablation.evaluation.significance import (
    compute_significance,
    compute_bootstrap_ci,
    generate_comparison_table,
)

__all__ = [
    # Surrogate quality metrics
    "compute_nlpd",
    "compute_rmse",
    "compute_spearman",
    "compute_kendall",
    "compute_r2",
    "compute_calibration_metrics",
    "compute_loocv_metrics",
    # Regret metrics
    "compute_simple_regret",
    "compute_cumulative_regret",
    "compute_auc_regret",
    "compute_steps_to_threshold",
    "compute_improvement_rate",
    "compute_all_regret_metrics",
    # Significance testing
    "compute_significance",
    "compute_bootstrap_ci",
    "generate_comparison_table",
]
