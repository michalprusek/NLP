"""
Multi-Model Hybrid Optimizer with Per-Model GP Selection.

This module extends hybrid_opro_hbbops for multiple models with:
- Per-model GP selection: Top 10 candidates independently for each model
- Shared ICM GP: Multi-output GP with Intrinsic Coregionalization Model kernel
- Batch per-model evaluation: Hoeffding bounds for early stopping

Main components:
- MultiModelHybridOptimizer: Main optimization class
- PerModelSelector: GP-based per-model candidate selection
- BatchPerModelEvaluator: Batch evaluation with Hoeffding bounds
- MultiModelHybridConfig: Configuration dataclass

Usage:
    >>> from multi_model_hybrid import MultiModelHybridOptimizer, MultiModelHybridConfig
    >>> config = MultiModelHybridConfig()
    >>> optimizer = MultiModelHybridOptimizer(config, validation_data, train_data)
    >>> best_inst, best_ex, best_acc, per_model = optimizer.run(num_iterations=10)

CLI:
    uv run python multi_model_hybrid/run_multi_model_hybrid.py --help
"""

from multi_model_hybrid.config import (
    MultiModelHybridConfig,
    MultiModelHybridCandidate,
    MultiModelHybridDesignPoint,
    PerModelSelectionConfig,
    PerModelCandidateScore,
)
from multi_model_hybrid.optimizer import MultiModelHybridOptimizer
from multi_model_hybrid.per_model_selector import PerModelSelector
from multi_model_hybrid.batch_evaluator import BatchPerModelEvaluator, HoeffdingDecision

__all__ = [
    # Main optimizer
    "MultiModelHybridOptimizer",
    # Selection
    "PerModelSelector",
    # Evaluation
    "BatchPerModelEvaluator",
    "HoeffdingDecision",
    # Config
    "MultiModelHybridConfig",
    "MultiModelHybridCandidate",
    "MultiModelHybridDesignPoint",
    "PerModelSelectionConfig",
    "PerModelCandidateScore",
]

__version__ = "1.0.0"
