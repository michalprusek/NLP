"""
BOLT Hyperparameter Tuning Module

Comprehensive Coordinate Descent optimization for BOLT pipeline:

Phases:
- Phase 1: VAE optimization (Retrieval Accuracy @K, Lipschitz)
- Phase 2: Scorer optimization (NDCG@K, diversity)
- Phase 3: GP optimization (Spearman Correlation, calibration)
- Phase 4: Inference optimization (Final Accuracy)

Features:
- 25 metrics across 5 categories
- 3-tier hyperparameter prioritization (Critical → Important → Finetune)
- Dual-GPU parallel execution
- Long-running support with checkpointing
- SQLite-backed results tracking

Usage:
    # Full tuning
    uv run python -m bolt.tuning.run_tuning --output-dir bolt/tuning/results

    # Quick test
    uv run python -m bolt.tuning.run_tuning --quick --output-dir bolt/tuning/test

    # Single phase
    uv run python -m bolt.tuning.run_tuning --phase vae --output-dir bolt/tuning/vae
"""

from .metrics import (
    MetricResult,
    MetricRegistry,
    MetricCategory,
    MetricTarget,
    TargetDirection,
    VAEMetrics,
    ScorerMetrics,
    GPMetrics,
    OptimizationMetrics,
    EndToEndMetrics,
)
from .hyperspace import (
    HyperparameterSpace,
    HyperparameterConfig,
    ParameterSpec,
    ParameterType,
    TuningPhase,
    TuningTier,
    CRITICAL_PARAMS,
    IMPORTANT_PARAMS,
    FINETUNE_PARAMS,
)
from .trial_runner import (
    TrialRunner,
    TrialResult,
    TrialState,
    run_trial,
)
from .parallel_executor import (
    DualGPUExecutor,
    SingleGPUExecutor,
    TrialTask,
    ExecutorStats,
    GPUInfo,
    GPUStatus,
)
from .coordinator import (
    CoordinateDescentTuner,
    PhaseConfig,
    PhaseResult,
    PhaseStatus,
    DEFAULT_PHASE_CONFIGS,
    run_quick_tune,
)
from .results_tracker import (
    ResultsTracker,
    TrialRecord,
)

__all__ = [
    # Metrics
    "MetricResult",
    "MetricRegistry",
    "MetricCategory",
    "MetricTarget",
    "TargetDirection",
    "VAEMetrics",
    "ScorerMetrics",
    "GPMetrics",
    "OptimizationMetrics",
    "EndToEndMetrics",
    # Hyperspace
    "HyperparameterSpace",
    "HyperparameterConfig",
    "ParameterSpec",
    "ParameterType",
    "TuningPhase",
    "TuningTier",
    "CRITICAL_PARAMS",
    "IMPORTANT_PARAMS",
    "FINETUNE_PARAMS",
    # Trial Runner
    "TrialRunner",
    "TrialResult",
    "TrialState",
    "run_trial",
    # Parallel Executor
    "DualGPUExecutor",
    "SingleGPUExecutor",
    "TrialTask",
    "ExecutorStats",
    "GPUInfo",
    "GPUStatus",
    # Coordinator
    "CoordinateDescentTuner",
    "PhaseConfig",
    "PhaseResult",
    "PhaseStatus",
    "DEFAULT_PHASE_CONFIGS",
    "run_quick_tune",
    # Results Tracker
    "ResultsTracker",
    "TrialRecord",
]
