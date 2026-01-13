"""
FlowPO: Active Learning with Flow Curvature Uncertainty.

Components:
- FCU Gating (Novel Contribution #3): Trajectory curvature for evaluation decisions
- Value Head: Cheap prediction for confident samples
- Acquisition: Cost-aware acquisition functions

NeurIPS 2026: FlowPO - Unified Flow Matching for Prompt Optimization
"""

from lido_pp.active_learning.curvature import (
    compute_flow_curvature,
    compute_fcu_with_threshold,
    FlowCurvatureEstimator,
    FCUResult,
)
from lido_pp.active_learning.value_head import (
    ValueHead,
    ValueHeadWithUncertainty,
    ValueHeadTrainer,
)
from lido_pp.active_learning.acquisition import (
    CostAwareAcquisition,
    AdaptiveAcquisition,
    AcquisitionResult,
)
from lido_pp.active_learning.gating import (
    EvaluationGate,
    AdaptiveGate,
    GatingDecision,
    EvaluationType,
)
from lido_pp.active_learning.fcu_gating import (
    FlowCurvatureUncertainty,
    AdaptiveEvaluationGate,
    FCUGatingResult,
    FCUStatistics,
    create_fcu_gating,
)

__all__ = [
    # === FlowPO FCU Gating (Novel Contribution #3) ===
    "FlowCurvatureUncertainty",
    "AdaptiveEvaluationGate",
    "FCUGatingResult",
    "FCUStatistics",
    "create_fcu_gating",
    # === Legacy Components ===
    # Curvature
    "compute_flow_curvature",
    "compute_fcu_with_threshold",
    "FlowCurvatureEstimator",
    "FCUResult",
    # Value Head
    "ValueHead",
    "ValueHeadWithUncertainty",
    "ValueHeadTrainer",
    # Acquisition
    "CostAwareAcquisition",
    "AdaptiveAcquisition",
    "AcquisitionResult",
    # Gating
    "EvaluationGate",
    "AdaptiveGate",
    "GatingDecision",
    "EvaluationType",
]
