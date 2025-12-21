"""
Configuration for Multi-Model Hybrid Optimizer with per-model GP selection.

This module defines dataclasses for:
- PerModelSelectionConfig: Per-model GP selection parameters
- MultiModelHybridConfig: Main configuration extending MultiModelConfig
- MultiModelHybridCandidate: Candidate with per-model GP predictions
- MultiModelHybridDesignPoint: Design point for GP training
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Import base config from multi_model_optimizer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_model_optimizer.config import MultiModelConfig


@dataclass
class PerModelSelectionConfig:
    """
    Configuration for per-model GP-based candidate selection.

    Each model independently selects its top-k candidates based on
    LCB (Lower Confidence Bound) scores from the GP predictions.
    Uses LCB = mean - kappa*std for minimizing error rate.
    The union of all per-model selections forms the evaluation pool.
    """

    top_k_per_model: int = 10
    """Number of top candidates to select per model."""

    max_union_candidates: int = 30
    """Maximum unique candidates from union of per-model selections."""

    use_uncertainty: bool = True
    """Whether to use LCB (mean - kappa*std) for selection vs mean-only."""

    ucb_kappa: float = 2.0
    """LCB exploration parameter. Higher = more exploration (wider bounds)."""

    def __post_init__(self):
        """Validate configuration."""
        if self.top_k_per_model < 1:
            raise ValueError("top_k_per_model must be >= 1")
        if self.max_union_candidates < self.top_k_per_model:
            raise ValueError("max_union_candidates must be >= top_k_per_model")
        if self.ucb_kappa < 0:
            raise ValueError("ucb_kappa must be non-negative")


@dataclass
class PerModelCandidateScore:
    """
    Score for a candidate on a specific model from GP prediction.

    Stores both mean prediction and uncertainty for LCB-based selection.
    LCB (Lower Confidence Bound) is used since we minimize error rate.
    """

    instruction_id: int
    """Instruction registry ID."""

    exemplar_id: int
    """Exemplar registry ID."""

    model_name: str
    """Model name this score is for."""

    gp_mean: float
    """GP predicted error rate (mean)."""

    gp_std: float
    """GP prediction uncertainty (standard deviation)."""

    lcb_score: float
    """LCB score: mean - kappa * std (lower is better, optimistic for low error)."""

    rank_in_model: int = -1
    """Rank within this model's selection (1 = best)."""


@dataclass
class MultiModelHybridCandidate:
    """
    Candidate for multi-model hybrid evaluation.

    Tracks per-model GP predictions and which models selected this candidate.
    """

    instruction: str
    """Instruction text."""

    instruction_id: int
    """Instruction registry ID."""

    instruction_embedding: np.ndarray
    """Instruction embedding (dimension depends on encoder model)."""

    exemplar: str
    """Exemplar text (few-shot examples)."""

    exemplar_id: int
    """Exemplar registry ID."""

    exemplar_embedding: np.ndarray
    """Exemplar embedding (dimension depends on encoder model)."""

    # Per-model GP predictions (filled during Phase 3)
    gp_predictions: Dict[str, PerModelCandidateScore] = field(default_factory=dict)
    """Map: model_name -> PerModelCandidateScore."""

    # Which models selected this candidate in their top-k
    selected_by_models: List[str] = field(default_factory=list)
    """List of model names that selected this candidate."""

    # Actual evaluation results (filled during Phase 4)
    actual_errors: Dict[str, float] = field(default_factory=dict)
    """Map: model_name -> actual error rate."""

    actual_fidelities: Dict[str, int] = field(default_factory=dict)
    """Map: model_name -> number of samples used."""

    # Hoeffding decisions per model
    decisions: Dict[str, str] = field(default_factory=dict)
    """Map: model_name -> 'drop'/'promote'/'continue'/'skipped'."""

    # Aggregated results
    aggregated_accuracy: Optional[float] = None
    """Aggregated accuracy across all models."""


@dataclass
class MultiModelHybridDesignPoint:
    """
    Design point for GP training with per-model tracking.

    Stores which models have actual evaluations vs GP predictions.
    """

    instruction_id: int
    """Instruction registry ID."""

    exemplar_id: int
    """Exemplar registry ID."""

    instruction_embedding: np.ndarray
    """Instruction embedding (dimension depends on encoder model)."""

    exemplar_embedding: np.ndarray
    """Exemplar embedding (dimension depends on encoder model)."""

    # Actual evaluations (model -> error_rate)
    actual_model_errors: Dict[str, float] = field(default_factory=dict)
    """Map: model_name -> actual error rate (from evaluation)."""

    actual_model_fidelities: Dict[str, int] = field(default_factory=dict)
    """Map: model_name -> number of samples used for evaluation."""

    # GP predictions for models not yet evaluated
    gp_predicted_errors: Dict[str, float] = field(default_factory=dict)
    """Map: model_name -> GP predicted error rate."""

    # Combined aggregated score
    aggregated_error: float = 1.0
    """Aggregated error rate across all models."""

    # Evaluation status
    evaluation_complete: bool = False
    """True if evaluated on ALL models."""


@dataclass
class MultiModelHybridConfig(MultiModelConfig):
    """
    Configuration for Multi-Model Hybrid Optimizer.

    Extends MultiModelConfig with per-model selection strategy.

    Key differences from MultiModelConfig:
    1. Per-model GP selection (not aggregated)
    2. Union-based candidate pooling
    3. Batch per-model evaluation with Hoeffding bounds
    """

    # Per-model selection configuration
    per_model_selection: PerModelSelectionConfig = field(
        default_factory=PerModelSelectionConfig
    )
    """Configuration for per-model GP selection."""

    # Override: GP selects per model, not globally
    # gp_top_k now means top_k per model (inherited from parent)

    # Hoeffding bounds configuration
    hoeffding_confidence: float = 0.95
    """Confidence level for Hoeffding bounds (1 - delta)."""

    hoeffding_min_samples: int = 10
    """Minimum samples before making Hoeffding decision."""

    hoeffding_min_promote_samples: int = 30
    """Minimum samples required to PROMOTE a candidate."""

    # Batch evaluation
    batch_per_model: bool = True
    """Always true for this module."""

    # APE Forward for initial instructions (Stage 1)
    use_ape_forward_init: bool = True
    """Use APE forward pass to generate initial instructions instead of loading from file."""

    ape_num_samples: int = 10
    """Number of input-output examples to show per APE generation."""

    ape_num_candidates: int = 100
    """Number of raw candidates to generate before clustering."""

    ape_num_final: int = 25
    """Number of final instructions after K-means clustering."""

    # Initial exemplars (Stage 1)
    num_initial_exemplars: int = 25
    """Number of initial exemplars to sample."""

    initial_exemplar_qa_pairs: int = 2
    """Number of Q/A pairs per initial exemplar (Stage 1 only)."""

    def __post_init__(self):
        """Validate and adjust configuration."""
        # Run parent validations first
        super().__post_init__()

        # Validate Hoeffding parameters
        if not 0 < self.hoeffding_confidence < 1:
            raise ValueError("hoeffding_confidence must be in (0, 1)")
        if self.hoeffding_min_samples < 1:
            raise ValueError("hoeffding_min_samples must be >= 1")
        if self.hoeffding_min_samples >= self.hoeffding_min_promote_samples:
            raise ValueError("hoeffding_min_samples must be < hoeffding_min_promote_samples")

        # Validate APE parameters
        if self.ape_num_final > self.ape_num_candidates:
            raise ValueError("ape_num_final must be <= ape_num_candidates")
        if self.ape_num_samples < 1:
            raise ValueError("ape_num_samples must be >= 1")

        # Ensure per-model selection matches gp_top_k
        self.per_model_selection.top_k_per_model = self.gp_top_k

        # Max union is at most num_models * top_k
        max_possible = len(self.target_models) * self.gp_top_k
        self.per_model_selection.max_union_candidates = min(
            self.per_model_selection.max_union_candidates,
            max_possible
        )

        # Ensure single_gpu is True for this module
        self.single_gpu = True
