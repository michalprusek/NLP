"""
Configuration dataclasses for Multi-Model Universal Prompt Optimizer.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from hybrid_opro_hbbops.config import HybridConfig


@dataclass
class MultiModelConfig(HybridConfig):
    """
    Configuration for multi-model universal prompt optimization.

    Extends HybridConfig with multi-model specific parameters.
    """

    # Target models (3 frontier models 7-8B)
    target_models: List[str] = field(
        default_factory=lambda: [
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]
    )

    # GPU assignment for each model (model_name -> GPU ID)
    gpu_assignment: Dict[str, int] = field(
        default_factory=lambda: {
            "Qwen/Qwen2.5-7B-Instruct": 0,
            "meta-llama/Llama-3.1-8B-Instruct": 1,
            "mistralai/Mistral-7B-Instruct-v0.3": 2,
        }
    )

    # Aggregation strategy: "average", "minimum", "weighted_softmin", "harmonic"
    aggregation: str = "weighted_softmin"

    # Temperature for weighted_softmin (lower = closer to min, higher = closer to avg)
    softmin_temperature: float = 0.1

    # Optional per-model weights (must sum to 1.0)
    model_weights: Optional[Dict[str, float]] = None

    # Multi-output GP configuration
    use_multi_output_gp: bool = True
    gp_num_tasks: int = 4  # One output per model
    gp_rank: int = 2  # Rank of ICM task covariance matrix

    # Parallel evaluation settings
    parallel_models: bool = True
    max_workers: int = 4  # Number of parallel model evaluations
    model_timeout: int = 300  # Timeout per model in seconds (5 min)

    # Single-GPU mode (for limited hardware)
    single_gpu: bool = False  # If True, switch models on one GPU
    single_gpu_id: int = 0  # GPU ID to use in single-GPU mode

    # Model-sequential mode (minimizes model switching)
    model_sequential_mode: bool = False  # Process each model completely before switching
    final_verification_top_k: int = 5  # Top candidates for final multi-model verification

    # Meta-model for OPRO (API-based)
    meta_model_backend: str = "vllm"  # "vllm", "gemini", "openai"
    meta_model_api_key: Optional[str] = None  # API key for external meta-model

    # Budget allocation
    budget_per_model: bool = False  # If True, total_llm_budget is per-model

    # Override defaults from parent
    total_llm_budget: int = 200000  # Higher budget for multi-model
    output_dir: str = "multi_model_optimizer/results"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure GPU assignment covers all target models
        for model in self.target_models:
            if model not in self.gpu_assignment:
                raise ValueError(f"Missing GPU assignment for model: {model}")

        # Validate aggregation strategy
        valid_strategies = {"average", "minimum", "weighted_softmin", "harmonic"}
        if self.aggregation not in valid_strategies:
            raise ValueError(
                f"Invalid aggregation: {self.aggregation}. "
                f"Must be one of: {valid_strategies}"
            )

        # Validate model weights if provided
        if self.model_weights is not None:
            if set(self.model_weights.keys()) != set(self.target_models):
                raise ValueError("model_weights must have weights for all target_models")
            total = sum(self.model_weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"model_weights must sum to 1.0, got {total}")

        # Update gp_num_tasks to match number of models
        self.gp_num_tasks = len(self.target_models)


@dataclass
class MultiModelDesignPoint:
    """
    Design point with scores from multiple models.

    Used for GP training data.
    """

    instruction_id: int
    exemplar_id: int
    instruction_embedding: np.ndarray
    exemplar_embedding: np.ndarray
    model_error_rates: Dict[str, float]  # model_name -> error_rate
    aggregated_error: float
    fidelity: int


@dataclass
class MultiModelPromptCandidate:
    """
    Prompt candidate with multi-model predictions and results.

    Used during GP screening and sequential testing.
    """

    instruction: str
    instruction_id: int
    instruction_embedding: np.ndarray
    exemplar: str
    exemplar_id: int
    exemplar_embedding: np.ndarray

    # GP predictions per model
    gp_predicted_errors: Optional[Dict[str, float]] = None
    gp_aggregated_prediction: Optional[float] = None

    # Actual evaluation results per model
    actual_accuracies: Optional[Dict[str, float]] = None
    aggregated_accuracy: Optional[float] = None

    # Sequential testing results
    samples_used: Optional[int] = None
    decision: Optional[str] = None  # "drop", "promote", or "full"
