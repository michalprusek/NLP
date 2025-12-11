"""
Configuration dataclasses for Hybrid OPRO + HbBoPs.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class HybridConfig:
    """Configuration for hybrid OPRO + HbBoPs optimization."""

    # Phase 1: Initial Hyperband (or load from file)
    initial_instructions_path: str = "datasets/hbbops/instructions_25.txt"
    initial_exemplars_path: str = "datasets/hbbops/examples_25.txt"
    bmin: int = 10
    eta: float = 2.0

    # Skip Phase 1 HbBoPs and load pre-computed results from file
    skip_phase1_hbbops: bool = False
    phase1_results_path: str = "datasets/hbbops/full_grid_combined.jsonl"

    # Phase 2: OPRO instruction generation
    opro_candidates_per_iter: int = 8
    opro_keep_top_k: int = 20
    meta_model: str = "Qwen/Qwen2.5-7B-Instruct"
    meta_temperature: float = 1.0
    meta_max_tokens: int = 500

    # Phase 3: GP screening
    num_dynamic_exemplars: int = 25
    exemplars_per_sample: int = 5  # Q/A pairs per exemplar
    gp_top_k: int = 10  # Top candidates by GP prediction

    # Phase 5: GP training
    gp_latent_dim: int = 10
    gp_train_epochs: int = 3000
    gp_lr: float = 0.01
    gp_patience: int = 10

    # Budget
    total_llm_budget: int = 50000  # Total LLM evaluation calls

    # Sequential testing with Hoeffding bounds (Phase 4)
    sequential_testing: bool = True  # Enable dynamic sample sizing
    sequential_confidence: float = 0.95  # Confidence level (95%)
    sequential_min_samples: int = 10  # Minimum samples before decisions
    sequential_min_promote_samples: int = 30  # Min samples for PROMOTE decision

    # Task model
    task_model: str = "Qwen/Qwen2.5-7B-Instruct"
    task_backend: str = "vllm"
    task_max_tokens: int = 1024

    # Encoder
    encoder_name: str = "bert-base-uncased"

    # Misc
    seed: int = 42
    device: str = "auto"
    output_dir: str = "hybrid_opro_hbbops/results"


@dataclass
class ScoredInstruction:
    """Instruction with its best accuracy across all exemplars."""

    instruction: str
    instruction_id: int
    best_accuracy: float  # 1 - error_rate
    best_exemplar_id: int
    embedding: Optional[np.ndarray] = None


@dataclass
class PromptCandidate:
    """A prompt candidate for GP screening."""

    instruction: str
    instruction_id: int
    instruction_embedding: np.ndarray
    exemplar: str
    exemplar_id: int
    exemplar_embedding: np.ndarray
    gp_predicted_accuracy: Optional[float] = None
    actual_accuracy: Optional[float] = None
    # Sequential testing results
    samples_used: Optional[int] = None  # Number of samples actually used
    decision: Optional[str] = None  # "drop", "promote", or "full"


@dataclass
class DesignPoint:
    """A single observation in GP design data."""

    instruction_id: int
    exemplar_id: int
    instruction_embedding: np.ndarray
    exemplar_embedding: np.ndarray
    error_rate: float
    fidelity: int  # Number of validation instances used
