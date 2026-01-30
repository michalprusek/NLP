"""
EcoFlow: Flow matching for SONAR embedding optimization.

This module implements:
- VelocityNetwork: DiT-style velocity network with AdaLN time conditioning
- FlowMatchingModel: ODE-based sampling wrapper
- Training infrastructure for OT-CFM objective
- SonarDecoder: SONAR embedding-to-text decoder
- Validation utilities for generation quality assessment
- SonarGPSurrogate: GP surrogate for 1024D SONAR Bayesian optimization
- GuidedFlowSampler: LCB-guided ODE sampling with CFG-Zero* schedule
- BOOptimizationLoop: Full BO pipeline orchestrator with checkpointing
"""

from src.ecoflow.velocity_network import VelocityNetwork
from src.ecoflow.flow_model import FlowMatchingModel
from src.ecoflow.data import SonarEmbeddingDataset, get_sonar_dataloader
from src.ecoflow.train_flow import EMAModel, train_flow
from src.ecoflow.decoder import SonarDecoder
from src.ecoflow.validate import (
    compute_sample_statistics,
    compute_diversity_metrics,
    compute_text_metrics,
    validate_generation,
    load_model_from_checkpoint,
)
from src.ecoflow.gp_surrogate import SonarGPSurrogate
from src.ecoflow.guided_flow import GuidedFlowSampler, cfg_zero_star_schedule
from src.ecoflow.optimization_loop import (
    BOOptimizationLoop,
    OptimizationState,
    MetricsTracker,
)
from src.ecoflow.batch_selection import select_batch_candidates
from src.ecoflow.flow_density import (
    compute_flow_log_density,
    filter_by_flow_density,
)

__all__ = [
    "VelocityNetwork",
    "FlowMatchingModel",
    "SonarEmbeddingDataset",
    "get_sonar_dataloader",
    "EMAModel",
    "train_flow",
    "SonarDecoder",
    "compute_sample_statistics",
    "compute_diversity_metrics",
    "compute_text_metrics",
    "validate_generation",
    "load_model_from_checkpoint",
    "SonarGPSurrogate",
    "GuidedFlowSampler",
    "cfg_zero_star_schedule",
    "BOOptimizationLoop",
    "OptimizationState",
    "MetricsTracker",
    "select_batch_candidates",
    "compute_flow_log_density",
    "filter_by_flow_density",
]
