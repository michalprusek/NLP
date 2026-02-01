"""Flow matching training infrastructure.

This module provides training utilities for flow matching experiments
on SONAR embeddings.

Public API:
    TrainingConfig: Configuration dataclass for training
    EarlyStopping: Early stopping callback
    EMAModel: Exponential moving average of model parameters
    get_cosine_schedule_with_warmup: Learning rate scheduler
    FlowTrainer: Training orchestrator (when available)
"""

from study.flow_matching.config import TrainingConfig
from study.flow_matching.utils import (
    EarlyStopping,
    EMAModel,
    get_cosine_schedule_with_warmup,
)

__all__ = [
    "TrainingConfig",
    "EarlyStopping",
    "EMAModel",
    "get_cosine_schedule_with_warmup",
]
