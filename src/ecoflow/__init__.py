"""
EcoFlow: Flow matching for SONAR embedding optimization.

This module implements:
- VelocityNetwork: DiT-style velocity network with AdaLN time conditioning
- FlowMatchingModel: ODE-based sampling wrapper
- Training infrastructure for OT-CFM objective
"""

from src.ecoflow.velocity_network import VelocityNetwork
from src.ecoflow.flow_model import FlowMatchingModel
from src.ecoflow.data import SonarEmbeddingDataset, get_sonar_dataloader
from src.ecoflow.train_flow import EMAModel, train_flow

__all__ = [
    "VelocityNetwork",
    "FlowMatchingModel",
    "SonarEmbeddingDataset",
    "get_sonar_dataloader",
    "EMAModel",
    "train_flow",
]
