"""
EcoFlow: Flow matching for SONAR embedding optimization.

This module implements:
- VelocityNetwork: DiT-style velocity network with AdaLN time conditioning
- FlowMatchingModel: ODE-based sampling wrapper
- Training infrastructure for OT-CFM objective
"""

from src.ecoflow.velocity_network import VelocityNetwork

__all__ = [
    "VelocityNetwork",
]

# Lazy imports for modules that may not exist yet
def __getattr__(name):
    if name == "FlowMatchingModel":
        from src.ecoflow.flow_model import FlowMatchingModel
        return FlowMatchingModel
    elif name == "SonarEmbeddingDataset":
        from src.ecoflow.data import SonarEmbeddingDataset
        return SonarEmbeddingDataset
    elif name == "get_sonar_dataloader":
        from src.ecoflow.data import get_sonar_dataloader
        return get_sonar_dataloader
    raise AttributeError(f"module 'src.ecoflow' has no attribute {name!r}")
