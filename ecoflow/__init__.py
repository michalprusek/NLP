"""EcoFlow: Flow matching + Bayesian optimization for prompt optimization."""

from ecoflow.velocity_network import VelocityNetwork
from ecoflow.flow_model import FlowMatchingModel
from ecoflow.data import SonarEmbeddingDataset, get_sonar_dataloader
from ecoflow.train_flow import EMAModel, train_flow
from ecoflow.decoder import SonarDecoder
from ecoflow.gp_surrogate import SonarGPSurrogate, BAxUSGPSurrogate, create_surrogate
from ecoflow.guided_flow import GuidedFlowSampler
from ecoflow.optimization_loop import BOOptimizationLoop, OptimizationState, MetricsTracker

__all__ = [
    "VelocityNetwork",
    "FlowMatchingModel",
    "SonarEmbeddingDataset",
    "get_sonar_dataloader",
    "EMAModel",
    "train_flow",
    "SonarDecoder",
    "SonarGPSurrogate",
    "BAxUSGPSurrogate",
    "create_surrogate",
    "GuidedFlowSampler",
    "BOOptimizationLoop",
    "OptimizationState",
    "MetricsTracker",
]
