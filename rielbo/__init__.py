"""EcoFlow: Flow matching + Bayesian optimization for prompt optimization."""

from rielbo.velocity_network import VelocityNetwork
from rielbo.flow_model import FlowMatchingModel
from rielbo.data import SonarEmbeddingDataset, get_sonar_dataloader
from rielbo.train_flow import EMAModel, train_flow
from rielbo.decoder import SonarDecoder
from rielbo.gp_surrogate import SonarGPSurrogate, BAxUSGPSurrogate, create_surrogate
from rielbo.guided_flow import GuidedFlowSampler
from rielbo.optimization_loop import BOOptimizationLoop, OptimizationState, MetricsTracker

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
