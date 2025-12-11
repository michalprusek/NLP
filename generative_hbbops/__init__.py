"""
HyLO: Hyperband Latent Optimization for Prompt Tuning.

A prompt optimization system that operates in the embedding space using
gradient-based optimization and Vec2Text inversion.
"""

from .config import HyLOConfig
from .encoder import GTREncoder
from .gp_model import FeatureExtractor, HyLOGP, GPTrainer
from .optimizer import CoordinateDescentOptimizer, GumbelSoftmaxOptimizer, OptimizationResult
from .inverter import Vec2TextInverter
from .visualizer import HyLOVisualizer
from .hylo import HyLO

__all__ = [
    "HyLOConfig",
    "GTREncoder",
    "FeatureExtractor",
    "HyLOGP",
    "GPTrainer",
    "CoordinateDescentOptimizer",
    "GumbelSoftmaxOptimizer",
    "OptimizationResult",
    "Vec2TextInverter",
    "HyLOVisualizer",
    "HyLO",
]

__version__ = "0.1.0"
