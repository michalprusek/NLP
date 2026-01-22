"""
FlowPO-HD: Manifold-Guided High-Dimensional Prompt Optimization.

Direct optimization in 1024D SONAR embedding space using:
- ManifoldKeeper: Flow Matching model to stay on valid instruction manifold
- TuRBO-1024: Trust regions adapted for high-dimensional optimization
- Flow-Guided Acquisition: GP gradient + manifold force for robust optimization

Key insight: No compression (1024D native), manifold force prevents adversarial examples.
"""

from flowpo_hd.config import FlowPOHDConfig, get_device
from flowpo_hd.manifold_keeper import ManifoldKeeperMLP

__all__ = [
    "FlowPOHDConfig",
    "get_device",
    "ManifoldKeeperMLP",
]
