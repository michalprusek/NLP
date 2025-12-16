"""
Multi-Model Universal Prompt Optimizer.

Finds a single prompt that works well across multiple frontier LLMs (4B-12B).
Uses hybrid OPRO + HbBoPs with multi-output GP and aggregated scoring.
"""

from .config import MultiModelConfig
from .aggregation import aggregate_scores
from .optimizer import MultiModelHybridOptimizer

__all__ = [
    "MultiModelConfig",
    "aggregate_scores",
    "MultiModelHybridOptimizer",
]
