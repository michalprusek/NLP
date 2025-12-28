"""Exemplar Selector with HbBoPs Deep Kernel GP - Reused from robust_vec2text.

Preserves the late fusion architecture:
- Instruction branch: 768 -> 64 -> 32
- Exemplar branch: 768 -> 64 -> 32
- Joint projection: 64 -> 10D for GP kernel

This structural-aware design is key to HbBoPs methodology and is unchanged
in the COWBOYS implementation.
"""

from robust_vec2text.exemplar_selector import (
    ExemplarSelector,
    DeepKernelGP,
    FeatureExtractor,
)

__all__ = ["ExemplarSelector", "DeepKernelGP", "FeatureExtractor"]
