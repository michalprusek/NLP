"""Reflow (Rectified Flow) implementation for straighter ODE paths.

Reflow iteratively straightens flow trajectories by generating synthetic
(noise, ODE_endpoint) pairs from a trained teacher model and retraining
on these pairs. This enables faster sampling with fewer ODE steps.

Components:
- ReflowPairGenerator: Generates synthetic training pairs from teacher ODE
- ReflowCoupling: Uses pre-generated pairs for training (via coupling/__init__.py)

Usage:
    >>> from study.flow_matching.reflow import ReflowPairGenerator
    >>> from study.flow_matching.coupling import create_coupling
    >>>
    >>> # Generate pairs from teacher
    >>> generator = ReflowPairGenerator(teacher_model, n_steps=100)
    >>> x0_pairs, x1_pairs = generator.generate_pairs(n_pairs=10000, device='cuda')
    >>>
    >>> # Create coupling for training
    >>> coupling = create_coupling('reflow', pair_tensors=(x0_pairs, x1_pairs))
"""

from study.flow_matching.reflow.pair_generator import ReflowPairGenerator

__all__ = ["ReflowPairGenerator"]
