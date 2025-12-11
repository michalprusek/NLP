"""
HbBoPs Improved 3: Multi-Fidelity GP with Heteroscedastic Noise and Output Warping

Improvements over hbbops_improved_2:
1. Wilson score variance for robust noise estimation
2. Logit transform with Delta method for output warping
3. Product kernel for multi-fidelity modeling
4. GP model persistence
"""

from .hbbops import HbBoPs, Prompt
from .gp_model import MultiFidelityDeepKernelGP, FeatureExtractor, prepare_gp_input
from .noise_estimation import (
    wilson_score_variance,
    logit_transform_with_delta_method,
    compute_heteroscedastic_noise
)
from .model_persistence import GPModelSaver

__all__ = [
    'HbBoPs',
    'Prompt',
    'MultiFidelityDeepKernelGP',
    'FeatureExtractor',
    'wilson_score_variance',
    'logit_transform_with_delta_method',
    'compute_heteroscedastic_noise',
    'GPModelSaver'
]
