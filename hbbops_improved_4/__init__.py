"""
HbBoPs Improved 4: Multi-Fidelity GP with Top 75% Fidelity Filtering

Based on Improved 3 but uses only top 75% of fidelity levels for GP training
(excludes bottom 25% lowest fidelity observations).

Features:
1. Wilson score variance for robust noise estimation
2. Logit transform with Delta method for output warping
3. Product kernel for multi-fidelity modeling
4. TOP 75% FIDELITY FILTERING - only higher fidelity data for GP training
5. GP model persistence
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
