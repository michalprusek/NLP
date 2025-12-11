"""
HbBoPs Improved 5: Multi-Fidelity GP WITHOUT Logit Transform

Based on Improved 4 but removes the logit output warping.
GP models accuracy directly with standard binomial variance.

Features:
1. Wilson score variance for robust noise estimation (at extreme p values)
2. NO LOGIT TRANSFORM - GP models accuracy directly
3. Standard binomial variance: p(1-p)/n
4. Product kernel for multi-fidelity modeling
5. Top 75% fidelity filtering
6. GP model persistence
"""

from .hbbops import HbBoPs, Prompt
from .gp_model import MultiFidelityDeepKernelGP, FeatureExtractor, prepare_gp_input
from .noise_estimation import (
    wilson_score_variance,
    compute_heteroscedastic_noise
)
from .model_persistence import GPModelSaver

__all__ = [
    'HbBoPs',
    'Prompt',
    'MultiFidelityDeepKernelGP',
    'FeatureExtractor',
    'wilson_score_variance',
    'compute_heteroscedastic_noise',
    'GPModelSaver'
]
