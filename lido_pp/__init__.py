"""
LID-O++: Latent Instruction Diffusion Optimization++

Unified architecture for latent instruction optimization using:
- Rectified Flow Matching with OAT-FM regularization
- GritLM unified backbone with NV-Embed Latent Attention
- Flow Curvature Uncertainty (FCU) for cost-aware active learning

Target: NeurIPS 2026
"""

from lido_pp.config import LIDOPPConfig

__version__ = "0.1.0"
__all__ = ["LIDOPPConfig"]
