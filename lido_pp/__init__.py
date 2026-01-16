"""
FlowPO: Flow Matching for Prompt Optimization

Unified architecture for latent instruction optimization using:
- Text Flow Autoencoder (TFA) with OT-CFM (Novel #1)
- SONAR text encoder (1024D reconstruction-optimized embeddings)
- GP-Guided Flow Generation for acquisition optimization (Novel #2)
- Flow Curvature Uncertainty (FCU) for cost-aware evaluation gating (Novel #3)
- Cross-Attention Decoder with memory slots for LLM conditioning (Novel #4)

Architecture:
    SONAR 1024D → TFA (OT-CFM) → 256D latent → GP-guided flow → decode → SONAR decoder

Target: NeurIPS 2026
"""

from lido_pp.config import FlowPOConfig, LIDOPPConfig  # LIDOPPConfig is alias

__version__ = "0.2.0"
__all__ = ["FlowPOConfig", "LIDOPPConfig"]
