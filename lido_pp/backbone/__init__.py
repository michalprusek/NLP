"""
FlowPO Backbone: Encoders and Decoders.

This module provides the core neural network components for FlowPO:

1. **SONAR Encoder**: Reconstruction-optimized embeddings (1024D)
   - Replaces GritLM (retrieval-optimized)
   - Better for text reconstruction due to DAE + translation training

2. **Text Flow Autoencoder (TFA)**: Flow-based compression
   - SONAR 1024D → 128D latent via simulation-free flow matching
   - Includes Lipschitz regularization for BO-friendly smoothness

3. **Cross-Attention Decoder**: ICAE-style memory conditioning
   - 16 K,V memory slots replace 4 prefix tokens
   - Position-specific cross-attention for better reconstruction

NeurIPS 2026: FlowPO - Unified Flow Matching for Prompt Optimization
"""

# === FlowPO Components (New Architecture) ===

from lido_pp.backbone.sonar_encoder import (
    SONAREncoder,
    SONARTextDecoder,
    create_sonar_encoder,
)

from lido_pp.backbone.cfm_encoder import (
    TextFlowAutoencoder,
    VelocityField,
    sliced_gw_distance,
    flow_matching_loss,
    compute_lipschitz_loss,
)

from lido_pp.backbone.cross_attention_decoder import (
    CrossAttentionProjector,
    CrossAttentionLayer,
    MemoryConditionedDecoder,
    create_cross_attention_projector,
)

__all__ = [
    # === FlowPO Core Components ===
    # SONAR Encoder (reconstruction-optimized)
    "SONAREncoder",
    "SONARTextDecoder",
    "create_sonar_encoder",
    # Text Flow Autoencoder (TFA)
    "TextFlowAutoencoder",
    "VelocityField",
    "sliced_gw_distance",
    "flow_matching_loss",
    "compute_lipschitz_loss",
    # Cross-Attention Decoder
    "CrossAttentionProjector",
    "CrossAttentionLayer",
    "MemoryConditionedDecoder",
    "create_cross_attention_projector",
]
