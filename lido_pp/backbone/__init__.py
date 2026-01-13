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
    # Legacy alias
    CoupledFlowEncoder,
    cfm_loss,
)

from lido_pp.backbone.cross_attention_decoder import (
    CrossAttentionProjector,
    CrossAttentionLayer,
    MemoryConditionedDecoder,
    create_cross_attention_projector,
)

# === Legacy Components (Deprecated but kept for compatibility) ===
# These will be removed in a future version

try:
    from lido_pp.backbone.latent_attention import (
        LatentAttentionPooling,
        AdaptiveLatentAttention,
    )
    _LATENT_ATTENTION_AVAILABLE = True
except ImportError:
    _LATENT_ATTENTION_AVAILABLE = False
    LatentAttentionPooling = None
    AdaptiveLatentAttention = None

try:
    from lido_pp.backbone.gritlm_encoder import (
        GritLMUnifiedEncoder,
        create_instruction_encoder,
    )
    _GRITLM_AVAILABLE = True
except ImportError:
    _GRITLM_AVAILABLE = False
    GritLMUnifiedEncoder = None
    create_instruction_encoder = None

try:
    from lido_pp.backbone.latent_injection import (
        LatentInjectionDecoder,
        LatentProjector,
        ProjectorTrainer,
        RoundTripEvaluator,
        LatentInjectionResult,
        create_latent_injection_decoder,
    )
    _LATENT_INJECTION_AVAILABLE = True
except ImportError:
    _LATENT_INJECTION_AVAILABLE = False
    LatentInjectionDecoder = None
    LatentProjector = None
    ProjectorTrainer = None
    RoundTripEvaluator = None
    LatentInjectionResult = None
    create_latent_injection_decoder = None


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
    # === Legacy Aliases (Deprecated) ===
    "CoupledFlowEncoder",  # Use TextFlowAutoencoder
    "cfm_loss",  # Use flow_matching_loss
]

# Conditionally add legacy exports if available
if _LATENT_ATTENTION_AVAILABLE:
    __all__.extend([
        "LatentAttentionPooling",
        "AdaptiveLatentAttention",
    ])

if _GRITLM_AVAILABLE:
    __all__.extend([
        "GritLMUnifiedEncoder",
        "create_instruction_encoder",
    ])

if _LATENT_INJECTION_AVAILABLE:
    __all__.extend([
        "LatentInjectionDecoder",
        "LatentProjector",
        "ProjectorTrainer",
        "RoundTripEvaluator",
        "LatentInjectionResult",
        "create_latent_injection_decoder",
    ])
