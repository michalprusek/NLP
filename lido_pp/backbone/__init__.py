"""Backbone encoders and decoders for LID-O++."""

from lido_pp.backbone.latent_attention import (
    LatentAttentionPooling,
    AdaptiveLatentAttention,
)
from lido_pp.backbone.gritlm_encoder import (
    GritLMUnifiedEncoder,
    create_instruction_encoder,
)
from lido_pp.backbone.latent_injection import (
    LatentInjectionDecoder,
    LatentProjector,
    ProjectorTrainer,
    RoundTripEvaluator,
    LatentInjectionResult,
    create_latent_injection_decoder,
)

__all__ = [
    # Encoder components
    "LatentAttentionPooling",
    "AdaptiveLatentAttention",
    "GritLMUnifiedEncoder",
    "create_instruction_encoder",
    # Decoder components
    "LatentInjectionDecoder",
    "LatentProjector",
    "ProjectorTrainer",
    "RoundTripEvaluator",
    "LatentInjectionResult",
    "create_latent_injection_decoder",
]
