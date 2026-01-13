"""
DEPRECATED: Legacy Translator Training (VAE + Projector).

This script is deprecated as FlowPO replaces VAE with Text Flow Autoencoder (TFA)
and prefix tokens with Cross-Attention decoder.

For the new FlowPO architecture, use:
- lido_pp/training/train_cfm.py (Train TFA)
- lido_pp/training/train_flow.py (Train Flow-DiT)

This file is kept for backward compatibility with existing checkpoints.
"""

import warnings

warnings.warn(
    "train_translator.py is DEPRECATED. "
    "FlowPO replaces VAE+Projector with TFA+CrossAttention. "
    "Use train_cfm.py for the new architecture.",
    DeprecationWarning,
    stacklevel=2,
)

raise ImportError(
    "train_translator.py is deprecated and no longer functional.\n"
    "The following components have been removed:\n"
    "  - lido_pp.vae.InstructionVAE → use lido_pp.backbone.cfm_encoder.TextFlowAutoencoder\n"
    "  - lido_pp.backbone.latent_injection.LatentProjector → use lido_pp.backbone.cross_attention_decoder.CrossAttentionProjector\n"
    "  - lido_pp.backbone.gritlm_encoder.GritLMUnifiedEncoder → use lido_pp.backbone.sonar_encoder.SONAREncoder\n"
    "\n"
    "Migration guide:\n"
    "  1. Train TFA: uv run python -m lido_pp.training.train_cfm --data lido_pp/data/sonar_embeddings.pt\n"
    "  2. Train Flow-DiT: (coming soon)\n"
    "  3. Train CrossAttn Projector: (coming soon)\n"
)
