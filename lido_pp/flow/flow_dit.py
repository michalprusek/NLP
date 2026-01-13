"""
Flow-DiT: Diffusion Transformer for Rectified Flow Matching.

This module implements the Flow-DiT architecture - a Transformer adapted
for Flow Matching that predicts velocity fields v(x_t, t, context).

Key components:
- AdaLayerNorm: Adaptive Layer Normalization conditioned on timestep
- FlowTransformerBlock: Transformer block with self-attention, cross-attention, MLP
- FlowDiT: Full model for velocity prediction

The architecture is designed for latent-space flow matching (32D VAE latent)
with context conditioning from GritLM embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from lido_pp.flow.timestep_embed import TimestepEmbedding


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep embedding.

    Instead of learned affine parameters, AdaLN uses the timestep embedding
    to generate scale (γ) and shift (β) parameters dynamically.

    Formula:
        AdaLN(x, t) = γ(t) * LayerNorm(x) + β(t)

    This allows the normalization to adapt based on the current timestep,
    which is crucial for flow matching where behavior changes across t ∈ [0, 1].

    Args:
        hidden_dim: Dimension of input features
        time_dim: Dimension of timestep embedding
    """

    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Layer norm without learnable affine (we'll use time-conditioned instead)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Project timestep embedding to scale and shift
        # Output: 2 * hidden_dim for scale and shift
        self.scale_shift_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 2),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize to identity transform at start."""
        # Initialize so that initial scale ≈ 1, shift ≈ 0
        nn.init.zeros_(self.scale_shift_proj[-1].weight)
        nn.init.zeros_(self.scale_shift_proj[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization.

        Args:
            x: Input tensor (B, ..., hidden_dim)
            t_emb: Timestep embedding (B, time_dim)

        Returns:
            Normalized and modulated tensor (B, ..., hidden_dim)
        """
        # Get scale and shift from timestep
        scale_shift = self.scale_shift_proj(t_emb)  # (B, hidden_dim * 2)
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each: (B, hidden_dim)

        # Expand for broadcasting if needed
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        # Apply: γ * norm(x) + β
        return self.norm(x) * (1 + scale) + shift


class FlowTransformerBlock(nn.Module):
    """
    Transformer block with AdaLayerNorm and optional cross-attention.

    Architecture:
        x → AdaLN → Self-Attention → + → AdaLN → Cross-Attention → + → AdaLN → MLP → +
           ↑                        ↑           (optional)         ↑              ↑
           └────────────────────────┘           ↑                  └──────────────┘
                                                context

    Args:
        hidden_dim: Dimension of input/output features
        time_dim: Dimension of timestep embedding
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension = hidden_dim * mlp_ratio
        dropout: Dropout rate
        cross_attention: Whether to include cross-attention to context
        context_dim: Dimension of context (for cross-attention)
    """

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        cross_attention: bool = True,
        context_dim: int = 768,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cross_attention_enabled = cross_attention

        # Self-attention block
        self.norm1 = AdaLayerNorm(hidden_dim, time_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention block (optional)
        if cross_attention:
            self.norm2 = AdaLayerNorm(hidden_dim, time_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            # Project context to hidden_dim if needed
            if context_dim != hidden_dim:
                self.context_proj = nn.Linear(context_dim, hidden_dim)
            else:
                self.context_proj = nn.Identity()

        # MLP block
        self.norm3 = AdaLayerNorm(hidden_dim, time_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Zero-init output projections for residual learning
        for module in [self.self_attn, self.mlp[-2]]:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor (B, L, hidden_dim)
            t_emb: Timestep embedding (B, time_dim)
            context: Optional context for cross-attention (B, num_ctx, context_dim)

        Returns:
            Output tensor (B, L, hidden_dim)
        """
        # Self-attention
        x_norm = self.norm1(x, t_emb)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        # Cross-attention (if enabled and context provided)
        if self.cross_attention_enabled and context is not None:
            x_norm = self.norm2(x, t_emb)
            context_proj = self.context_proj(context)
            cross_out, _ = self.cross_attn(
                x_norm, context_proj, context_proj, need_weights=False
            )
            x = x + cross_out

        # MLP
        x = x + self.mlp(self.norm3(x, t_emb))

        return x


class FlowDiT(nn.Module):
    """
    Flow-DiT: Diffusion Transformer for Flow Matching.

    Predicts velocity v(x_t, t, context) for ODE integration:
        dx/dt = v(x_t, t, c)

    For Rectified Flow, the target velocity is v* = x_1 - x_0 (straight line).

    Architecture:
        x_t (latent_dim) → Linear → (hidden_dim)
        t → TimestepEmbedding → (time_dim)
        context → [optional projection]

        Then through N transformer blocks:
        FlowTransformerBlock × num_layers

        Finally:
        LayerNorm → Linear → (latent_dim) = v_θ

    Args:
        latent_dim: Dimension of VAE latent space (default: 32)
        hidden_dim: Transformer hidden dimension (default: 512)
        num_layers: Number of transformer blocks (default: 6)
        num_heads: Number of attention heads (default: 8)
        time_embed_dim: Timestep embedding dimension (default: 256)
        context_dim: Context embedding dimension (default: 768)
        num_context_tokens: Number of context tokens (default: 4)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        cross_attention: Use cross-attention to context (default: True)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        time_embed_dim: int = 256,
        context_dim: int = 768,
        num_context_tokens: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        cross_attention: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection: latent_dim → hidden_dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(
            embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,  # Match transformer hidden dim
        )

        # Context embedding (optional projection)
        if context_dim != hidden_dim:
            self.context_embed = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.context_embed = nn.LayerNorm(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(
                hidden_dim=hidden_dim,
                time_dim=hidden_dim,  # time_embed outputs hidden_dim
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                cross_attention=cross_attention,
                context_dim=hidden_dim,  # After context_embed projection
            )
            for _ in range(num_layers)
        ])

        # Output projection: hidden_dim → latent_dim
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Input projection
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.02)
        nn.init.zeros_(self.input_proj.bias)

        # Output projection: zero-init for residual learning
        # This makes initial predictions close to zero (identity flow)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity at (x_t, t) conditioned on context.

        Args:
            x_t: Noisy/interpolated latent (B, latent_dim)
            t: Timesteps (B,) in range [0, 1]
            context: Optional context embeddings (B, num_ctx, context_dim)

        Returns:
            v: Predicted velocity (B, latent_dim)
        """
        batch_size = x_t.shape[0]

        # Project input to hidden dimension
        # x_t: (B, latent_dim) → (B, hidden_dim)
        h = self.input_proj(x_t)

        # Add sequence dimension for transformer
        # (B, hidden_dim) → (B, 1, hidden_dim)
        h = h.unsqueeze(1)

        # Get timestep embedding
        # t: (B,) → (B, hidden_dim)
        t_emb = self.time_embed(t)

        # Process context if provided
        if context is not None:
            context = self.context_embed(context)

        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, context)

        # Remove sequence dimension
        # (B, 1, hidden_dim) → (B, hidden_dim)
        h = h.squeeze(1)

        # Output projection to velocity
        # (B, hidden_dim) → (B, latent_dim)
        v = self.output_proj(self.output_norm(h))

        return v

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class FlowDiTWithValueHead(nn.Module):
    """
    FlowDiT with auxiliary Value Head for reward prediction.

    Extends FlowDiT with a separate head that predicts the expected
    reward (negative error rate) of a latent vector. This is used
    for cost-aware acquisition in active learning.

    The Value Head shares the transformer backbone but has its own
    output projection.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        time_embed_dim: int = 256,
        context_dim: int = 768,
        num_context_tokens: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        cross_attention: bool = True,
        value_head_hidden: int = 128,
    ):
        super().__init__()

        # Main FlowDiT model
        self.flow_dit = FlowDiT(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            time_embed_dim=time_embed_dim,
            context_dim=context_dim,
            num_context_tokens=num_context_tokens,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            cross_attention=cross_attention,
        )

        # Value Head: predicts reward from latent
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, value_head_hidden),
            nn.GELU(),
            nn.LayerNorm(value_head_hidden),
            nn.Linear(value_head_hidden, value_head_hidden),
            nn.GELU(),
            nn.LayerNorm(value_head_hidden),
            nn.Linear(value_head_hidden, 1),
            nn.Sigmoid(),  # Output in [0, 1] (error rate)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for velocity prediction."""
        return self.flow_dit(x_t, t, context)

    def predict_value(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict error rate for latent vector.

        Args:
            z: Latent vector (B, latent_dim)

        Returns:
            error_rate: Predicted error rate (B,) in [0, 1]
        """
        return self.value_head(z).squeeze(-1)


if __name__ == "__main__":
    # Test FlowDiT
    print("Testing FlowDiT...")

    batch_size = 4
    latent_dim = 32
    context_dim = 768
    num_context = 4

    model = FlowDiT(
        latent_dim=latent_dim,
        hidden_dim=512,
        num_layers=6,
        context_dim=context_dim,
    )

    # Random inputs
    x_t = torch.randn(batch_size, latent_dim)
    t = torch.rand(batch_size)
    context = torch.randn(batch_size, num_context, context_dim)

    # Forward pass
    v = model(x_t, t, context)

    print(f"Input x_t shape: {x_t.shape}")
    print(f"Input t: {t}")
    print(f"Context shape: {context.shape}")
    print(f"Output v shape: {v.shape}")
    print(f"Output v norm: {v.norm(dim=-1)}")

    # Parameter count
    num_params = model.get_num_params()
    print(f"\nTotal parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Test without context
    v_no_ctx = model(x_t, t, context=None)
    print(f"\nWithout context - v shape: {v_no_ctx.shape}")

    # Test FlowDiT with Value Head
    print("\n" + "=" * 50)
    print("Testing FlowDiTWithValueHead...")

    model_with_value = FlowDiTWithValueHead(
        latent_dim=latent_dim,
        hidden_dim=512,
        num_layers=6,
        context_dim=context_dim,
    )

    v = model_with_value(x_t, t, context)
    value = model_with_value.predict_value(x_t)

    print(f"Velocity shape: {v.shape}")
    print(f"Value (error rate) shape: {value.shape}")
    print(f"Value predictions: {value}")
