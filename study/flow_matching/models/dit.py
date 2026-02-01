"""DiT-style velocity network for flow matching with AdaLN-Zero conditioning.

This module provides DiTVelocityNetwork, a Diffusion Transformer (DiT) velocity network
ported from ecoflow/velocity_network.py with scaled-down configuration for the baseline
architecture study.

Key fix (from code review): The 1024-dim embedding is now chunked into 16 tokens of 64 dims
each, so attention can actually learn cross-chunk dependencies (unlike single-token which
trivially returns q@k^T@v = v).

Improvements (NeurIPS 2026):
- QK-normalization for improved training stability (optional, enabled by default)

Target: ~9.3M parameters with default configuration (hidden_dim=384, num_layers=3, num_heads=6).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from study.flow_matching.models.mlp import timestep_embedding


class QKNormAttention(nn.Module):
    """Multi-head self-attention with QK-normalization.

    QK-normalization applies LayerNorm to Q and K vectors before computing
    attention scores. This improves training stability and allows for larger
    learning rates, especially beneficial for flow matching where the velocity
    field must be learned accurately.

    Reference: Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters"
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """Initialize QK-normalized attention.

        Args:
            embed_dim: Total dimension of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)

        # QK normalization (per-head LayerNorm)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with QK-normalized attention.

        Args:
            x: Input tensor [B, S, D].

        Returns:
            Output tensor [B, S, D].
        """
        B, S, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply QK-normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, H, S, head_dim]
        out = out.transpose(1, 2).reshape(B, S, D)

        return self.out_proj(out)


class AdaLNBlock(nn.Module):
    """Transformer block with AdaLN-Zero (6 modulation params: shift/scale/gate for attn and mlp).

    Supports optional QK-normalization for improved training stability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = False,
    ):
        """Initialize AdaLN transformer block.

        Args:
            hidden_dim: Hidden dimension for attention and MLP.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for MLP hidden dimension.
            use_qk_norm: If True, use QK-normalized attention.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_qk_norm = use_qk_norm

        # Layer norms without learnable parameters (AdaLN provides modulation)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Self-attention (with or without QK-norm)
        if use_qk_norm:
            self.attn = QKNormAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True,
            )

        # MLP with GELU activation
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

        # AdaLN modulation: 6 parameters (shift, scale, gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # Zero-init the modulation output layer
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, S, D] where S is sequence length.
            c: Conditioning tensor [B, D] (time embedding).

        Returns:
            Output tensor [B, S, D].
        """
        # Get modulation parameters from conditioning
        mod = self.adaLN_modulation(c)  # [B, 6*D]
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Attention block with AdaLN
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)

        # Apply attention (different interface for QK-norm vs standard)
        if self.use_qk_norm:
            attn_out = self.attn(x_mod)
        else:
            attn_out, _ = self.attn(x_mod, x_mod, x_mod)

        x = x + gate_attn.unsqueeze(1) * attn_out

        # MLP block with AdaLN
        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_mod)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class DiTVelocityNetwork(nn.Module):
    """DiT-style velocity network for flow matching with AdaLN-Zero time conditioning.

    Ported from ecoflow/velocity_network.py with scaled-down configuration for baseline study.

    Architecture:
    - Chunk 1024-dim embedding into 16 tokens of 64 dims each (enables meaningful attention)
    - Sinusoidal time embedding -> time MLP -> hidden_dim
    - Per-token projection to hidden dim
    - N transformer blocks with AdaLN-Zero conditioning
    - Final AdaLN layer norm and output projection
    - Zero-init on output layers for stable training start

    ~9.3M params with hidden_dim=384, num_layers=3, num_heads=6 (default configuration).

    Attributes:
        input_dim: Input/output dimension (1024 for SONAR embeddings).
        hidden_dim: Hidden layer dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        time_embed_dim: Dimension for sinusoidal time embeddings.
        num_tokens: Number of tokens to chunk embedding into (16 for 1024/64).
        token_dim: Dimension per token (64 for 1024/16).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 384,
        num_layers: int = 3,
        num_heads: int = 6,
        time_embed_dim: int = 256,
        num_tokens: int = 16,  # Chunk 1024 into 16 tokens of 64 dims
        use_qk_norm: bool = True,  # QK-normalization for stability
    ):
        """Initialize DiTVelocityNetwork.

        Args:
            input_dim: Input/output dimension (1024 for SONAR).
            hidden_dim: Hidden layer dimension (384 for ~9.3M params).
            num_layers: Number of transformer blocks (3 for ~9.3M params).
            num_heads: Number of attention heads (6 for 64 dim per head).
            time_embed_dim: Dimension for sinusoidal time embeddings.
            num_tokens: Number of tokens to split input into (default 16).
            use_qk_norm: If True, use QK-normalized attention (recommended).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.num_tokens = num_tokens
        self.token_dim = input_dim // num_tokens
        self.use_qk_norm = use_qk_norm

        assert input_dim % num_tokens == 0, f"input_dim ({input_dim}) must be divisible by num_tokens ({num_tokens})"

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-token input projection (from token_dim to hidden_dim)
        self.input_proj = nn.Linear(self.token_dim, hidden_dim)

        # Learnable position embedding for tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # AdaLN transformer blocks with optional QK-norm
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, num_heads, use_qk_norm=use_qk_norm)
            for _ in range(num_layers)
        ])

        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # shift and scale only
        )
        # Per-token output projection (from hidden_dim back to token_dim)
        self.output_proj = nn.Linear(hidden_dim, self.token_dim)

        # Zero-init final layers for stable training
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity for flow matching.

        Args:
            x: Input tensor [B, input_dim] (noisy embedding at time t).
            t: Time tensor [B], [B, 1], or scalar, values in [0, 1].

        Returns:
            Velocity tensor [B, input_dim].
        """
        batch_size = x.shape[0]

        # Handle t shape
        if t.dim() == 2:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        # Time embedding
        t_emb = timestep_embedding(t, self.time_embed_dim)  # [B, time_embed_dim]
        c = self.time_embed(t_emb)  # [B, hidden_dim]

        # Chunk input into tokens: [B, input_dim] -> [B, num_tokens, token_dim]
        h = x.view(batch_size, self.num_tokens, self.token_dim)

        # Project each token to hidden dim: [B, num_tokens, token_dim] -> [B, num_tokens, hidden_dim]
        h = self.input_proj(h)

        # Add position embeddings
        h = h + self.pos_embed

        # Apply transformer blocks (now attention across 16 tokens is meaningful!)
        for block in self.blocks:
            h = block(h, c)

        # Final norm with AdaLN (applied per token)
        h = self.final_norm(h)  # [B, num_tokens, hidden_dim]
        mod = self.final_adaLN(c)  # [B, 2*hidden_dim]
        shift, scale = mod.chunk(2, dim=-1)
        # Broadcast to all tokens
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Project back to token_dim: [B, num_tokens, hidden_dim] -> [B, num_tokens, token_dim]
        h = self.output_proj(h)

        # Reassemble: [B, num_tokens, token_dim] -> [B, input_dim]
        v = h.view(batch_size, self.input_dim)

        return v
