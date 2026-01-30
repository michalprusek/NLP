"""
Velocity network for flow matching with DiT-style AdaLN time conditioning.

Architecture:
- Sinusoidal timestep embeddings (same as transformer positional encoding)
- AdaLN-Zero blocks with adaptive layer normalization
- Zero-initialized output projection for stable training
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        t: Tensor of timesteps, shape [B]
        dim: Embedding dimension
        max_period: Maximum period for frequencies

    Returns:
        Embedding tensor of shape [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class AdaLNBlock(nn.Module):
    """
    Transformer block with Adaptive Layer Normalization (AdaLN-Zero).

    Uses 6 modulation parameters per block:
    - shift_attn, scale_attn, gate_attn for attention
    - shift_mlp, scale_mlp, gate_mlp for MLP
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Layer norms without learnable parameters (AdaLN provides modulation)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Self-attention
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
        """
        Forward pass with adaptive layer normalization.

        Args:
            x: Input tensor [B, S, D]
            c: Conditioning tensor [B, D] (time embedding)

        Returns:
            Output tensor [B, S, D]
        """
        # Get modulation parameters from conditioning
        mod = self.adaLN_modulation(c)  # [B, 6*D]
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Attention block with AdaLN
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_attn.unsqueeze(1) * attn_out

        # MLP block with AdaLN
        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_mod)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class VelocityNetwork(nn.Module):
    """
    DiT-style velocity network for flow matching.

    Takes input x and timestep t, outputs velocity v of same shape as x.
    Uses AdaLN-Zero blocks for time conditioning.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embed_dim = time_embed_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # AdaLN transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # shift and scale only
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Zero-init final layers for stable training
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict velocity.

        Args:
            x: Input tensor [B, input_dim]
            t: Timesteps [B] or [B, 1]

        Returns:
            Velocity tensor [B, input_dim]
        """
        # Handle t shape
        if t.dim() == 2:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # Time embedding
        t_emb = timestep_embedding(t, self.time_embed_dim)  # [B, time_embed_dim]
        c = self.time_embed(t_emb)  # [B, hidden_dim]

        # Project input to hidden dim and add sequence dimension
        h = self.input_proj(x)  # [B, hidden_dim]
        h = h.unsqueeze(1)  # [B, 1, hidden_dim]

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, c)

        # Remove sequence dimension
        h = h.squeeze(1)  # [B, hidden_dim]

        # Final norm with AdaLN and output projection
        h = self.final_norm(h)
        mod = self.final_adaLN(c)
        shift, scale = mod.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        v = self.output_proj(h)

        return v
