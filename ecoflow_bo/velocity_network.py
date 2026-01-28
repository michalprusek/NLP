"""
DiT-based Velocity Network for Conditional Flow Matching.

Predicts the velocity field v_θ(x_t, t, z) that transforms noise to data.

Architecture (Diffusion Transformer):
1. InputTokenizer: 768D → 12 tokens (each 64D → projected to hidden_dim)
2. LatentExpander: 16D latent → 16 tokens (each hidden_dim)
3. DiTBlock: Self-attention + Cross-attention with AdaLN modulation
4. OutputProjector: 12 tokens → 768D

Key insight: Multi-token input representation enables meaningful self-attention,
while cross-attention to latent tokens provides conditional control.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import DiTVelocityNetConfig


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time t ∈ [0, 1]."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor [B] or [B, 1] with values in [0, 1]

        Returns:
            Time embedding [B, dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() == 2 and t.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"Expected t shape [B] or [B,1], got {t.shape}")

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )

        args = t * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding


class InputTokenizer(nn.Module):
    """
    Tokenize 768D embedding into multiple tokens for self-attention.

    Splits input into n_tokens chunks and projects each to hidden_dim.
    Adds learned positional embeddings to preserve order information.

    This enables meaningful self-attention (vs single-token bottleneck).
    """

    def __init__(
        self,
        data_dim: int = 768,
        n_tokens: int = 12,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.n_tokens = n_tokens
        self.token_dim = data_dim // n_tokens  # 64D per token
        self.hidden_dim = hidden_dim

        assert data_dim % n_tokens == 0, f"data_dim ({data_dim}) must be divisible by n_tokens ({n_tokens})"

        # Project each token chunk to hidden_dim
        self.token_proj = nn.Linear(self.token_dim, hidden_dim)

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input data [B, data_dim]

        Returns:
            tokens: [B, n_tokens, hidden_dim]
        """
        B = x.shape[0]
        # [B, 768] → [B, 12, 64]
        x = x.view(B, self.n_tokens, self.token_dim)
        # [B, 12, 64] → [B, 12, hidden_dim]
        tokens = self.token_proj(x)
        return tokens + self.pos_embed


class OutputProjector(nn.Module):
    """
    Project multiple tokens back to data dimension.

    Inverse of InputTokenizer: combines token representations into single vector.
    """

    def __init__(
        self,
        data_dim: int = 768,
        n_tokens: int = 12,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.n_tokens = n_tokens
        self.token_dim = data_dim // n_tokens
        self.hidden_dim = hidden_dim

        # Project each token back to original chunk size
        self.token_proj = nn.Linear(hidden_dim, self.token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, n_tokens, hidden_dim]

        Returns:
            x: [B, data_dim]
        """
        B = tokens.shape[0]
        # [B, 12, hidden_dim] → [B, 12, 64]
        x = self.token_proj(tokens)
        # [B, 12, 64] → [B, 768]
        return x.view(B, self.data_dim)


class LatentExpander(nn.Module):
    """
    Expand 16 latent scalars into 16 tokens for cross-attention.

    Each latent dimension becomes a separate token with learned positional embedding.
    This allows the transformer to selectively attend to different latent dimensions.

    Matryoshka-aware: When z[i] = 0 (masked), the corresponding token is zeroed out
    entirely, ensuring masked dimensions don't contribute to cross-attention.
    """

    def __init__(self, latent_dim: int = 16, token_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.token_dim = token_dim

        # Each scalar → token_dim vector (zero-init bias for clean Matryoshka masking)
        self.expander = nn.Linear(1, token_dim)
        nn.init.zeros_(self.expander.bias)

        # Learnable positional embeddings for each latent position
        self.pos_embed = nn.Parameter(torch.randn(1, latent_dim, token_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent codes [B, latent_dim], may have trailing zeros for Matryoshka

        Returns:
            tokens: Expanded latent tokens [B, latent_dim, token_dim]
                    Tokens for zero z dimensions are zeroed out.
        """
        B = z.shape[0]
        # [B, 16] → [B, 16, 1] → [B, 16, token_dim]
        tokens = self.expander(z.unsqueeze(-1))
        tokens = tokens + self.pos_embed

        # Matryoshka masking: zero out tokens where z was zero
        # This ensures masked dimensions don't contribute to cross-attention
        z_mask = (z != 0).float().unsqueeze(-1)  # [B, 16, 1]
        tokens = tokens * z_mask

        return tokens


class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with AdaLN (Adaptive Layer Norm).

    Components:
    1. Self-attention on x tokens
    2. Cross-attention (x attends to latent tokens)
    3. MLP with GELU activation

    All modulated by time embedding via AdaLN (scale & shift).
    """

    def __init__(self, dim: int = 256, n_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention (x attends to latent tokens)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )

        # MLP
        mlp_hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

        # AdaLN modulation from time embedding
        # 6 parameters: (scale, shift) for each of 3 norms
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, dim]
            context: Latent tokens for cross-attention [B, latent_dim, dim]
            time_emb: Time embedding [B, dim]

        Returns:
            x: Output tokens [B, N, dim]
        """
        # AdaLN modulation parameters
        # [B, 6*dim] → 6 x [B, 1, dim] for broadcasting
        modulation = self.adaLN(time_emb).unsqueeze(1)
        shift1, scale1, shift2, scale2, shift3, scale3 = modulation.chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = self.norm1(x) * (1 + scale1) + shift1
        x = x + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # Cross-attention with AdaLN (x attends to latent context)
        x_norm = self.norm2(x) * (1 + scale2) + shift2
        x = x + self.cross_attn(x_norm, context, context, need_weights=False)[0]

        # MLP with AdaLN
        x_norm = self.norm3(x) * (1 + scale3) + shift3
        x = x + self.mlp(x_norm)

        return x


class VelocityNetwork(nn.Module):
    """
    DiT-based velocity predictor v_θ(x_t, t, z) for flow matching.

    Architecture:
    1. Tokenize x_t (768D) into 12 tokens via InputTokenizer
    2. Expand latent z (16D) into 16 tokens via LatentExpander
    3. Create time embedding with sinusoidal encoding + MLP
    4. Stack DiT blocks: self-attention on x tokens + cross-attention to z tokens
    5. Project back to data dimension (768D) via OutputProjector

    Multi-token representation enables meaningful self-attention.
    Cross-attention to latent tokens provides conditional control.
    """

    def __init__(self, config: Optional[DiTVelocityNetConfig] = None):
        super().__init__()
        if config is None:
            config = DiTVelocityNetConfig()

        self.config = config
        self.data_dim = config.data_dim
        self.condition_dim = config.condition_dim
        self.hidden_dim = config.hidden_dim
        self.n_input_tokens = config.n_input_tokens

        # Input tokenization: 768D → 12 tokens of hidden_dim
        self.input_tokenizer = InputTokenizer(
            data_dim=config.data_dim,
            n_tokens=config.n_input_tokens,
            hidden_dim=config.hidden_dim,
        )

        # Output projection: 12 tokens → 768D
        self.output_projector = OutputProjector(
            data_dim=config.data_dim,
            n_tokens=config.n_input_tokens,
            hidden_dim=config.hidden_dim,
        )

        # Latent expansion: 16 scalars → 16 tokens
        self.latent_expander = LatentExpander(config.condition_dim, config.hidden_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=config.hidden_dim,
                n_heads=config.n_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable but learnable training."""
        # Small random init for output projection (NOT zero - allows gradients to flow)
        nn.init.normal_(self.output_projector.token_proj.weight, std=0.02)
        nn.init.zeros_(self.output_projector.token_proj.bias)

        # Small init for AdaLN modulation (near-identity but learnable)
        for block in self.blocks:
            nn.init.normal_(block.adaLN[-1].weight, std=0.02)
            nn.init.zeros_(block.adaLN[-1].bias)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity at (x_t, t) conditioned on z.

        Args:
            x_t: Noisy data at time t [B, data_dim]
            t: Time [B] or [B, 1], values in [0, 1]
            z: Latent condition [B, condition_dim]

        Returns:
            v: Predicted velocity [B, data_dim]
        """
        # Tokenize input: [B, 768] → [B, 12, hidden_dim]
        h = self.input_tokenizer(x_t)

        # Expand latent to tokens for cross-attention: [B, 16, hidden_dim]
        z_tokens = self.latent_expander(z)

        # Time embedding: [B, hidden_dim]
        t_emb = self.time_embed(t)

        # DiT blocks with self-attention on x and cross-attention to z
        for block in self.blocks:
            h = block(h, context=z_tokens, time_emb=t_emb)

        # Final projection: [B, 12, hidden_dim] → [B, 768]
        h = self.final_norm(h)
        v = self.output_projector(h)

        return v

    def forward_with_matryoshka(
        self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor, active_dims: int
    ) -> torch.Tensor:
        """
        Forward with only first `active_dims` of z active.

        Used during coarse-to-fine decoding and Matryoshka training.
        Inactive z dimensions are set to 0.

        Args:
            x_t: Noisy data [B, data_dim]
            t: Time [B]
            z: Full latent [B, condition_dim]
            active_dims: Number of active z dimensions

        Returns:
            v: Predicted velocity [B, data_dim]
        """
        z_masked = z.clone()
        z_masked[:, active_dims:] = 0.0
        return self.forward(x_t, t, z_masked)
