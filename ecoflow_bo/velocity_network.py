"""
Velocity Network for Conditional Flow Matching / Rectified Flow.

Predicts the velocity field v_θ(x_t, t, z) that transforms noise to data.
Uses FiLM-style conditioning with sinusoidal time embeddings.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from .config import VelocityNetConfig


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

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation block.

    Modulates the hidden state using conditioning information:
    h = γ * h + β, where γ and β come from conditioning.
    """

    def __init__(self, hidden_dim: int, condition_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Conditioning to scale and shift
        self.cond_proj = nn.Linear(condition_dim, hidden_dim * 2)

    def forward(
        self, h: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: Hidden state [B, hidden_dim]
            cond: Conditioning [B, condition_dim]

        Returns:
            Modulated hidden state [B, hidden_dim]
        """
        # Get scale (γ) and shift (β) from conditioning
        gamma_beta = self.cond_proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma + 1.0  # Initialize scale around 1

        # FiLM modulation
        h_norm = self.norm(h)
        h_mod = gamma * h_norm + beta

        # MLP with residual
        out = self.fc1(h_mod)
        out = torch.nn.functional.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)

        return h + out


class VelocityNetwork(nn.Module):
    """
    Velocity predictor v_θ(x_t, t, z) for flow matching.

    Architecture:
    1. Project x_t to hidden dimension
    2. Create time embedding
    3. Create condition embedding from z
    4. Stack FiLM blocks that combine all information
    5. Project to output velocity
    """

    def __init__(self, config: Optional[VelocityNetConfig] = None):
        super().__init__()
        if config is None:
            config = VelocityNetConfig()

        self.config = config
        self.data_dim = config.data_dim
        self.condition_dim = config.condition_dim
        self.hidden_dim = config.hidden_dim

        # Input projection
        self.input_proj = nn.Linear(config.data_dim, config.hidden_dim)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Condition (z) embedding
        self.cond_proj = nn.Sequential(
            nn.Linear(config.condition_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Combined conditioning (time + z)
        self.combine_cond = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # FiLM blocks
        self.blocks = nn.ModuleList([
            FiLMBlock(config.hidden_dim, config.hidden_dim, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.data_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Zero-initialize output projection for residual-like behavior
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

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
        # Project input
        h = self.input_proj(x_t)

        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_proj(t_emb)

        # Condition embedding
        z_emb = self.cond_proj(z)

        # Combine time and condition
        cond = self.combine_cond(torch.cat([t_emb, z_emb], dim=-1))

        # FiLM blocks
        for block in self.blocks:
            h = block(h, cond)

        # Output velocity
        v = self.output_proj(h)

        return v

    def forward_with_matryoshka(
        self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor, active_dims: int
    ) -> torch.Tensor:
        """
        Forward with only first `active_dims` of z active.

        Used during coarse-to-fine decoding. Inactive z dimensions
        are set to 0.

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
