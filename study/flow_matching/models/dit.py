"""DiT-style velocity network for flow matching with AdaLN-Zero conditioning.

This module provides DiTVelocityNetwork, a Diffusion Transformer (DiT) velocity network
ported from ecoflow/velocity_network.py with scaled-down configuration for the baseline
architecture study.

Target: ~9.3M parameters with default configuration (hidden_dim=384, num_layers=3, num_heads=6).
"""

import torch
import torch.nn as nn

from study.flow_matching.models.mlp import timestep_embedding


class AdaLNBlock(nn.Module):
    """Transformer block with AdaLN-Zero (6 modulation params: shift/scale/gate for attn and mlp)."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        """Initialize AdaLN transformer block.

        Args:
            hidden_dim: Hidden dimension for attention and MLP.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for MLP hidden dimension.
        """
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
    - Sinusoidal time embedding -> time MLP -> hidden_dim
    - Input projection to hidden dim
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
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 384,
        num_layers: int = 3,
        num_heads: int = 6,
        time_embed_dim: int = 256,
    ):
        """Initialize DiTVelocityNetwork.

        Args:
            input_dim: Input/output dimension (1024 for SONAR).
            hidden_dim: Hidden layer dimension (384 for ~9.3M params).
            num_layers: Number of transformer blocks (3 for ~9.3M params).
            num_heads: Number of attention heads (6 for 64 dim per head).
            time_embed_dim: Dimension for sinusoidal time embeddings.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
        """Predict velocity for flow matching.

        Args:
            x: Input tensor [B, input_dim] (noisy embedding at time t).
            t: Time tensor [B], [B, 1], or scalar, values in [0, 1].

        Returns:
            Velocity tensor [B, input_dim].
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
