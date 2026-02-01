"""Experimental Mamba velocity network for flow matching.

This module provides MambaVelocityNetwork, an experimental selective state-space
model (SSM) adapted for velocity prediction on SONAR embeddings.

WARNING: This is EXPERIMENTAL. Mamba is designed for sequential data with causal
structure; applying it to unordered embedding transformations may not work well.
Results (positive or negative) are valid research contributions.

Target: ~2M parameters with default configuration (hidden_dim=256, num_layers=4).

Requires mamba-ssm package (pip install mamba-ssm). If unavailable, MAMBA_AVAILABLE
will be False and MambaVelocityNetwork will raise ImportError on instantiation.
"""

import torch
import torch.nn as nn

from study.flow_matching.models.mlp import timestep_embedding

# Try to import mamba-ssm; fallback gracefully if not available
try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None  # type: ignore


class MambaVelocityNetwork(nn.Module):
    """Experimental Mamba-based velocity network for flow matching.

    Key adaptations for embedding-to-embedding (non-sequential):
    1. Treat 1024-dim embedding as "sequence" of chunks (16 chunks of 64 dims)
    2. Use bidirectional processing (forward + backward Mamba)
    3. Add time conditioning after each layer
    4. Aggregate with linear projection

    Architecture (~2M params with defaults):
    - Reshape [B, 1024] -> [B, 16, 64] (virtual sequence)
    - Project to hidden: Linear(chunk_size, hidden_dim)
    - Bidirectional Mamba layers with time conditioning
    - Concatenate forward/backward outputs
    - Project back: Linear(hidden_dim*2, chunk_size)
    - Reshape [B, 16, 64] -> [B, 1024]

    WARNING: This is experimental. Mamba is designed for sequences,
    not for unordered embedding transformations. Document results either way.

    Attributes:
        input_dim: Input/output dimension (1024 for SONAR embeddings).
        hidden_dim: Hidden layer dimension for Mamba.
        d_state: SSM state dimension (internal Mamba parameter).
        d_conv: Convolution dimension (internal Mamba parameter).
        expand: Expansion factor (internal Mamba parameter).
        num_layers: Number of Mamba layers (forward and backward each).
        time_embed_dim: Dimension for sinusoidal time embeddings.
        chunk_size: Size of each chunk (input_dim must be divisible by this).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        time_embed_dim: int = 256,
        chunk_size: int = 64,
    ):
        """Initialize MambaVelocityNetwork.

        Args:
            input_dim: Input/output dimension (1024 for SONAR).
            hidden_dim: Hidden dimension for Mamba layers.
            d_state: SSM state dimension.
            d_conv: Convolution dimension.
            expand: Expansion factor for inner dimension.
            num_layers: Number of bidirectional Mamba layers.
            time_embed_dim: Dimension for sinusoidal time embeddings.
            chunk_size: Chunk size for virtual sequence.

        Raises:
            ImportError: If mamba-ssm is not installed.
            ValueError: If input_dim is not divisible by chunk_size.
        """
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is not installed. Install with: pip install mamba-ssm\n"
                "Note: mamba-ssm requires Linux, NVIDIA GPU, and CUDA 11.6+."
            )

        if input_dim % chunk_size != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by chunk_size ({chunk_size}). "
                f"Valid chunk_sizes for 1024: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.chunk_size = chunk_size
        self.n_chunks = input_dim // chunk_size

        # Time embedding MLP: sinusoidal -> hidden
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project chunks to hidden dim
        self.input_proj = nn.Linear(chunk_size, hidden_dim)

        # Mamba layers (bidirectional: forward + backward)
        self.mamba_fwd = nn.ModuleList(
            [
                Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(num_layers)
            ]
        )
        self.mamba_bwd = nn.ModuleList(
            [
                Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(num_layers)
            ]
        )

        # Time conditioning projections (one per layer)
        self.time_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Output projection (zero-init for stable training)
        self.output_proj = nn.Linear(hidden_dim * 2, chunk_size)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity for flow matching.

        Args:
            x: Input tensor [B, 1024] (noisy embedding at time t).
            t: Time tensor [B], [B, 1], or scalar, values in [0, 1].

        Returns:
            Velocity tensor [B, 1024].
        """
        B = x.shape[0]

        # Handle t shape variations
        if t.dim() == 2:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)

        # Time embedding
        t_emb = timestep_embedding(t, self.time_embed_dim)
        cond = self.time_mlp(t_emb)  # [B, hidden_dim]

        # Reshape embedding to sequence of chunks
        # [B, 1024] -> [B, n_chunks, chunk_size]
        h = x.view(B, self.n_chunks, self.chunk_size)

        # Project to hidden dim
        h = self.input_proj(h)  # [B, n_chunks, hidden_dim]

        # Bidirectional Mamba processing
        h_fwd = h
        h_bwd = h.flip(dims=[1])  # Reverse sequence for backward direction

        for i, (mamba_f, mamba_b, time_proj) in enumerate(
            zip(self.mamba_fwd, self.mamba_bwd, self.time_projs)
        ):
            # Forward direction
            h_fwd = mamba_f(h_fwd)
            # Backward direction (on reversed sequence)
            h_bwd = mamba_b(h_bwd)

            # Add time conditioning (broadcast across sequence)
            t_cond = time_proj(cond).unsqueeze(1)  # [B, 1, hidden_dim]
            h_fwd = h_fwd + t_cond
            h_bwd = h_bwd + t_cond

        # Reverse backward output to match forward direction
        h_bwd = h_bwd.flip(dims=[1])

        # Concatenate forward and backward
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, n_chunks, hidden_dim*2]

        # Project back to chunk size
        v = self.output_proj(h)  # [B, n_chunks, chunk_size]

        # Reshape to original dimension
        v = v.view(B, self.input_dim)  # [B, 1024]

        return v
