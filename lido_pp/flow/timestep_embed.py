"""
Timestep Embedding for Rectified Flow Matching.

This module implements sinusoidal timestep embeddings similar to those used
in diffusion models, adapted for flow matching where t ∈ [0, 1].

The embedding provides positional information about the current timestep
to the velocity prediction network.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding (fixed, not learned).

    Maps scalar timestep t to high-dimensional embedding using sine and cosine
    functions at different frequencies.

    Formula:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Args:
        dim: Embedding dimension (should be even)
        max_period: Maximum period for frequencies (default: 10000)
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # Precompute frequencies (not trainable)
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal embedding for timesteps.

        Args:
            t: Timesteps (B,) in range [0, 1]

        Returns:
            embedding: (B, dim) sinusoidal embedding
        """
        # Scale timestep to have good frequency coverage
        # t is in [0, 1], we want good coverage across frequencies
        args = t.unsqueeze(-1) * self.freqs  # (B, half_dim)

        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding with learnable MLP transformation.

    Combines sinusoidal position embedding with a learned MLP to produce
    a timestep embedding that can modulate the transformer layers.

    Architecture:
        t (scalar) → Sinusoidal (dim) → Linear → SiLU → Linear → output (hidden_dim)

    Args:
        embed_dim: Dimension of sinusoidal embedding (default: 256)
        hidden_dim: Output dimension (default: 512)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Sinusoidal embedding
        self.sinusoidal = SinusoidalPositionEmbedding(embed_dim)

        # Learnable MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),  # Swish activation (works well with transformers)
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute timestep embedding.

        Args:
            t: Timesteps (B,) in range [0, 1]

        Returns:
            embedding: (B, hidden_dim) timestep embedding
        """
        # Get sinusoidal embedding
        sinusoidal_emb = self.sinusoidal(t)

        # Project through MLP
        return self.mlp(sinusoidal_emb)


class FourierFeatures(nn.Module):
    """
    Random Fourier features for timestep embedding.

    Alternative to sinusoidal embedding that uses random frequencies.
    Can provide better coverage of the embedding space.

    Reference: "Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains" (Tancik et al., NeurIPS 2020)

    Args:
        dim: Output dimension
        scale: Standard deviation of random frequencies (default: 16.0)
    """

    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.dim = dim

        # Random frequencies (fixed after initialization)
        B = torch.randn(dim // 2) * scale
        self.register_buffer("B", B)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features.

        Args:
            t: Timesteps (B,) in range [0, 1]

        Returns:
            features: (B, dim) Fourier features
        """
        # Scale t to [0, 2π]
        t_proj = 2 * math.pi * t.unsqueeze(-1) * self.B  # (B, dim//2)

        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


if __name__ == "__main__":
    # Test timestep embedding
    batch_size = 4
    embed_dim = 256
    hidden_dim = 512

    embedding = TimestepEmbedding(embed_dim, hidden_dim)

    # Test with various timesteps
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    emb = embedding(t)

    print(f"Input timesteps: {t}")
    print(f"Output shape: {emb.shape}")
    print(f"Output norm: {emb.norm(dim=-1)}")

    # Visualize embedding similarity across timesteps
    print("\nTimestep embedding similarity (cosine):")
    emb_norm = emb / emb.norm(dim=-1, keepdim=True)
    sim = emb_norm @ emb_norm.T
    for i in range(len(t)):
        print(f"  t={t[i]:.2f}: {sim[i].tolist()}")
