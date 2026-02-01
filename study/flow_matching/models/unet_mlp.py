"""U-Net MLP velocity network with FiLM time conditioning.

This module provides UNetMLP, an encoder-decoder MLP architecture with skip
connections and Feature-wise Linear Modulation (FiLM) for time conditioning.

The U-Net structure may better preserve embedding structure through the
encoder-decoder pipeline compared to simple MLP. FiLM provides efficient
time conditioning via affine modulation (gamma*x + beta).

Target: ~6.9M parameters with default configuration (hidden_dims=(512, 256)).
Note: Concatenative skip connections increase params vs original 2.5M estimate.
"""

import torch
import torch.nn as nn

from study.flow_matching.models.mlp import normalize_timestep, timestep_embedding


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.

    Applies affine transformation: out = gamma * x + beta
    where gamma and beta are predicted from conditioning (timestep).

    The layer is initialized to identity transformation (gamma=1, beta=0)
    to ensure stable training start.

    Reference: https://arxiv.org/abs/1709.07871

    Attributes:
        feature_dim: Dimension of features to modulate.
        conditioning_dim: Dimension of conditioning input.
    """

    def __init__(self, feature_dim: int, conditioning_dim: int):
        """Initialize FiLM layer.

        Args:
            feature_dim: Dimension of features to modulate.
            conditioning_dim: Dimension of conditioning input (e.g., time embedding).
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim

        # Predict gamma (scale) and beta (shift) from conditioning
        self.film_params = nn.Linear(conditioning_dim, 2 * feature_dim)

        # Initialize to identity transformation (gamma=1, beta=0)
        # This prevents training instability from random modulation
        nn.init.zeros_(self.film_params.weight)
        nn.init.zeros_(self.film_params.bias)
        self.film_params.bias.data[:feature_dim] = 1.0  # gamma = 1

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Feature tensor [B, D].
            cond: Conditioning tensor [B, cond_dim] (e.g., time embedding).

        Returns:
            Modulated features [B, D].
        """
        params = self.film_params(cond)  # [B, 2*D]
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * x + beta


class UNetMLPBlock(nn.Module):
    """Single U-Net MLP block with FiLM conditioning.

    Architecture:
    - Linear1: in_dim -> out_dim
    - FiLM modulation based on time conditioning
    - Linear2: out_dim -> out_dim
    - Residual connection if in_dim == out_dim

    Attributes:
        in_dim: Input dimension.
        out_dim: Output dimension.
        cond_dim: Conditioning dimension for FiLM.
    """

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        """Initialize U-Net MLP block.

        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            cond_dim: Conditioning dimension for FiLM layer.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.film = FiLMLayer(out_dim, cond_dim)
        self.act = nn.SiLU()

        # Residual connection if dimensions match
        self.residual = in_dim == out_dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass through block.

        Args:
            x: Input tensor [B, in_dim].
            cond: Conditioning tensor [B, cond_dim].

        Returns:
            Output tensor [B, out_dim].
        """
        h = self.act(self.linear1(x))
        h = self.film(h, cond)
        h = self.act(self.linear2(h))
        if self.residual:
            h = h + x
        return h


class UNetMLP(nn.Module):
    """U-Net MLP velocity network with FiLM time conditioning.

    Architecture (~6.9M params with default config):
    - Encoder: 1024 -> 512 -> 256 (bottleneck)
    - Decoder: 256 -> 512 -> 1024 with skip connections
    - FiLM modulation at each block
    - Output layer zero-initialized for stable training

    The skip connections concatenate encoder outputs with decoder inputs,
    potentially preserving embedding structure through the network.

    Attributes:
        input_dim: Input/output dimension (1024 for SONAR embeddings).
        hidden_dims: Tuple of encoder dimensions (512, 256).
        time_embed_dim: Dimension for sinusoidal time embeddings.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: tuple = (512, 256),
        time_embed_dim: int = 256,
    ):
        """Initialize U-Net MLP velocity network.

        Args:
            input_dim: Input/output dimension (1024 for SONAR).
            hidden_dims: Encoder dimension sequence, decoder mirrors this.
            time_embed_dim: Dimension for sinusoidal time embeddings.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.time_embed_dim = time_embed_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Encoder blocks: input_dim -> hidden_dims[0] -> hidden_dims[1] -> ...
        encoder_dims = [input_dim] + list(hidden_dims)
        self.encoder = nn.ModuleList([
            UNetMLPBlock(encoder_dims[i], encoder_dims[i + 1], time_embed_dim)
            for i in range(len(hidden_dims))
        ])

        # Decoder blocks with skip connections
        # Reverse dimensions: hidden_dims[-1] -> ... -> input_dim
        decoder_dims = list(reversed(hidden_dims)) + [input_dim]
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dims)):
            # Skip connection doubles input dimension (except bottleneck)
            in_dim = decoder_dims[i] if i == 0 else decoder_dims[i] * 2
            out_dim = decoder_dims[i + 1]
            self.decoder.append(UNetMLPBlock(in_dim, out_dim, time_embed_dim))

        # Output projection with input skip (zero-init for stable start)
        self.output_proj = nn.Linear(input_dim * 2, input_dim)
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
        t = normalize_timestep(t, x.shape[0])
        cond = self.time_mlp(timestep_embedding(t, self.time_embed_dim))

        # Encoder with skip storage
        skips = [x]  # Store input as first skip
        h = x
        for block in self.encoder:
            h = block(h, cond)
            skips.append(h)

        # Decoder with skip connections
        # Don't include bottleneck output in skips to itself
        skips = skips[:-1]
        for i, block in enumerate(self.decoder):
            if i > 0:  # Skip for non-bottleneck layers
                skip = skips.pop()
                h = torch.cat([h, skip], dim=-1)
            h = block(h, cond)

        # Final output with input skip
        h = torch.cat([h, skips.pop()], dim=-1)  # Concat with original input
        return self.output_proj(h)
