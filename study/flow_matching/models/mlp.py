"""Simple MLP velocity network for flow matching.

This module provides SimpleMLP, a multi-layer perceptron velocity network
with sinusoidal time embeddings for flow matching on SONAR embeddings.

Target: ~985K parameters with default configuration (hidden_dim=256, num_layers=5).
"""

import math

import torch
import torch.nn as nn


def normalize_timestep(t: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize timestep tensor to shape [B].

    Handles various input shapes: scalar, [B], or [B, 1].

    Args:
        t: Timestep tensor (scalar, [B], or [B, 1]).
        batch_size: Expected batch size for scalar expansion.

    Returns:
        Timestep tensor of shape [B].
    """
    if t.dim() == 2:
        t = t.squeeze(-1)
    if t.dim() == 0:
        t = t.unsqueeze(0).expand(batch_size)
    return t


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Following the standard diffusion/flow matching pattern from DiT and HuggingFace diffusers.

    Args:
        t: Timestep tensor of shape [B] with values in [0, 1].
        dim: Embedding dimension (must be even for clean split).
        max_period: Controls frequency range (default 10000 from Transformer).

    Returns:
        Embedding tensor of shape [B, dim].
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


class SimpleMLP(nn.Module):
    """Simple MLP velocity network for flow matching.

    Architecture:
    - Sinusoidal time embedding -> time MLP -> hidden_dim
    - Concatenate [x, time_emb] -> hidden layers with SiLU -> output
    - Output layer initialized near zero for stable training start

    ~985K params with hidden_dim=256, num_layers=5 (default configuration).

    Attributes:
        input_dim: Input/output dimension (1024 for SONAR embeddings).
        hidden_dim: Hidden layer dimension.
        num_layers: Number of main network layers.
        time_embed_dim: Dimension for sinusoidal time embeddings.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 5,
        time_embed_dim: int = 256,
    ):
        """Initialize SimpleMLP velocity network.

        Args:
            input_dim: Input/output dimension (1024 for SONAR).
            hidden_dim: Hidden layer dimension.
            num_layers: Number of main network layers.
            time_embed_dim: Dimension for sinusoidal time embeddings.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim

        # Time embedding MLP: sinusoidal -> hidden
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main network: [x, t_emb] -> hidden -> ... -> output
        layers = []
        in_dim = input_dim + hidden_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.SiLU(),
            ])
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training.

        Uses Kaiming initialization for hidden layers and near-zero
        initialization for output layer to prevent large initial velocities.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Output layer: small weights for near-zero initial velocity
        # This prevents NaN loss from large initial predictions
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity for flow matching.

        Args:
            x: Input tensor [B, 1024] (noisy embedding at time t).
            t: Time tensor [B], [B, 1], or scalar, values in [0, 1].

        Returns:
            Velocity tensor [B, 1024].
        """
        t = normalize_timestep(t, x.shape[0])
        t_emb = self.time_mlp(timestep_embedding(t, self.time_embed_dim))
        return self.net(torch.cat([x, t_emb], dim=-1))
