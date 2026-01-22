"""
FlowDiT: Flow Matching DiT for manifold projection.

Simple DiT architecture without attention - MLP blocks with AdaLN.
"""

import math
import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] timesteps in [0, 1]
        Returns:
            [B, hidden_dim] time embeddings
        """
        # Sinusoidal embedding
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return self.mlp(emb)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] input
            cond: [B, cond_dim] conditioning
        Returns:
            [B, D] modulated output
        """
        params = self.proj(cond)
        scale, shift = params.chunk(2, dim=-1)

        # LayerNorm
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-6)

        return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """DiT block: AdaLN + MLP."""

    def __init__(self, hidden_dim: int, cond_dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        mlp_hidden = int(hidden_dim * mlp_ratio)

        self.adaln = AdaLN(hidden_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        x_norm = self.adaln(x, cond)
        return x + self.mlp(x_norm)


class FlowDiT(nn.Module):
    """
    Flow Matching DiT for SONAR space.

    Maps noise → data via learned velocity field v(x, t).
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        time_embed_dim: int = 256,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Time embedding: 256 → 2048
        cond_dim = int(hidden_dim * mlp_ratio)
        self.time_embed = TimestepEmbedding(time_embed_dim, cond_dim)

        # Input/output projections
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, cond_dim, mlp_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity field v(x, t).

        Args:
            x: [B, latent_dim] noisy input
            t: [B] timesteps in [0, 1]

        Returns:
            [B, latent_dim] velocity field
        """
        # Time conditioning
        cond = self.time_embed(t)

        # Project input
        h = self.input_proj(x)

        # Process through blocks
        for block in self.blocks:
            h = block(h, cond)

        # Output projection
        return self.output_proj(h)


def integrate_euler(
    model: FlowDiT,
    x0: torch.Tensor,
    num_steps: int = 20,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """
    Integrate ODE from t_start to t_end using Euler method.

    For flow matching: dx/dt = v(x, t)

    Args:
        model: FlowDiT velocity field
        x0: [B, D] initial state (noise at t=0)
        num_steps: number of Euler steps
        t_start: start time
        t_end: end time

    Returns:
        [B, D] final state (data at t=1)
    """
    device = x0.device
    dtype = x0.dtype
    B = x0.shape[0]

    dt = (t_end - t_start) / num_steps
    x = x0

    for i in range(num_steps):
        t = t_start + i * dt
        t_batch = torch.full((B,), t, device=device, dtype=dtype)

        v = model(x, t_batch)
        x = x + v * dt

    return x


if __name__ == "__main__":
    # Test loading checkpoint
    import sys

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "flowpo_hd/checkpoints_mega_aux2/best.pt"

    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = FlowDiT(
        latent_dim=1024,
        hidden_dim=1024,
        num_layers=4,
        time_embed_dim=256,
        mlp_ratio=2.0,
    )

    model.load_state_dict(ckpt['model_state_dict'])
    print("Loaded successfully!")

    # Test forward pass
    x = torch.randn(2, 1024)
    t = torch.rand(2)
    v = model(x, t)
    print(f"Forward pass: {x.shape} → {v.shape}")

    # Test integration
    x0 = torch.randn(2, 1024)
    x1 = integrate_euler(model, x0, num_steps=20)
    print(f"Integration: {x0.shape} → {x1.shape}")
