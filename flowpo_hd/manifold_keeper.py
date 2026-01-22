"""
ManifoldKeeper: MLP Velocity Field for FlowPO-HD.

The ManifoldKeeper is a flow matching model that learns the velocity field
on the manifold of valid instruction embeddings. During optimization, it
provides a "manifold force" that keeps candidates on this manifold.

Architecture (NO bottleneck):
    Input: x(1024D) + t → AdaLN conditioning
    Block×3: Linear(1024→2048) → GELU → Dropout → Linear(2048→1024) + Residual
    Output: 1024D velocity (zero-init for stability)
    ~15M parameters

Key insight: Unlike autoencoders, we don't compress. The flow model learns
the structure of the 1024D manifold directly, providing meaningful gradients
for staying on it.

Reference:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- OT-CFM: Improving Training of Rectified Flows (Liu et al., 2024)
"""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timestep embedding."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or scalar timestep in [0, 1]

        Returns:
            encoding: (B, dim) sinusoidal encoding
        """
        t = t.reshape(-1, 1)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )

        args = t * freqs
        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            encoding = F.pad(encoding, (0, 1))

        return encoding


class TimestepEmbedding(nn.Module):
    """MLP to embed timestep into hidden dimension."""

    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.positional = SinusoidalPositionalEncoding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        pos = self.positional(t)
        return self.mlp(pos)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization with learned scale and shift from timestep.

    AdaLN(x, t) = γ(t) * LayerNorm(x) + β(t)

    This allows the network to modulate its behavior based on the timestep,
    which is crucial for flow matching where dynamics change along the flow.
    """

    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # Project time embedding to scale and shift
        self.proj = nn.Linear(time_emb_dim, dim * 2)

        # Initialize to identity transform
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) input features
            t_emb: (B, time_emb_dim) timestep embedding

        Returns:
            (B, D) normalized and modulated features
        """
        # Get scale and shift from timestep
        scale_shift = self.proj(t_emb)  # (B, D*2)
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each (B, D)

        # AdaLN: scale * norm(x) + shift
        # Adding 1 to scale so init is identity
        return (1 + scale) * self.norm(x) + shift


class ManifoldResBlock(nn.Module):
    """Residual block with AdaLN conditioning for ManifoldKeeper.

    Architecture:
        x → AdaLN(t) → Linear → GELU → Dropout → Linear → + x

    The AdaLN allows timestep-dependent processing, critical for flow matching.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.adaln = AdaLN(dim, time_emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # Zero-init the last layer for stable residual connection
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) input features
            t_emb: (B, time_emb_dim) timestep embedding

        Returns:
            (B, D) output features
        """
        h = self.adaln(x, t_emb)
        h = self.mlp(h)
        return x + h  # Residual connection


class ManifoldKeeperMLP(nn.Module):
    """
    MLP Velocity Field for learning the instruction manifold.

    This model learns v_θ(x, t) such that integrating the ODE
    dx/dt = v_θ(x, t) from noise x_0 at t=0 to t=1 produces
    samples on the manifold of valid instruction embeddings.

    During optimization, we use v_θ(x, t=0.9) as a "manifold force"
    to push candidates towards the manifold.

    Architecture (~15M parameters):
        Input: x(1024D) concatenated with t_emb(256D)
        Blocks: 3× ManifoldResBlock(1024→2048→1024)
        Output: 1024D velocity

    Why t=0.9 for manifold force:
    - At t=1.0, samples are exactly on data manifold
    - At t=0.9, velocity points strongly towards manifold
    - At t=0.5, velocity is less informative (middle of flow)
    - At t=0.0, we're in noise space
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int = 2048,
        time_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize ManifoldKeeper.

        Args:
            dim: Input/output dimension (1024 for SONAR)
            hidden_dim: Hidden dimension in residual blocks
            time_dim: Timestep embedding dimension
            num_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_blocks = num_blocks

        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_dim, hidden_dim)

        # Input projection (concat x and t_emb, then project)
        self.input_proj = nn.Linear(dim, dim)

        # Residual blocks with AdaLN
        self.blocks = nn.ModuleList([
            ManifoldResBlock(dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(dim, dim)

        # Zero-init output for stable training (identity flow at init)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Count parameters
        self._num_params = sum(p.numel() for p in self.parameters())

    @property
    def num_params(self) -> int:
        """Total number of parameters."""
        return self._num_params

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity at (x, t).

        Args:
            t: (B,) or scalar timestep in [0, 1]
            x: (B, dim) current position

        Returns:
            v: (B, dim) velocity
        """
        # Handle scalar or mismatched batch size
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])

        # Embed timestep
        t_emb = self.time_embed(t)  # (B, hidden_dim)

        # Input projection
        h = self.input_proj(x)  # (B, dim)

        # Process through residual blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # Output velocity
        v = self.output_proj(h)

        return v

    @torch.no_grad()
    def integrate(
        self,
        x_0: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
        num_steps: int = 50,
        method: Literal["euler", "heun"] = "heun",
    ) -> torch.Tensor:
        """
        Integrate the ODE dx/dt = v(x, t) from t_start to t_end.

        Args:
            x_0: (B, dim) initial position (noise at t=0)
            t_start: Starting time
            t_end: Ending time
            num_steps: Number of integration steps
            method: Integration method ("euler" or "heun")

        Returns:
            x_T: (B, dim) final position
        """
        dt = (t_end - t_start) / num_steps
        x = x_0

        for i in range(num_steps):
            t = t_start + i * dt
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)

            if method == "euler":
                v = self.forward(t_tensor, x)
                x = x + dt * v
            elif method == "heun":
                # Heun's method (RK2) - more accurate
                v1 = self.forward(t_tensor, x)
                x_euler = x + dt * v1

                t_next = t + dt
                t_next_tensor = torch.full((x.shape[0],), t_next, device=x.device, dtype=x.dtype)
                v2 = self.forward(t_next_tensor, x_euler)

                x = x + dt * 0.5 * (v1 + v2)
            else:
                raise ValueError(f"Unknown method: {method}")

        return x

    @torch.no_grad()
    def get_manifold_velocity(
        self,
        x: torch.Tensor,
        t: float = 0.9,
    ) -> torch.Tensor:
        """
        Get velocity pointing towards manifold at time t.

        This is the "manifold force" used during optimization.
        At t=0.9 (near the data manifold), the velocity gives a
        meaningful direction towards valid instruction embeddings.

        Args:
            x: (B, dim) current position in embedding space
            t: Time at which to evaluate (default 0.9)

        Returns:
            v: (B, dim) velocity towards manifold
        """
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        return self.forward(t_tensor, x)

    @torch.no_grad()
    def project_to_manifold(
        self,
        x: torch.Tensor,
        num_steps: int = 10,
        t_current: float = 0.5,
    ) -> torch.Tensor:
        """
        Project a point onto the manifold by integrating to t=1.

        Useful for final refinement of optimized candidates.

        Args:
            x: (B, dim) point to project
            num_steps: Integration steps
            t_current: Assumed current time (how far along the flow)

        Returns:
            x_proj: (B, dim) projected point on manifold
        """
        return self.integrate(x, t_start=t_current, t_end=1.0, num_steps=num_steps)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Sample from the learned manifold by integrating from noise.

        Args:
            batch_size: Number of samples
            device: Target device
            num_steps: Integration steps

        Returns:
            samples: (B, dim) samples on the manifold
        """
        # Sample from standard normal (noise prior)
        x_0 = torch.randn(batch_size, self.dim, device=device)
        # Integrate to data manifold
        return self.integrate(x_0, t_start=0.0, t_end=1.0, num_steps=num_steps)


def create_manifold_keeper(config) -> ManifoldKeeperMLP:
    """Factory function to create ManifoldKeeper from config.

    Args:
        config: FlowPOHDConfig with mk_* parameters

    Returns:
        Initialized ManifoldKeeperMLP
    """
    model = ManifoldKeeperMLP(
        dim=config.sonar_dim,
        hidden_dim=config.mk_hidden_dim,
        time_dim=config.mk_time_dim,
        num_blocks=config.mk_num_blocks,
        dropout=config.mk_dropout,
    )

    print(f"ManifoldKeeper created: {model.num_params:,} parameters")
    print(f"  dim={config.sonar_dim}, hidden={config.mk_hidden_dim}, blocks={config.mk_num_blocks}")

    return model


if __name__ == "__main__":
    print("Testing ManifoldKeeperMLP...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create model
    model = ManifoldKeeperMLP(
        dim=1024,
        hidden_dim=2048,
        time_dim=256,
        num_blocks=3,
        dropout=0.1,
    ).to(device)

    print(f"\nParameters: {model.num_params:,} ({model.num_params / 1e6:.2f}M)")

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 1024, device=device)
    t = torch.rand(batch_size, device=device)

    print("\n--- Forward Pass ---")
    v = model(t, x)
    print(f"Input x: {x.shape}, t: {t.shape}")
    print(f"Output v: {v.shape}")
    print(f"v norm: {v.norm(dim=-1).mean():.4f}")

    # Test integration
    print("\n--- ODE Integration ---")
    x_0 = torch.randn(batch_size, 1024, device=device)
    x_1 = model.integrate(x_0, num_steps=20)
    print(f"x_0 norm: {x_0.norm(dim=-1).mean():.4f}")
    print(f"x_1 norm: {x_1.norm(dim=-1).mean():.4f}")

    # Test manifold velocity
    print("\n--- Manifold Velocity (t=0.9) ---")
    v_manifold = model.get_manifold_velocity(x, t=0.9)
    print(f"Manifold velocity norm: {v_manifold.norm(dim=-1).mean():.4f}")

    # Test sampling
    print("\n--- Sampling ---")
    samples = model.sample(4, device, num_steps=50)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample norms: {samples.norm(dim=-1)}")

    print("\n[OK] ManifoldKeeperMLP tests passed!")
