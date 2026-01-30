"""
Latent-Query Transformer (Perceiver-style) Decoder for EcoFlow-BO.

This architecture is optimized for text embeddings where dimensions have no
spatial relationship (unlike image pixels). Key advantages:

1. No patching bias: Each of 768 output dimensions has its own learned query
2. Compute efficiency: Deep processing happens in cheap 48-token latent space
3. Semantic precision: Output queries learn "what does each GTR dimension mean"
4. Matryoshka-ready: Works naturally with masked z dimensions
5. Residual Latent: Supports z_full = [z_core (16D) | z_detail (32D)] = 48D

Architecture:
1. LatentExpander: z (48D) → 48 rich tokens (with learned position embeddings!)
2. LatentProcessor: Deep self-attention on 48 tokens (still cheap!)
3. CrossAttentionReadout: 768 learned queries attend to 48 tokens → 768D output

Position embeddings are CRITICAL: They encode dimension importance.
For Matryoshka, position 0 > position 1 > ... > position 15 (z_core ordering).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class PerceiverDecoderConfig:
    """Configuration for Latent-Query Transformer decoder.

    Supports residual latent: z_full = [z_core (16D) | z_detail (32D)] = 48D
    """
    latent_dim: int = 48  # Input z dimension (z_core + z_detail)
    output_dim: int = 768  # Output embedding dimension (GTR)
    hidden_size: int = 1024  # Token dimension (can be large - only 48 tokens!)
    depth: int = 12  # Number of self-attention layers in processor
    num_heads: int = 16  # Attention heads
    mlp_ratio: float = 4.0  # MLP hidden = hidden_size * mlp_ratio
    dropout: float = 0.1

    # Readout config
    readout_heads: int = 8  # Heads for cross-attention readout

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.hidden_size % self.readout_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"readout_heads ({self.readout_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {self.latent_dim}")


class LatentExpander(nn.Module):
    """
    Expand latent scalars into rich token vectors (48D → 48×1024 tokens).

    Each z dimension gets its own learned expansion + LEARNED POSITIONAL EMBEDDING.

    Position embeddings are CRITICAL:
    - They encode the "importance" of each dimension
    - For Matryoshka: pos_embed[0] encodes "most important", pos_embed[15] encodes "fine detail"
    - This allows the decoder to learn dimension-specific semantics

    Matryoshka-aware: tokens for z[i]=0 are zeroed out, ensuring masked
    dimensions don't contribute to cross-attention.
    """

    def __init__(self, latent_dim: int = 16, hidden_size: int = 1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        # Separate expander for each latent dimension (more expressive)
        self.expanders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(latent_dim)
        ])

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, latent_dim, hidden_size) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent codes [B, latent_dim], may have zeros for Matryoshka

        Returns:
            tokens: [B, latent_dim, hidden_size] with zeros for masked dims
        """
        B = z.shape[0]
        tokens = []

        for i in range(self.latent_dim):
            scalar = z[:, i:i+1]  # [B, 1]
            token = self.expanders[i](scalar)  # [B, hidden_size]
            tokens.append(token.unsqueeze(1))  # [B, 1, hidden_size]

        tokens = torch.cat(tokens, dim=1)  # [B, 16, hidden_size]
        tokens = tokens + self.pos_embed

        # Matryoshka masking: zero out tokens where z was zero
        z_mask = (z != 0).float().unsqueeze(-1)  # [B, 16, 1]
        tokens = tokens * z_mask

        return tokens


class ProcessorBlock(nn.Module):
    """
    Standard Transformer block for latent processing.

    Self-attention + MLP with pre-norm (more stable).
    Processes only 16 tokens - very efficient!
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent tokens [B, 16, hidden_size]

        Returns:
            x: Processed tokens [B, 16, hidden_size]
        """
        # Self-attention (16×16 - very cheap!)
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class CrossAttentionReadout(nn.Module):
    """
    Generate 768D output by having 768 learned queries attend to latent tokens.

    Each query learns: "What information from z do I need to produce dimension i?"
    This is semantically cleaner than patching - no artificial grouping.

    Attention cost: 768×16 (one-time, not per layer) - very efficient!
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        output_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size

        # 768 learned queries - each represents one output dimension
        self.output_queries = nn.Parameter(
            torch.randn(1, output_dim, hidden_size) * 0.02
        )

        # Cross-attention: queries attend to latent tokens
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Project each query's output to a single scalar value
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),  # Each query → 1 number
        )

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_tokens: Processed latent tokens [B, 16, hidden_size]

        Returns:
            embedding: Output embedding [B, 768]
        """
        B = latent_tokens.shape[0]

        # Expand queries for batch
        queries = self.output_queries.expand(B, -1, -1)  # [B, 768, hidden_size]

        # Cross-attention: 768 queries attend to 16 latent tokens
        # This is cheap: attention matrix is 768×16
        q = self.norm_q(queries)
        kv = self.norm_kv(latent_tokens)

        attended, _ = self.cross_attn(q, kv, kv, need_weights=False)
        # attended: [B, 768, hidden_size]

        # Residual connection
        attended = attended + queries

        # Project to scalar values
        values = self.output_mlp(attended).squeeze(-1)  # [B, 768]

        return values


class PerceiverDecoder(nn.Module):
    """
    Latent-Query Transformer: Perceiver-style decoder for EcoFlow-BO.

    Architecture:
    1. LatentExpander: z (16D) → 16 tokens
    2. LatentProcessor: Deep self-attention on 16 tokens (all the "thinking")
    3. CrossAttentionReadout: 768 queries → 768D embedding

    Why this is better than DiT for embeddings:
    - No patching bias (dimensions are treated independently)
    - Compute-efficient (deep processing on 16 tokens only)
    - Each output dimension has dedicated learned query
    - Naturally Matryoshka-compatible
    """

    def __init__(self, config: Optional[PerceiverDecoderConfig] = None):
        super().__init__()
        if config is None:
            config = PerceiverDecoderConfig()

        self.config = config

        # 1. Latent Expansion
        self.expander = LatentExpander(config.latent_dim, config.hidden_size)

        # 2. Deep Latent Processing
        # Can afford many layers because we only have 16 tokens!
        self.processor = nn.ModuleList([
            ProcessorBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.depth)
        ])

        # 3. Cross-Attention Readout
        self.readout = CrossAttentionReadout(
            hidden_size=config.hidden_size,
            num_heads=config.readout_heads,
            output_dim=config.output_dim,
            dropout=config.dropout,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Small init for output projection
        for m in self.readout.output_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z to embedding.

        Args:
            z: Latent codes [B, latent_dim]

        Returns:
            embedding: Reconstructed embedding [B, output_dim]
        """
        # 1. Expand z to rich latent tokens
        latents = self.expander(z)  # [B, 16, hidden_size]

        # 2. Deep processing (self-attention among 16 tokens)
        for block in self.processor:
            latents = block(latents)

        # 3. Generate output via query-based readout
        embedding = self.readout(latents)  # [B, 768]

        return embedding

    def forward_with_matryoshka(self, z: torch.Tensor, active_dims: int) -> torch.Tensor:
        """
        Forward with only first active_dims of z.

        Args:
            z: Full latent [B, latent_dim]
            active_dims: Number of active dimensions

        Returns:
            embedding: [B, output_dim]
        """
        z_masked = z.clone()
        z_masked[:, active_dims:] = 0.0
        return self.forward(z_masked)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the architecture
    config = PerceiverDecoderConfig(
        latent_dim=16,
        output_dim=768,
        hidden_size=1024,
        depth=12,
        num_heads=16,
    )

    model = PerceiverDecoder(config)
    print(f"PerceiverDecoder parameters: {count_parameters(model):,}")

    # Test forward pass
    z = torch.randn(4, 16)
    out = model(z)
    print(f"Input: {z.shape} → Output: {out.shape}")

    # Test Matryoshka masking
    z_masked = z.clone()
    z_masked[:, 8:] = 0.0
    out_masked = model(z_masked)
    print(f"Matryoshka (8D): {out_masked.shape}")

    # Verify masked tokens are handled
    z_4d = z.clone()
    z_4d[:, 4:] = 0.0
    out_4d = model(z_4d)
    print(f"Matryoshka (4D): {out_4d.shape}")
