"""Set Transformer for permutation-invariant exemplar encoding.

Based on Lee et al., 2019: "Set Transformer: A Framework for
Attention-based Permutation-Invariant Neural Networks"

Key components:
- ISAB (Induced Set Attention Block): O(n) attention via inducing points
- PMA (Pooling by Multihead Attention): Variable-size input to fixed-size output
- ExemplarSetEncoder: GTR embeddings → 16D VAE latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiheadAttentionBlock(nn.Module):
    """Multihead Attention Block (MAB).

    MAB(X, Y) = LayerNorm(H + rFF(H))
    where H = LayerNorm(X + Multihead(X, Y, Y))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, n_query, embed_dim)
            key: (batch, n_key, embed_dim)
            key_padding_mask: (batch, n_key), True = ignore

        Returns:
            output: (batch, n_query, embed_dim)
        """
        # Self-attention with residual
        attn_out, _ = self.attn(
            query=query,
            key=key,
            value=key,
            key_padding_mask=key_padding_mask,
        )
        h = self.ln1(query + attn_out)

        # Feedforward with residual
        ff_out = self.ff(h)
        return self.ln2(h + ff_out)


class InducingSetAttentionBlock(nn.Module):
    """ISAB: Induced Set Attention Block.

    Uses M inducing points to reduce O(n²) attention to O(nM).

    ISAB(X) = MAB(X, H) where H = MAB(I, X)
    - I: Learned inducing points (M, embed_dim)
    - First MAB: inducing points attend to set
    - Second MAB: set attends to inducing representations
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_inducing: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_inducing = num_inducing
        self.hidden_dim = hidden_dim

        # Learned inducing points
        self.inducing_points = nn.Parameter(
            torch.randn(num_inducing, hidden_dim) * 0.02
        )

        # Input projection (if dimensions differ)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # MAB: I -> X (inducing attends to set)
        self.mab_i_to_x = MultiheadAttentionBlock(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # MAB: X -> H (set attends to inducing representations)
        self.mab_x_to_h = MultiheadAttentionBlock(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Set elements (batch, K, input_dim)
            mask: Valid element mask (batch, K), True = valid

        Returns:
            output: Encoded set (batch, K, hidden_dim)
        """
        batch_size = x.shape[0]
        K = x.shape[1]
        device = x.device

        # Project input
        x = self.input_proj(x)  # (batch, K, hidden_dim)

        # Handle empty samples in mixed batch
        # PyTorch MHA produces NaN when key_padding_mask is all True
        if mask is not None:
            # Find samples with no valid elements
            has_valid = mask.any(dim=1)  # (batch,)

            # If no valid elements in any sample, return zeros
            if not has_valid.any():
                return torch.zeros(batch_size, K, self.hidden_dim, device=device)

            # If mixed batch, process only non-empty samples
            if not has_valid.all():
                # Create output tensor
                output = torch.zeros(batch_size, K, self.hidden_dim, device=device)

                # Get indices of non-empty samples
                valid_idx = has_valid.nonzero(as_tuple=True)[0]

                # Process only non-empty samples
                x_valid = x[valid_idx]
                mask_valid = mask[valid_idx]

                # Recursive call with only valid samples
                output_valid = self._forward_impl(x_valid, mask_valid)

                # Scatter back
                output[valid_idx] = output_valid
                return output

        # Normal path: all samples have valid elements
        return self._forward_impl(x, mask)

    def _forward_impl(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal forward - assumes all samples have at least one valid element."""
        batch_size = x.shape[0]

        # Expand inducing points for batch
        I = self.inducing_points.unsqueeze(0).expand(batch_size, -1, -1)

        # Convert mask to key_padding_mask format (True = ignore)
        key_padding_mask = ~mask if mask is not None else None

        # MAB(I, X): inducing points attend to set
        H = self.mab_i_to_x(I, x, key_padding_mask=key_padding_mask)

        # MAB(X, H): set attends to inducing representations
        output = self.mab_x_to_h(x, H)

        return output


class PoolingMultiheadAttention(nn.Module):
    """PMA: Pooling by Multihead Attention.

    Produces fixed-size output from variable-size input.
    Uses learned seed vectors as queries.

    PMA(X) = MAB(S, rFF(X))
    where S are learned seed vectors.

    Optional instruction conditioning: seeds can be modulated by instruction embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_seeds: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        instruction_dim: int = 768,  # GTR embedding dimension
        use_instruction_conditioning: bool = True,
    ):
        super().__init__()
        self.num_seeds = num_seeds
        self.use_instruction_conditioning = use_instruction_conditioning

        # Learned seed vectors (queries for pooling)
        self.seeds = nn.Parameter(torch.randn(num_seeds, embed_dim) * 0.02)

        # Instruction conditioning: project instruction to seed modulation
        if use_instruction_conditioning:
            self.instruction_proj = nn.Sequential(
                nn.Linear(instruction_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        # Pre-pooling feedforward (rFF in paper)
        self.pre_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # Multihead attention for pooling
        self.mab = MultiheadAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        instruction_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Set elements (batch, K, embed_dim)
            mask: Valid element mask (batch, K), True = valid
            instruction_emb: Optional instruction embedding (batch, 768) for conditioning

        Returns:
            output: Pooled representation (batch, num_seeds, embed_dim)
        """
        batch_size = x.shape[0]
        embed_dim = x.shape[2]
        device = x.device

        # Handle empty samples - return zeros
        if mask is not None:
            has_valid = mask.any(dim=1)  # (batch,)

            if not has_valid.any():
                return torch.zeros(batch_size, self.num_seeds, embed_dim, device=device)

            if not has_valid.all():
                # Mixed batch - process only non-empty samples
                output = torch.zeros(batch_size, self.num_seeds, embed_dim, device=device)
                valid_idx = has_valid.nonzero(as_tuple=True)[0]

                x_valid = x[valid_idx]
                mask_valid = mask[valid_idx]
                inst_valid = instruction_emb[valid_idx] if instruction_emb is not None else None

                output_valid = self._forward_impl(x_valid, mask_valid, inst_valid)
                output[valid_idx] = output_valid
                return output

        return self._forward_impl(x, mask, instruction_emb)

    def _forward_impl(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        instruction_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal forward - assumes all samples have at least one valid element."""
        batch_size = x.shape[0]

        # Pre-pooling transform
        x = self.pre_ff(x)

        # Expand seeds for batch
        S = self.seeds.unsqueeze(0).expand(batch_size, -1, -1)

        # Condition seeds on instruction (if provided and enabled)
        if instruction_emb is not None and self.use_instruction_conditioning:
            # Project instruction to seed space and add as modulation
            inst_context = self.instruction_proj(instruction_emb)  # (batch, embed_dim)
            S = S + inst_context.unsqueeze(1)  # (batch, num_seeds, embed_dim)

        # Convert mask to key_padding_mask
        key_padding_mask = ~mask if mask is not None else None

        # Seeds attend to set
        output = self.mab(S, x, key_padding_mask=key_padding_mask)

        return output


class ExemplarSetEncoder(nn.Module):
    """Set Transformer encoder for variable-length exemplar sets.

    Architecture:
        GTR embeddings (K × 768) → ISAB₁ → ISAB₂ → PMA → MLP → (μ, logvar)

    Permutation invariant: order of exemplars doesn't affect encoding.
    Handles empty sets (K=0) with learned empty embedding.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        num_inducing: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # Two-layer Set Transformer
        self.isab1 = InducingSetAttentionBlock(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_inducing=num_inducing,
            dropout=dropout,
        )

        self.isab2 = InducingSetAttentionBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,  # 64
            num_heads=num_heads,
            num_inducing=num_inducing,
            dropout=dropout,
        )

        # Pooling to fixed-size
        self.pma = PoolingMultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_seeds=1,
            num_heads=num_heads,
            dropout=dropout,
        )

        # VAE heads
        intermediate_dim = hidden_dim // 2
        self.fc_mu = nn.Linear(intermediate_dim, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim, latent_dim)

        # Learned embedding for empty set
        self.empty_mu = nn.Parameter(torch.zeros(latent_dim))
        self.empty_logvar = nn.Parameter(torch.full((latent_dim,), -2.0))

    def forward(
        self,
        exemplar_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        instruction_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            exemplar_embeddings: GTR embeddings (batch, K, 768)
            mask: Validity mask (batch, K), True = valid exemplar
            instruction_emb: Optional instruction embedding (batch, 768) for conditioning

        Returns:
            mu, logvar: VAE parameters (batch, latent_dim)
        """
        batch_size = exemplar_embeddings.shape[0]
        K = exemplar_embeddings.shape[1]
        device = exemplar_embeddings.device

        # Handle empty set case (K=0)
        if K == 0:
            mu = self.empty_mu.unsqueeze(0).expand(batch_size, -1)
            logvar = self.empty_logvar.unsqueeze(0).expand(batch_size, -1)
            return mu, logvar

        # Determine which samples are empty (all mask False)
        if mask is not None:
            has_valid = mask.any(dim=1)  # (batch,) - True if sample has at least one valid exemplar
        else:
            has_valid = torch.ones(batch_size, dtype=torch.bool, device=device)

        # If all samples are empty, return empty embeddings
        if not has_valid.any():
            mu = self.empty_mu.unsqueeze(0).expand(batch_size, -1)
            logvar = self.empty_logvar.unsqueeze(0).expand(batch_size, -1)
            return mu, logvar

        # Set Transformer encoding
        # Note: ISAB and PMA now handle empty samples internally
        h = self.isab1(exemplar_embeddings, mask)  # (batch, K, hidden)
        h = self.isab2(h, mask)  # (batch, K, hidden/2)

        # Pool to fixed size (with optional instruction conditioning)
        pooled = self.pma(h, mask, instruction_emb=instruction_emb)  # (batch, 1, hidden/2)
        pooled = pooled.squeeze(1)  # (batch, hidden/2)

        # VAE parameters
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)

        # Handle mixed batch: replace empty samples with learned empty embedding
        if not has_valid.all():
            empty_mask = ~has_valid  # (batch,) - True for empty samples
            mu = torch.where(
                empty_mask.unsqueeze(1),
                self.empty_mu.unsqueeze(0).expand(batch_size, -1),
                mu,
            )
            logvar = torch.where(
                empty_mask.unsqueeze(1),
                self.empty_logvar.unsqueeze(0).expand(batch_size, -1),
                logvar,
            )

        return mu, logvar

    def encode_deterministic(
        self,
        exemplar_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode to mu only (for GP, no sampling)."""
        mu, _ = self.forward(exemplar_embeddings, mask)
        return mu
