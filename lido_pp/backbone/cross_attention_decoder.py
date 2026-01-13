"""
Cross-Attention Decoder for FlowPO.

ICAE-style memory slot projection replacing prefix tokens for high-fidelity
reconstruction conditioning. Instead of 4 prefix tokens that cannot influence
attention patterns, we project to 16 K,V pairs for position-specific cross-attention.

Key insight: Prefix tokens are limited because:
1. They compete for attention with actual content
2. Fixed positions lack flexibility for content-dependent conditioning
3. Only 4 tokens = severe information bottleneck

Memory slots solve these by:
1. Separate cross-attention pathway (no competition)
2. Position embeddings provide slot-specific semantics
3. 16 slots with 4096D hidden = 65K parameters of conditioning

Reference:
- ICAE: In-Context Autoencoder (ICLR 2024)
- DiT: Diffusion Transformer (ICCV 2023)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossAttentionProjector(nn.Module):
    """
    ICAE-style memory slot projection for cross-attention conditioning.

    Projects compact latent (128D) to K,V memory slots that condition
    the decoder via cross-attention rather than prefix token concatenation.

    Architecture:
    - Latent (128D) → Linear → 16 Keys (each 4096D)
    - Latent (128D) → Linear → 16 Values (each 4096D)
    - Learnable position embeddings added to each slot
    - LayerNorm for training stability

    Advantages over prefix tokens:
    - Position-specific conditioning (each slot attends differently)
    - No attention competition with content tokens
    - More expressive (16 × 4096 vs 4 × 4096)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 4096,
        num_memory_slots: int = 16,
        dropout: float = 0.1,
        use_gate: bool = True,
    ):
        """
        Initialize cross-attention projector.

        Args:
            latent_dim: Input latent dimension (128D from TFA)
            hidden_dim: Hidden dimension for K,V (match decoder dim, typically 4096)
            num_memory_slots: Number of memory slots (16 recommended)
            dropout: Dropout rate for regularization
            use_gate: Whether to use gating mechanism for K,V
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_memory_slots
        self.use_gate = use_gate

        # Projection layers: latent → K,V for all slots
        # Total output: num_slots × hidden_dim for each of K and V
        self.to_keys = nn.Linear(latent_dim, num_memory_slots * hidden_dim)
        self.to_values = nn.Linear(latent_dim, num_memory_slots * hidden_dim)

        # Learnable position embeddings for each slot
        # These give each slot a distinct "role" in conditioning
        self.pos_embed = nn.Parameter(
            torch.randn(num_memory_slots, hidden_dim) * 0.02
        )

        # Normalization layers for training stability
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.value_norm = nn.LayerNorm(hidden_dim)

        # Optional gating mechanism (like GLU)
        if use_gate:
            self.key_gate = nn.Linear(latent_dim, num_memory_slots * hidden_dim)
            self.value_gate = nn.Linear(latent_dim, num_memory_slots * hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"CrossAttentionProjector: {latent_dim}D → {num_memory_slots} slots × {hidden_dim}D"
        )

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        # Small initialization prevents large initial conditioning signal
        nn.init.xavier_uniform_(self.to_keys.weight, gain=0.1)
        nn.init.xavier_uniform_(self.to_values.weight, gain=0.1)
        nn.init.zeros_(self.to_keys.bias)
        nn.init.zeros_(self.to_values.bias)

        if self.use_gate:
            # Gate biases initialized to 0 (sigmoid(0) = 0.5)
            nn.init.xavier_uniform_(self.key_gate.weight, gain=0.1)
            nn.init.xavier_uniform_(self.value_gate.weight, gain=0.1)
            nn.init.zeros_(self.key_gate.bias)
            nn.init.zeros_(self.value_gate.bias)

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project latent to K,V memory slots for cross-attention.

        Args:
            latent: (B, latent_dim) compact latent representation

        Returns:
            keys: (B, num_slots, hidden_dim) key vectors for cross-attention
            values: (B, num_slots, hidden_dim) value vectors for cross-attention
        """
        B = latent.shape[0]

        # Project to keys: (B, num_slots × hidden_dim) → (B, num_slots, hidden_dim)
        keys = self.to_keys(latent).view(B, self.num_slots, self.hidden_dim)

        # Project to values
        values = self.to_values(latent).view(B, self.num_slots, self.hidden_dim)

        # Apply gating if enabled (GLU-style)
        if self.use_gate:
            key_gates = torch.sigmoid(
                self.key_gate(latent).view(B, self.num_slots, self.hidden_dim)
            )
            value_gates = torch.sigmoid(
                self.value_gate(latent).view(B, self.num_slots, self.hidden_dim)
            )
            keys = keys * key_gates
            values = values * value_gates

        # Add position embeddings (broadcast across batch)
        keys = keys + self.pos_embed.unsqueeze(0)
        values = values + self.pos_embed.unsqueeze(0)

        # Normalize for training stability
        keys = self.key_norm(keys)
        values = self.value_norm(values)

        # Apply dropout
        keys = self.dropout(keys)
        values = self.dropout(values)

        return keys, values


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for decoder integration.

    Implements multi-head cross-attention between decoder hidden states
    and memory slot K,V pairs from CrossAttentionProjector.

    This layer can be inserted into a frozen decoder (e.g., GritLM, LLaMA)
    to enable latent conditioning without modifying the base model.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention layer.

        Args:
            hidden_dim: Hidden dimension (must match decoder)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query projection for decoder hidden states
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention
        self.scale = self.head_dim ** -0.5

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, seq, hidden) to (B, num_heads, seq, head_dim)."""
        B, seq_len, _ = x.shape
        return x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, num_heads, seq, head_dim) to (B, seq, hidden)."""
        B, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-attention conditioning.

        Args:
            hidden_states: (B, seq_len, hidden_dim) decoder hidden states
            memory_keys: (B, num_slots, hidden_dim) from CrossAttentionProjector
            memory_values: (B, num_slots, hidden_dim) from CrossAttentionProjector
            attention_mask: Optional mask for memory slots

        Returns:
            (B, seq_len, hidden_dim) conditioned hidden states
        """
        # Pre-norm
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        # Reshape Q, K, V to multi-head format: (B, num_heads, seq, head_dim)
        Q = self._split_heads(self.q_proj(hidden_states))
        K = self._split_heads(memory_keys)
        V = self._split_heads(memory_values)

        # Compute attention scores: (B, num_heads, seq_len, num_slots)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )

        # Softmax and apply to values
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values and merge heads
        attn_output = self._merge_heads(torch.matmul(attn_probs, V))

        # Output projection with dropout
        attn_output = self.dropout(self.out_proj(attn_output))

        # Residual connection
        return residual + attn_output


class MemoryConditionedDecoder(nn.Module):
    """
    Wrapper that adds cross-attention conditioning to a frozen decoder.

    This module inserts CrossAttentionLayer modules between decoder layers
    to enable latent-based conditioning without modifying the base model.

    Usage:
    1. Load frozen decoder (e.g., GritLM text_decoder_embed_tokens)
    2. Wrap with MemoryConditionedDecoder
    3. Only train the cross-attention layers (via LoRA or full)
    """

    def __init__(
        self,
        cross_attention_projector: CrossAttentionProjector,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        num_cross_attn_layers: int = 4,
        insert_every_n_layers: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize memory-conditioned decoder wrapper.

        Args:
            cross_attention_projector: Projector that produces K,V from latent
            hidden_dim: Decoder hidden dimension
            num_heads: Number of attention heads
            num_cross_attn_layers: Number of cross-attention layers to insert
            insert_every_n_layers: Insert cross-attention every N decoder layers
            dropout: Dropout rate
        """
        super().__init__()
        self.projector = cross_attention_projector
        self.insert_every_n = insert_every_n_layers

        # Create cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_cross_attn_layers)
        ])

        logger.info(
            f"MemoryConditionedDecoder: {num_cross_attn_layers} cross-attn layers, "
            f"inserted every {insert_every_n_layers} decoder layers"
        )

    def get_memory_kv(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get memory K,V from latent for conditioning."""
        return self.projector(latent)

    def apply_cross_attention(
        self,
        hidden_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Apply cross-attention conditioning at specified layer.

        Args:
            hidden_states: Decoder hidden states
            memory_keys: Memory keys from projector
            memory_values: Memory values from projector
            layer_idx: Current decoder layer index

        Returns:
            Conditioned hidden states
        """
        # Determine which cross-attention layer to use
        cross_attn_idx = layer_idx // self.insert_every_n
        if cross_attn_idx < len(self.cross_attn_layers):
            hidden_states = self.cross_attn_layers[cross_attn_idx](
                hidden_states, memory_keys, memory_values
            )
        return hidden_states


def create_cross_attention_projector(
    latent_dim: int = 128,
    hidden_dim: int = 4096,
    num_slots: int = 16,
    **kwargs,
) -> CrossAttentionProjector:
    """
    Factory function to create cross-attention projector.

    Args:
        latent_dim: Input latent dimension (from TFA)
        hidden_dim: Decoder hidden dimension
        num_slots: Number of memory slots
        **kwargs: Additional arguments for CrossAttentionProjector

    Returns:
        Initialized CrossAttentionProjector
    """
    return CrossAttentionProjector(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_memory_slots=num_slots,
        **kwargs,
    )


if __name__ == "__main__":
    print("Testing Cross-Attention Decoder...")
    print()

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create projector
    projector = CrossAttentionProjector(
        latent_dim=128,
        hidden_dim=4096,
        num_memory_slots=16,
    ).to(device)

    # Test projection
    print("\n--- CrossAttentionProjector Test ---")
    latent = torch.randn(2, 128, device=device)
    keys, values = projector(latent)
    print(f"Input latent: {latent.shape}")
    print(f"Output keys: {keys.shape}")
    print(f"Output values: {values.shape}")

    # Verify shapes
    assert keys.shape == (2, 16, 4096), f"Keys shape mismatch: {keys.shape}"
    assert values.shape == (2, 16, 4096), f"Values shape mismatch: {values.shape}"

    # Test cross-attention layer
    print("\n--- CrossAttentionLayer Test ---")
    cross_attn = CrossAttentionLayer(
        hidden_dim=4096,
        num_heads=32,
    ).to(device)

    hidden_states = torch.randn(2, 64, 4096, device=device)  # (B, seq_len, hidden)
    output = cross_attn(hidden_states, keys, values)
    print(f"Input hidden states: {hidden_states.shape}")
    print(f"Output hidden states: {output.shape}")

    assert output.shape == hidden_states.shape, f"Output shape mismatch: {output.shape}"

    # Parameter count
    print("\n--- Parameter Counts ---")
    proj_params = sum(p.numel() for p in projector.parameters())
    attn_params = sum(p.numel() for p in cross_attn.parameters())
    print(f"CrossAttentionProjector: {proj_params:,} params")
    print(f"CrossAttentionLayer: {attn_params:,} params")

    print("\n[OK] Cross-Attention Decoder tests passed!")
