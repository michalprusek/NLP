"""Matryoshka Funnel Flow for hierarchical latent space compression.

Extends FunnelFlow with Matryoshka importance ordering - earlier dimensions
encode the most important information, enabling staged Bayesian optimization.

Architecture:
    768D GTR embedding
    → 6× AffineCoupling + Permutation (dimension-preserving)
    → MatryoshkaFunnelLayer (768D → 64D with multi-level decoders)
    → 4× AffineCoupling + Permutation (in 64D)
    → 64D importance-ordered latent (dim 0 > dim 1 > ... > dim 63)

Key insight: Multi-level decoders learn p(z[k:latent_dim] | z[0:k]) for each k
in matryoshka_dims[:-1], enabling reconstruction from partial latent prefixes
with quality degradation proportional to the prefix size.

References:
    - Matryoshka Representation Learning (NeurIPS 2022): https://arxiv.org/abs/2205.13147
    - Expected Coordinate Improvement (2024): https://arxiv.org/abs/2404.11917
    - Funnels: Exact MLE with Dimensionality Reduction (NeurIPS 2021)
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from vec2text_vae.funnel_flow import (
    AffineCouplingLayer,
    FunnelFlowOutput,
    MLP,
    PermutationLayer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Matryoshka Funnel Layer
# =============================================================================


class MatryoshkaFunnelLayer(nn.Module):
    """Surjective funnel layer with Matryoshka-aware multi-level decoders.

    Unlike the standard FunnelLayer which has a single decoder, this layer
    maintains per-level decoders that learn to reconstruct discarded dimensions
    from different prefix sizes.

    This enables:
    1. Reconstruction from partial latent (z[:16] → 768D, z[:32] → 768D, etc.)
    2. Importance ordering where smaller prefixes encode coarse semantics
    3. Staged Bayesian optimization where GP optimizes 16D at a time

    Forward pass: Keep first 64 dimensions, compute log probability of discarded
    Inverse pass: Use level-specific decoder based on active_dim to reconstruct
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 64,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64),
        hidden_dims: List[int] = [512, 512],
    ):
        """Initialize Matryoshka Funnel Layer.

        Args:
            input_dim: Input dimension (768 for GTR)
            output_dim: Output dimension (64 for latent)
            matryoshka_dims: Nested dimensions for multi-level decoding
            hidden_dims: Hidden layer sizes for decoder MLPs
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_discard = input_dim - output_dim
        self.matryoshka_dims = matryoshka_dims

        assert self.n_discard > 0, "Funnel must reduce dimensions"
        assert matryoshka_dims[-1] == output_dim, (
            f"Last matryoshka dim ({matryoshka_dims[-1]}) must equal output_dim ({output_dim})"
        )
        assert matryoshka_dims == tuple(sorted(matryoshka_dims)), (
            "matryoshka_dims must be in ascending order"
        )

        # Per-level decoders: each predicts p(z_discarded | z[:level])
        # Output: mean and log_std for the discarded 704 dimensions
        self.level_decoders = nn.ModuleDict()
        for level in matryoshka_dims[:-1]:  # 16, 32, 48 (not 64 - that's full)
            self.level_decoders[str(level)] = MLP(
                input_dim=level,
                hidden_dims=hidden_dims,
                output_dim=2 * self.n_discard,  # mean and log_std
            )
            logger.debug(f"MatryoshkaFunnelLayer: decoder for {level}D → {self.n_discard}D")

        # Full decoder (64D → 704D): standard FunnelLayer behavior
        self.full_decoder = MLP(
            input_dim=output_dim,
            hidden_dims=hidden_dims,
            output_dim=2 * self.n_discard,
        )

        logger.info(
            f"MatryoshkaFunnelLayer: {input_dim}D → {output_dim}D, "
            f"levels={matryoshka_dims}, discarding {self.n_discard}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: keep first output_dim dimensions.

        The log determinant is computed using the FULL decoder (64D) since
        during training we observe the complete latent.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            z_kept: Kept dimensions (batch, output_dim)
            log_det: Log determinant contribution (negative log prob of discarded)
        """
        z_kept = x[:, :self.output_dim]
        z_discarded = x[:, self.output_dim:]

        # Use full decoder for log_det computation during forward
        params = self.full_decoder(z_kept)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-6

        # Log probability of discarded dimensions
        dist = Normal(mean, std)
        log_prob = dist.log_prob(z_discarded).sum(dim=-1)

        # For surjective flows: log_det = -log p(z_discard | z_keep)
        log_det = -log_prob

        return z_kept, log_det

    def inverse(
        self,
        z_kept: torch.Tensor,
        active_dim: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Inverse pass with Matryoshka awareness.

        If active_dim is specified, uses the corresponding level decoder and
        zeros out dimensions beyond active_dim.

        Args:
            z_kept: Kept dimensions (batch, output_dim)
            active_dim: Active Matryoshka level (16, 32, 48, or 64/None for full)
            deterministic: Use mean instead of sampling

        Returns:
            x: Reconstructed full vector (batch, input_dim)
        """
        # Determine which decoder to use
        if active_dim is None or active_dim >= self.output_dim:
            # Use full decoder
            params = self.full_decoder(z_kept)
        elif active_dim in [int(d) for d in self.level_decoders.keys()]:
            # Use level-specific decoder
            # Zero out inactive dimensions for conditioning
            z_active = z_kept.clone()
            z_active[:, active_dim:] = 0.0
            params = self.level_decoders[str(active_dim)](z_active[:, :active_dim])
        else:
            # Find closest smaller level
            valid_levels = [int(d) for d in self.level_decoders.keys()]
            closest = max([l for l in valid_levels if l <= active_dim], default=valid_levels[-1])
            z_active = z_kept.clone()
            z_active[:, closest:] = 0.0
            params = self.level_decoders[str(closest)](z_active[:, :closest])

        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-6

        if deterministic:
            z_discarded = mean
        else:
            dist = Normal(mean, std)
            z_discarded = dist.rsample()

        x = torch.cat([z_kept, z_discarded], dim=-1)
        return x

    def get_level_params(
        self,
        z_kept: torch.Tensor,
        level: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get decoder parameters for a specific level.

        Useful for computing Matryoshka loss at different levels.

        Args:
            z_kept: Full kept dimensions (batch, output_dim)
            level: Matryoshka level (16, 32, 48, 64)

        Returns:
            mean, std: Decoder distribution parameters
        """
        if level >= self.output_dim:
            params = self.full_decoder(z_kept)
        else:
            z_active = z_kept.clone()
            z_active[:, level:] = 0.0
            params = self.level_decoders[str(level)](z_active[:, :level])

        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-6
        return mean, std


# =============================================================================
# Cascading Matryoshka Funnel Layer (Improved Architecture)
# =============================================================================


class CascadingMatryoshkaFunnelLayer(nn.Module):
    """Cascading surjective funnel with hierarchical level-to-level prediction.

    Instead of each level predicting ALL discarded dimensions (704D), this
    architecture predicts incrementally:
        z[:16] → predict z[16:32] (16D)
        z[:32] → predict z[32:48] (16D)
        z[:48] → predict z[48:64] (16D)
        z[:64] → predict z[64:768] (704D)

    Benefits:
    1. Each intermediate decoder only predicts 16 dimensions (simpler task)
    2. GP can iteratively refine: optimize z[0:16], predict z[16:32], refine, ...
    3. Hierarchical autoregressive structure enables better gradient flow
    4. Fewer parameters (uniform 16→16 decoders for intermediate levels)

    For LSBO, this enables a "predict-then-refine" loop:
        Stage 1: GP optimizes z[0:16]
        Predict: z[16:32] = decoder_16_32(z[0:16])
        Stage 2: GP refines z[16:32] starting from prediction
        Predict: z[32:48] = decoder_32_48(z[0:32])
        Stage 3: GP refines z[32:48] starting from prediction
        ...
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 64,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64),
        hidden_dims: List[int] = [256, 256],
        final_hidden_dims: List[int] = [512, 512],
    ):
        """Initialize Cascading Matryoshka Funnel Layer.

        Args:
            input_dim: Input dimension (768 for GTR)
            output_dim: Output dimension (64 for latent)
            matryoshka_dims: Nested dimensions for cascading prediction
            hidden_dims: Hidden dims for inter-level decoders (smaller)
            final_hidden_dims: Hidden dims for final 64→704 decoder (larger)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_discard = input_dim - output_dim
        self.matryoshka_dims = matryoshka_dims

        assert self.n_discard > 0, "Funnel must reduce dimensions"
        assert matryoshka_dims[-1] == output_dim
        assert matryoshka_dims == tuple(sorted(matryoshka_dims))

        # Cascading decoders: each level predicts the NEXT level only
        # z[:16] → z[16:32], z[:32] → z[32:48], z[:48] → z[48:64]
        self.cascade_decoders = nn.ModuleDict()
        for i, (level_in, level_out) in enumerate(zip(matryoshka_dims[:-1], matryoshka_dims[1:])):
            step_size = level_out - level_in
            self.cascade_decoders[f"{level_in}_{level_out}"] = MLP(
                input_dim=level_in,
                hidden_dims=hidden_dims,
                output_dim=2 * step_size,  # mean and log_std for step_size dims
            )
            logger.debug(f"CascadeDecoder: z[:{level_in}] → z[{level_in}:{level_out}] ({step_size}D)")

        # Final decoder: z[:64] → z[64:768] (704 dimensions)
        self.final_decoder = MLP(
            input_dim=output_dim,
            hidden_dims=final_hidden_dims,
            output_dim=2 * self.n_discard,
        )

        logger.info(
            f"CascadingMatryoshkaFunnelLayer: {input_dim}D → {output_dim}D, "
            f"cascade={matryoshka_dims}, final_discard={self.n_discard}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: keep first output_dim dimensions.

        Computes log_det using cascading factorization:
        log p(x) = log p(z[:64]) + log p(z[64:] | z[:64])
                 = log p(z[:16]) + log p(z[16:32] | z[:16]) + ... + log p(z[64:] | z[:64])

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            z_kept: Kept dimensions (batch, output_dim)
            log_det: Log determinant (sum of cascade log probs)
        """
        z_kept = x[:, :self.output_dim]
        z_discarded = x[:, self.output_dim:]

        log_det = torch.zeros(x.size(0), device=x.device)

        # Cascade log probabilities for latent dimensions
        for i, (level_in, level_out) in enumerate(zip(self.matryoshka_dims[:-1], self.matryoshka_dims[1:])):
            z_cond = x[:, :level_in]  # Conditioning prefix
            z_target = x[:, level_in:level_out]  # Target to predict

            params = self.cascade_decoders[f"{level_in}_{level_out}"](z_cond)
            mean, log_std = params.chunk(2, dim=-1)
            std = F.softplus(log_std) + 1e-6

            dist = Normal(mean, std)
            log_prob = dist.log_prob(z_target).sum(dim=-1)
            log_det = log_det - log_prob  # Negative because surjective

        # Final decoder log prob for discarded dimensions
        params = self.final_decoder(z_kept)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-6

        dist = Normal(mean, std)
        log_prob = dist.log_prob(z_discarded).sum(dim=-1)
        log_det = log_det - log_prob

        return z_kept, log_det

    def inverse(
        self,
        z_kept: torch.Tensor,
        active_dim: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Inverse pass with cascading reconstruction.

        If active_dim < 64, uses cascade decoders to fill in missing latent dims,
        then uses final decoder to reconstruct discarded 704 dims.

        Args:
            z_kept: Kept dimensions (batch, output_dim) - may have zeros beyond active_dim
            active_dim: Active Matryoshka level (16, 32, 48, or 64/None)
            deterministic: Use mean instead of sampling

        Returns:
            x: Reconstructed full vector (batch, input_dim)
        """
        batch_size = z_kept.size(0)
        device = z_kept.device

        # Start with the input z_kept (may have zeros beyond active_dim)
        z_full = z_kept.clone()

        # If active_dim specified, cascade-fill the rest of the latent
        if active_dim is not None and active_dim < self.output_dim:
            # Find which cascades to apply
            for level_in, level_out in zip(self.matryoshka_dims[:-1], self.matryoshka_dims[1:]):
                if level_in >= active_dim:
                    # This cascade starts at or after our active_dim
                    # We need to predict from the previous level
                    continue
                if level_out <= active_dim:
                    # This cascade ends before our active_dim, skip
                    continue

                # Apply cascade: z[:level_in] → z[level_in:level_out]
                z_cond = z_full[:, :level_in]
                params = self.cascade_decoders[f"{level_in}_{level_out}"](z_cond)
                mean, log_std = params.chunk(2, dim=-1)
                std = F.softplus(log_std) + 1e-6

                if deterministic:
                    z_pred = mean
                else:
                    z_pred = Normal(mean, std).rsample()

                # Only fill dimensions that are currently zero (beyond active_dim)
                fill_start = max(level_in, active_dim)
                fill_end = level_out
                if fill_start < fill_end:
                    pred_start = fill_start - level_in
                    pred_end = fill_end - level_in
                    z_full[:, fill_start:fill_end] = z_pred[:, pred_start:pred_end]

            # Now cascade-fill any remaining latent dims
            for level_in, level_out in zip(self.matryoshka_dims[:-1], self.matryoshka_dims[1:]):
                if level_in < active_dim:
                    continue  # Already handled above

                z_cond = z_full[:, :level_in]
                params = self.cascade_decoders[f"{level_in}_{level_out}"](z_cond)
                mean, log_std = params.chunk(2, dim=-1)
                std = F.softplus(log_std) + 1e-6

                if deterministic:
                    z_pred = mean
                else:
                    z_pred = Normal(mean, std).rsample()

                z_full[:, level_in:level_out] = z_pred

        # Final decoder: z[:64] → z[64:768]
        params = self.final_decoder(z_full)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-6

        if deterministic:
            z_discarded = mean
        else:
            z_discarded = Normal(mean, std).rsample()

        x = torch.cat([z_full, z_discarded], dim=-1)
        return x

    def predict_next_level(
        self,
        z_prefix: torch.Tensor,
        current_level: int,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Predict the next level dimensions from current prefix.

        This is the key method for LSBO predict-then-refine loop:
        1. GP optimizes z[0:16]
        2. Call predict_next_level(z[:16], 16) → get z[16:32]
        3. GP can refine z[16:32] starting from this prediction

        Args:
            z_prefix: Current latent prefix (batch, current_level)
            current_level: Current level (16, 32, or 48)
            deterministic: Use mean instead of sampling

        Returns:
            z_next: Predicted next level dimensions (batch, step_size)
        """
        # Find the next level
        try:
            idx = list(self.matryoshka_dims).index(current_level)
            next_level = self.matryoshka_dims[idx + 1]
        except (ValueError, IndexError):
            raise ValueError(f"current_level {current_level} not valid for cascade prediction")

        decoder_key = f"{current_level}_{next_level}"
        params = self.cascade_decoders[decoder_key](z_prefix)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-6

        if deterministic:
            return mean
        else:
            return Normal(mean, std).rsample()

    def cascade_fill(
        self,
        z_partial: torch.Tensor,
        start_level: int,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Fill all latent dimensions from a partial prefix using cascades.

        Args:
            z_partial: Partial latent (batch, start_level)
            start_level: Starting level (16, 32, 48)
            deterministic: Use mean instead of sampling

        Returns:
            z_full: Complete 64D latent
        """
        batch_size = z_partial.size(0)
        device = z_partial.device

        z_full = torch.zeros(batch_size, self.output_dim, device=device)
        z_full[:, :start_level] = z_partial

        # Apply cascades
        for level_in, level_out in zip(self.matryoshka_dims[:-1], self.matryoshka_dims[1:]):
            if level_in < start_level:
                continue

            z_next = self.predict_next_level(z_full[:, :level_in], level_in, deterministic)
            z_full[:, level_in:level_out] = z_next

        return z_full


# =============================================================================
# Cascading Encoder (Hierarchical - each level conditions on previous)
# =============================================================================


class CascadingEncoder(nn.Module):
    """Hierarchical cascading encoder for importance-ordered latent space.

    Unlike standard encoders that map input → latent in one shot, this encoder
    builds the latent hierarchically:

        z[:16] = encoder_16(x)                    # Coarsest representation
        z[16:32] = encoder_32(x, z[:16])          # Residual detail
        z[32:48] = encoder_48(x, z[:32])          # More detail
        ...

    Each level explicitly conditions on all previous levels, ensuring:
    1. z[:16] is a self-contained coarse representation
    2. z[16:32] adds detail that complements z[:16]
    3. GP can optimize level-by-level knowing each level is meaningful

    This is the key insight: the encoder mirrors the decoder's cascade structure.
    """

    def __init__(
        self,
        input_dim: int = 768,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64, 80, 96, 112, 128),
        hidden_dims: List[int] = [512, 256],
    ):
        """Initialize Cascading Encoder.

        Args:
            input_dim: Input dimension (768 for GTR)
            matryoshka_dims: Cascade levels (16, 32, 48, ...)
            hidden_dims: Hidden dimensions for encoder MLPs
        """
        super().__init__()
        self.input_dim = input_dim
        self.matryoshka_dims = matryoshka_dims
        self.output_dim = matryoshka_dims[-1]

        # First encoder: x → z[:first_level] (no conditioning)
        first_level = matryoshka_dims[0]
        self.first_encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=first_level,
        )

        # Subsequent encoders: (x, z[:prev_level]) → z[prev_level:curr_level]
        self.cascade_encoders = nn.ModuleDict()
        for level_in, level_out in zip(matryoshka_dims[:-1], matryoshka_dims[1:]):
            step_size = level_out - level_in
            # Input: original x + all previous z dimensions
            encoder_input_dim = input_dim + level_in
            self.cascade_encoders[f"{level_in}_{level_out}"] = MLP(
                input_dim=encoder_input_dim,
                hidden_dims=hidden_dims,
                output_dim=step_size,
            )

        logger.info(
            f"CascadingEncoder: {input_dim}D → {self.output_dim}D, "
            f"levels={matryoshka_dims}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input hierarchically.

        Args:
            x: Input embeddings (batch, input_dim)

        Returns:
            z: Hierarchically-built latent (batch, output_dim)
        """
        # First level: coarsest representation
        z_parts = [self.first_encoder(x)]

        # Build rest hierarchically
        for level_in, level_out in zip(self.matryoshka_dims[:-1], self.matryoshka_dims[1:]):
            # Concatenate x with all z so far
            z_so_far = torch.cat(z_parts, dim=-1)  # (batch, level_in)
            encoder_input = torch.cat([x, z_so_far], dim=-1)  # (batch, input_dim + level_in)

            # Encode next level
            z_next = self.cascade_encoders[f"{level_in}_{level_out}"](encoder_input)
            z_parts.append(z_next)

        return torch.cat(z_parts, dim=-1)

    def encode_to_level(self, x: torch.Tensor, target_level: int) -> torch.Tensor:
        """Encode only up to a specific level.

        Useful for partial encoding during LSBO.

        Args:
            x: Input embeddings (batch, input_dim)
            target_level: Target Matryoshka level (16, 32, 48, ...)

        Returns:
            z_partial: Partial latent (batch, target_level)
        """
        z_parts = [self.first_encoder(x)]
        current_level = self.matryoshka_dims[0]

        if current_level >= target_level:
            return z_parts[0][:, :target_level]

        for level_in, level_out in zip(self.matryoshka_dims[:-1], self.matryoshka_dims[1:]):
            if level_in >= target_level:
                break

            z_so_far = torch.cat(z_parts, dim=-1)
            encoder_input = torch.cat([x, z_so_far], dim=-1)
            z_next = self.cascade_encoders[f"{level_in}_{level_out}"](encoder_input)
            z_parts.append(z_next)

            if level_out >= target_level:
                break

        z_full = torch.cat(z_parts, dim=-1)
        return z_full[:, :target_level]


# =============================================================================
# Cascading Decoder (mirrors encoder structure)
# =============================================================================


class CascadingDecoder(nn.Module):
    """Hierarchical cascading decoder that mirrors CascadingEncoder.

    Decodes from latent back to input space, with ability to decode from
    partial latent (using cascade prediction for missing levels).

        x_hat = decoder_final(z[:output_dim])    # Full latent → embedding

    For partial decoding (z[:16] only):
        z[16:32] = predict_32(z[:16])            # Predict missing
        z[32:48] = predict_48(z[:32])            # Predict more
        ...
        x_hat = decoder_final(z_completed)       # Decode completed latent
    """

    def __init__(
        self,
        output_dim: int = 768,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64, 80, 96, 112, 128),
        hidden_dims: List[int] = [256, 512],
        predictor_hidden_dims: List[int] = [256, 256],
    ):
        """Initialize Cascading Decoder.

        Args:
            output_dim: Output dimension (768 for GTR)
            matryoshka_dims: Cascade levels
            hidden_dims: Hidden dims for main decoder
            predictor_hidden_dims: Hidden dims for level predictors
        """
        super().__init__()
        self.output_dim = output_dim
        self.matryoshka_dims = matryoshka_dims
        self.latent_dim = matryoshka_dims[-1]

        # Level predictors: z[:level_in] → z[level_in:level_out]
        self.level_predictors = nn.ModuleDict()
        for level_in, level_out in zip(matryoshka_dims[:-1], matryoshka_dims[1:]):
            step_size = level_out - level_in
            self.level_predictors[f"{level_in}_{level_out}"] = MLP(
                input_dim=level_in,
                hidden_dims=predictor_hidden_dims,
                output_dim=step_size * 2,  # mean and log_std for sampling
            )

        # Final decoder: full latent → output
        self.final_decoder = MLP(
            input_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

        logger.info(
            f"CascadingDecoder: {self.latent_dim}D → {output_dim}D, "
            f"levels={matryoshka_dims}"
        )

    def forward(
        self,
        z: torch.Tensor,
        active_dim: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Decode latent to output.

        If active_dim specified, completes missing dimensions using predictors.

        Args:
            z: Latent (batch, latent_dim) - may have zeros beyond active_dim
            active_dim: Active dimensions (rest will be predicted)
            deterministic: Use mean for predictions

        Returns:
            x_hat: Reconstructed output (batch, output_dim)
        """
        # Complete latent if needed
        if active_dim is not None and active_dim < self.latent_dim:
            z = self.complete_latent(z, active_dim, deterministic)

        return self.final_decoder(z)

    def complete_latent(
        self,
        z_partial: torch.Tensor,
        active_dim: int,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Complete partial latent using cascade predictors.

        Args:
            z_partial: Partial latent (batch, latent_dim) with zeros beyond active_dim
            active_dim: Number of active (non-zero) dimensions
            deterministic: Use mean instead of sampling

        Returns:
            z_complete: Completed latent (batch, latent_dim)
        """
        # Collect all parts to concatenate at the end (avoids in-place operations)
        z_parts = [z_partial[:, :active_dim]]

        for level_in, level_out in zip(self.matryoshka_dims[:-1], self.matryoshka_dims[1:]):
            if level_in < active_dim:
                continue  # This level is already filled

            # Predict this level from all previous
            z_so_far = torch.cat(z_parts, dim=-1)
            params = self.level_predictors[f"{level_in}_{level_out}"](z_so_far)
            mean, log_std = params.chunk(2, dim=-1)

            if deterministic:
                z_pred = mean
            else:
                std = F.softplus(log_std) + 1e-6
                z_pred = Normal(mean, std).rsample()

            z_parts.append(z_pred)

        return torch.cat(z_parts, dim=-1)

    def predict_next_level(
        self,
        z_prefix: torch.Tensor,
        current_level: int,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Predict next level dimensions.

        Args:
            z_prefix: Current latent prefix (batch, current_level)
            current_level: Current level
            deterministic: Use mean

        Returns:
            z_next: Next level dimensions (batch, step_size)
        """
        idx = list(self.matryoshka_dims).index(current_level)
        next_level = self.matryoshka_dims[idx + 1]

        params = self.level_predictors[f"{current_level}_{next_level}"](z_prefix)
        mean, log_std = params.chunk(2, dim=-1)

        if deterministic:
            return mean
        else:
            std = F.softplus(log_std) + 1e-6
            return Normal(mean, std).rsample()


# =============================================================================
# Full Cascading Matryoshka Flow (Encoder + Decoder)
# =============================================================================


class CascadingMatryoshkaGTRFunnelFlow(nn.Module):
    """Fully cascading Matryoshka Flow with hierarchical encoder AND decoder.

    Key innovation: Both encoder and decoder are cascading, ensuring:
    1. z[:16] is a self-contained coarse representation (encoder builds it first)
    2. z[16:32] adds detail conditioned on z[:16] (encoder conditions on previous)
    3. Decoder can reconstruct from any prefix using cascade prediction

    Architecture:
        ENCODE:
        768D → encoder_16(x) → z[:16]
             → encoder_32(x, z[:16]) → z[16:32]
             → encoder_48(x, z[:32]) → z[32:48]
             → ... → z[:128]

        DECODE:
        z[:k] → predict z[k:k+16] → predict z[k+16:k+32] → ... → z[:128]
        z[:128] → final_decoder → 768D

    For LSBO:
        Stage 1: GP optimizes z[:16] (encoder_16 output is meaningful alone!)
        Predict: z[16:32] from z[:16]
        Stage 2: GP refines z[16:32]
        ...
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 128,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64, 80, 96, 112, 128),
        encoder_hidden_dims: List[int] = [512, 256],
        decoder_hidden_dims: List[int] = [256, 512],
        predictor_hidden_dims: List[int] = [256, 256],
    ):
        """Initialize Cascading Matryoshka Flow.

        Args:
            input_dim: Input dimension (768)
            latent_dim: Latent dimension (128)
            matryoshka_dims: Cascade levels
            encoder_hidden_dims: Hidden dims for encoder MLPs
            decoder_hidden_dims: Hidden dims for final decoder
            predictor_hidden_dims: Hidden dims for level predictors
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.matryoshka_dims = matryoshka_dims

        assert matryoshka_dims[-1] == latent_dim, (
            f"Last matryoshka dim ({matryoshka_dims[-1]}) must equal latent_dim ({latent_dim})"
        )

        # Cascading encoder
        self.encoder = CascadingEncoder(
            input_dim=input_dim,
            matryoshka_dims=matryoshka_dims,
            hidden_dims=encoder_hidden_dims,
        )

        # Cascading decoder
        self.decoder = CascadingDecoder(
            output_dim=input_dim,
            matryoshka_dims=matryoshka_dims,
            hidden_dims=decoder_hidden_dims,
            predictor_hidden_dims=predictor_hidden_dims,
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"CascadingMatryoshkaGTRFunnelFlow: {input_dim}D → {latent_dim}D, "
            f"cascade={matryoshka_dims}, {n_params:,} parameters"
        )

    def forward(self, x: torch.Tensor) -> FunnelFlowOutput:
        """Forward pass: encode to latent.

        Args:
            x: Input embeddings (batch, input_dim)

        Returns:
            FunnelFlowOutput with z and placeholder log_prob
        """
        z = self.encoder(x)

        # Simple regularization: encourage standard normal
        log_prob = -0.5 * (z ** 2).sum(dim=-1)

        return FunnelFlowOutput(z=z, log_det=torch.zeros_like(log_prob), log_prob=log_prob)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent."""
        return self.encoder(x)

    def encode_to_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Encode only up to a specific Matryoshka level."""
        return self.encoder.encode_to_level(x, level)

    def decode(
        self,
        z: torch.Tensor,
        active_dim: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Decode latent to output.

        Args:
            z: Latent (batch, latent_dim)
            active_dim: Active dimensions (predicts rest if < latent_dim)
            deterministic: Use mean for predictions

        Returns:
            x_hat: Reconstructed output (batch, input_dim)
        """
        return self.decoder(z, active_dim=active_dim, deterministic=deterministic)

    def reconstruct(
        self,
        x: torch.Tensor,
        active_dim: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Full reconstruction pipeline."""
        z = self.encode(x)

        if active_dim is not None and active_dim < self.latent_dim:
            z_masked = z.clone()
            z_masked[:, active_dim:] = 0.0
            return self.decode(z_masked, active_dim=active_dim, deterministic=deterministic)

        return self.decode(z, deterministic=deterministic)

    def predict_and_decode(
        self,
        z_partial: torch.Tensor,
        start_level: int,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict missing dims and decode.

        Key method for LSBO warm-start:
        1. GP optimized z[:start_level]
        2. This predicts z[start_level:] and decodes

        Args:
            z_partial: Partial latent (batch, start_level)
            start_level: Where known latent ends
            deterministic: Use mean for predictions

        Returns:
            z_full: Completed latent (batch, latent_dim)
            x_hat: Decoded output (batch, input_dim)
        """
        # Pad to full latent dim
        batch_size = z_partial.size(0)
        z_padded = torch.zeros(batch_size, self.latent_dim, device=z_partial.device)
        z_padded[:, :start_level] = z_partial

        # Complete and decode
        z_full = self.decoder.complete_latent(z_padded, start_level, deterministic)
        x_hat = self.decoder.final_decoder(z_full)

        return z_full, x_hat

    def get_warm_start_for_level(
        self,
        z_prefix: torch.Tensor,
        target_level: int,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Get warm start for GP optimization at next level.

        Args:
            z_prefix: Current optimized prefix (batch, current_level)
            target_level: Target level to predict

        Returns:
            z_next: Predicted values for next level (batch, step_size)
        """
        current_level = z_prefix.size(-1)

        # Find the cascade step
        for level_in, level_out in zip(self.matryoshka_dims[:-1], self.matryoshka_dims[1:]):
            if level_in == current_level and level_out == target_level:
                return self.decoder.predict_next_level(z_prefix, current_level, deterministic)

        raise ValueError(f"No cascade from {current_level} to {target_level}")


# =============================================================================
# Matryoshka GTR Funnel Flow (Original - kept for backward compatibility)
# =============================================================================


class MatryoshkaGTRFunnelFlow(nn.Module):
    """Funnel Flow with Matryoshka importance ordering for GTR embeddings.

    Extends GTRFunnelFlow by replacing FunnelLayer with MatryoshkaFunnelLayer,
    enabling reconstruction from partial latent prefixes.

    Architecture:
        Input: 768D GTR embedding
        → Pre-funnel bijective layers (768D)
        → MatryoshkaFunnelLayer (768D → 64D with multi-level decoders)
        → Post-funnel bijective layers (64D)
        Output: 64D importance-ordered latent

    Training objective: Multi-scale reconstruction loss + NLL
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 64,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64),
        n_pre_layers: int = 6,
        n_post_layers: int = 4,
        hidden_dims: List[int] = [512, 512],
    ):
        """Initialize Matryoshka GTR Funnel Flow.

        Args:
            input_dim: GTR embedding dimension (768)
            latent_dim: Latent space dimension (64)
            matryoshka_dims: Nested dimensions for importance ordering
            n_pre_layers: Number of bijective layers before funnel
            n_post_layers: Number of bijective layers after funnel
            hidden_dims: Hidden layer sizes for flow components
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.matryoshka_dims = matryoshka_dims

        # Pre-funnel bijective layers (in input_dim space)
        self.pre_layers = nn.ModuleList()
        for _ in range(n_pre_layers):
            self.pre_layers.append(AffineCouplingLayer(input_dim, hidden_dims))
            self.pre_layers.append(PermutationLayer(input_dim))

        # Matryoshka-aware funnel layer
        self.funnel = MatryoshkaFunnelLayer(
            input_dim=input_dim,
            output_dim=latent_dim,
            matryoshka_dims=matryoshka_dims,
            hidden_dims=hidden_dims,
        )

        # Post-funnel bijective layers (in latent_dim space)
        self.post_layers = nn.ModuleList()
        for _ in range(n_post_layers):
            self.post_layers.append(AffineCouplingLayer(latent_dim, hidden_dims[:1]))
            self.post_layers.append(PermutationLayer(latent_dim))

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MatryoshkaGTRFunnelFlow: {input_dim}D → {latent_dim}D, "
            f"levels={matryoshka_dims}, {n_pre_layers} pre + {n_post_layers} post layers, "
            f"{n_params:,} parameters"
        )

    def forward(self, x: torch.Tensor) -> FunnelFlowOutput:
        """Forward pass: embed → latent.

        Args:
            x: GTR embeddings (batch, 768)

        Returns:
            FunnelFlowOutput with z, log_det, log_prob
        """
        log_det_total = torch.zeros(x.size(0), device=x.device)

        # Pre-funnel bijective layers
        z = x
        for layer in self.pre_layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det

        # Matryoshka funnel (dimension reduction)
        z, log_det = self.funnel(z)
        log_det_total = log_det_total + log_det

        # Post-funnel bijective layers
        for layer in self.post_layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det

        # Log probability under standard normal prior
        log_prob_prior = Normal(0, 1).log_prob(z).sum(dim=-1)

        # Total log probability
        log_prob = log_prob_prior + log_det_total

        return FunnelFlowOutput(z=z, log_det=log_det_total, log_prob=log_prob)

    def inverse(
        self,
        z: torch.Tensor,
        active_dim: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Inverse pass: latent → embed with Matryoshka awareness.

        Args:
            z: Latent vectors (batch, latent_dim)
            active_dim: Active Matryoshka level (affects funnel inverse)
            deterministic: Use mean for stochastic components

        Returns:
            Reconstructed embeddings (batch, 768)
        """
        # Inverse post-funnel layers (reverse order)
        x = z
        for layer in reversed(self.post_layers):
            x = layer.inverse(x)

        # Inverse funnel with Matryoshka awareness
        x = self.funnel.inverse(x, active_dim=active_dim, deterministic=deterministic)

        # Inverse pre-funnel layers (reverse order)
        for layer in reversed(self.pre_layers):
            x = layer.inverse(x)

        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode embeddings to latent (convenience method)."""
        return self.forward(x).z

    def decode(
        self,
        z: torch.Tensor,
        active_dim: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Decode latent to embeddings with Matryoshka awareness.

        Args:
            z: Latent vectors (batch, latent_dim)
            active_dim: Active Matryoshka level (16, 32, 48, or None for full)
            deterministic: Use mean for stochastic components

        Returns:
            Reconstructed embeddings (batch, 768)
        """
        return self.inverse(z, active_dim=active_dim, deterministic=deterministic)

    def reconstruct(
        self,
        x: torch.Tensor,
        active_dim: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Full reconstruction with optional Matryoshka level."""
        z = self.encode(x)

        # Zero out dimensions beyond active_dim if specified
        if active_dim is not None and active_dim < self.latent_dim:
            z = z.clone()
            z[:, active_dim:] = 0.0

        return self.decode(z, active_dim=active_dim, deterministic=deterministic)

    def get_intermediate_after_funnel(self, x: torch.Tensor) -> torch.Tensor:
        """Get representation after funnel but before post-layers.

        Useful for computing per-level loss on the "raw" funneled representation.
        """
        z = x
        for layer in self.pre_layers:
            z, _ = layer(z)
        z, _ = self.funnel(z)
        return z


# =============================================================================
# Matryoshka Funnel Loss
# =============================================================================


class MatryoshkaFunnelLoss(nn.Module):
    """Multi-scale loss for Matryoshka Funnel Flow training.

    Computes reconstruction loss at each Matryoshka level with importance
    weighting - smaller prefixes get higher weights to enforce importance
    ordering.

    Loss = Σ_k weight_k * (1 - cos_sim(x, decode(z[:k])))

    Default weights with decay=0.5:
        16D → 0.533 (most important - must capture core semantics)
        32D → 0.267
        48D → 0.133
        64D → 0.067 (full reconstruction)
    """

    def __init__(
        self,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64),
        decay: float = 0.5,
        nll_weight: float = 0.1,
    ):
        """Initialize Matryoshka loss.

        Args:
            matryoshka_dims: Nested dimensions
            decay: Geometric decay factor for weights (smaller = more emphasis on small prefixes)
            nll_weight: Weight for negative log-likelihood term
        """
        super().__init__()
        self.matryoshka_dims = matryoshka_dims
        self.decay = decay
        self.nll_weight = nll_weight

        # Compute weights with geometric decay
        # Smaller dimensions get higher weights
        n_levels = len(matryoshka_dims)
        raw_weights = [decay ** i for i in range(n_levels)]
        total = sum(raw_weights)
        self.weights = {dim: w / total for dim, w in zip(matryoshka_dims, raw_weights)}

        weights_str = ", ".join([f"{k}: {v:.3f}" for k, v in self.weights.items()])
        logger.info(
            f"MatryoshkaFunnelLoss: dims={matryoshka_dims}, decay={decay}, "
            f"weights={{{weights_str}}}"
        )

    def forward(
        self,
        flow: MatryoshkaGTRFunnelFlow,
        x: torch.Tensor,
        current_decay: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-scale Matryoshka loss.

        Args:
            flow: Matryoshka Funnel Flow model
            x: Input embeddings (batch, 768)
            current_decay: Override decay for progressive scheduling

        Returns:
            total_loss: Combined loss
            metrics: Per-level metrics
        """
        # Encode to latent
        output = flow.forward(x)
        z = output.z

        # Compute weights (possibly with updated decay)
        if current_decay is not None and current_decay != self.decay:
            n_levels = len(self.matryoshka_dims)
            raw_weights = [current_decay ** i for i in range(n_levels)]
            total = sum(raw_weights)
            weights = {dim: w / total for dim, w in zip(self.matryoshka_dims, raw_weights)}
        else:
            weights = self.weights

        # Multi-scale reconstruction loss
        recon_loss = torch.tensor(0.0, device=x.device)
        metrics = {}

        for level in self.matryoshka_dims:
            weight = weights[level]

            # Mask latent beyond active level
            z_masked = z.clone()
            if level < flow.latent_dim:
                z_masked[:, level:] = 0.0

            # Decode with level-specific decoder
            x_recon = flow.decode(z_masked, active_dim=level, deterministic=True)

            # Cosine similarity loss
            cos_sim = F.cosine_similarity(x, x_recon, dim=-1)
            level_loss = (1 - cos_sim).mean()

            recon_loss = recon_loss + weight * level_loss
            metrics[f'cos_sim_{level}D'] = cos_sim.mean().item()
            metrics[f'loss_{level}D'] = level_loss.item()

        # NLL loss (optional regularization)
        nll_loss = -output.log_prob.mean()
        metrics['nll'] = nll_loss.item()

        # Total loss
        total_loss = recon_loss + self.nll_weight * nll_loss
        metrics['total_loss'] = total_loss.item()
        metrics['recon_loss'] = recon_loss.item()

        return total_loss, metrics


# =============================================================================
# Cascading Matryoshka Funnel Loss
# =============================================================================


class CascadingMatryoshkaFunnelLoss(nn.Module):
    """Loss for Cascading Matryoshka Funnel Flow training.

    Unlike MatryoshkaFunnelLoss which evaluates reconstruction at each level,
    this loss focuses on:
    1. Cascade prediction accuracy (how well each level predicts the next)
    2. Final reconstruction quality
    3. Progressive importance weighting

    This encourages the model to learn good autoregressive structure in latent space.
    """

    def __init__(
        self,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64, 80, 96, 112, 128),
        cascade_weight: float = 1.0,
        recon_weight: float = 1.0,
        nll_weight: float = 0.01,
        progressive_cascade: bool = True,
    ):
        """Initialize Cascading loss.

        Args:
            matryoshka_dims: Nested dimensions for cascading
            cascade_weight: Weight for cascade prediction loss
            recon_weight: Weight for final reconstruction loss
            nll_weight: Weight for NLL regularization
            progressive_cascade: Weight earlier cascades more heavily
        """
        super().__init__()
        self.matryoshka_dims = matryoshka_dims
        self.cascade_weight = cascade_weight
        self.recon_weight = recon_weight
        self.nll_weight = nll_weight
        self.progressive_cascade = progressive_cascade

        # Cascade weights: earlier cascades more important
        n_cascades = len(matryoshka_dims) - 1
        if progressive_cascade:
            # Exponential decay: first cascade most important
            raw = [0.7 ** i for i in range(n_cascades)]
            total = sum(raw)
            self.cascade_weights = [w / total for w in raw]
        else:
            self.cascade_weights = [1.0 / n_cascades] * n_cascades

        logger.info(
            f"CascadingMatryoshkaFunnelLoss: dims={matryoshka_dims}, "
            f"cascade_weights={[f'{w:.3f}' for w in self.cascade_weights]}"
        )

    def forward(
        self,
        flow: "CascadingMatryoshkaGTRFunnelFlow",
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute cascading loss.

        Args:
            flow: Cascading Matryoshka Flow model (with CascadingEncoder/Decoder)
            x: Input embeddings (batch, 768)

        Returns:
            total_loss: Combined loss
            metrics: Detailed metrics
        """
        # Forward pass - encode hierarchically
        output = flow.forward(x)
        z = output.z

        metrics = {}

        # 1. Cascade prediction loss
        # Decoder should predict next level from prefix
        cascade_loss = torch.tensor(0.0, device=x.device)

        for i, (level_in, level_out) in enumerate(
            zip(self.matryoshka_dims[:-1], self.matryoshka_dims[1:])
        ):
            # Get ground truth for this cascade step (from encoder)
            z_target = z[:, level_in:level_out].detach()

            # Predict using decoder's level predictor
            z_pred = flow.decoder.predict_next_level(
                z[:, :level_in], level_in, deterministic=True
            )

            # MSE loss for cascade prediction
            cascade_mse = F.mse_loss(z_pred, z_target)
            cascade_loss = cascade_loss + self.cascade_weights[i] * cascade_mse

            metrics[f'cascade_{level_in}_{level_out}_mse'] = cascade_mse.item()

        metrics['cascade_loss'] = cascade_loss.item()

        # 2. Multi-level reconstruction loss
        # Evaluate reconstruction from different starting points
        recon_loss = torch.tensor(0.0, device=x.device)
        n_levels = len(self.matryoshka_dims)
        latent_dim = self.matryoshka_dims[-1]

        for i, level in enumerate(self.matryoshka_dims):
            # Decode from this level (decoder completes the rest)
            # Use concatenation instead of in-place assignment to avoid gradient issues
            if level < latent_dim:
                z_partial = torch.cat([
                    z[:, :level],
                    torch.zeros(z.size(0), latent_dim - level, device=z.device, dtype=z.dtype)
                ], dim=-1)
            else:
                z_partial = z
            x_recon = flow.decode(z_partial, active_dim=level, deterministic=True)

            # Cosine similarity
            cos_sim = F.cosine_similarity(x, x_recon, dim=-1)
            level_loss = (1 - cos_sim).mean()

            # Progressive weight: later levels slightly more important for recon
            weight = (i + 1) / sum(range(1, n_levels + 1))
            recon_loss = recon_loss + weight * level_loss

            metrics[f'cos_sim_{level}D'] = cos_sim.mean().item()

        metrics['recon_loss'] = recon_loss.item()

        # 3. NLL regularization
        nll_loss = -output.log_prob.mean()
        metrics['nll'] = nll_loss.item()

        # Total loss
        total_loss = (
            self.cascade_weight * cascade_loss +
            self.recon_weight * recon_loss +
            self.nll_weight * nll_loss
        )
        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


# =============================================================================
# Progressive Training Scheduler
# =============================================================================


class ProgressiveMatryoshkaScheduler:
    """Progressive training scheduler for Matryoshka importance ordering.

    Training phases:
    1. Warmup (epochs 0-30): Equal weights on all levels (decay=1.0)
    2. Progressive (epochs 30-70): Gradually increase importance of smaller prefixes
    3. Full ordering (epochs 70+): Final weights (decay=0.5)

    This prevents the model from collapsing to only using larger prefixes early
    in training, ensuring all level decoders learn meaningful representations.
    """

    def __init__(
        self,
        total_epochs: int = 100,
        warmup_epochs: int = 30,
        progressive_epochs: int = 40,
        initial_decay: float = 1.0,
        final_decay: float = 0.5,
    ):
        """Initialize progressive scheduler.

        Args:
            total_epochs: Total training epochs
            warmup_epochs: Epochs with equal weights
            progressive_epochs: Epochs to transition from initial to final decay
            initial_decay: Starting decay (1.0 = equal weights)
            final_decay: Final decay (0.5 = strong importance ordering)
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.progressive_epochs = progressive_epochs
        self.initial_decay = initial_decay
        self.final_decay = final_decay

        logger.info(
            f"ProgressiveMatryoshkaScheduler: warmup={warmup_epochs}, "
            f"progressive={progressive_epochs}, decay: {initial_decay} → {final_decay}"
        )

    def get_decay(self, epoch: int) -> float:
        """Get decay value for current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            Decay value in [final_decay, initial_decay]
        """
        if epoch < self.warmup_epochs:
            # Warmup: equal weights
            return self.initial_decay

        elif epoch < self.warmup_epochs + self.progressive_epochs:
            # Progressive: linear interpolation
            progress = (epoch - self.warmup_epochs) / self.progressive_epochs
            return self.initial_decay + progress * (self.final_decay - self.initial_decay)

        else:
            # Full ordering
            return self.final_decay


# =============================================================================
# Evaluation utilities
# =============================================================================


def evaluate_matryoshka_reconstruction(
    flow: MatryoshkaGTRFunnelFlow,
    embeddings: torch.Tensor,
    matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64),
) -> Dict[str, float]:
    """Evaluate reconstruction quality at each Matryoshka level.

    Args:
        flow: Trained Matryoshka Flow
        embeddings: Test embeddings (n, 768)
        matryoshka_dims: Levels to evaluate

    Returns:
        Dictionary with cos_sim for each level
    """
    flow.eval()
    results = {}

    with torch.no_grad():
        z = flow.encode(embeddings)

        for level in matryoshka_dims:
            z_masked = z.clone()
            if level < flow.latent_dim:
                z_masked[:, level:] = 0.0

            x_recon = flow.decode(z_masked, active_dim=level, deterministic=True)
            cos_sim = F.cosine_similarity(embeddings, x_recon, dim=-1).mean().item()
            results[f'{level}D'] = cos_sim

    return results
