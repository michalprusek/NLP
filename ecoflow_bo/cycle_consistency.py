"""
Cycle Consistency Checker: Hallucination detection before evaluation.

After decoding z → x, we re-encode x → z' and check if ||z - z'|| is small.
Large discrepancy indicates the decoder produced something off-manifold
(a "hallucination") that the encoder can't recognize.

This prevents wasting expensive objective evaluations on invalid candidates.
"""

import torch
from typing import Optional, Tuple, List, Union

from .encoder import MatryoshkaEncoder
from .cfm_decoder import RectifiedFlowDecoder
from .config import CycleConfig


class CycleConsistencyChecker:
    """
    Checks if decoded embeddings are valid by re-encoding and measuring error.

    Workflow:
    1. Decode: z → x_decoded
    2. Re-encode: x_decoded → z_reencoded
    3. Check: ||z - z_reencoded|| < threshold

    If error is large, the decoder likely "hallucinated" - produced an embedding
    that doesn't correspond to the input z. This happens when z is far from the
    training distribution.
    """

    def __init__(
        self,
        encoder: MatryoshkaEncoder,
        decoder: RectifiedFlowDecoder,
        config: Optional[CycleConfig] = None,
    ):
        if config is None:
            config = CycleConfig()

        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.error_threshold = config.error_threshold
        self.max_retries = config.max_retries

    @torch.no_grad()
    def compute_cycle_error(
        self,
        z: torch.Tensor,
        active_dims: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute cycle consistency error.

        Args:
            z: Latent codes [B, latent_dim]
            active_dims: Which dimensions to compare. Can be:
                - None: use all dimensions
                - int: use first N dimensions (legacy, for contiguous prefixes)
                - List[int]: use specific dimension indices (supports non-contiguous)

        Returns:
            x_decoded: Decoded embeddings [B, data_dim]
            z_reencoded: Re-encoded latents [B, latent_dim]
            error: L2 error ||z - z_reencoded|| [B]
        """
        self.encoder.eval()
        self.decoder.velocity_net.eval()

        # Decode
        x_decoded = self.decoder.decode_deterministic(z)

        # Re-encode (deterministic)
        z_reencoded = self.encoder.encode_deterministic(x_decoded)

        # Compute error on active dimensions only
        if active_dims is not None:
            if isinstance(active_dims, int):
                # Legacy: int means "first N dimensions"
                z_compare = z[:, :active_dims]
                z_re_compare = z_reencoded[:, :active_dims]
            else:
                # List of indices: use advanced indexing
                z_compare = z[:, active_dims]
                z_re_compare = z_reencoded[:, active_dims]
        else:
            z_compare = z
            z_re_compare = z_reencoded

        error = torch.norm(z_compare - z_re_compare, dim=-1)

        return x_decoded, z_reencoded, error

    def is_valid(
        self,
        z: torch.Tensor,
        active_dims: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if z decodes to a valid embedding.

        Args:
            z: Latent codes [B, latent_dim]
            active_dims: Which dims to check (int for prefix, List[int] for indices)

        Returns:
            valid_mask: Boolean mask [B] - True if valid
            errors: Cycle errors [B]
        """
        _, _, errors = self.compute_cycle_error(z, active_dims)
        valid_mask = errors < self.error_threshold
        return valid_mask, errors

    def filter_valid(
        self,
        z_candidates: torch.Tensor,
        active_dims: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Filter candidates to only valid ones.

        Args:
            z_candidates: Candidate latents [N, latent_dim]
            active_dims: Which dims to check (int for prefix, List[int] for indices)

        Returns:
            z_valid: Valid candidates [M, latent_dim]
            x_decoded: Decoded embeddings for valid [M, data_dim]
            errors: Errors for valid [M]
        """
        x_decoded, _, errors = self.compute_cycle_error(z_candidates, active_dims)
        valid_mask = errors < self.error_threshold

        if not valid_mask.any():
            # No valid candidates - return best (lowest error)
            best_idx = errors.argmin()
            return (
                z_candidates[best_idx:best_idx + 1],
                x_decoded[best_idx:best_idx + 1],
                errors[best_idx:best_idx + 1],
            )

        return (
            z_candidates[valid_mask],
            x_decoded[valid_mask],
            errors[valid_mask],
        )

    def check_and_decode(
        self,
        z: torch.Tensor,
        active_dims: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[Optional[torch.Tensor], float, bool]:
        """
        Decode z and check validity. Single sample version.

        Args:
            z: Single latent [latent_dim] or [1, latent_dim]
            active_dims: Which dims to check (int for prefix, List[int] for indices)

        Returns:
            x_decoded: Decoded embedding or None if invalid
            error: Cycle error
            is_valid: Whether the decode is valid
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        x_decoded, _, error = self.compute_cycle_error(z, active_dims)
        error_val = error.item()
        valid = error_val < self.error_threshold

        if valid:
            return x_decoded.squeeze(0), error_val, True
        else:
            return None, error_val, False

    def select_valid_from_ranked(
        self,
        z_candidates: torch.Tensor,
        acq_values: torch.Tensor,
        active_dims: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        """
        Select first valid candidate from ranked list.

        Tries candidates in order of acquisition value until finding
        a valid one or exhausting max_retries.

        Args:
            z_candidates: Candidates sorted by acquisition [N, latent_dim]
            acq_values: Acquisition values [N]
            active_dims: Which dims to check (int for prefix, List[int] for indices)

        Returns:
            z_selected: Selected latent [latent_dim]
            x_decoded: Decoded embedding [data_dim]
            error: Cycle error
            n_tried: Number of candidates tried
        """
        for i in range(min(self.max_retries, len(z_candidates))):
            z_i = z_candidates[i]
            x_decoded, error, valid = self.check_and_decode(z_i, active_dims)

            if valid:
                return z_i, x_decoded, error, i + 1

        # No valid found - return best (lowest error) from tried
        z_tried = z_candidates[: self.max_retries]
        _, _, errors = self.compute_cycle_error(z_tried, active_dims)
        best_idx = errors.argmin()

        z_best = z_tried[best_idx]
        x_decoded = self.decoder.decode_deterministic(z_best.unsqueeze(0)).squeeze(0)
        error = errors[best_idx].item()

        return z_best, x_decoded, error, self.max_retries


class AdaptiveCycleChecker(CycleConsistencyChecker):
    """
    Adaptive cycle consistency checker that adjusts threshold based on
    observed error distribution.

    Useful when the optimal threshold depends on the specific encoder/decoder
    quality achieved during training.
    """

    def __init__(
        self,
        encoder: MatryoshkaEncoder,
        decoder: RectifiedFlowDecoder,
        config: Optional[CycleConfig] = None,
        percentile: float = 0.95,
    ):
        super().__init__(encoder, decoder, config)
        self.percentile = percentile
        self.observed_errors: List[float] = []
        self.adaptive_threshold: Optional[float] = None

    def calibrate(
        self,
        z_samples: torch.Tensor,
        active_dims: Optional[Union[int, List[int]]] = None,
    ):
        """
        Calibrate threshold based on error distribution on known-good samples.

        Args:
            z_samples: Samples from encoder (should be valid) [N, latent_dim]
            active_dims: Which dims to use (int for prefix, List[int] for indices)
        """
        _, _, errors = self.compute_cycle_error(z_samples, active_dims)

        # Set threshold at percentile of observed errors
        sorted_errors = errors.sort().values
        idx = int(self.percentile * len(sorted_errors))
        self.adaptive_threshold = sorted_errors[idx].item()

        # Add buffer
        self.adaptive_threshold *= 1.2

        print(f"[Cycle] Calibrated threshold: {self.adaptive_threshold:.4f} "
              f"(median={errors.median():.4f}, 95th={sorted_errors[idx]:.4f})")

    def is_valid(
        self,
        z: torch.Tensor,
        active_dims: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check validity using adaptive or fixed threshold."""
        _, _, errors = self.compute_cycle_error(z, active_dims)

        threshold = (
            self.adaptive_threshold
            if self.adaptive_threshold is not None
            else self.error_threshold
        )

        valid_mask = errors < threshold
        return valid_mask, errors

    def update_stats(self, error: float):
        """Track error for online threshold adjustment."""
        self.observed_errors.append(error)

        # Periodically update threshold
        if len(self.observed_errors) % 50 == 0:
            sorted_errors = sorted(self.observed_errors)
            idx = int(self.percentile * len(sorted_errors))
            new_threshold = sorted_errors[idx] * 1.2

            if self.adaptive_threshold is None:
                self.adaptive_threshold = new_threshold
            else:
                # Exponential moving average
                self.adaptive_threshold = (
                    0.9 * self.adaptive_threshold + 0.1 * new_threshold
                )
