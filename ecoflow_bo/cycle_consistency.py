"""
Cycle Consistency Checker: Hallucination detection before evaluation.

After decoding z → x, we re-encode x → z' and check if ||z - z'|| is small.
Large discrepancy indicates the decoder produced something off-manifold
(a "hallucination") that the encoder can't recognize.

This prevents wasting expensive objective evaluations on invalid candidates.

Residual Latent Support:
- For residual latent architecture (z_full = [z_core, z_detail])
- GP operates on z_core only (16D) - tractable optimization
- Decoder needs z_full (48D) - high-fidelity reconstruction
- Cycle checker bridges the gap via detail_retriever
- Cycle error is computed on z_core only (what GP optimizes)
"""

import torch
from typing import Optional, Tuple, List, Union, TYPE_CHECKING

from .encoder import MatryoshkaEncoder
from .cfm_decoder import RectifiedFlowDecoder
from .config import CycleConfig

if TYPE_CHECKING:
    from .detail_retriever import SimpleDetailRetriever


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

    Supports two modes:
    - Fixed threshold (adaptive=False): Use config.error_threshold directly
    - Adaptive threshold (adaptive=True): Calibrate threshold from observed errors
    """

    def __init__(
        self,
        encoder: MatryoshkaEncoder,
        decoder: RectifiedFlowDecoder,
        config: Optional[CycleConfig] = None,
        adaptive: bool = False,
        percentile: float = 0.95,
        detail_retriever: Optional["SimpleDetailRetriever"] = None,
        core_dim: int = 16,
    ):
        """
        Initialize cycle consistency checker.

        Args:
            encoder: Matryoshka encoder
            decoder: Rectified flow decoder
            config: Cycle configuration
            adaptive: Whether to use adaptive thresholding
            percentile: Percentile for adaptive threshold
            detail_retriever: Optional retriever for z_detail (for residual latent mode)
            core_dim: Dimension of z_core (default 16) for residual latent mode
        """
        if config is None:
            config = CycleConfig()

        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.error_threshold = config.error_threshold
        self.max_retries = config.max_retries

        # Adaptive mode settings
        self.adaptive = adaptive
        self.percentile = percentile
        self.observed_errors: List[float] = []
        self.adaptive_threshold: Optional[float] = None

        # Residual latent mode settings
        self.detail_retriever = detail_retriever
        self.core_dim = core_dim
        self.residual_mode = detail_retriever is not None

    @property
    def effective_threshold(self) -> float:
        """Get the effective threshold (adaptive or fixed)."""
        if self.adaptive and self.adaptive_threshold is not None:
            return self.adaptive_threshold
        return self.error_threshold

    def set_detail_retriever(
        self,
        detail_retriever: "SimpleDetailRetriever",
        core_dim: int = 16,
    ):
        """
        Set detail retriever for residual latent mode.

        Called after initialization when training data is available.

        Args:
            detail_retriever: Retriever for z_detail from training set
            core_dim: Dimension of z_core
        """
        self.detail_retriever = detail_retriever
        self.core_dim = core_dim
        self.residual_mode = True

    def _prepare_z_for_decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Prepare z for decoding, handling residual latent mode.

        In residual mode with z_core input:
        - Retrieves z_detail from training set
        - Returns z_full = [z_core, z_detail]

        In standard mode:
        - Returns z unchanged

        Args:
            z: Latent codes [B, latent_dim] (z_core or z_full)

        Returns:
            z_for_decode: [B, full_dim] ready for decoder
        """
        if not self.residual_mode or self.detail_retriever is None:
            return z

        # Check if input is z_core only (needs detail retrieval)
        if z.shape[-1] == self.core_dim:
            # Get z_detail from training set and concatenate
            return self.detail_retriever.get_full_latent(z)
        else:
            # Input is already z_full
            return z

    @torch.no_grad()
    def compute_cycle_error(
        self,
        z: torch.Tensor,
        active_dims: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute cycle consistency error.

        For residual latent mode:
        - If z is z_core (16D), retrieves z_detail and creates z_full for decoding
        - Cycle error is computed on z_core only (what GP optimizes)
        - Returns z_core (not z_full) in z_reencoded

        Args:
            z: Latent codes [B, latent_dim] - can be z_core (16D) or z_full (48D)
            active_dims: Which dimensions to compare. Can be:
                - None: use all dimensions (or all z_core dims in residual mode)
                - int: use first N dimensions (legacy, for contiguous prefixes)
                - List[int]: use specific dimension indices (supports non-contiguous)

        Returns:
            x_decoded: Decoded embeddings [B, data_dim]
            z_reencoded: Re-encoded latents [B, core_dim or latent_dim]
            error: L2 error ||z_core - z_core_reencoded|| [B]
        """
        self.encoder.eval()
        self.decoder.velocity_net.eval()

        # In residual mode, we need to track the original z_core for comparison
        z_core_original = None
        if self.residual_mode and z.shape[-1] == self.core_dim:
            z_core_original = z  # Save for comparison
            z_for_decode = self._prepare_z_for_decode(z)
        else:
            z_for_decode = z

        # Decode using z_full (or z if not in residual mode)
        x_decoded = self.decoder.decode_deterministic(z_for_decode)

        # Re-encode (deterministic) - returns z_core in residual mode
        z_reencoded = self.encoder.encode_deterministic(x_decoded)

        # Determine what to compare based on mode
        if z_core_original is not None:
            # Residual mode: compare z_core only
            z_compare = z_core_original
            z_re_compare = z_reencoded
        else:
            z_compare = z
            z_re_compare = z_reencoded

        # Compute error on active dimensions only
        if active_dims is not None:
            if isinstance(active_dims, int):
                # Legacy: int means "first N dimensions"
                z_compare = z_compare[:, :active_dims]
                z_re_compare = z_re_compare[:, :active_dims]
            else:
                # List of indices: use advanced indexing
                z_compare = z_compare[:, active_dims]
                z_re_compare = z_re_compare[:, active_dims]

        error = torch.norm(z_compare - z_re_compare, dim=-1)

        return x_decoded, z_reencoded, error

    def calibrate(
        self,
        z_samples: torch.Tensor,
        active_dims: Optional[Union[int, List[int]]] = None,
    ):
        """
        Calibrate threshold based on error distribution on known-good samples.

        Only has effect when adaptive=True. In fixed mode, this is a no-op.

        Args:
            z_samples: Samples from encoder (should be valid) [N, latent_dim]
            active_dims: Which dims to use (int for prefix, List[int] for indices)
        """
        if not self.adaptive:
            return  # No-op for fixed mode

        _, _, errors = self.compute_cycle_error(z_samples, active_dims)

        # Set threshold at percentile of observed errors
        sorted_errors = errors.sort().values
        idx = int(self.percentile * len(sorted_errors))
        self.adaptive_threshold = sorted_errors[idx].item()

        # Add buffer
        self.adaptive_threshold *= 1.2

        print(f"[Cycle] Calibrated threshold: {self.adaptive_threshold:.4f} "
              f"(median={errors.median():.4f}, 95th={sorted_errors[idx]:.4f})")

    def update_stats(self, error: float):
        """
        Track error for online threshold adjustment.

        Only has effect when adaptive=True. In fixed mode, this is a no-op.
        """
        if not self.adaptive:
            return  # No-op for fixed mode

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
        valid_mask = errors < self.effective_threshold
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
        threshold = self.effective_threshold
        valid_mask = errors < threshold

        if not valid_mask.any():
            # No valid candidates - critical error condition
            import logging
            logger = logging.getLogger(__name__)
            best_idx = errors.argmin()
            error_stats = f"min={errors.min():.3f}, median={errors.median():.3f}, max={errors.max():.3f}"
            logger.error(
                f"CRITICAL: All {len(z_candidates)} candidates failed cycle consistency! "
                f"Threshold={threshold:.3f}, error stats: {error_stats}. "
                f"This indicates decoder hallucinations or encoder-decoder misalignment. "
                f"Returning best invalid candidate with error={errors[best_idx].item():.3f}."
            )
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
        valid = error_val < self.effective_threshold

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
        threshold = self.effective_threshold

        for i in range(min(self.max_retries, len(z_candidates))):
            z_i = z_candidates[i]
            x_decoded, error, valid = self.check_and_decode(z_i, active_dims)

            if valid:
                return z_i, x_decoded, error, i + 1

        # No valid found - return best (lowest error) from tried
        import logging
        logger = logging.getLogger(__name__)

        z_tried = z_candidates[: self.max_retries]
        _, _, errors = self.compute_cycle_error(z_tried, active_dims)
        best_idx = errors.argmin()
        error = errors[best_idx].item()

        logger.warning(
            f"[Cycle] No valid candidate found after {self.max_retries} tries. "
            f"Returning best invalid candidate with error={error:.4f} "
            f"(threshold={self.effective_threshold:.4f}). "
            f"Error stats: min={errors.min():.4f}, max={errors.max():.4f}."
        )

        z_best = z_tried[best_idx]
        # Prepare z for decode (handles residual latent mode)
        z_for_decode = self._prepare_z_for_decode(z_best.unsqueeze(0))
        x_decoded = self.decoder.decode_deterministic(z_for_decode).squeeze(0)

        return z_best, x_decoded, error, self.max_retries


# Backwards compatibility alias (deprecated)
AdaptiveCycleChecker = CycleConsistencyChecker
