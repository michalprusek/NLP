"""Token-level text augmentation for contrastive learning.

Provides augmentation strategies that create positive pairs for InfoNCE loss
by applying transformations directly at the token level (no detokenization needed).

Key design principles:
- All operations work on tokenized inputs (input_ids, attention_mask)
- Special tokens (BOS, EOS, PAD) are preserved
- Augmentations are stochastic for training diversity
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import Tensor


@dataclass
class AugmentationConfig:
    """Configuration for text augmentation strategies."""

    # Span masking: replace contiguous spans with mask token
    span_mask_prob: float = 0.15  # Probability of masking a span
    span_mask_max_tokens: int = 5  # Maximum span length

    # Word dropout: randomly drop individual tokens
    word_dropout_rate: float = 0.1  # Probability of dropping each token

    # Local shuffle: shuffle tokens within local windows
    word_shuffle_window: int = 3  # Window size for local shuffling

    # Which augmentations to apply (any combination)
    use_span_mask: bool = True
    use_word_dropout: bool = True
    use_local_shuffle: bool = False  # Disabled by default - can hurt semantic content

    # Special token IDs (will be auto-detected if not set)
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None  # Used as mask token


class TextAugmenter:
    """Token-level augmentation for creating contrastive positive pairs.

    All augmentations operate directly on token IDs, preserving sequence structure
    and avoiding expensive detokenization/retokenization cycles.

    Usage:
        augmenter = TextAugmenter(AugmentationConfig())
        aug_ids, aug_mask = augmenter.augment(input_ids, attention_mask)

    The augmented sequence represents a "positive pair" - semantically similar
    to the original but with surface-level variations that force the encoder
    to learn robust representations.
    """

    def __init__(self, config: AugmentationConfig):
        """Initialize augmenter with configuration.

        Args:
            config: AugmentationConfig with augmentation parameters
        """
        self.config = config

    def _get_special_mask(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Create mask for positions that should NOT be augmented.

        Returns True for special tokens (BOS, EOS, PAD) and False for regular tokens.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Boolean mask where True = protected position (batch, seq_len)
        """
        cfg = self.config
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Start with padding mask (0 in attention_mask = padding)
        special_mask = attention_mask == 0

        # Protect BOS token (first position of each sequence)
        if cfg.bos_token_id is not None:
            special_mask = special_mask | (input_ids == cfg.bos_token_id)
        else:
            # Assume first token is BOS
            first_token_mask = torch.zeros_like(special_mask)
            first_token_mask[:, 0] = True
            special_mask = special_mask | first_token_mask

        # Protect EOS token
        if cfg.eos_token_id is not None:
            special_mask = special_mask | (input_ids == cfg.eos_token_id)

        # Protect PAD token
        if cfg.pad_token_id is not None:
            special_mask = special_mask | (input_ids == cfg.pad_token_id)

        return special_mask

    def span_mask(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply span masking: replace random contiguous spans with UNK/mask token.

        Inspired by SpanBERT - masks contiguous spans to force encoder to learn
        from context rather than individual tokens.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            (masked_ids, attention_mask) - mask unchanged since tokens not removed
        """
        if self.config.span_mask_prob <= 0:
            return input_ids.clone(), attention_mask.clone()

        cfg = self.config
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get mask token (use UNK or PAD as fallback)
        mask_token = cfg.unk_token_id if cfg.unk_token_id is not None else (
            cfg.pad_token_id if cfg.pad_token_id is not None else 0
        )

        # Get special token mask (these positions won't be augmented)
        special_mask = self._get_special_mask(input_ids, attention_mask)

        # Clone for output
        masked_ids = input_ids.clone()

        # Process each sequence in batch
        for b in range(batch_size):
            # Get valid positions for this sequence
            valid_positions = (~special_mask[b]).nonzero(as_tuple=True)[0]
            if len(valid_positions) < 2:
                continue

            # Calculate number of tokens to mask
            num_valid = len(valid_positions)
            num_to_mask = int(num_valid * cfg.span_mask_prob)

            if num_to_mask == 0:
                continue

            # Select random span start positions
            masked_count = 0
            attempts = 0
            max_attempts = num_to_mask * 3  # Prevent infinite loops

            while masked_count < num_to_mask and attempts < max_attempts:
                attempts += 1

                # Random span length
                span_len = torch.randint(1, cfg.span_mask_max_tokens + 1, (1,)).item()

                # Random start position within valid range
                max_start = num_valid - span_len
                if max_start <= 0:
                    span_len = 1
                    max_start = num_valid - 1

                start_idx = torch.randint(0, max_start + 1, (1,)).item()

                # Map to actual positions
                span_start = valid_positions[start_idx].item()
                span_end = min(span_start + span_len, seq_len)

                # Mask the span
                for pos in range(span_start, span_end):
                    if not special_mask[b, pos]:
                        masked_ids[b, pos] = mask_token
                        masked_count += 1

        return masked_ids, attention_mask.clone()

    def word_dropout(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply word dropout: randomly replace tokens with mask/UNK token.

        Unlike span masking, this affects individual tokens independently,
        creating a different type of noise pattern.

        Fully vectorized implementation - no Python loops.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            (dropped_ids, attention_mask) - mask unchanged since tokens not removed
        """
        if self.config.word_dropout_rate <= 0:
            return input_ids.clone(), attention_mask.clone()  # Clone for consistency

        cfg = self.config

        # Get mask token
        mask_token = cfg.unk_token_id if cfg.unk_token_id is not None else (
            cfg.pad_token_id if cfg.pad_token_id is not None else 0
        )

        # Get special token mask (vectorized)
        special_mask = self._get_special_mask(input_ids, attention_mask)

        # Create dropout mask directly (fused operation)
        # True = keep, False = drop
        keep_prob = torch.empty_like(input_ids, dtype=torch.float).uniform_()
        keep_mask = (keep_prob >= cfg.word_dropout_rate) | special_mask

        # Apply dropout (single where operation)
        dropped_ids = torch.where(keep_mask, input_ids, mask_token)

        return dropped_ids, attention_mask

    def local_shuffle(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply local shuffling: permute tokens within small windows.

        Preserves global structure while adding local variation.
        Based on "Sentence Shuffling" but at sub-sentence level.

        Note: This can significantly hurt semantic content for instruction
        understanding, so it's disabled by default.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            (shuffled_ids, attention_mask) - mask unchanged
        """
        if self.config.word_shuffle_window <= 1:
            return input_ids.clone(), attention_mask.clone()

        cfg = self.config
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get special token mask
        special_mask = self._get_special_mask(input_ids, attention_mask)

        # Clone for output
        shuffled_ids = input_ids.clone()

        # Process each sequence
        for b in range(batch_size):
            # Get valid positions
            valid_positions = (~special_mask[b]).nonzero(as_tuple=True)[0].tolist()
            if len(valid_positions) < 2:
                continue

            # Shuffle within windows
            for start_idx in range(0, len(valid_positions), cfg.word_shuffle_window):
                end_idx = min(start_idx + cfg.word_shuffle_window, len(valid_positions))
                window_positions = valid_positions[start_idx:end_idx]

                if len(window_positions) < 2:
                    continue

                # Get tokens in window
                window_tokens = [shuffled_ids[b, pos].item() for pos in window_positions]

                # Shuffle tokens
                perm = torch.randperm(len(window_tokens)).tolist()
                shuffled_tokens = [window_tokens[i] for i in perm]

                # Put back
                for pos, tok in zip(window_positions, shuffled_tokens):
                    shuffled_ids[b, pos] = tok

        return shuffled_ids, attention_mask.clone()

    def augment(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply configured augmentation pipeline.

        Applies enabled augmentations in sequence:
        1. Span masking (if enabled)
        2. Word dropout (if enabled)
        3. Local shuffle (if enabled)

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            (augmented_ids, attention_mask) tuple
        """
        cfg = self.config
        aug_ids = input_ids.clone()
        aug_mask = attention_mask.clone()

        # Apply augmentations in sequence
        if cfg.use_span_mask:
            aug_ids, aug_mask = self.span_mask(aug_ids, aug_mask)

        if cfg.use_word_dropout:
            aug_ids, aug_mask = self.word_dropout(aug_ids, aug_mask)

        if cfg.use_local_shuffle:
            aug_ids, aug_mask = self.local_shuffle(aug_ids, aug_mask)

        return aug_ids, aug_mask

    def configure_from_tokenizer(self, tokenizer) -> "TextAugmenter":
        """Configure special token IDs from a tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer with special token attributes

        Returns:
            Self for chaining
        """
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            self.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            self.config.bos_token_id = tokenizer.bos_token_id
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            self.config.eos_token_id = tokenizer.eos_token_id
        if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
            self.config.unk_token_id = tokenizer.unk_token_id

        return self
