"""Batch collator for VAE training.

Handles batching of TokenizedExample objects with proper padding.
"""

from typing import List, Dict
from dataclasses import dataclass

import torch

from soft_prompt_vae.data.tokenization import TokenizedExample


@dataclass
class VAEBatch:
    """Batched data for VAE training."""

    # Encoder inputs (instruction)
    instruction_ids: torch.Tensor  # (batch_size, max_instr_len)
    instruction_attention_mask: torch.Tensor  # (batch_size, max_instr_len)

    # Decoder inputs/targets (response)
    response_ids: torch.Tensor  # (batch_size, max_resp_len)
    response_attention_mask: torch.Tensor  # (batch_size, max_resp_len)

    # Labels for loss computation (-100 for ignored positions)
    labels: torch.Tensor  # (batch_size, max_resp_len)

    def to(self, device: torch.device) -> "VAEBatch":
        """Move batch to device."""
        return VAEBatch(
            instruction_ids=self.instruction_ids.to(device),
            instruction_attention_mask=self.instruction_attention_mask.to(device),
            response_ids=self.response_ids.to(device),
            response_attention_mask=self.response_attention_mask.to(device),
            labels=self.labels.to(device),
        )

    def pin_memory(self) -> "VAEBatch":
        """Pin memory for faster GPU transfer."""
        return VAEBatch(
            instruction_ids=self.instruction_ids.pin_memory(),
            instruction_attention_mask=self.instruction_attention_mask.pin_memory(),
            response_ids=self.response_ids.pin_memory(),
            response_attention_mask=self.response_attention_mask.pin_memory(),
            labels=self.labels.pin_memory(),
        )

    def __len__(self) -> int:
        """Get batch size."""
        return self.instruction_ids.size(0)


class VAECollator:
    """Collator for VAE training batches.

    Stacks TokenizedExample objects into VAEBatch.
    Since we use fixed-length tokenization, no additional padding is needed.
    """

    def __init__(self, pad_token_id: int = 0):
        """Initialize collator.

        Args:
            pad_token_id: Padding token ID (for reference)
        """
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[TokenizedExample]) -> VAEBatch:
        """Collate examples into batch.

        Args:
            examples: List of TokenizedExample

        Returns:
            VAEBatch
        """
        return VAEBatch(
            instruction_ids=torch.stack([e.instruction_ids for e in examples]),
            instruction_attention_mask=torch.stack(
                [e.instruction_attention_mask for e in examples]
            ),
            response_ids=torch.stack([e.response_ids for e in examples]),
            response_attention_mask=torch.stack(
                [e.response_attention_mask for e in examples]
            ),
            labels=torch.stack([e.labels for e in examples]),
        )


class DynamicPaddingCollator:
    """Collator with dynamic padding to maximum length in batch.

    More memory efficient than fixed-length padding when sequence
    lengths vary significantly within batches.
    """

    def __init__(self, pad_token_id: int = 0):
        """Initialize collator.

        Args:
            pad_token_id: Padding token ID
        """
        self.pad_token_id = pad_token_id

    def _pad_tensors(
        self,
        tensors: List[torch.Tensor],
        pad_value: int = 0,
    ) -> torch.Tensor:
        """Pad tensors to maximum length in batch.

        Args:
            tensors: List of 1D tensors
            pad_value: Value to use for padding

        Returns:
            Padded and stacked tensor (batch_size, max_len)
        """
        max_len = max(t.size(0) for t in tensors)
        batch_size = len(tensors)

        padded = torch.full(
            (batch_size, max_len),
            pad_value,
            dtype=tensors[0].dtype,
        )

        for i, t in enumerate(tensors):
            padded[i, : t.size(0)] = t

        return padded

    def __call__(self, examples: List[TokenizedExample]) -> VAEBatch:
        """Collate examples into batch with dynamic padding.

        Args:
            examples: List of TokenizedExample

        Returns:
            VAEBatch with dynamic padding
        """
        # Find actual lengths (non-padded)
        instr_lengths = [
            e.instruction_attention_mask.sum().item() for e in examples
        ]
        resp_lengths = [
            e.response_attention_mask.sum().item() for e in examples
        ]

        max_instr_len = max(instr_lengths)
        max_resp_len = max(resp_lengths)

        # Truncate to actual max lengths
        instruction_ids = torch.stack(
            [e.instruction_ids[:max_instr_len] for e in examples]
        )
        instruction_attention_mask = torch.stack(
            [e.instruction_attention_mask[:max_instr_len] for e in examples]
        )
        response_ids = torch.stack(
            [e.response_ids[:max_resp_len] for e in examples]
        )
        response_attention_mask = torch.stack(
            [e.response_attention_mask[:max_resp_len] for e in examples]
        )
        labels = torch.stack([e.labels[:max_resp_len] for e in examples])

        return VAEBatch(
            instruction_ids=instruction_ids,
            instruction_attention_mask=instruction_attention_mask,
            response_ids=response_ids,
            response_attention_mask=response_attention_mask,
            labels=labels,
        )
