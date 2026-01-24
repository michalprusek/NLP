"""Llama-3 tokenizer wrapper for VAE training.

Handles proper tokenization for instruction-response pairs
with attention to special tokens and padding.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from soft_prompt_vae.config import ModelConfig, DataConfig

logger = logging.getLogger(__name__)


@dataclass
class TokenizedExample:
    """Tokenized instruction-response pair."""

    # Instruction (for encoding)
    instruction_ids: torch.Tensor
    instruction_attention_mask: torch.Tensor

    # Response (for reconstruction target)
    response_ids: torch.Tensor
    response_attention_mask: torch.Tensor

    # Full sequence for decoder (soft_prompt + response)
    # Labels are response_ids with -100 for soft prompt positions
    labels: torch.Tensor


# Fallback tokenizers for preprocessing when main model isn't accessible
FALLBACK_TOKENIZERS = [
    "Qwen/Qwen2.5-7B-Instruct",  # Open model with similar vocab
    "microsoft/Phi-3-mini-4k-instruct",  # Another option
    "gpt2",  # Last resort
]


class LlamaTokenizerWrapper:
    """Wrapper for Llama-3 tokenizer with VAE-specific methods."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        fallback: bool = True,
    ):
        """Initialize tokenizer.

        Args:
            model_config: Model configuration
            data_config: Data configuration
            fallback: Try fallback tokenizers if main fails
        """
        self.model_config = model_config
        self.data_config = data_config

        # Try loading main tokenizer
        tokenizer_name = model_config.model_name
        self.tokenizer = None
        self._using_fallback = False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
            )
            logger.info(f"Loaded tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to load {tokenizer_name}: {e}")

            if fallback:
                # Try fallback tokenizers
                for fallback_name in FALLBACK_TOKENIZERS:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            fallback_name,
                            trust_remote_code=True,
                        )
                        self._using_fallback = True
                        logger.warning(
                            f"Using fallback tokenizer: {fallback_name}. "
                            "Token counts may differ slightly from Llama."
                        )
                        break
                    except Exception as e2:
                        logger.debug(f"Fallback {fallback_name} failed: {e2}")
                        continue

        if self.tokenizer is None:
            raise RuntimeError(
                f"Could not load tokenizer {tokenizer_name} or any fallback. "
                "Please login to HuggingFace: `huggingface-cli login`"
            )

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(
            f"Tokenizer ready: vocab_size={self.tokenizer.vocab_size}, "
            f"pad_token={self.tokenizer.pad_token}"
        )

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id

    def tokenize_instruction(
        self,
        instruction: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize instruction for encoding.

        Args:
            instruction: Instruction text
            max_length: Maximum sequence length

        Returns:
            Dict with input_ids and attention_mask
        """
        max_length = max_length or self.data_config.max_instruction_length

        encoded = self.tokenizer(
            instruction,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def tokenize_response(
        self,
        response: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize response for reconstruction target.

        Args:
            response: Response text
            max_length: Maximum sequence length

        Returns:
            Dict with input_ids and attention_mask
        """
        max_length = max_length or self.data_config.max_response_length

        # Add EOS token to response
        encoded = self.tokenizer(
            response + self.tokenizer.eos_token,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def tokenize_pair(
        self,
        instruction: str,
        response: str,
    ) -> TokenizedExample:
        """Tokenize instruction-response pair for VAE training.

        Args:
            instruction: Instruction text
            response: Response text

        Returns:
            TokenizedExample with all necessary tensors
        """
        # Tokenize instruction (encoder input)
        instr_encoded = self.tokenize_instruction(instruction)

        # Tokenize response (decoder target)
        resp_encoded = self.tokenize_response(response)

        # Create labels (response tokens, -100 for padding)
        labels = resp_encoded["input_ids"].clone()
        labels[resp_encoded["attention_mask"] == 0] = -100

        return TokenizedExample(
            instruction_ids=instr_encoded["input_ids"],
            instruction_attention_mask=instr_encoded["attention_mask"],
            response_ids=resp_encoded["input_ids"],
            response_attention_mask=resp_encoded["attention_mask"],
            labels=labels,
        )

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode batch of token IDs to texts.

        Args:
            token_ids: Token IDs tensor (batch_size, seq_len)
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )


def create_tokenizer(
    model_config: ModelConfig,
    data_config: DataConfig,
) -> LlamaTokenizerWrapper:
    """Create tokenizer from configs.

    Args:
        model_config: Model configuration
        data_config: Data configuration

    Returns:
        LlamaTokenizerWrapper instance
    """
    return LlamaTokenizerWrapper(model_config, data_config)
