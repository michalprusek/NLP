"""
SONAR decoder wrapper for converting embeddings back to text.

Uses the SONAR text decoder pipeline to decode 1024D embeddings
into English text.
"""

import logging
from typing import List, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class SonarDecoder:
    """
    Wrapper for SONAR embedding-to-text decoder.

    Decodes 1024D SONAR embeddings back to English text using
    the text_sonar_basic_decoder model.
    """

    def __init__(self, device: str = "cuda:0"):
        """
        Initialize the SONAR decoder.

        Args:
            device: Device to run decoder on (default: cuda:0)
        """
        self.device = device
        logger.info(f"Initializing SonarDecoder on device: {device}")

        # Import and create the decoder pipeline
        from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

        self.decoder = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
        )
        logger.info("SonarDecoder initialized successfully")

    def decode(
        self,
        embeddings: Tensor,
        max_seq_len: int = 256,
        beam_size: int = 5,
    ) -> List[str]:
        """
        Decode embeddings to text.

        Args:
            embeddings: Tensor of shape [N, 1024] containing SONAR embeddings
            max_seq_len: Maximum sequence length for generation (default: 256)
            beam_size: Beam size for beam search decoding (default: 5)

        Returns:
            List of decoded strings, one per embedding
        """
        # Ensure embeddings are on the correct device
        if embeddings.device != torch.device(self.device):
            embeddings = embeddings.to(self.device)

        # Ensure 2D tensor
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        logger.info(f"Decoding {embeddings.shape[0]} embeddings...")

        # Decode using SONAR pipeline
        decoded_texts = self.decoder.predict(
            embeddings,
            target_lang="eng_Latn",
            max_seq_len=max_seq_len,
            beam_size=beam_size,
        )

        # Convert to list if necessary
        if isinstance(decoded_texts, str):
            decoded_texts = [decoded_texts]

        logger.info(f"Decoded {len(decoded_texts)} texts")
        return list(decoded_texts)

    def decode_batch(
        self,
        embeddings: Tensor,
        batch_size: int = 32,
        max_seq_len: int = 256,
        beam_size: int = 5,
    ) -> List[str]:
        """
        Decode embeddings in batches for memory efficiency.

        Args:
            embeddings: Tensor of shape [N, 1024] containing SONAR embeddings
            batch_size: Number of embeddings to decode at once
            max_seq_len: Maximum sequence length for generation
            beam_size: Beam size for beam search decoding

        Returns:
            List of decoded strings, one per embedding
        """
        all_texts = []
        n_samples = embeddings.shape[0]

        for i in range(0, n_samples, batch_size):
            batch = embeddings[i : i + batch_size]
            texts = self.decode(batch, max_seq_len=max_seq_len, beam_size=beam_size)
            all_texts.extend(texts)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Decoded {min(i + batch_size, n_samples)}/{n_samples}")

        return all_texts
