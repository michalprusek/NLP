"""SONAR decoder for converting embeddings back to text."""

import logging
from typing import List, Sequence

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class SonarDecoder:
    """Decodes 1024D SONAR embeddings to English text with n-gram repeat blocking."""

    def __init__(self, device: str = "cuda:0", ngram_block_size: int = 3):
        """
        Initialize the SONAR decoder.

        Args:
            device: Device to run decoder on.
            ngram_block_size: Block repeating n-grams of this size (0 to disable).
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.ngram_block_size = ngram_block_size
        logger.info(f"Initializing SonarDecoder on device: {self.device}")

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
            embeddings: Tensor of shape [N, 1024] containing SONAR embeddings.
            max_seq_len: Maximum sequence length for generation.
            beam_size: Beam size for beam search decoding.

        Returns:
            List of decoded strings, one per embedding.
        """
        if embeddings.device != torch.device(self.device):
            embeddings = embeddings.to(self.device)

        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        logger.info(f"Decoding {embeddings.shape[0]} embeddings...")

        generator_kwargs = {
            "max_seq_len": max_seq_len,
            "beam_size": beam_size,
        }

        if self.ngram_block_size > 0:
            from fairseq2.generation.step_processor import NGramRepeatBlockProcessor

            step_processors: Sequence = [NGramRepeatBlockProcessor(self.ngram_block_size)]
            generator_kwargs["step_processors"] = step_processors

        decoded_texts = self.decoder.predict(
            embeddings,
            target_lang="eng_Latn",
            **generator_kwargs,
        )

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
            embeddings: Tensor of shape [N, 1024] containing SONAR embeddings.
            batch_size: Number of embeddings to decode at once.
            max_seq_len: Maximum sequence length for generation.
            beam_size: Beam size for beam search decoding.

        Returns:
            List of decoded strings, one per embedding.
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
