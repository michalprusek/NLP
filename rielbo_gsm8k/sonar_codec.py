"""SONAR text encoder/decoder for prompt optimization.

Wraps Meta's SONAR sentence embeddings for use as the latent space
in SphericalSubspaceBOv2. Provides encode/decode matching the
MolecularCodec interface pattern from shared/guacamol/codec.py.

SONAR produces 1024D embeddings suitable for Bayesian optimization
in a continuous latent space, analogous to SELFIES VAE for molecules.

Usage:
    codec = SonarCodec(device="cuda:1")
    embeddings = codec.encode(["Solve step by step.", "Think carefully."])  # [2, 1024]
    texts = codec.decode(embeddings)  # ["Solve step by step.", "Think carefully."]
"""

import logging

import torch

logger = logging.getLogger(__name__)


class SonarCodec:
    """SONAR text encoder/decoder for 1024D sentence embeddings.

    Uses Meta's SONAR (Sentence-level multimodal and language-Agnostic
    Representations) for encoding text to continuous vectors and decoding
    back to text.
    """

    EMBEDDING_DIM = 1024

    def __init__(self, device: str = "cuda:1"):
        """Initialize SONAR encoder and decoder.

        Args:
            device: Device for SONAR models. Default cuda:1 to keep
                    GPU 0 free for vLLM task model.
        """
        from sonar.inference_pipelines.text import (
            EmbeddingToTextModelPipeline,
            TextToEmbeddingModelPipeline,
        )

        self.device = torch.device(device)

        logger.info(f"Loading SONAR encoder on {device}...")
        self.encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
        )

        logger.info(f"Loading SONAR decoder on {device}...")
        self.decoder = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_decoder",
            device=self.device,
        )

        logger.info("SONAR codec ready")

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode text prompts to 1024D embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            Embeddings tensor [N, 1024]
        """
        if not texts:
            return torch.empty(0, self.EMBEDDING_DIM, device=self.device)

        embeddings = self.encoder.predict(texts, source_lang="eng_Latn")
        return embeddings.to(self.device)

    def decode(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
        max_seq_len: int = 512,
    ) -> list[str]:
        """Decode embeddings to text strings.

        Args:
            embeddings: Embeddings tensor [N, 1024]
            temperature: Not used (SONAR decoder is deterministic)
            max_seq_len: Maximum sequence length for decoded text

        Returns:
            List of decoded text strings. Returns "" for decode failures.
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        embeddings = embeddings.to(self.device)

        try:
            decoded = self.decoder.predict(
                embeddings,
                target_lang="eng_Latn",
                max_seq_len=max_seq_len,
            )
        except Exception as e:
            logger.warning(f"SONAR decode failed: {e}")
            return [""] * embeddings.shape[0]

        # Filter garbled outputs (very short = likely decode failure)
        results = []
        for text in decoded:
            if len(text.strip()) < 5:
                results.append("")
            else:
                results.append(text.strip())

        return results
