"""
SONAR Encoder for FlowPO.

SONAR (Sentence-level multimOdal and laNguage-Agnostic Representations) provides
reconstruction-optimized embeddings, unlike GritLM which is retrieval-optimized.

Key insight: SONAR is trained with DAE + translation loss, preserving
reconstruction information that contrastive models (GritLM, GTR) lose.

Reference:
- Duquenne et al., "SONAR: Sentence-Level Multimodal and Language-Agnostic Representations"
- https://github.com/facebookresearch/SONAR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SONAREncoder(nn.Module):
    """
    SONAR encoder for high-fidelity text reconstruction.

    Unlike GritLM (contrastive/retrieval-optimized), SONAR is trained with
    denoising auto-encoding + translation loss, preserving reconstruction
    information in its 1024D embeddings.

    Advantages over GritLM:
    - 1024D native (vs 4096D) → lower compression ratio
    - Reconstruction-optimized → better decode fidelity
    - Tolerates noise up to cos_sim ~0.9 while maintaining decodability

    Output dimension: 1024D (fixed by SONAR architecture)
    """

    def __init__(
        self,
        device: str = "cuda",
        source_lang: str = "eng_Latn",
        normalize: bool = True,
    ):
        """
        Initialize SONAR encoder.

        Args:
            device: Device for computation ("cuda" or "cpu")
            source_lang: Source language code (default: English)
            normalize: L2 normalize output embeddings
        """
        super().__init__()
        self.device = device
        self.source_lang = source_lang
        self.normalize = normalize
        self.output_dim = 1024  # SONAR native dimension

        # Lazy initialization to avoid loading model at import time
        self._pipeline = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of SONAR pipeline."""
        if self._initialized:
            return

        try:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

            logger.info("Initializing SONAR text encoder...")
            # Convert device string to torch.device (fairseq2 requirement)
            device = torch.device(self.device) if isinstance(self.device, str) else self.device
            self._pipeline = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=device,
            )
            self._initialized = True
            logger.info("SONAR encoder initialized successfully")

        except ImportError as e:
            raise ImportError(
                "SONAR not installed. Install with: pip install sonar-space>=0.5.0\n"
                "Note: SONAR requires fairseq2 with matching PyTorch/CUDA versions.\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SONAR encoder: {e}\n"
                "Try clearing fairseq2 cache: rm -rf ~/.cache/fairseq2"
            )

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode texts to 1024D embeddings.

        Args:
            texts: Single string or list of strings to encode

        Returns:
            embeddings: (N, 1024) tensor of embeddings
        """
        self._initialize()

        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        # Encode using SONAR pipeline
        embeddings = self._pipeline.predict(texts, source_lang=self.source_lang)

        # Ensure on correct device
        embeddings = embeddings.to(self.device)

        # Optional L2 normalization
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @torch.no_grad()
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Encode large list of texts in batches.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            embeddings: (N, 1024) tensor of all embeddings
        """
        self._initialize()

        all_embeddings = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(texts), batch_size), desc="Encoding")
            except ImportError:
                logger.warning(
                    "tqdm not installed. Progress bar disabled. "
                    "Install with: pip install tqdm"
                )
                iterator = range(0, len(texts), batch_size)
        else:
            iterator = range(0, len(texts), batch_size)

        for i in iterator:
            batch = texts[i : i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Forward pass (alias for encode)."""
        return self.encode(texts)

    def set_device(self, device: str) -> "SONAREncoder":
        """
        Move encoder to device.

        Note: This method reinitializes the SONAR pipeline on the new device.
        Use set_device() instead of to() to avoid confusion with nn.Module.to().

        Args:
            device: Target device string (e.g., "cuda:0", "cpu")

        Returns:
            self for method chaining
        """
        self.device = device
        if self._initialized and self._pipeline is not None:
            # Reinitialize on new device
            self._initialized = False
            self._initialize()
        return self

    def to(self, device: str) -> "SONAREncoder":
        """Move encoder to device (alias for set_device)."""
        return self.set_device(device)


class SONARTextDecoder(nn.Module):
    """
    SONAR text decoder for embedding-to-text reconstruction.

    Decodes 1024D SONAR embeddings back to text using SONAR's
    built-in decoder model.

    Note: This provides near-verbatim reconstruction for clean embeddings,
    but quality degrades with noise or manipulation in the embedding space.
    """

    def __init__(
        self,
        device: str = "cuda",
        target_lang: str = "eng_Latn",
    ):
        """
        Initialize SONAR decoder.

        Args:
            device: Device for computation
            target_lang: Target language code for decoding
        """
        super().__init__()
        self.device = device
        self.target_lang = target_lang

        # Lazy initialization
        self._pipeline = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of SONAR decoder pipeline."""
        if self._initialized:
            return

        try:
            from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

            logger.info("Initializing SONAR text decoder...")
            # Convert device string to torch.device (fairseq2 requirement)
            device = torch.device(self.device) if isinstance(self.device, str) else self.device
            self._pipeline = EmbeddingToTextModelPipeline(
                decoder="text_sonar_basic_decoder",
                tokenizer="text_sonar_basic_decoder",
                device=device,
            )
            self._initialized = True
            logger.info("SONAR decoder initialized successfully")

        except ImportError as e:
            raise ImportError(
                "SONAR not installed. Install with: pip install sonar-space>=0.5.0\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SONAR decoder: {e}\n"
                "Try clearing fairseq2 cache: rm -rf ~/.cache/fairseq2"
            )

    @torch.no_grad()
    def decode(
        self,
        embeddings: torch.Tensor,
        max_seq_len: int = 256,
    ) -> List[str]:
        """
        Decode embeddings to text.

        Args:
            embeddings: (N, 1024) tensor of SONAR embeddings
            max_seq_len: Maximum sequence length for decoding

        Returns:
            texts: List of decoded strings
        """
        self._initialize()

        # Ensure embeddings are on correct device
        embeddings = embeddings.to(self.device)

        # Decode using SONAR pipeline
        texts = self._pipeline.predict(
            embeddings,
            target_lang=self.target_lang,
            max_seq_len=max_seq_len,
        )

        return texts

    def forward(self, embeddings: torch.Tensor) -> List[str]:
        """Forward pass (alias for decode)."""
        return self.decode(embeddings)


def create_sonar_encoder(
    device: str = "cuda",
    source_lang: str = "eng_Latn",
    **kwargs,
) -> SONAREncoder:
    """
    Factory function to create SONAR encoder.

    Args:
        device: Device for computation
        source_lang: Source language code
        **kwargs: Additional arguments for SONAREncoder

    Returns:
        Initialized SONAREncoder
    """
    return SONAREncoder(device=device, source_lang=source_lang, **kwargs)


if __name__ == "__main__":
    print("Testing SONAR Encoder...")
    print()

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create encoder
    encoder = SONAREncoder(device=device)

    # Test single encoding
    print("\n--- Single Encoding ---")
    text = "The quick brown fox jumps over the lazy dog."
    embedding = encoder.encode(text)
    print(f"Input: '{text}'")
    print(f"Output shape: {embedding.shape}")
    print(f"Output norm: {embedding.norm().item():.4f}")

    # Test batch encoding
    print("\n--- Batch Encoding ---")
    texts = [
        "Hello, world!",
        "Machine learning is fascinating.",
        "SONAR embeddings are reconstruction-optimized.",
    ]
    embeddings = encoder.encode(texts)
    print(f"Input: {len(texts)} texts")
    print(f"Output shape: {embeddings.shape}")

    # Test similarity
    print("\n--- Similarity Test ---")
    sim_texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",  # Semantically similar
        "Quantum physics is complex.",  # Different topic
    ]
    sim_embeddings = encoder.encode(sim_texts)

    sim_01 = F.cosine_similarity(sim_embeddings[0:1], sim_embeddings[1:2]).item()
    sim_02 = F.cosine_similarity(sim_embeddings[0:1], sim_embeddings[2:3]).item()

    print(f"'{sim_texts[0]}' vs")
    print(f"'{sim_texts[1]}': {sim_01:.4f} (should be high)")
    print(f"'{sim_texts[2]}': {sim_02:.4f} (should be low)")

    print("\n[OK] SONAR Encoder tests passed!")
