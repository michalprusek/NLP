"""GTR Prompt Encoder for Vec2Text compatibility.

Uses sentence-transformers/gtr-t5-base which produces L2-normalized 768D embeddings
that are compatible with Vec2Text inversion models.
"""

import numpy as np
import torch
from typing import List, Optional, Union


class GTRPromptEncoder:
    """Encode prompts using GTR-T5-Base with mean pooling.

    Compatible with Vec2Text inversion models (ielabgroup/vec2text_gtr-base-st_*).
    Uses SentenceTransformer internally for guaranteed compatibility.

    Key differences from BERT PromptEncoder:
    - Uses mean pooling (not [CLS] token)
    - Produces L2-normalized embeddings
    - 768D output (same dim as BERT, but different embedding space)

    Attributes:
        model_name: The sentence-transformers model name
        embedding_dim: Output embedding dimension (768)
        device: Device for computation
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/gtr-t5-base",
        normalize: bool = True,
        device: str = "auto",
    ):
        """Initialize GTR encoder.

        Args:
            model_name: SentenceTransformer model name
            normalize: Whether to L2-normalize embeddings (required for Vec2Text)
            device: Device to use ("auto", "cuda", "cpu", "mps")
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self.embedding_dim = 768
        self.device = self._get_device(device)

        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)

    def _get_device(self, device: str) -> str:
        """Determine device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def encode(self, text: str) -> np.ndarray:
        """Encode text to 768D GTR embedding.

        Args:
            text: Input text string

        Returns:
            768-dimensional numpy array (L2-normalized if normalize=True)
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embedding

    def encode_tensor(self, text: str) -> torch.Tensor:
        """Encode text to 768D tensor on device.

        Args:
            text: Input text string

        Returns:
            768-dimensional torch tensor on device
        """
        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
        )
        return embedding.to(self.device)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch encode texts to embeddings.

        Args:
            texts: List of input strings
            batch_size: Batch size for encoding

        Returns:
            (N, 768) numpy array
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def encode_batch_tensor(
        self, texts: List[str], batch_size: int = 32
    ) -> torch.Tensor:
        """Batch encode texts to tensor.

        Args:
            texts: List of input strings
            batch_size: Batch size for encoding

        Returns:
            (N, 768) torch tensor on device
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.to(self.device)

    @staticmethod
    def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding (768D)
            emb2: Second embedding (768D)

        Returns:
            Cosine similarity in [-1, 1]
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
