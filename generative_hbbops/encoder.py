"""
GTR-Base encoder compatible with Vec2Text inversion.

Uses SentenceTransformer for proper embedding generation that matches
Vec2Text pre-trained models from ielabgroup.

IMPORTANT: Previous implementation used raw T5 encoder with custom mean pooling,
which produced incompatible embeddings (cosine similarity ~0.02 with SentenceTransformer).
This version uses the official SentenceTransformer implementation.
"""
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union


class GTREncoder:
    """Encode text using GTR-T5-Base via SentenceTransformer.

    This encoder produces embeddings compatible with Vec2Text inversion models:
    - ielabgroup/vec2text_gtr-base-st_inversion
    - ielabgroup/vec2text_gtr-base-st_corrector

    Architecture:
        Text -> SentenceTransformer(gtr-t5-base) -> 768-dim normalized embedding
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/gtr-t5-base",
        device: str = "auto",
        max_length: int = 128
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            device: "auto", "cuda", "cpu", or "mps"
            max_length: Maximum token sequence length
        """
        self.model_name = model_name
        self.max_length = max_length

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load SentenceTransformer model
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model.max_seq_length = max_length

        # Embedding dimension
        self.embedding_dim = 768

    def encode(self, text: str) -> np.ndarray:
        """Encode single text to 768-dim embedding.

        Args:
            text: Input text string

        Returns:
            768-dim numpy array (normalized)
        """
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts to embeddings.

        More efficient than encoding texts one by one.

        Args:
            texts: List of input strings

        Returns:
            (N, 768) numpy array
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def encode_tensor(self, text: str) -> torch.Tensor:
        """Encode text and return as torch tensor on device.

        Returns tensor that can be used for further computation.
        Note: This does NOT enable gradients through the encoder.
        For gradient-based optimization, optimize the embedding directly.

        Args:
            text: Input text string

        Returns:
            768-dim tensor on self.device
        """
        return self.model.encode(text, convert_to_tensor=True)

    def compute_cosine_similarity(
        self,
        emb1: Union[np.ndarray, torch.Tensor],
        emb2: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Used to verify inversion quality by comparing original
        embedding with re-embedded inverted text.

        Args:
            emb1: First embedding (768-dim)
            emb2: Second embedding (768-dim)

        Returns:
            Cosine similarity in range [-1, 1]
        """
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.cpu().numpy()

        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))
