"""
GTR-Base encoder compatible with Vec2Text inversion.

Uses mean pooling over token embeddings (not [CLS] token) for compatibility
with Vec2Text pre-trained models from ielabgroup.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional


class GTREncoder:
    """Encode text using GTR-T5-Base with mean pooling.

    This encoder produces embeddings compatible with Vec2Text inversion models:
    - ielabgroup/vec2text_gtr-base-st_inversion
    - ielabgroup/vec2text_gtr-base-st_corrector

    Architecture:
        Text -> T5 Encoder -> Mean Pooling -> 768-dim embedding

    The mean pooling is critical for Vec2Text compatibility. BERT-style [CLS]
    token extraction does NOT work with Vec2Text.
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
        self.device = self._get_device(device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model - GTR uses T5 encoder only
        full_model = AutoModel.from_pretrained(model_name)
        if hasattr(full_model, 'encoder'):
            self.model = full_model.encoder
        else:
            self.model = full_model

        self.model.eval()
        self.model.to(self.device)

        # Embedding dimension
        self.embedding_dim = 768

    def _get_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _mean_pooling(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling to hidden states.

        Mean pooling is computed as:
            embedding = sum(hidden * mask) / sum(mask)

        This is the pooling method required by Vec2Text.

        Args:
            hidden_state: (batch, seq_len, hidden_dim) encoder outputs
            attention_mask: (batch, seq_len) mask for valid tokens

        Returns:
            (batch, hidden_dim) mean pooled embeddings
        """
        # Expand mask for broadcasting: (batch, seq_len) -> (batch, seq_len, 1)
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # Sum embeddings weighted by mask
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)

        # Sum of mask (number of valid tokens per sequence)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def encode(self, text: str) -> np.ndarray:
        """Encode single text to 768-dim embedding.

        Uses mean pooling over non-padding tokens for Vec2Text compatibility.

        Args:
            text: Input text string

        Returns:
            768-dim numpy array
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            ).to(self.device)

            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            embedding = self._mean_pooling(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )

        return embedding.cpu().numpy().squeeze()

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts to embeddings.

        More efficient than encoding texts one by one.

        Args:
            texts: List of input strings

        Returns:
            (N, 768) numpy array
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            ).to(self.device)

            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            embeddings = self._mean_pooling(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )

        return embeddings.cpu().numpy()

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
        embedding = self.encode(text)
        return torch.tensor(embedding, dtype=torch.float32, device=self.device)

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
