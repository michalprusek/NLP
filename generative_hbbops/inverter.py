"""
Vec2Text embedding inversion for HyLO.

Converts optimized embeddings back to text using:
- ielabgroup/vec2text_gtr-base-st_inversion (initial hypothesis)
- ielabgroup/vec2text_gtr-base-st_corrector (iterative refinement)

Vec2Text uses an iterative correction process where:
1. Inversion model generates initial text hypothesis
2. Corrector model iteratively refines the hypothesis
3. Each correction step re-encodes and corrects towards target embedding

Reference:
    Morris et al. "Text Embeddings Reveal (Almost) As Much As Text" (EMNLP 2023)
"""
import torch
from typing import Optional, Tuple
import warnings


class Vec2TextInverter:
    """Wrapper for Vec2Text embedding inversion.

    Uses 50 correction iterations with beam search to achieve
    high-quality text reconstruction from GTR-base embeddings.

    The inversion quality depends on:
    - Number of correction steps (more = better but slower)
    - Beam width (wider = more options explored)
    - How well the embedding matches the training distribution
    """

    def __init__(
        self,
        num_steps: int = 50,
        beam_width: int = 4,
        device: str = "auto"
    ):
        """
        Args:
            num_steps: Number of correction iterations (default: 50)
            beam_width: Beam search width (default: 4)
            device: Device for inference ("auto", "cuda", "cpu")
        """
        self.num_steps = num_steps
        self.beam_width = beam_width
        self.device_str = device

        # Lazy loading of models
        self._corrector = None
        self._device = None

    def _get_device(self) -> torch.device:
        """Resolve device string to torch.device."""
        if self._device is not None:
            return self._device

        if self.device_str == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self.device_str)

        return self._device

    def _load_models(self) -> None:
        """Load inversion and corrector models from HuggingFace.

        Models:
            - ielabgroup/vec2text_gtr-base-st_inversion
            - ielabgroup/vec2text_gtr-base-st_corrector

        These models are trained on GTR-base embeddings and require ~2GB download.

        Note: Uses manual loading with safetensors to avoid meta tensor issues
        in newer versions of transformers.
        """
        if self._corrector is not None:
            return

        try:
            import vec2text
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
            from vec2text.models.config import InversionConfig
            from vec2text.models.inversion import InversionModel
            from vec2text.models.corrector_encoder import CorrectorEncoderModel
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Install with: uv add vec2text safetensors huggingface_hub"
            )

        print("Loading Vec2Text models (this may take a while on first run)...")
        device = self._get_device()

        # Load InversionModel
        inv_weights = hf_hub_download(
            "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
        )
        inv_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_inversion"
        )
        inversion_model = InversionModel(inv_config)
        inversion_model.load_state_dict(load_file(inv_weights), strict=False)
        inversion_model = inversion_model.to(device).eval()

        # Load CorrectorEncoderModel
        corr_weights = hf_hub_download(
            "ielabgroup/vec2text_gtr-base-st_corrector", "model.safetensors"
        )
        corr_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_corrector"
        )
        corrector_model = CorrectorEncoderModel(corr_config)
        corrector_model.load_state_dict(load_file(corr_weights), strict=False)
        corrector_model = corrector_model.to(device).eval()

        # Create corrector pipeline
        self._corrector = vec2text.load_corrector(inversion_model, corrector_model)
        print("Loaded ielabgroup Vec2Text models successfully.")

    def invert(self, embedding: torch.Tensor) -> str:
        """Invert single embedding to text.

        Uses iterative correction with beam search for high-quality inversion.

        Args:
            embedding: (768,) GTR-base embedding tensor

        Returns:
            Reconstructed text string
        """
        self._load_models()

        try:
            import vec2text
        except ImportError:
            raise ImportError("vec2text not installed")

        # Ensure proper shape and device
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        device = self._get_device()

        # Move to model's device
        try:
            model_device = next(self._corrector.model.parameters()).device
        except:
            model_device = device

        embedding = embedding.to(model_device)

        # Invert using vec2text API
        with torch.no_grad():
            results = vec2text.invert_embeddings(
                embeddings=embedding,
                corrector=self._corrector,
                num_steps=self.num_steps,
                sequence_beam_width=self.beam_width
            )

        return results[0] if results else ""

    def invert_instruction(
        self,
        instruction_emb: torch.Tensor,
        verbose: bool = True
    ) -> str:
        """Invert instruction embedding to text.

        Args:
            instruction_emb: (768,) instruction embedding
            verbose: Print progress

        Returns:
            Reconstructed instruction text
        """
        if verbose:
            print(f"Inverting instruction embedding ({self.num_steps} steps, beam={self.beam_width})...")

        text = self.invert(instruction_emb)

        if verbose:
            print(f"Inverted instruction: {text[:100]}...")

        return text

    def verify_reconstruction(
        self,
        original_emb: torch.Tensor,
        reconstructed_text: str,
        encoder: 'GTREncoder'
    ) -> Tuple[float, torch.Tensor]:
        """Verify quality of reconstruction.

        Re-encodes reconstructed text and computes cosine similarity
        with original embedding. Higher similarity = better inversion.

        Args:
            original_emb: (768,) original optimized embedding
            reconstructed_text: Text from inversion
            encoder: GTREncoder instance for re-encoding

        Returns:
            (cosine_similarity, re_embedded_vector)
        """
        # Re-encode the reconstructed text
        re_embedded = encoder.encode_tensor(reconstructed_text)

        # Compute cosine similarity
        cosine_sim = encoder.compute_cosine_similarity(original_emb, re_embedded)

        return cosine_sim, re_embedded


class NearestNeighborInverter:
    """Fallback inverter using nearest neighbor lookup.

    When Vec2Text is not available or produces poor results, this inverter
    finds the nearest text from a known set of embeddings.

    This guarantees valid text but may have large "inversion gap" if the
    optimized embedding is far from any known text.
    """

    def __init__(
        self,
        texts: list,
        embeddings: torch.Tensor,
        device: torch.device = None
    ):
        """
        Args:
            texts: List of known texts
            embeddings: (N, 768) embeddings corresponding to texts
            device: Torch device
        """
        self.texts = texts
        self.embeddings = embeddings.to(device) if device else embeddings
        self.device = device

    def invert(self, embedding: torch.Tensor) -> Tuple[str, int, float]:
        """Find nearest text to the given embedding.

        Args:
            embedding: (768,) target embedding

        Returns:
            (text, index, distance)
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        embedding = embedding.to(self.embeddings.device)

        # Compute L2 distances
        distances = torch.norm(self.embeddings - embedding, dim=1)

        # Find nearest
        nearest_idx = distances.argmin().item()
        nearest_dist = distances[nearest_idx].item()
        nearest_text = self.texts[nearest_idx]

        return nearest_text, nearest_idx, nearest_dist
