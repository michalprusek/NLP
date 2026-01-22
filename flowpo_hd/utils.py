"""
Utility functions for FlowPO-HD.

Includes:
- SONAR encoding/decoding helpers
- Evaluation utilities
- Checkpoint management
- Logging helpers
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# RANDOM SEED
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


# =============================================================================
# SONAR HELPERS
# =============================================================================

class SONARHelper:
    """Helper class for SONAR encoding and decoding."""

    def __init__(
        self,
        device: str = "cuda",
        source_lang: str = "eng_Latn",
        target_lang: str = "eng_Latn",
        normalize: bool = False,
    ):
        """
        Initialize SONAR helper.

        Args:
            device: Torch device (string like 'cuda' or 'cuda:0')
            source_lang: Source language for encoding
            target_lang: Target language for decoding
            normalize: Whether to L2 normalize embeddings
        """
        self.device = torch.device(device)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.normalize = normalize

        # Lazy initialization
        self._encoder = None
        self._decoder = None

    def _init_encoder(self):
        """Initialize SONAR encoder."""
        if self._encoder is not None:
            return

        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        self._encoder = TextToEmbeddingModelPipeline(
            encoder='text_sonar_basic_encoder',
            tokenizer='text_sonar_basic_encoder',
            device=self.device,
        )
        logger.info("SONAR encoder initialized")

    def _init_decoder(self):
        """Initialize SONAR decoder."""
        if self._decoder is not None:
            return

        from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
        self._decoder = EmbeddingToTextModelPipeline(
            decoder='text_sonar_basic_decoder',
            tokenizer='text_sonar_basic_decoder',
            device=self.device,
        )
        logger.info("SONAR decoder initialized")

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode texts to SONAR embeddings."""
        self._init_encoder()
        if isinstance(texts, str):
            texts = [texts]
        emb = self._encoder.predict(texts, source_lang=self.source_lang)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    @torch.no_grad()
    def decode(
        self,
        embeddings: torch.Tensor,
        max_seq_len: int = 256,
    ) -> List[str]:
        """Decode SONAR embeddings to text."""
        self._init_decoder()
        return self._decoder.predict(embeddings, target_lang=self.target_lang, max_seq_len=max_seq_len)

    @torch.no_grad()
    def roundtrip(
        self,
        texts: Union[str, List[str]],
    ) -> Tuple[List[str], float]:
        """
        Encode and decode texts, returning cosine similarity.

        Args:
            texts: Input texts

        Returns:
            (decoded_texts, mean_cosine_similarity)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Encode
        embeddings = self.encode(texts)

        # Decode
        decoded = self.decode(embeddings)

        # Re-encode decoded for similarity
        decoded_embeddings = self.encode(decoded)

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(embeddings, decoded_embeddings, dim=-1)

        return decoded, cos_sim.mean().item()


# =============================================================================
# FLOW-DiT HELPER (for manifold projection)
# =============================================================================

class FlowDiTHelper:
    """
    Helper class for FlowDiT manifold projection.

    Maps noise to instruction manifold in 1024D SONAR space.
    """

    def __init__(
        self,
        checkpoint_path: str = "flowpo_hd/checkpoints_mega_aux2/best.pt",
        device: str = "cuda",
        num_steps: int = 20,
    ):
        self.device = device
        self.num_steps = num_steps
        self.checkpoint_path = checkpoint_path
        self._model = None

    def _init_model(self):
        """Lazy initialization of FlowDiT model."""
        if self._model is not None:
            return

        from flowpo_hd.flow_dit import FlowDiT

        # Load checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        # Create model with inferred architecture
        self._model = FlowDiT(
            latent_dim=1024,
            hidden_dim=1024,
            num_layers=4,
            time_embed_dim=256,
            mlp_ratio=2.0,
        )

        self._model.load_state_dict(ckpt['model_state_dict'])
        self._model.to(self.device)
        self._model.eval()

        logger.info(f"FlowDiT loaded from {self.checkpoint_path}")

    @torch.no_grad()
    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points to instruction manifold via flow integration.

        For points near the manifold, we integrate from t=0 to t=1.
        This maps noise→data, so input is treated as "noisy" version.

        Args:
            x: Points in SONAR space [B, 1024]

        Returns:
            Points on manifold [B, 1024]
        """
        self._init_model()
        from flowpo_hd.flow_dit import integrate_euler

        x = x.to(device=self.device, dtype=torch.float32)

        # ODE integration t=0 (noise) → t=1 (data)
        return integrate_euler(
            self._model,
            x,
            num_steps=self.num_steps,
        )

    @torch.no_grad()
    def generate_from_noise(self, n_samples: int) -> torch.Tensor:
        """
        Generate manifold points from pure noise.

        Args:
            n_samples: Number of samples

        Returns:
            Points on manifold [n_samples, 1024]
        """
        self._init_model()
        from flowpo_hd.flow_dit import integrate_euler

        # Sample noise
        z = torch.randn(n_samples, 1024, device=self.device)

        # ODE integration: noise → manifold
        return integrate_euler(
            self._model,
            z,
            num_steps=self.num_steps,
        )


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def extract_last_number(text: str) -> Optional[float]:
    """Extract the last number from text for GSM8K evaluation."""
    import re

    # Find all numbers (including decimals, negatives)
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))

    if not numbers:
        return None

    try:
        return float(numbers[-1])
    except ValueError:
        return None


def evaluate_gsm8k_accuracy(
    predictions: List[str],
    targets: List[str],
    tolerance: float = 1e-6,
) -> float:
    """
    Evaluate accuracy on GSM8K by comparing extracted numbers.

    Args:
        predictions: Model output texts
        targets: Ground truth answers (containing correct number)
        tolerance: Numerical tolerance for comparison

    Returns:
        Accuracy (0.0 to 1.0)
    """
    correct = 0

    for pred, target in zip(predictions, targets):
        pred_num = extract_last_number(pred)
        target_num = extract_last_number(target)

        if pred_num is None or target_num is None:
            continue

        if abs(pred_num - target_num) < tolerance:
            correct += 1
        elif target_num != 0 and abs((pred_num - target_num) / target_num) < tolerance:
            correct += 1

    return correct / len(predictions) if predictions else 0.0


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Any] = None,
):
    """
    Save a training checkpoint.

    Args:
        path: Save path
        model: Model to save
        optimizer: Optional optimizer
        epoch: Current epoch
        step: Current step
        metrics: Optional metrics dict
        config: Optional config object
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "model_state_dict": model_state,
        "epoch": epoch,
        "step": step,
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["config"] = vars(config) if hasattr(config, '__dict__') else config

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load into
        optimizer: Optional optimizer to load into
        device: Target device

    Returns:
        Checkpoint dict with epoch, step, metrics, etc.
    """
    checkpoint = torch.load(path, map_location=device)

    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint from {path}, epoch={checkpoint.get('epoch', 0)}")

    return checkpoint


# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_results(
    path: Union[str, Path],
    results: Dict[str, Any],
    append: bool = False,
):
    """
    Save results to JSON file.

    Args:
        path: Save path
        results: Results dict
        append: If True, append to existing file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        with open(path, 'r') as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing.append(results)
        else:
            existing = [existing, results]
        results = existing

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved results to {path}")


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# METRIC HELPERS
# =============================================================================

def compute_embedding_stats(embeddings: torch.Tensor) -> Dict[str, float]:
    """Compute statistics of embeddings."""
    norms = embeddings.norm(dim=-1)
    return {
        "n_samples": embeddings.shape[0],
        "dim": embeddings.shape[1],
        "mean_norm": norms.mean().item(),
        "std_norm": norms.std().item(),
        "min_norm": norms.min().item(),
        "max_norm": norms.max().item(),
    }


def compute_pairwise_distances(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute pairwise L2 distances."""
    if y is None:
        y = x
    return torch.cdist(x, y, p=2)


def compute_diversity(embeddings: torch.Tensor) -> float:
    """Compute diversity as mean pairwise distance."""
    if embeddings.shape[0] < 2:
        return 0.0
    dists = compute_pairwise_distances(embeddings)
    # Exclude diagonal
    n = dists.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
    return dists[mask].mean().item()


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


class ProgressLogger:
    """Simple progress logger for long-running tasks."""

    def __init__(self, total: int, desc: str = "", log_interval: int = 10):
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.start_time = datetime.now()

    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0

            logger.info(
                f"{self.desc} {self.current}/{self.total} "
                f"({100*self.current/self.total:.1f}%) "
                f"[{elapsed:.1f}s elapsed, ~{eta:.1f}s remaining]"
            )


if __name__ == "__main__":
    print("Testing utils...")

    # Test seed setting
    print("\n--- Seed Setting ---")
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.allclose(a, b), "Seed not working"
    print("Seed setting works correctly")

    # Test number extraction
    print("\n--- Number Extraction ---")
    test_cases = [
        ("The answer is 42.", 42.0),
        ("Result: -3.14", -3.14),
        ("No numbers here", None),
        ("1, 2, 3, final: 100", 100.0),
    ]
    for text, expected in test_cases:
        result = extract_last_number(text)
        print(f"'{text}' -> {result} (expected {expected})")
        assert result == expected, f"Expected {expected}, got {result}"

    # Test embedding stats
    print("\n--- Embedding Stats ---")
    embeddings = torch.randn(100, 1024) * 0.2
    stats = compute_embedding_stats(embeddings)
    print(f"Stats: {stats}")

    # Test diversity
    print("\n--- Diversity ---")
    div = compute_diversity(embeddings)
    print(f"Diversity: {div:.4f}")

    print("\n[OK] utils tests passed!")
