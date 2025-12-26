"""Evaluation metrics for Vec2Text inversion quality.

Implements standard text reconstruction metrics:
- Exact match
- BLEU score (1-4 grams)
- Cosine similarity (embedding space)
- Character Error Rate (CER)
- Token accuracy
"""

import numpy as np
from typing import List, Tuple, Union
import warnings


def exact_match(original: str, reconstructed: str, normalize: bool = True) -> float:
    """Check if reconstruction exactly matches original.

    Args:
        original: Original text string
        reconstructed: Reconstructed text from Vec2Text
        normalize: If True, lowercase, strip whitespace, and collapse multiple spaces

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if normalize:
        import re
        original = re.sub(r'\s+', ' ', original.lower().strip())
        reconstructed = re.sub(r'\s+', ' ', reconstructed.lower().strip())
    return 1.0 if original == reconstructed else 0.0


def bleu_score(
    original: str,
    reconstructed: str,
    max_n: int = 4,
    weights: Tuple[float, ...] = None
) -> float:
    """Compute BLEU score between original and reconstructed text.

    Uses NLTK's sentence_bleu with smoothing to handle short sequences.

    Args:
        original: Reference text
        reconstructed: Hypothesis text
        max_n: Maximum n-gram order (default: 4 for BLEU-4)
        weights: N-gram weights (default: uniform)

    Returns:
        BLEU score in [0, 1]
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        warnings.warn("NLTK not installed. Install with: pip install nltk")
        return 0.0

    # Tokenize (simple whitespace split)
    reference = [original.lower().split()]
    hypothesis = reconstructed.lower().split()

    if not hypothesis:
        return 0.0

    if weights is None:
        # Uniform weights up to max_n
        weights = tuple(1.0 / max_n for _ in range(max_n))

    # Use smoothing to handle short sequences
    smoothing = SmoothingFunction().method1

    try:
        score = sentence_bleu(
            reference,
            hypothesis,
            weights=weights,
            smoothing_function=smoothing
        )
    except Exception:
        score = 0.0

    return score


def character_error_rate(original: str, reconstructed: str) -> float:
    """Compute Character Error Rate (CER) using Levenshtein distance.

    CER = (insertions + deletions + substitutions) / len(original)

    Lower is better. 0.0 = perfect reconstruction.

    Args:
        original: Reference text
        reconstructed: Hypothesis text

    Returns:
        CER value (can be > 1.0 if reconstruction is much longer)
    """
    import re
    # Normalize whitespace before comparison
    original = re.sub(r'\s+', ' ', original.lower().strip())
    reconstructed = re.sub(r'\s+', ' ', reconstructed.lower().strip())

    if not original:
        return 0.0 if not reconstructed else float(len(reconstructed))

    # Compute Levenshtein distance (edit distance)
    distance = _levenshtein_distance(original, reconstructed)

    return distance / len(original)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings.

    Dynamic programming implementation with O(min(m,n)) space.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    current_row = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row[0] = i + 1

        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            current_row[j + 1] = min(insertions, deletions, substitutions)

        previous_row, current_row = current_row, previous_row

    return previous_row[-1]


def token_accuracy(original: str, reconstructed: str) -> float:
    """Compute token-level accuracy (word overlap).

    Returns fraction of original tokens present in reconstruction.

    Args:
        original: Reference text
        reconstructed: Hypothesis text

    Returns:
        Accuracy in [0, 1]
    """
    orig_tokens = set(original.lower().split())
    recon_tokens = set(reconstructed.lower().split())

    if not orig_tokens:
        return 1.0 if not recon_tokens else 0.0

    overlap = len(orig_tokens & recon_tokens)
    return overlap / len(orig_tokens)


def cosine_similarity(
    emb1: Union[np.ndarray, "torch.Tensor"],
    emb2: Union[np.ndarray, "torch.Tensor"]
) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        emb1: First embedding vector
        emb2: Second embedding vector

    Returns:
        Cosine similarity in [-1, 1]
    """
    # Convert torch tensors if needed
    if hasattr(emb1, 'cpu'):
        emb1 = emb1.cpu().numpy()
    if hasattr(emb2, 'cpu'):
        emb2 = emb2.cpu().numpy()

    emb1 = np.asarray(emb1).flatten()
    emb2 = np.asarray(emb2).flatten()

    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0

    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def compute_all_metrics(
    original: str,
    reconstructed: str,
    original_embedding: np.ndarray = None,
    reconstructed_embedding: np.ndarray = None
) -> dict:
    """Compute all available metrics for a single sample.

    Args:
        original: Original text
        reconstructed: Reconstructed text
        original_embedding: Original text embedding (optional)
        reconstructed_embedding: Reconstructed text embedding (optional)

    Returns:
        Dictionary with all metric values
    """
    metrics = {
        "exact_match": exact_match(original, reconstructed),
        "bleu": bleu_score(original, reconstructed),
        "cer": character_error_rate(original, reconstructed),
        "token_accuracy": token_accuracy(original, reconstructed),
    }

    if original_embedding is not None and reconstructed_embedding is not None:
        metrics["cosine_similarity"] = cosine_similarity(
            original_embedding, reconstructed_embedding
        )

    return metrics


def aggregate_metrics(results: List[dict]) -> dict:
    """Aggregate metrics across multiple samples.

    Args:
        results: List of metric dictionaries

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not results:
        return {}

    # Only aggregate numeric metrics
    numeric_metrics = [
        "exact_match", "bleu", "cer", "token_accuracy", "cosine_similarity"
    ]

    aggregated = {}

    for name in numeric_metrics:
        values = [r[name] for r in results if name in r and isinstance(r[name], (int, float))]
        if values:
            aggregated[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return aggregated
