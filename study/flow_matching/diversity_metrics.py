"""Diversity metrics for evaluating generated text variation.

This module provides diversity metrics for generated samples:
- Self-BLEU: Average BLEU score of each sample against all others (lower = more diverse)
- Distinct-1/2: Ratio of unique n-grams to total n-grams (higher = more diverse)
- Type-Token Ratio (TTR): Ratio of unique words to total words
- Coverage: Fraction of reference vocabulary covered by generated samples

Usage:
    from study.flow_matching.diversity_metrics import DiversityMetrics

    metrics = DiversityMetrics()
    results = metrics.compute_all(generated_texts)
    print(results)
    # {'self_bleu': 0.35, 'distinct_1': 0.72, 'distinct_2': 0.89, 'ttr': 0.45}
"""

import logging
from collections import Counter
from typing import Dict, List, Optional

import torch
from torchmetrics.text import BLEUScore

logger = logging.getLogger(__name__)


def get_ngrams(text: str, n: int) -> List[tuple]:
    """Extract n-grams from text.

    Args:
        text: Input text string.
        n: N-gram size (1 for unigrams, 2 for bigrams, etc.).

    Returns:
        List of n-gram tuples.
    """
    words = text.lower().split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def compute_distinct_n(texts: List[str], n: int) -> float:
    """Compute Distinct-n score.

    Distinct-n is the ratio of unique n-grams to total n-grams across all texts.
    Higher values indicate more diversity.

    Args:
        texts: List of text strings.
        n: N-gram size.

    Returns:
        Distinct-n score (0-1).
    """
    all_ngrams = []
    for text in texts:
        all_ngrams.extend(get_ngrams(text, n))

    if not all_ngrams:
        return 0.0

    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)


def compute_self_bleu(texts: List[str], n_gram: int = 4) -> float:
    """Compute Self-BLEU score.

    Self-BLEU measures how similar each text is to all other texts.
    Lower values indicate more diversity.

    For each text i, compute BLEU(text_i, [all texts except i]).
    Return the average.

    Args:
        texts: List of text strings.
        n_gram: Maximum n-gram for BLEU (default 4).

    Returns:
        Self-BLEU score (0-1). Lower = more diverse.
    """
    if len(texts) < 2:
        logger.warning("Self-BLEU requires at least 2 texts, returning 0")
        return 0.0

    bleu = BLEUScore(n_gram=n_gram)
    scores = []

    for i, text in enumerate(texts):
        # All other texts as references
        references = texts[:i] + texts[i + 1 :]
        # BLEUScore expects list of lists for references
        refs = [[ref] for ref in references]
        # Compute BLEU of this text against all others
        score = bleu([text], refs).item()
        scores.append(score)

    return sum(scores) / len(scores)


def compute_ttr(texts: List[str]) -> float:
    """Compute Type-Token Ratio.

    TTR is the ratio of unique words to total words across all texts.
    Higher values indicate more vocabulary diversity.

    Args:
        texts: List of text strings.

    Returns:
        TTR score (0-1). Higher = more vocabulary diversity.
    """
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())

    if not all_words:
        return 0.0

    unique_words = set(all_words)
    return len(unique_words) / len(all_words)


def compute_vocab_coverage(
    generated: List[str],
    reference: List[str],
) -> float:
    """Compute vocabulary coverage.

    Measures what fraction of the reference vocabulary appears in generated texts.

    Args:
        generated: List of generated text strings.
        reference: List of reference text strings (from dataset).

    Returns:
        Coverage ratio (0-1). Higher = better vocabulary coverage.
    """
    # Build reference vocabulary
    ref_vocab = set()
    for text in reference:
        ref_vocab.update(text.lower().split())

    if not ref_vocab:
        return 0.0

    # Check how much of ref vocab is covered by generated
    gen_vocab = set()
    for text in generated:
        gen_vocab.update(text.lower().split())

    covered = ref_vocab & gen_vocab
    return len(covered) / len(ref_vocab)


def compute_entropy(texts: List[str], n: int = 1) -> float:
    """Compute n-gram entropy.

    Higher entropy indicates more uniform distribution of n-grams = more diversity.

    Args:
        texts: List of text strings.
        n: N-gram size.

    Returns:
        Entropy value in bits.
    """
    import math

    all_ngrams = []
    for text in texts:
        all_ngrams.extend(get_ngrams(text, n))

    if not all_ngrams:
        return 0.0

    # Count n-gram frequencies
    counts = Counter(all_ngrams)
    total = len(all_ngrams)

    # Compute entropy: -sum(p * log2(p))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


class DiversityMetrics:
    """Compute diversity metrics for generated text samples.

    Evaluates how varied the generated samples are, which is important
    for generative models to avoid mode collapse.

    Metrics:
    - Self-BLEU: Similarity among generated samples (lower = more diverse)
    - Distinct-1/2: Unique unigrams/bigrams ratio (higher = more diverse)
    - TTR: Type-token ratio for vocabulary diversity
    - Entropy: N-gram entropy (higher = more uniform distribution)
    """

    def compute_self_bleu(self, texts: List[str], n_gram: int = 4) -> float:
        """Compute Self-BLEU score."""
        return compute_self_bleu(texts, n_gram)

    def compute_distinct(self, texts: List[str]) -> Dict[str, float]:
        """Compute Distinct-1 and Distinct-2 scores."""
        return {
            "distinct_1": compute_distinct_n(texts, 1),
            "distinct_2": compute_distinct_n(texts, 2),
        }

    def compute_ttr(self, texts: List[str]) -> float:
        """Compute Type-Token Ratio."""
        return compute_ttr(texts)

    def compute_entropy(self, texts: List[str]) -> Dict[str, float]:
        """Compute unigram and bigram entropy."""
        return {
            "entropy_1": compute_entropy(texts, 1),
            "entropy_2": compute_entropy(texts, 2),
        }

    def compute_coverage(
        self,
        generated: List[str],
        reference: Optional[List[str]] = None,
    ) -> float:
        """Compute vocabulary coverage against reference."""
        if reference is None:
            return 0.0
        return compute_vocab_coverage(generated, reference)

    def compute_all(
        self,
        generated: List[str],
        reference: Optional[List[str]] = None,
        skip_self_bleu: bool = False,
    ) -> Dict[str, float]:
        """Compute all diversity metrics.

        Args:
            generated: List of generated text strings.
            reference: Optional reference texts for coverage computation.
            skip_self_bleu: Skip Self-BLEU (slow for large datasets).

        Returns:
            Dict with all diversity metrics.
        """
        if len(generated) == 0:
            logger.warning("Empty input, returning zeros")
            return {
                "self_bleu": 0.0,
                "distinct_1": 0.0,
                "distinct_2": 0.0,
                "ttr": 0.0,
                "entropy_1": 0.0,
                "entropy_2": 0.0,
                "vocab_coverage": 0.0,
            }

        logger.info(f"Computing diversity metrics for {len(generated)} samples...")

        results = {}

        # Self-BLEU (can be slow)
        if not skip_self_bleu:
            results["self_bleu"] = self.compute_self_bleu(generated)
            logger.info(f"Self-BLEU: {results['self_bleu']:.4f} (lower = more diverse)")
        else:
            results["self_bleu"] = 0.0

        # Distinct-n
        distinct_results = self.compute_distinct(generated)
        results.update(distinct_results)
        logger.info(
            f"Distinct-1/2: {distinct_results['distinct_1']:.4f} / "
            f"{distinct_results['distinct_2']:.4f}"
        )

        # TTR
        results["ttr"] = self.compute_ttr(generated)
        logger.info(f"TTR: {results['ttr']:.4f}")

        # Entropy
        entropy_results = self.compute_entropy(generated)
        results.update(entropy_results)
        logger.info(
            f"Entropy (1/2): {entropy_results['entropy_1']:.2f} / "
            f"{entropy_results['entropy_2']:.2f} bits"
        )

        # Vocabulary coverage
        results["vocab_coverage"] = self.compute_coverage(generated, reference)
        if reference:
            logger.info(f"Vocab coverage: {results['vocab_coverage']:.4f}")

        return results


@torch.no_grad()
def evaluate_generation_diversity(
    model: torch.nn.Module,
    test_embeddings: torch.Tensor,
    stats: dict,
    decoder,
    n_samples: int = 100,
    n_steps: int = 100,
    device: str = "cuda:0",
    skip_self_bleu: bool = False,
) -> Dict[str, float]:
    """Evaluate diversity of generated samples.

    End-to-end evaluation: generate embeddings, decode to text, compute diversity.

    Args:
        model: Velocity network in eval mode.
        test_embeddings: Normalized test embeddings [N, 1024] (for reference vocab).
        stats: Normalization statistics dict.
        decoder: SonarDecoder instance.
        n_samples: Number of samples to generate.
        n_steps: ODE integration steps.
        device: Computation device.
        skip_self_bleu: Skip Self-BLEU (slow).

    Returns:
        Dict with all diversity metrics.
    """
    from study.flow_matching.evaluate import euler_ode_integrate
    from study.data.normalize import denormalize

    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    # Generate embeddings from random noise (not from test set)
    logger.info(f"Generating {n_samples} samples with {n_steps} steps...")
    x0 = torch.randn(n_samples, 1024, device=device)
    x1_gen_normalized = euler_ode_integrate(model, x0, n_steps, device, show_progress=True)

    # Denormalize
    x1_gen = denormalize(x1_gen_normalized, stats)

    # Decode to text
    logger.info("Decoding generated embeddings...")
    generated_texts = decoder.decode(x1_gen)

    # Optionally decode reference texts for vocab coverage
    reference_texts = None
    n_ref = min(n_samples, test_embeddings.shape[0])
    if n_ref > 0:
        logger.info("Decoding reference embeddings for vocab coverage...")
        x1_ref = denormalize(test_embeddings[:n_ref].to(device), stats)
        reference_texts = decoder.decode(x1_ref)

    # Compute diversity metrics
    metrics = DiversityMetrics()
    results = metrics.compute_all(
        generated_texts,
        reference=reference_texts,
        skip_self_bleu=skip_self_bleu,
    )

    return results


if __name__ == "__main__":
    # Quick test
    generated = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models process large amounts of data.",
        "Python is a popular programming language.",
        "JavaScript is widely used for web development.",
        "The sun rises in the east and sets in the west.",
        "Stars twinkle in the night sky above us.",
    ]

    reference = [
        "The brown fox jumped quickly over the sleeping dog.",
        "AI systems can learn patterns from data.",
        "Python and JavaScript are both programming languages.",
        "The sun and moon are celestial bodies.",
    ]

    metrics = DiversityMetrics()
    results = metrics.compute_all(generated, reference=reference, skip_self_bleu=False)

    print("\nDiversity Metrics Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
