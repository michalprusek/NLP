"""Quantitative text generation metrics for flow matching evaluation.

This module provides text quality metrics for evaluating generated text:
- BLEU score (n-gram overlap)
- ROUGE scores (recall-oriented metrics)
- BERTScore (semantic similarity using BERT)
- SentenceTransformer cosine similarity

All metrics compare generated text against reference text from the dataset.

Usage:
    from study.flow_matching.text_metrics import TextMetrics

    metrics = TextMetrics(device="cuda:0")
    results = metrics.compute_all(generated_texts, reference_texts)
    print(results)
    # {'bleu': 0.45, 'rouge1': 0.52, 'rouge2': 0.31, 'rougeL': 0.48,
    #  'bertscore_f1': 0.87, 'sbert_similarity': 0.82}
"""

import logging
from typing import Dict, List, Optional

import torch
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextMetrics:
    """Compute text quality metrics for generated vs reference texts.

    Provides BLEU, ROUGE, BERTScore, and SentenceTransformer similarity.
    All metrics are computed in batch for efficiency.

    Attributes:
        device: Computation device (cuda/cpu).
        bleu: TorchMetrics BLEU scorer.
        rouge: TorchMetrics ROUGE scorer.
        sbert: SentenceTransformer model for semantic similarity.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        sbert_model: str = "all-MiniLM-L6-v2",
        load_sbert: bool = True,
        load_bertscore: bool = True,
    ):
        """Initialize text metrics.

        Args:
            device: Computation device.
            sbert_model: SentenceTransformer model name for semantic similarity.
            load_sbert: Whether to load SentenceTransformer model.
            load_bertscore: Whether to load BERTScore model.
        """
        self.device = device

        # BLEU and ROUGE (lightweight, always loaded)
        self.bleu = BLEUScore(n_gram=4)
        self.rouge = ROUGEScore()

        # SentenceTransformer for semantic similarity
        self.sbert = None
        if load_sbert:
            logger.info(f"Loading SentenceTransformer: {sbert_model}")
            self.sbert = SentenceTransformer(sbert_model, device=device)

        # BERTScore (optional, heavy)
        self.bertscore_model = None
        if load_bertscore:
            try:
                from torchmetrics.text.bert import BERTScore
                logger.info("Loading BERTScore model...")
                self.bertscore_model = BERTScore(
                    model_name_or_path="microsoft/deberta-xlarge-mnli",
                    device=device,
                )
            except ImportError:
                logger.warning("BERTScore not available, skipping")

    def compute_bleu(
        self,
        generated: List[str],
        references: List[str],
    ) -> float:
        """Compute BLEU score.

        Args:
            generated: List of generated texts.
            references: List of reference texts.

        Returns:
            BLEU score (0-1).
        """
        # BLEU expects references as list of lists
        refs = [[ref] for ref in references]
        return self.bleu(generated, refs).item()

    def compute_rouge(
        self,
        generated: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute ROUGE scores (1, 2, L).

        Args:
            generated: List of generated texts.
            references: List of reference texts.

        Returns:
            Dict with rouge1, rouge2, rougeL F1 scores.
        """
        result = self.rouge(generated, references)
        return {
            "rouge1": result["rouge1_fmeasure"].item(),
            "rouge2": result["rouge2_fmeasure"].item(),
            "rougeL": result["rougeL_fmeasure"].item(),
        }

    def compute_bertscore(
        self,
        generated: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute BERTScore (precision, recall, F1).

        Args:
            generated: List of generated texts.
            references: List of reference texts.

        Returns:
            Dict with bertscore_precision, bertscore_recall, bertscore_f1.
        """
        if self.bertscore_model is None:
            logger.warning("BERTScore model not loaded, returning zeros")
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
            }

        result = self.bertscore_model(generated, references)
        return {
            "bertscore_precision": result["precision"].mean().item(),
            "bertscore_recall": result["recall"].mean().item(),
            "bertscore_f1": result["f1"].mean().item(),
        }

    def compute_sbert_similarity(
        self,
        generated: List[str],
        references: List[str],
    ) -> float:
        """Compute average cosine similarity using SentenceTransformer.

        Args:
            generated: List of generated texts.
            references: List of reference texts.

        Returns:
            Mean cosine similarity (0-1).
        """
        if self.sbert is None:
            logger.warning("SentenceTransformer not loaded, returning 0")
            return 0.0

        # Encode both sets
        gen_embeddings = self.sbert.encode(
            generated,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        ref_embeddings = self.sbert.encode(
            references,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # Compute pairwise cosine similarity
        # Normalize embeddings
        gen_norm = gen_embeddings / gen_embeddings.norm(dim=1, keepdim=True)
        ref_norm = ref_embeddings / ref_embeddings.norm(dim=1, keepdim=True)

        # Element-wise dot product for paired similarity
        similarities = (gen_norm * ref_norm).sum(dim=1)
        return similarities.mean().item()

    def compute_all(
        self,
        generated: List[str],
        references: List[str],
        skip_bertscore: bool = False,
    ) -> Dict[str, float]:
        """Compute all text quality metrics.

        Args:
            generated: List of generated texts.
            references: List of reference texts.
            skip_bertscore: Skip BERTScore computation (slow).

        Returns:
            Dict with all metrics: bleu, rouge1, rouge2, rougeL,
            bertscore_precision/recall/f1, sbert_similarity.
        """
        if len(generated) != len(references):
            raise ValueError(
                f"Length mismatch: {len(generated)} generated vs {len(references)} references"
            )

        if len(generated) == 0:
            logger.warning("Empty input, returning zeros")
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "sbert_similarity": 0.0,
            }

        logger.info(f"Computing text metrics for {len(generated)} samples...")

        results = {}

        # BLEU
        results["bleu"] = self.compute_bleu(generated, references)
        logger.info(f"BLEU: {results['bleu']:.4f}")

        # ROUGE
        rouge_results = self.compute_rouge(generated, references)
        results.update(rouge_results)
        logger.info(
            f"ROUGE-1/2/L: {rouge_results['rouge1']:.4f} / "
            f"{rouge_results['rouge2']:.4f} / {rouge_results['rougeL']:.4f}"
        )

        # BERTScore (optional, slow)
        if not skip_bertscore:
            bertscore_results = self.compute_bertscore(generated, references)
            results.update(bertscore_results)
            logger.info(f"BERTScore F1: {bertscore_results['bertscore_f1']:.4f}")
        else:
            results.update({
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
            })

        # SentenceTransformer similarity
        results["sbert_similarity"] = self.compute_sbert_similarity(generated, references)
        logger.info(f"SBERT similarity: {results['sbert_similarity']:.4f}")

        return results


@torch.no_grad()
def evaluate_generation_quality(
    model: torch.nn.Module,
    test_embeddings: torch.Tensor,
    stats: dict,
    decoder,
    n_samples: int = 100,
    n_steps: int = 100,
    device: str = "cuda:0",
    skip_bertscore: bool = False,
) -> Dict[str, float]:
    """Evaluate text generation quality against test set.

    End-to-end evaluation: generate embeddings, decode to text, compute metrics.

    Args:
        model: Velocity network in eval mode.
        test_embeddings: Normalized test embeddings [N, 1024].
        stats: Normalization statistics dict.
        decoder: SonarDecoder instance.
        n_samples: Number of samples to evaluate.
        n_steps: ODE integration steps.
        device: Computation device.
        skip_bertscore: Skip BERTScore (slow).

    Returns:
        Dict with all text quality metrics.
    """
    from study.flow_matching.evaluate import euler_ode_integrate
    from study.data.normalize import denormalize

    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    # Limit samples
    n_actual = min(n_samples, test_embeddings.shape[0])
    test_subset = test_embeddings[:n_actual].to(device)

    # Generate embeddings from noise
    logger.info(f"Generating {n_actual} samples with {n_steps} steps...")
    x0 = torch.randn_like(test_subset)
    x1_gen_normalized = euler_ode_integrate(model, x0, n_steps, device, show_progress=True)

    # Denormalize generated embeddings
    x1_gen = denormalize(x1_gen_normalized, stats)

    # Decode generated embeddings to text
    logger.info("Decoding generated embeddings...")
    generated_texts = decoder.decode(x1_gen)

    # Decode reference embeddings to text
    logger.info("Decoding reference embeddings...")
    x1_ref = denormalize(test_subset, stats)
    reference_texts = decoder.decode(x1_ref)

    # Compute text metrics
    metrics = TextMetrics(device=device)
    results = metrics.compute_all(generated_texts, reference_texts, skip_bertscore=skip_bertscore)

    return results


if __name__ == "__main__":
    # Quick test
    generated = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]
    references = [
        "A fast brown fox leaps over a sleepy dog.",
        "Machine learning belongs to the field of AI.",
        "Python is widely used for programming.",
    ]

    metrics = TextMetrics(device="cuda:0" if torch.cuda.is_available() else "cpu")
    results = metrics.compute_all(generated, references, skip_bertscore=True)
    print("\nText Metrics Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
