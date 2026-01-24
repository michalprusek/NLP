"""Evaluation metrics for Soft-Prompt VAE.

Implements:
- Active Units (AU): Dimensions with variance > threshold
- BLEU/ROUGE: Reconstruction quality
- Interpolation smoothness: Semantic consistency along latent paths
- Latent space statistics
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore

logger = logging.getLogger(__name__)


@dataclass
class VAEMetrics:
    """Container for all VAE evaluation metrics."""

    # Reconstruction quality
    bleu: float
    rouge_l: float

    # Latent space quality
    active_units: int
    active_unit_ratio: float
    total_kl: float
    mean_kl_per_dim: float

    # Interpolation quality
    interpolation_smoothness: Optional[float] = None

    # Statistics
    mu_mean: float = 0.0
    mu_std: float = 0.0
    logvar_mean: float = 0.0

    def __str__(self) -> str:
        return (
            f"BLEU: {self.bleu:.4f}, ROUGE-L: {self.rouge_l:.4f}, "
            f"AU: {self.active_units} ({self.active_unit_ratio:.1%}), "
            f"KL: {self.total_kl:.2f}"
        )


class ActiveUnitsCounter:
    """Count active units in latent space.

    A dimension is "active" if its aggregated posterior variance
    differs significantly from the prior (unit Gaussian).

    Based on "Avoiding Latent Variable Collapse" (Burda et al., 2015)
    """

    def __init__(self, threshold: float = 0.01):
        """Initialize counter.

        Args:
            threshold: Variance threshold for considering a dimension active
        """
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._mu_sum = None
        self._mu_sq_sum = None
        self._count = 0

    def update(self, mu: torch.Tensor) -> None:
        """Update with batch of latent means.

        Args:
            mu: Latent means (batch, latent_dim)
        """
        batch_size = mu.size(0)

        if self._mu_sum is None:
            self._mu_sum = torch.zeros_like(mu[0])
            self._mu_sq_sum = torch.zeros_like(mu[0])

        self._mu_sum += mu.sum(dim=0)
        self._mu_sq_sum += (mu ** 2).sum(dim=0)
        self._count += batch_size

    def compute(self) -> Tuple[int, float, torch.Tensor]:
        """Compute active units.

        Returns:
            active_units: Number of active dimensions
            ratio: Ratio of active dimensions
            variances: Per-dimension variance
        """
        if self._count == 0:
            return 0, 0.0, torch.tensor([])

        # Compute mean and variance across all samples
        mean = self._mu_sum / self._count
        var = self._mu_sq_sum / self._count - mean ** 2

        # Count dimensions with variance above threshold
        active = (var > self.threshold).sum().item()
        ratio = active / len(var)

        return int(active), ratio, var


class ReconstructionMetrics:
    """BLEU and ROUGE metrics for reconstruction quality."""

    def __init__(self):
        """Initialize metrics."""
        self.bleu = BLEUScore(n_gram=4)
        self.rouge = ROUGEScore()
        self.reset()

    def reset(self) -> None:
        """Reset accumulated scores."""
        self._predictions: List[str] = []
        self._references: List[str] = []

    def update(
        self,
        predictions: List[str],
        references: List[str],
    ) -> None:
        """Update with batch of predictions and references.

        Args:
            predictions: Generated texts
            references: Ground truth texts
        """
        self._predictions.extend(predictions)
        self._references.extend(references)

    def compute(self) -> Tuple[float, float]:
        """Compute BLEU and ROUGE-L scores.

        Returns:
            bleu: BLEU-4 score
            rouge_l: ROUGE-L F1 score
        """
        if not self._predictions:
            return 0.0, 0.0

        # BLEU expects list of references per prediction
        bleu_refs = [[ref] for ref in self._references]
        bleu_score = self.bleu(self._predictions, bleu_refs).item()

        # ROUGE
        rouge_scores = self.rouge(self._predictions, self._references)
        rouge_l = rouge_scores["rougeL_fmeasure"].item()

        return bleu_score, rouge_l


class InterpolationSmoothness:
    """Measure semantic smoothness of latent interpolations.

    Computes cosine similarity between consecutive generations
    along an interpolation path. Higher = smoother transitions.
    """

    def __init__(self, embedding_model: Optional[torch.nn.Module] = None):
        """Initialize smoothness metric.

        Args:
            embedding_model: Optional sentence encoder for semantic similarity.
                           If None, uses simple token overlap.
        """
        self.embedding_model = embedding_model

    def _token_overlap(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity of word sets."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union

    def compute(
        self,
        generated_texts: List[str],
    ) -> float:
        """Compute interpolation smoothness.

        Args:
            generated_texts: List of texts generated along interpolation path

        Returns:
            Average similarity between consecutive texts
        """
        if len(generated_texts) < 2:
            return 1.0

        similarities = []
        for i in range(len(generated_texts) - 1):
            sim = self._token_overlap(generated_texts[i], generated_texts[i + 1])
            similarities.append(sim)

        return sum(similarities) / len(similarities)


class LatentStatistics:
    """Compute statistics over latent space."""

    def __init__(self):
        """Initialize statistics tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._mu_values: List[torch.Tensor] = []
        self._logvar_values: List[torch.Tensor] = []
        self._kl_values: List[torch.Tensor] = []

    def update(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> None:
        """Update with batch of latent parameters.

        Args:
            mu: Latent means (batch, latent_dim)
            logvar: Latent log variances (batch, latent_dim)
        """
        self._mu_values.append(mu.detach().cpu())
        self._logvar_values.append(logvar.detach().cpu())

        # Compute per-sample KL
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(dim=1)
        self._kl_values.append(kl.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute latent space statistics.

        Returns:
            Dictionary with statistics
        """
        if not self._mu_values:
            return {}

        all_mu = torch.cat(self._mu_values, dim=0)
        all_logvar = torch.cat(self._logvar_values, dim=0)
        all_kl = torch.cat(self._kl_values, dim=0)

        return {
            "mu_mean": all_mu.mean().item(),
            "mu_std": all_mu.std().item(),
            "mu_abs_mean": all_mu.abs().mean().item(),
            "logvar_mean": all_logvar.mean().item(),
            "logvar_std": all_logvar.std().item(),
            "kl_mean": all_kl.mean().item(),
            "kl_std": all_kl.std().item(),
            "kl_total": all_kl.mean().item(),  # Per-sample mean KL
        }


def compute_all_metrics(
    model: torch.nn.Module,
    dataloader,
    tokenizer,
    device: torch.device,
    max_samples: int = 1000,
    compute_generation: bool = True,
) -> VAEMetrics:
    """Compute all VAE metrics on a dataset.

    Args:
        model: VAE model
        dataloader: Evaluation dataloader
        tokenizer: Tokenizer for decoding
        device: Compute device
        max_samples: Maximum samples to evaluate
        compute_generation: Whether to compute generation metrics (slow)

    Returns:
        VAEMetrics with all scores
    """
    model.eval()

    # Metric trackers
    au_counter = ActiveUnitsCounter()
    recon_metrics = ReconstructionMetrics()
    latent_stats = LatentStatistics()

    samples_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= max_samples:
                break

            # Move batch to device
            batch = batch.to(device)

            # Forward pass
            output = model(
                instruction_ids=batch.instruction_ids,
                instruction_attention_mask=batch.instruction_attention_mask,
                response_ids=batch.response_ids,
                response_attention_mask=batch.response_attention_mask,
            )

            # Update latent metrics
            au_counter.update(output.mu)
            latent_stats.update(output.mu, output.logvar)

            # Compute generation metrics if requested
            if compute_generation:
                # Generate reconstructions
                generated_ids = model.generate(
                    output.z,
                    max_length=batch.response_ids.size(1),
                    temperature=0.7,
                    do_sample=True,
                )

                # Decode
                predictions = tokenizer.batch_decode(generated_ids)
                references = tokenizer.batch_decode(
                    batch.response_ids * (batch.response_attention_mask)
                )

                recon_metrics.update(predictions, references)

            samples_processed += batch.instruction_ids.size(0)

    # Compute final metrics
    active_units, au_ratio, _ = au_counter.compute()
    bleu, rouge_l = recon_metrics.compute() if compute_generation else (0.0, 0.0)
    latent_dict = latent_stats.compute()

    return VAEMetrics(
        bleu=bleu,
        rouge_l=rouge_l,
        active_units=active_units,
        active_unit_ratio=au_ratio,
        total_kl=latent_dict.get("kl_total", 0.0),
        mean_kl_per_dim=latent_dict.get("kl_mean", 0.0) / 64,  # Assuming 64D latent
        mu_mean=latent_dict.get("mu_mean", 0.0),
        mu_std=latent_dict.get("mu_std", 0.0),
        logvar_mean=latent_dict.get("logvar_mean", 0.0),
    )
