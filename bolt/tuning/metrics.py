"""
Comprehensive Metrics Module for BOLT Hyperparameter Tuning

25 metrics across 5 categories:
- VAE Quality (6 metrics)
- Scorer/Selection (5 metrics)
- GP Quality (5 metrics)
- Optimization Quality (5 metrics)
- End-to-End (4 metrics)

All metrics have:
- Unified API (compute, target, passed)
- Serialization support
- Historical tracking
"""

from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


class MetricCategory(Enum):
    """Categories of metrics for organization."""
    VAE = "vae"
    SCORER = "scorer"
    GP = "gp"
    OPTIMIZATION = "optimization"
    END_TO_END = "end_to_end"


class TargetDirection(Enum):
    """Direction of optimization for the metric."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    RANGE = "range"  # Target is a range [low, high]


@dataclass
class MetricTarget:
    """Specification for metric target."""
    direction: TargetDirection
    value: Optional[float] = None  # For MINIMIZE/MAXIMIZE threshold
    low: Optional[float] = None    # For RANGE
    high: Optional[float] = None   # For RANGE

    def is_passed(self, actual: float) -> bool:
        """Check if actual value meets target."""
        if self.direction == TargetDirection.MAXIMIZE:
            return actual >= self.value if self.value is not None else True
        elif self.direction == TargetDirection.MINIMIZE:
            return actual <= self.value if self.value is not None else True
        elif self.direction == TargetDirection.RANGE:
            low_ok = actual >= self.low if self.low is not None else True
            high_ok = actual <= self.high if self.high is not None else True
            return low_ok and high_ok
        return True

    def distance_to_target(self, actual: float) -> float:
        """Compute distance to target (0 = perfect, >0 = not met)."""
        if self.direction == TargetDirection.MAXIMIZE:
            if self.value is None:
                return 0.0
            return max(0.0, self.value - actual)
        elif self.direction == TargetDirection.MINIMIZE:
            if self.value is None:
                return 0.0
            return max(0.0, actual - self.value)
        elif self.direction == TargetDirection.RANGE:
            if self.low is not None and actual < self.low:
                return self.low - actual
            if self.high is not None and actual > self.high:
                return actual - self.high
            return 0.0
        return 0.0


@dataclass
class MetricResult:
    """Result of a single metric computation."""
    name: str
    value: float
    target: MetricTarget
    passed: bool
    category: MetricCategory
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "target": {
                "direction": self.target.direction.value,
                "value": self.target.value,
                "low": self.target.low,
                "high": self.target.high,
            },
            "passed": self.passed,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MetricResult:
        """Deserialize from dictionary."""
        target = MetricTarget(
            direction=TargetDirection(d["target"]["direction"]),
            value=d["target"]["value"],
            low=d["target"]["low"],
            high=d["target"]["high"],
        )
        return cls(
            name=d["name"],
            value=d["value"],
            target=target,
            passed=d["passed"],
            category=MetricCategory(d["category"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            metadata=d.get("metadata", {}),
        )


class BaseMetric(ABC):
    """Base class for all metrics."""

    def __init__(
        self,
        name: str,
        category: MetricCategory,
        target: MetricTarget,
        description: str = "",
    ):
        self.name = name
        self.category = category
        self.target = target
        self.description = description
        self.history: List[MetricResult] = []

    @abstractmethod
    def compute(self, **kwargs) -> float:
        """Compute the metric value."""
        pass

    def evaluate(self, **kwargs) -> MetricResult:
        """Compute and create result with target checking."""
        start_time = time.time()
        value = self.compute(**kwargs)
        compute_time = time.time() - start_time

        passed = self.target.is_passed(value)
        result = MetricResult(
            name=self.name,
            value=value,
            target=self.target,
            passed=passed,
            category=self.category,
            metadata={"compute_time_seconds": compute_time},
        )
        self.history.append(result)
        return result

    def get_history(self) -> List[MetricResult]:
        """Get historical values."""
        return self.history

    def get_trend(self, window: int = 5) -> Optional[float]:
        """Get trend over last N evaluations (positive = improving)."""
        if len(self.history) < 2:
            return None

        recent = self.history[-window:]
        values = [r.value for r in recent]

        # Compute linear regression slope
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)

        # Flip sign if minimizing (so positive = improving)
        if self.target.direction == TargetDirection.MINIMIZE:
            slope = -slope

        return float(slope)


# =============================================================================
# VAE METRICS (6 metrics)
# =============================================================================

class ReconstructionCosineMetric(BaseMetric):
    """Mean cosine similarity between input and reconstructed embeddings."""

    def __init__(self):
        super().__init__(
            name="vae_reconstruction_cosine",
            category=MetricCategory.VAE,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.93),
            description="Mean cosine similarity of VAE reconstruction",
        )

    def compute(
        self,
        original_embeddings: torch.Tensor,
        reconstructed_embeddings: torch.Tensor,
        **kwargs,
    ) -> float:
        """Compute mean cosine similarity."""
        original_embeddings = F.normalize(original_embeddings, dim=-1)
        reconstructed_embeddings = F.normalize(reconstructed_embeddings, dim=-1)

        cos_sim = (original_embeddings * reconstructed_embeddings).sum(dim=-1)
        return float(cos_sim.mean().item())


class ReconstructionMSEMetric(BaseMetric):
    """Mean squared error of VAE reconstruction."""

    def __init__(self):
        super().__init__(
            name="vae_reconstruction_mse",
            category=MetricCategory.VAE,
            target=MetricTarget(TargetDirection.MINIMIZE, value=0.05),
            description="MSE of VAE reconstruction",
        )

    def compute(
        self,
        original_embeddings: torch.Tensor,
        reconstructed_embeddings: torch.Tensor,
        **kwargs,
    ) -> float:
        """Compute MSE."""
        mse = F.mse_loss(reconstructed_embeddings, original_embeddings)
        return float(mse.item())


class KLDivergenceMetric(BaseMetric):
    """KL divergence of VAE latent distribution from N(0,I)."""

    def __init__(self):
        super().__init__(
            name="vae_kl_divergence",
            category=MetricCategory.VAE,
            target=MetricTarget(TargetDirection.RANGE, low=0.1, high=1.0),
            description="KL divergence (too low = posterior collapse, too high = poor reconstruction)",
        )

    def compute(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        **kwargs,
    ) -> float:
        """Compute KL divergence."""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return float(kl.mean().item())


class RetrievalAccuracyMetric(BaseMetric):
    """Retrieval accuracy: encode K items, decode, retrieve from pool."""

    def __init__(self, k: int = 8):
        self.k = k
        super().__init__(
            name=f"vae_retrieval_accuracy_at_{k}",
            category=MetricCategory.VAE,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.85),
            description=f"Retrieval accuracy @{k} (encode→decode→retrieve)",
        )

    def compute(
        self,
        vae_encoder: Callable,
        vae_decoder: Callable,
        pool_embeddings: torch.Tensor,
        n_trials: int = 100,
        **kwargs,
    ) -> float:
        """
        Compute retrieval accuracy.

        For n_trials:
        1. Sample K random items from pool
        2. Encode to latent
        3. Decode back to embedding space
        4. Find K nearest neighbors in pool
        5. Measure overlap with original K
        """
        pool_embeddings = F.normalize(pool_embeddings, dim=-1)
        n_pool = pool_embeddings.shape[0]

        accuracies = []
        for _ in range(n_trials):
            # Sample K random indices
            indices = torch.randperm(n_pool)[:self.k]
            original_embs = pool_embeddings[indices]

            # Encode and decode
            with torch.no_grad():
                z = vae_encoder(original_embs)
                if isinstance(z, tuple):
                    z = z[0]  # Handle (mu, logvar) output
                reconstructed = vae_decoder(z)
                reconstructed = F.normalize(reconstructed, dim=-1)

            # Find nearest neighbors for each reconstructed embedding
            retrieved_indices = set()
            for recon_emb in reconstructed:
                similarities = torch.matmul(pool_embeddings, recon_emb)
                top_idx = similarities.argmax().item()
                retrieved_indices.add(top_idx)

            # Compute overlap
            original_set = set(indices.tolist())
            overlap = len(original_set & retrieved_indices)
            accuracies.append(overlap / self.k)

        return float(np.mean(accuracies))


class LipschitzConstantMetric(BaseMetric):
    """Approximate Lipschitz constant of VAE decoder (smoothness measure)."""

    def __init__(self):
        super().__init__(
            name="vae_lipschitz_constant",
            category=MetricCategory.VAE,
            target=MetricTarget(TargetDirection.MINIMIZE, value=10.0),
            description="Lipschitz constant (lower = smoother latent space)",
        )

    def compute(
        self,
        vae_decoder: Callable,
        z_samples: torch.Tensor,
        epsilon: float = 0.01,
        n_directions: int = 100,
        **kwargs,
    ) -> float:
        """
        Estimate Lipschitz constant via random perturbations.

        L ≈ max(||f(z+ε) - f(z)|| / ||ε||)
        """
        lipschitz_estimates = []

        with torch.no_grad():
            for z in z_samples:
                for _ in range(n_directions):
                    # Random direction, normalized to epsilon norm
                    direction = torch.randn_like(z)
                    direction = direction / direction.norm() * epsilon

                    z_perturbed = z + direction

                    # Decode both
                    out_orig = vae_decoder(z.unsqueeze(0))
                    out_pert = vae_decoder(z_perturbed.unsqueeze(0))

                    if isinstance(out_orig, tuple):
                        out_orig = out_orig[0]
                    if isinstance(out_pert, tuple):
                        out_pert = out_pert[0]

                    # Compute output change
                    output_change = (out_orig - out_pert).norm().item()

                    # Lipschitz estimate
                    lipschitz_estimates.append(output_change / epsilon)

        return float(max(lipschitz_estimates))


class Percentile10CosineMetric(BaseMetric):
    """10th percentile of cosine similarity (worst-case reconstruction)."""

    def __init__(self):
        super().__init__(
            name="vae_percentile10_cosine",
            category=MetricCategory.VAE,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.70),
            description="10th percentile cosine similarity (worst-case quality)",
        )

    def compute(
        self,
        original_embeddings: torch.Tensor,
        reconstructed_embeddings: torch.Tensor,
        **kwargs,
    ) -> float:
        """Compute 10th percentile cosine similarity."""
        original_embeddings = F.normalize(original_embeddings, dim=-1)
        reconstructed_embeddings = F.normalize(reconstructed_embeddings, dim=-1)

        cos_sim = (original_embeddings * reconstructed_embeddings).sum(dim=-1)
        percentile_10 = float(np.percentile(cos_sim.cpu().numpy(), 10))
        return percentile_10


# =============================================================================
# SCORER/SELECTION METRICS (5 metrics) - BOLT only
# =============================================================================

class NDCGMetric(BaseMetric):
    """Normalized Discounted Cumulative Gain for exemplar ranking."""

    def __init__(self, k: int = 8):
        self.k = k
        super().__init__(
            name=f"scorer_ndcg_at_{k}",
            category=MetricCategory.SCORER,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.70),
            description=f"NDCG@{k} for exemplar ranking quality",
        )

    def compute(
        self,
        predicted_scores: torch.Tensor,
        relevance_labels: torch.Tensor,
        **kwargs,
    ) -> float:
        """
        Compute NDCG@K.

        Args:
            predicted_scores: (batch, n_items) predicted relevance scores
            relevance_labels: (batch, n_items) ground truth relevance
        """
        batch_size = predicted_scores.shape[0]
        ndcg_scores = []

        for i in range(batch_size):
            scores = predicted_scores[i]
            labels = relevance_labels[i]

            # Sort by predicted scores
            _, sorted_indices = torch.sort(scores, descending=True)
            sorted_labels = labels[sorted_indices[:self.k]]

            # DCG - ensure tensors are on same device
            device = sorted_labels.device
            positions = torch.arange(1, self.k + 1, dtype=torch.float32, device=device)
            discounts = torch.log2(positions + 1)
            dcg = (sorted_labels / discounts).sum().item()

            # Ideal DCG
            ideal_sorted, _ = torch.sort(labels, descending=True)
            ideal_labels = ideal_sorted[:self.k]
            idcg = (ideal_labels / discounts).sum().item()

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores))


class MRRMetric(BaseMetric):
    """Mean Reciprocal Rank for exemplar selection."""

    def __init__(self):
        super().__init__(
            name="scorer_mrr",
            category=MetricCategory.SCORER,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.50),
            description="Mean Reciprocal Rank for best exemplar",
        )

    def compute(
        self,
        predicted_scores: torch.Tensor,
        relevance_labels: torch.Tensor,
        **kwargs,
    ) -> float:
        """
        Compute MRR.

        Finds rank of the most relevant item in predicted ranking.
        """
        batch_size = predicted_scores.shape[0]
        reciprocal_ranks = []

        for i in range(batch_size):
            scores = predicted_scores[i]
            labels = relevance_labels[i]

            # Find index of best true item
            best_true_idx = labels.argmax().item()

            # Sort by predicted scores and find rank of best_true_idx
            _, sorted_indices = torch.sort(scores, descending=True)
            rank = (sorted_indices == best_true_idx).nonzero(as_tuple=True)[0].item() + 1

            reciprocal_ranks.append(1.0 / rank)

        return float(np.mean(reciprocal_ranks))


class SelectionLossMetric(BaseMetric):
    """ListMLE selection loss (lower = better ranking)."""

    def __init__(self):
        super().__init__(
            name="scorer_selection_loss",
            category=MetricCategory.SCORER,
            target=MetricTarget(TargetDirection.MINIMIZE, value=2.0),
            description="ListMLE ranking loss",
        )

    def compute(self, loss_value: float, **kwargs) -> float:
        """Return the provided loss value."""
        return loss_value


class ExemplarDiversityMetric(BaseMetric):
    """Mean pairwise distance between selected exemplars."""

    def __init__(self):
        super().__init__(
            name="scorer_exemplar_diversity",
            category=MetricCategory.SCORER,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.30),
            description="Mean pairwise cosine distance between selected exemplars",
        )

    def compute(
        self,
        selected_embeddings: torch.Tensor,
        **kwargs,
    ) -> float:
        """
        Compute mean pairwise cosine distance.

        Args:
            selected_embeddings: (K, D) embeddings of selected exemplars
        """
        selected_embeddings = F.normalize(selected_embeddings, dim=-1)

        # Compute pairwise cosine similarities
        similarity_matrix = torch.matmul(selected_embeddings, selected_embeddings.T)

        # Extract upper triangle (excluding diagonal)
        k = selected_embeddings.shape[0]
        triu_indices = torch.triu_indices(k, k, offset=1)
        pairwise_sims = similarity_matrix[triu_indices[0], triu_indices[1]]

        # Convert to distances (1 - similarity)
        pairwise_dists = 1 - pairwise_sims

        return float(pairwise_dists.mean().item())


class ExemplarVarianceMetric(BaseMetric):
    """Variance in exemplar selection across BO iterations."""

    def __init__(self):
        super().__init__(
            name="scorer_exemplar_variance",
            category=MetricCategory.SCORER,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.0),  # Any variance > 0
            description="Variance in exemplar IDs across iterations (>0 = exploring)",
        )

    def compute(
        self,
        exemplar_id_history: List[List[int]],
        **kwargs,
    ) -> float:
        """
        Compute variance in exemplar selections.

        Args:
            exemplar_id_history: List of exemplar ID lists per iteration
        """
        if len(exemplar_id_history) < 2:
            return 0.0

        # Flatten and compute unique IDs per iteration
        unique_counts = [len(set(ids)) for ids in exemplar_id_history]

        # Also compute Jaccard distance between consecutive iterations
        jaccard_distances = []
        for i in range(1, len(exemplar_id_history)):
            set_prev = set(exemplar_id_history[i - 1])
            set_curr = set(exemplar_id_history[i])

            intersection = len(set_prev & set_curr)
            union = len(set_prev | set_curr)

            jaccard = 1 - (intersection / union if union > 0 else 0)
            jaccard_distances.append(jaccard)

        # Return mean Jaccard distance (0 = identical, 1 = completely different)
        return float(np.mean(jaccard_distances)) if jaccard_distances else 0.0


# =============================================================================
# GP METRICS (5 metrics)
# =============================================================================

class SpearmanCorrelationMetric(BaseMetric):
    """Spearman rank correlation between GP predictions and actual values."""

    def __init__(self):
        super().__init__(
            name="gp_spearman_correlation",
            category=MetricCategory.GP,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.40),
            description="Spearman correlation (rank agreement between predictions and actuals)",
        )

    def compute(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        **kwargs,
    ) -> float:
        """Compute Spearman correlation."""
        if len(predictions) < 3:
            return 0.0

        correlation, p_value = stats.spearmanr(predictions, actuals)

        # Handle NaN (e.g., constant predictions)
        if np.isnan(correlation):
            return 0.0

        return float(correlation)


class GPRMSEMetric(BaseMetric):
    """Root Mean Squared Error of GP predictions."""

    def __init__(self):
        super().__init__(
            name="gp_rmse",
            category=MetricCategory.GP,
            target=MetricTarget(TargetDirection.MINIMIZE, value=0.05),
            description="RMSE of GP predictions",
        )

    def compute(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        **kwargs,
    ) -> float:
        """Compute RMSE."""
        mse = np.mean((predictions - actuals) ** 2)
        return float(np.sqrt(mse))


class CalibrationErrorMetric(BaseMetric):
    """Calibration error: |predicted confidence - actual accuracy|."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        super().__init__(
            name="gp_calibration_error",
            category=MetricCategory.GP,
            target=MetricTarget(TargetDirection.MINIMIZE, value=0.10),
            description="Expected calibration error (well-calibrated = low)",
        )

    def compute(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        actuals: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Compute Expected Calibration Error.

        For regression, we check if actual values fall within predicted intervals.
        """
        # Compute prediction errors
        errors = np.abs(predictions - actuals)

        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_errors = errors[sorted_indices]
        sorted_uncertainties = uncertainties[sorted_indices]

        # Bin by uncertainty
        bin_edges = np.linspace(0, len(errors), self.n_bins + 1, dtype=int)

        calibration_errors = []
        for i in range(self.n_bins):
            start, end = bin_edges[i], bin_edges[i + 1]
            if end <= start:
                continue

            bin_errors = sorted_errors[start:end]
            bin_uncertainties = sorted_uncertainties[start:end]

            # Expected: error ≈ uncertainty (for well-calibrated model)
            mean_error = np.mean(bin_errors)
            mean_uncertainty = np.mean(bin_uncertainties)

            calibration_errors.append(abs(mean_error - mean_uncertainty))

        return float(np.mean(calibration_errors)) if calibration_errors else 0.0


class LengthscaleRatioMetric(BaseMetric):
    """Ratio of instruction to exemplar lengthscales (balance check)."""

    def __init__(self):
        super().__init__(
            name="gp_lengthscale_ratio",
            category=MetricCategory.GP,
            target=MetricTarget(TargetDirection.RANGE, low=0.5, high=2.0),
            description="Ratio of mean instruction/exemplar lengthscales",
        )

    def compute(
        self,
        instruction_lengthscales: np.ndarray,
        exemplar_lengthscales: np.ndarray,
        **kwargs,
    ) -> float:
        """Compute ratio of mean lengthscales."""
        inst_mean = np.mean(instruction_lengthscales)
        ex_mean = np.mean(exemplar_lengthscales)

        if ex_mean < 1e-6:
            return float('inf')

        return float(inst_mean / ex_mean)


class GPNLLMetric(BaseMetric):
    """Negative Log-Likelihood on held-out data."""

    def __init__(self):
        super().__init__(
            name="gp_nll",
            category=MetricCategory.GP,
            target=MetricTarget(TargetDirection.MINIMIZE, value=None),  # Minimize without threshold
            description="Negative log-likelihood on held-out data",
        )

    def compute(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        actuals: np.ndarray,
        **kwargs,
    ) -> float:
        """Compute NLL assuming Gaussian predictions."""
        # NLL = 0.5 * log(2π) + 0.5 * log(σ²) + (y - μ)² / (2σ²)
        # Add small epsilon to prevent division by zero
        variance = uncertainties ** 2 + 1e-10

        nll = 0.5 * np.log(2 * np.pi * variance) + 0.5 * (actuals - predictions) ** 2 / variance

        return float(np.mean(nll))


# =============================================================================
# OPTIMIZATION METRICS (5 metrics)
# =============================================================================

class SystemGapMetric(BaseMetric):
    """L2 distance between optimized latent and realized latent."""

    def __init__(self):
        super().__init__(
            name="opt_system_gap",
            category=MetricCategory.OPTIMIZATION,
            target=MetricTarget(TargetDirection.MINIMIZE, value=0.5),
            description="||z_opt - z_realized|| (optimization faithfulness)",
        )

    def compute(
        self,
        z_optimized: torch.Tensor,
        z_realized: torch.Tensor,
        **kwargs,
    ) -> float:
        """Compute L2 distance."""
        distance = (z_optimized - z_realized).norm(dim=-1)
        return float(distance.mean().item())


class RejectionRateMetric(BaseMetric):
    """Fraction of candidates rejected during inference."""

    def __init__(self):
        super().__init__(
            name="opt_rejection_rate",
            category=MetricCategory.OPTIMIZATION,
            target=MetricTarget(TargetDirection.MINIMIZE, value=0.30),
            description="Rejection rate during candidate generation",
        )

    def compute(
        self,
        rejected_count: int,
        total_attempts: int,
        **kwargs,
    ) -> float:
        """Compute rejection rate."""
        if total_attempts == 0:
            return 0.0
        return float(rejected_count / total_attempts)


class ImprovementRateMetric(BaseMetric):
    """Fraction of BO iterations that improved over previous best."""

    def __init__(self):
        super().__init__(
            name="opt_improvement_rate",
            category=MetricCategory.OPTIMIZATION,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.20),
            description="Fraction of iterations that improved",
        )

    def compute(
        self,
        error_history: List[float],
        **kwargs,
    ) -> float:
        """Compute improvement rate."""
        if len(error_history) < 2:
            return 0.0

        improvements = 0
        best_so_far = error_history[0]

        for error in error_history[1:]:
            if error < best_so_far:
                improvements += 1
                best_so_far = error

        return float(improvements / (len(error_history) - 1))


class BestErrorRateMetric(BaseMetric):
    """Best (minimum) error rate achieved."""

    def __init__(self):
        super().__init__(
            name="opt_best_error_rate",
            category=MetricCategory.OPTIMIZATION,
            target=MetricTarget(TargetDirection.MINIMIZE, value=0.085),
            description="Best error rate achieved (1 - accuracy)",
        )

    def compute(
        self,
        error_history: List[float],
        **kwargs,
    ) -> float:
        """Return minimum error."""
        if not error_history:
            return 1.0
        return float(min(error_history))


class ConvergenceSpeedMetric(BaseMetric):
    """Number of iterations to reach 90% of final improvement."""

    def __init__(self):
        super().__init__(
            name="opt_convergence_speed",
            category=MetricCategory.OPTIMIZATION,
            target=MetricTarget(TargetDirection.MINIMIZE, value=20.0),
            description="Iterations to 90% of final improvement",
        )

    def compute(
        self,
        error_history: List[float],
        **kwargs,
    ) -> float:
        """Compute iterations to 90% improvement."""
        if len(error_history) < 2:
            return float(len(error_history))

        initial_error = error_history[0]
        final_error = min(error_history)

        if initial_error <= final_error:
            return float(len(error_history))  # No improvement

        total_improvement = initial_error - final_error
        target_error = initial_error - 0.9 * total_improvement

        for i, error in enumerate(error_history):
            if error <= target_error:
                return float(i + 1)

        return float(len(error_history))


# =============================================================================
# END-TO-END METRICS (4 metrics)
# =============================================================================

class FinalAccuracyMetric(BaseMetric):
    """Final accuracy achieved (1 - best_error)."""

    def __init__(self):
        super().__init__(
            name="e2e_final_accuracy",
            category=MetricCategory.END_TO_END,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=0.915),
            description="Final accuracy (1 - best error rate)",
        )

    def compute(self, best_error_rate: float, **kwargs) -> float:
        """Compute accuracy."""
        return 1.0 - best_error_rate


class SampleEfficiencyMetric(BaseMetric):
    """Accuracy per 1000 LLM calls."""

    def __init__(self):
        super().__init__(
            name="e2e_sample_efficiency",
            category=MetricCategory.END_TO_END,
            target=MetricTarget(TargetDirection.MAXIMIZE, value=None),
            description="Accuracy per 1000 LLM calls (higher = more efficient)",
        )

    def compute(
        self,
        best_error_rate: float,
        total_llm_calls: int,
        **kwargs,
    ) -> float:
        """Compute sample efficiency."""
        accuracy = 1.0 - best_error_rate
        if total_llm_calls == 0:
            return 0.0
        return float(accuracy / (total_llm_calls / 1000))


class WallClockTimeMetric(BaseMetric):
    """Total wall-clock time in seconds."""

    def __init__(self):
        super().__init__(
            name="e2e_wall_clock_time",
            category=MetricCategory.END_TO_END,
            target=MetricTarget(TargetDirection.MINIMIZE, value=None),
            description="Total wall-clock time (seconds)",
        )

    def compute(self, elapsed_seconds: float, **kwargs) -> float:
        """Return elapsed time."""
        return elapsed_seconds


class GPUMemoryPeakMetric(BaseMetric):
    """Peak GPU memory usage in GB."""

    def __init__(self):
        super().__init__(
            name="e2e_gpu_memory_peak",
            category=MetricCategory.END_TO_END,
            target=MetricTarget(TargetDirection.MINIMIZE, value=20.0),
            description="Peak GPU memory usage (GB)",
        )

    def compute(self, **kwargs) -> float:
        """Get peak GPU memory."""
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            return float(peak_bytes / (1024 ** 3))
        return 0.0


# =============================================================================
# METRIC COLLECTIONS
# =============================================================================

class VAEMetrics:
    """Collection of VAE quality metrics."""

    def __init__(self, retrieval_k: int = 8):
        self.reconstruction_cosine = ReconstructionCosineMetric()
        self.reconstruction_mse = ReconstructionMSEMetric()
        self.kl_divergence = KLDivergenceMetric()
        self.retrieval_accuracy = RetrievalAccuracyMetric(k=retrieval_k)
        self.lipschitz_constant = LipschitzConstantMetric()
        self.percentile10_cosine = Percentile10CosineMetric()

    def all_metrics(self) -> List[BaseMetric]:
        return [
            self.reconstruction_cosine,
            self.reconstruction_mse,
            self.kl_divergence,
            self.retrieval_accuracy,
            self.lipschitz_constant,
            self.percentile10_cosine,
        ]

    def checkpoint_metrics(self) -> List[BaseMetric]:
        """Metrics required for phase checkpoint."""
        return [
            self.retrieval_accuracy,
            self.lipschitz_constant,
        ]


class ScorerMetrics:
    """Collection of scorer/selection metrics (BOLT only)."""

    def __init__(self, ndcg_k: int = 8):
        self.ndcg = NDCGMetric(k=ndcg_k)
        self.mrr = MRRMetric()
        self.selection_loss = SelectionLossMetric()
        self.exemplar_diversity = ExemplarDiversityMetric()
        self.exemplar_variance = ExemplarVarianceMetric()

    def all_metrics(self) -> List[BaseMetric]:
        return [
            self.ndcg,
            self.mrr,
            self.selection_loss,
            self.exemplar_diversity,
            self.exemplar_variance,
        ]

    def checkpoint_metrics(self) -> List[BaseMetric]:
        """Metrics required for phase checkpoint."""
        return [self.ndcg]


class GPMetrics:
    """Collection of GP quality metrics."""

    def __init__(self):
        self.spearman = SpearmanCorrelationMetric()
        self.rmse = GPRMSEMetric()
        self.calibration_error = CalibrationErrorMetric()
        self.lengthscale_ratio = LengthscaleRatioMetric()
        self.nll = GPNLLMetric()

    def all_metrics(self) -> List[BaseMetric]:
        return [
            self.spearman,
            self.rmse,
            self.calibration_error,
            self.lengthscale_ratio,
            self.nll,
        ]

    def checkpoint_metrics(self) -> List[BaseMetric]:
        """Metrics required for phase checkpoint."""
        return [self.spearman]


class OptimizationMetrics:
    """Collection of optimization quality metrics."""

    def __init__(self):
        self.system_gap = SystemGapMetric()
        self.rejection_rate = RejectionRateMetric()
        self.improvement_rate = ImprovementRateMetric()
        self.best_error_rate = BestErrorRateMetric()
        self.convergence_speed = ConvergenceSpeedMetric()

    def all_metrics(self) -> List[BaseMetric]:
        return [
            self.system_gap,
            self.rejection_rate,
            self.improvement_rate,
            self.best_error_rate,
            self.convergence_speed,
        ]

    def checkpoint_metrics(self) -> List[BaseMetric]:
        """Metrics required for phase checkpoint."""
        return [self.best_error_rate]


class EndToEndMetrics:
    """Collection of end-to-end metrics."""

    def __init__(self):
        self.final_accuracy = FinalAccuracyMetric()
        self.sample_efficiency = SampleEfficiencyMetric()
        self.wall_clock_time = WallClockTimeMetric()
        self.gpu_memory_peak = GPUMemoryPeakMetric()

    def all_metrics(self) -> List[BaseMetric]:
        return [
            self.final_accuracy,
            self.sample_efficiency,
            self.wall_clock_time,
            self.gpu_memory_peak,
        ]

    def checkpoint_metrics(self) -> List[BaseMetric]:
        """Metrics required for final checkpoint."""
        return [self.final_accuracy]


# =============================================================================
# METRIC REGISTRY
# =============================================================================

class MetricRegistry:
    """
    Central registry for all metrics.

    Provides:
    - Unified access to all metrics
    - Serialization/deserialization
    - History tracking
    - Checkpoint validation
    """

    def __init__(self, retrieval_k: int = 8, ndcg_k: int = 8):
        self.vae = VAEMetrics(retrieval_k=retrieval_k)
        self.scorer = ScorerMetrics(ndcg_k=ndcg_k)
        self.gp = GPMetrics()
        self.optimization = OptimizationMetrics()
        self.e2e = EndToEndMetrics()

        self._all_metrics: Dict[str, BaseMetric] = {}
        self._build_registry()

    def _build_registry(self):
        """Build flat registry of all metrics."""
        for collection in [self.vae, self.scorer, self.gp, self.optimization, self.e2e]:
            for metric in collection.all_metrics():
                self._all_metrics[metric.name] = metric

    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """Get metric by name."""
        return self._all_metrics.get(name)

    def get_all_metrics(self) -> List[BaseMetric]:
        """Get all metrics."""
        return list(self._all_metrics.values())

    def get_metrics_by_category(self, category: MetricCategory) -> List[BaseMetric]:
        """Get metrics by category."""
        return [m for m in self._all_metrics.values() if m.category == category]

    def get_checkpoint_metrics(self, phase: str) -> List[BaseMetric]:
        """Get checkpoint metrics for a phase."""
        phase_map = {
            "vae": self.vae.checkpoint_metrics(),
            "scorer": self.scorer.checkpoint_metrics(),
            "gp": self.gp.checkpoint_metrics(),
            "optimization": self.optimization.checkpoint_metrics(),
            "inference": self.e2e.checkpoint_metrics(),
            "e2e": self.e2e.checkpoint_metrics(),
        }
        return phase_map.get(phase, [])

    def check_checkpoint(self, phase: str, results: Dict[str, MetricResult]) -> Tuple[bool, List[str]]:
        """
        Check if all checkpoint metrics for a phase are passed.

        Returns:
            (all_passed, list_of_failed_metric_names)
        """
        checkpoint_metrics = self.get_checkpoint_metrics(phase)
        failed = []

        for metric in checkpoint_metrics:
            result = results.get(metric.name)
            if result is None or not result.passed:
                failed.append(metric.name)

        return len(failed) == 0, failed

    def save_history(self, path: Path):
        """Save all metric histories to JSON."""
        history = {}
        for name, metric in self._all_metrics.items():
            history[name] = [r.to_dict() for r in metric.history]

        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def load_history(self, path: Path):
        """Load metric histories from JSON."""
        with open(path, "r") as f:
            history = json.load(f)

        for name, results in history.items():
            metric = self._all_metrics.get(name)
            if metric:
                metric.history = [MetricResult.from_dict(r) for r in results]

    def summary(self) -> Dict[str, Any]:
        """Generate summary of all metrics."""
        summary = {
            "total_metrics": len(self._all_metrics),
            "by_category": {},
            "checkpoint_status": {},
        }

        for category in MetricCategory:
            metrics = self.get_metrics_by_category(category)
            summary["by_category"][category.value] = {
                "count": len(metrics),
                "metrics": [m.name for m in metrics],
            }

        for phase in ["vae", "scorer", "gp", "inference"]:
            checkpoint_metrics = self.get_checkpoint_metrics(phase)
            summary["checkpoint_status"][phase] = {
                "required": [m.name for m in checkpoint_metrics],
                "targets": {m.name: m.target.value for m in checkpoint_metrics},
            }

        return summary
