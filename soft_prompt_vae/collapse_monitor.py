"""Adaptive collapse detection and intervention for VAE training.

Monitors for posterior collapse (decoder ignoring latent codes) and triggers
corrective interventions during training. Implements three key metrics:

1. Active Units (AU) - dimensions with variance above threshold
2. Mutual Information estimate via InfoNCE bound
3. KL divergence trends

Based on:
- "Avoiding Latent Variable Collapse with Generative Skip Models" (Dieng et al., 2019)
- "Lagging Inference Networks and Posterior Collapse in VAEs" (He et al., 2019)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class CollapseMetrics:
    """Metrics for assessing latent space health."""

    # Active Units metrics
    active_units: int  # Number of dimensions with variance > threshold
    active_unit_ratio: float  # Ratio of active to total dimensions (0-1)

    # Information metrics
    mutual_info_estimate: float  # Lower bound on I(X; Z) from InfoNCE

    # KL metrics
    mean_kl: float  # Mean KL per dimension

    # Collapse assessment
    is_collapsing: bool  # Whether intervention is recommended
    intervention_level: int  # 0=none, 1=mild, 2=moderate, 3=severe

    # Raw variance for debugging
    variance_per_dim: Optional[Tensor] = None


@dataclass
class CollapseMonitorConfig:
    """Configuration for collapse monitoring."""

    # Active Units thresholds
    au_variance_threshold: float = 0.01  # Minimum variance to count as "active"
    au_ratio_warning: float = 0.3  # Below this triggers level 1
    au_ratio_critical: float = 0.1  # Below this triggers level 3

    # Mutual Information thresholds (nats)
    mi_warning: float = 1.0  # Below this triggers warning
    mi_critical: float = 0.5  # Below this triggers critical

    # KL thresholds
    kl_minimum: float = 0.1  # KL below this suggests severe collapse

    # History for trend detection
    history_size: int = 100  # Number of steps to track


class CollapseMonitor:
    """Monitors latent space utilization and detects posterior collapse.

    Posterior collapse occurs when the VAE decoder learns to ignore the latent
    codes z, instead relying entirely on autoregressive context. This manifests as:
    - Low variance in latent means (all inputs map to similar z)
    - Low KL divergence (posterior ≈ prior)
    - Low mutual information I(X; Z)

    This monitor tracks these metrics and triggers interventions when collapse
    is detected, before it becomes irreversible.

    Usage:
        monitor = CollapseMonitor(latent_dim=64)
        for batch in dataloader:
            output = model(batch)
            metrics = monitor.update(output.mu, output.logvar, output.mu_augmented)
            if metrics.is_collapsing:
                # Trigger intervention
                scheduler.intervene(metrics)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        config: Optional[CollapseMonitorConfig] = None,
    ):
        """Initialize collapse monitor.

        Args:
            latent_dim: Dimensionality of latent space
            config: Monitoring configuration (uses defaults if None)
        """
        self.latent_dim = latent_dim
        self.config = config or CollapseMonitorConfig()

        # Running statistics for Active Units computation
        self._mu_sum = None  # Sum of mu values (latent_dim,)
        self._mu_sq_sum = None  # Sum of mu^2 values (latent_dim,)
        self._count = 0

        # History tracking for trends
        self._au_history: deque = deque(maxlen=self.config.history_size)
        self._kl_history: deque = deque(maxlen=self.config.history_size)
        self._mi_history: deque = deque(maxlen=self.config.history_size)

        # Last computed metrics
        self._last_metrics: Optional[CollapseMetrics] = None
        self._last_variance: Optional[Tensor] = None  # Keep on GPU

    def reset(self) -> None:
        """Reset running statistics (call at epoch boundaries if desired)."""
        self._mu_sum = None
        self._mu_sq_sum = None
        self._count = 0

    def update(
        self,
        mu: Tensor,
        logvar: Tensor,
        mu_augmented: Optional[Tensor] = None,
    ) -> CollapseMetrics:
        """Update statistics and compute collapse metrics.

        Args:
            mu: Latent means (batch, latent_dim)
            logvar: Latent log variances (batch, latent_dim)
            mu_augmented: Augmented latent means for MI estimate (batch, latent_dim)

        Returns:
            CollapseMetrics with current health assessment
        """
        batch_size = mu.size(0)
        device = mu.device

        # Detach for monitoring (don't affect gradients)
        mu = mu.detach()
        logvar = logvar.detach()
        if mu_augmented is not None:
            mu_augmented = mu_augmented.detach()

        # Update running statistics for Active Units
        if self._mu_sum is None:
            self._mu_sum = torch.zeros(self.latent_dim, device=device)
            self._mu_sq_sum = torch.zeros(self.latent_dim, device=device)

        # Move to same device if needed
        self._mu_sum = self._mu_sum.to(device)
        self._mu_sq_sum = self._mu_sq_sum.to(device)

        # Accumulate statistics
        self._mu_sum += mu.sum(dim=0)
        self._mu_sq_sum += (mu ** 2).sum(dim=0)
        self._count += batch_size

        # Compute per-dimension variance
        mean = self._mu_sum / self._count
        variance = (self._mu_sq_sum / self._count) - (mean ** 2)
        variance = torch.clamp(variance, min=0)  # Numerical safety

        # Active Units: dimensions with variance > threshold
        cfg = self.config
        active_mask = variance > cfg.au_variance_threshold
        active_units = active_mask.sum().item()
        au_ratio = active_units / self.latent_dim

        # Compute mean KL per dimension
        # KL = 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        mean_kl = kl_per_dim.mean().item()

        # Estimate Mutual Information using InfoNCE bound
        mi_estimate = self._estimate_mutual_info(mu, mu_augmented)

        # Update histories
        self._au_history.append(au_ratio)
        self._kl_history.append(mean_kl)
        self._mi_history.append(mi_estimate)

        # Assess collapse
        is_collapsing, intervention_level = self._assess_collapse(
            au_ratio, mi_estimate, mean_kl
        )

        # Build metrics (avoid CPU transfer unless debugging)
        metrics = CollapseMetrics(
            active_units=int(active_units),
            active_unit_ratio=au_ratio,
            mutual_info_estimate=mi_estimate,
            mean_kl=mean_kl,
            is_collapsing=is_collapsing,
            intervention_level=intervention_level,
            variance_per_dim=None,  # Only set on demand to avoid CPU transfer
        )

        self._last_metrics = metrics
        self._last_variance = variance  # Keep on GPU for potential debugging
        return metrics

    def _estimate_mutual_info(
        self,
        mu: Tensor,
        mu_augmented: Optional[Tensor],
    ) -> float:
        """Estimate mutual information I(X; Z) using InfoNCE bound.

        When augmented latents are available, we use them as positive pairs
        to compute the InfoNCE lower bound on MI. Without augmentation,
        we estimate from KL divergence (less accurate).

        Args:
            mu: Original latent means (batch, latent_dim)
            mu_augmented: Augmented latent means (batch, latent_dim), optional

        Returns:
            Estimated mutual information in nats
        """
        if mu_augmented is None:
            # Fallback: estimate from batch statistics
            # MI ≈ log(N) - avg NCE loss, but without pairs we estimate from variance
            variance = mu.var(dim=0).mean().item()
            # Higher variance → higher MI (rough approximation)
            return max(0.0, variance * 2.0)

        # Compute InfoNCE-style estimate
        batch_size = mu.size(0)
        if batch_size < 2:
            return 0.0

        # Normalize latents for cosine similarity
        mu_norm = torch.nn.functional.normalize(mu, dim=1)
        mu_aug_norm = torch.nn.functional.normalize(mu_augmented, dim=1)

        # Similarity matrix: (batch, batch)
        similarity = torch.mm(mu_norm, mu_aug_norm.t())

        # Temperature (0.1 is common for InfoNCE)
        temperature = 0.1
        similarity = similarity / temperature

        # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
        # Positive pairs are on diagonal
        labels = torch.arange(batch_size, device=mu.device)
        nce_loss = torch.nn.functional.cross_entropy(similarity, labels)

        # MI lower bound: log(N) - NCE_loss
        mi_estimate = max(0.0, torch.log(torch.tensor(batch_size, dtype=torch.float)).item() - nce_loss.item())

        return mi_estimate

    def _assess_collapse(
        self,
        au_ratio: float,
        mi_estimate: float,
        mean_kl: float,
    ) -> Tuple[bool, int]:
        """Determine if collapse is occurring and intervention level.

        Args:
            au_ratio: Active unit ratio (0-1)
            mi_estimate: Estimated mutual information
            mean_kl: Mean KL per dimension

        Returns:
            (is_collapsing, intervention_level) where level is 0-3
        """
        cfg = self.config

        # Score each metric (higher = worse)
        au_severity = 0
        if au_ratio < cfg.au_ratio_critical:
            au_severity = 3
        elif au_ratio < cfg.au_ratio_warning:
            au_severity = 1

        mi_severity = 0
        if mi_estimate < cfg.mi_critical:
            mi_severity = 3
        elif mi_estimate < cfg.mi_warning:
            mi_severity = 1

        kl_severity = 0
        if mean_kl < cfg.kl_minimum:
            kl_severity = 2

        # Check for downward trends (collapse getting worse)
        trend_severity = 0
        if len(self._au_history) >= 10:
            recent = list(self._au_history)[-10:]
            if recent[-1] < recent[0] * 0.8:  # 20% decline
                trend_severity = 1

        # Combine severities (take max with slight weighting)
        overall_severity = max(au_severity, mi_severity, kl_severity, trend_severity)

        # Require at least two indicators for high severity
        num_indicators = sum([
            au_severity > 0,
            mi_severity > 0,
            kl_severity > 0,
            trend_severity > 0,
        ])

        if overall_severity >= 2 and num_indicators < 2:
            overall_severity = 1

        is_collapsing = overall_severity > 0

        return is_collapsing, overall_severity

    def get_trend_summary(self) -> Dict[str, float]:
        """Get summary of recent trends.

        Returns:
            Dictionary with trend information
        """
        def _trend(history: deque) -> float:
            if len(history) < 5:
                return 0.0
            recent = list(history)
            start_avg = sum(recent[:5]) / 5
            end_avg = sum(recent[-5:]) / 5
            if start_avg == 0:
                return 0.0
            return (end_avg - start_avg) / start_avg

        return {
            "au_trend": _trend(self._au_history),
            "kl_trend": _trend(self._kl_history),
            "mi_trend": _trend(self._mi_history),
            "au_current": self._au_history[-1] if self._au_history else 0.0,
            "kl_current": self._kl_history[-1] if self._kl_history else 0.0,
            "mi_current": self._mi_history[-1] if self._mi_history else 0.0,
        }


class AdaptiveInterventionScheduler:
    """Adjusts training parameters in response to collapse detection.

    Implements a graduated response system:
    - Level 1 (Mild): Small adjustments to encourage latent usage
    - Level 2 (Moderate): Stronger adjustments + contrastive boost
    - Level 3 (Severe): Maximum intervention to rescue from collapse

    All adjustments are multiplicative and reversible to allow recovery
    when collapse metrics improve.
    """

    def __init__(
        self,
        initial_bow_weight: float = 1.5,
        initial_contrastive_weight: float = 0.1,
        initial_word_dropout: float = 0.4,
        max_bow_weight: float = 3.0,
        max_contrastive_weight: float = 0.5,
        min_word_dropout: float = 0.1,
    ):
        """Initialize intervention scheduler.

        Args:
            initial_bow_weight: Starting BoW loss weight
            initial_contrastive_weight: Starting contrastive loss weight
            initial_word_dropout: Starting word dropout rate
            max_bow_weight: Maximum BoW weight during intervention
            max_contrastive_weight: Maximum contrastive weight
            min_word_dropout: Minimum word dropout (reduce to give decoder more signal)
        """
        self.initial_bow_weight = initial_bow_weight
        self.initial_contrastive_weight = initial_contrastive_weight
        self.initial_word_dropout = initial_word_dropout

        self.max_bow_weight = max_bow_weight
        self.max_contrastive_weight = max_contrastive_weight
        self.min_word_dropout = min_word_dropout

        # Current values
        self.current_bow_weight = initial_bow_weight
        self.current_contrastive_weight = initial_contrastive_weight
        self.current_word_dropout = initial_word_dropout

        # Intervention history
        self._intervention_count = 0
        self._last_level = 0

    def intervene(self, metrics: CollapseMetrics) -> Dict[str, float]:
        """Apply intervention based on collapse metrics.

        Args:
            metrics: Current collapse metrics

        Returns:
            Dictionary of adjusted training parameters
        """
        level = metrics.intervention_level

        if level == 0:
            # No intervention needed - gradually return to defaults
            return self._relax_intervention()

        self._intervention_count += 1

        if level >= 3:
            # Severe: Maximum intervention
            self.current_bow_weight = self.max_bow_weight
            self.current_contrastive_weight = self.max_contrastive_weight
            self.current_word_dropout = self.min_word_dropout
            logger.warning(
                f"Severe collapse detected (AU={metrics.active_unit_ratio:.1%}, "
                f"MI={metrics.mutual_info_estimate:.2f}). Maximum intervention applied."
            )

        elif level == 2:
            # Moderate: Significant adjustment
            self.current_bow_weight = min(
                self.current_bow_weight * 1.3,
                self.max_bow_weight
            )
            self.current_contrastive_weight = min(
                self.current_contrastive_weight * 1.3,
                self.max_contrastive_weight
            )
            self.current_word_dropout = max(
                self.current_word_dropout * 0.8,
                self.min_word_dropout
            )
            logger.info(
                f"Moderate collapse detected. Adjusting: BoW={self.current_bow_weight:.2f}, "
                f"contrastive={self.current_contrastive_weight:.2f}"
            )

        elif level == 1:
            # Mild: Small adjustment
            self.current_bow_weight = min(
                self.current_bow_weight * 1.1,
                self.max_bow_weight
            )
            self.current_contrastive_weight = min(
                self.current_contrastive_weight * 1.1,
                self.max_contrastive_weight
            )
            self.current_word_dropout = max(
                self.current_word_dropout * 0.95,
                self.min_word_dropout
            )

        self._last_level = level

        return {
            "bow_weight": self.current_bow_weight,
            "contrastive_weight": self.current_contrastive_weight,
            "word_dropout": self.current_word_dropout,
            "intervention_level": level,
        }

    def _relax_intervention(self) -> Dict[str, float]:
        """Gradually return to initial values when no collapse detected."""
        # Exponential decay back to initial values
        decay = 0.95

        self.current_bow_weight = (
            decay * self.current_bow_weight +
            (1 - decay) * self.initial_bow_weight
        )
        self.current_contrastive_weight = (
            decay * self.current_contrastive_weight +
            (1 - decay) * self.initial_contrastive_weight
        )
        self.current_word_dropout = (
            decay * self.current_word_dropout +
            (1 - decay) * self.initial_word_dropout
        )

        return {
            "bow_weight": self.current_bow_weight,
            "contrastive_weight": self.current_contrastive_weight,
            "word_dropout": self.current_word_dropout,
            "intervention_level": 0,
        }

    def get_status(self) -> Dict[str, float]:
        """Get current intervention status.

        Returns:
            Dictionary with current parameter values and counts
        """
        return {
            "bow_weight": self.current_bow_weight,
            "contrastive_weight": self.current_contrastive_weight,
            "word_dropout": self.current_word_dropout,
            "intervention_count": self._intervention_count,
            "last_level": self._last_level,
        }
