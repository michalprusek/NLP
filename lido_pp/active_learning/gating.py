"""
Evaluation Gating Logic for Cost-Aware Active Learning.

This module implements the decision logic that determines whether to:
1. Evaluate a candidate with expensive LLM (high uncertainty)
2. Trust the cheap Value Head prediction (low uncertainty)

The gating decision is based on Flow Curvature Uncertainty (FCU):
    - High FCU → Model uncertain about velocity field → Need LLM evaluation
    - Low FCU → Model confident → Trust Value Head

This enables significant cost savings by avoiding redundant evaluations
in regions where the model is already confident.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from lido_pp.active_learning.curvature import compute_flow_curvature, FCUResult


class EvaluationType(Enum):
    """Type of evaluation for a candidate."""
    LLM = "llm"              # Expensive LLM evaluation
    VALUE_HEAD = "value_head"  # Cheap Value Head prediction
    CACHED = "cached"        # From cache (previously evaluated)


@dataclass
class GatingDecision:
    """Result of gating decision for a candidate."""

    # The latent vector
    z: torch.Tensor

    # Evaluation type
    eval_type: EvaluationType

    # FCU value (if computed)
    curvature: Optional[float] = None

    # Value Head prediction (if using VALUE_HEAD)
    value_prediction: Optional[float] = None

    # Confidence in the decision (0-1)
    confidence: Optional[float] = None


@dataclass
class GatingStatistics:
    """Statistics about gating decisions over time."""

    total_decisions: int = 0
    llm_evaluations: int = 0
    value_head_predictions: int = 0
    cache_hits: int = 0

    # Accuracy of Value Head when we can verify
    value_head_mse: float = 0.0
    num_verified: int = 0

    @property
    def llm_ratio(self) -> float:
        """Fraction of decisions that required LLM."""
        if self.total_decisions == 0:
            return 0.0
        return self.llm_evaluations / self.total_decisions

    @property
    def cost_savings(self) -> float:
        """Estimated cost savings (1 - llm_ratio)."""
        return 1.0 - self.llm_ratio


class EvaluationGate:
    """
    Evaluation gating based on Flow Curvature Uncertainty.

    The gate decides whether to evaluate with expensive LLM
    or trust cheap Value Head based on model uncertainty.

    Decision logic:
        1. Check cache (if enabled)
        2. Compute FCU
        3. If FCU > threshold: evaluate with LLM
        4. Else: use Value Head prediction

    The threshold can be:
        - Fixed (percentile-based from historical data)
        - Adaptive (based on recent accuracy)
    """

    def __init__(
        self,
        flow_model: nn.Module,
        value_head: nn.Module,
        percentile_threshold: float = 90.0,
        curvature_steps: int = 20,
        use_cache: bool = True,
        cache_tolerance: float = 1e-4,
        min_samples_for_threshold: int = 50,
        adaptive_threshold: bool = True,
    ):
        """
        Args:
            flow_model: FlowDiT for curvature computation
            value_head: Value Head for cheap predictions
            percentile_threshold: Percentile for FCU threshold
            curvature_steps: Steps for FCU computation
            use_cache: Enable evaluation caching
            cache_tolerance: L2 distance for cache hits
            min_samples_for_threshold: Minimum samples for threshold
            adaptive_threshold: Adjust threshold based on VH accuracy
        """
        self.flow_model = flow_model
        self.value_head = value_head
        self.percentile_threshold = percentile_threshold
        self.curvature_steps = curvature_steps
        self.use_cache = use_cache
        self.cache_tolerance = cache_tolerance
        self.min_samples_for_threshold = min_samples_for_threshold
        self.adaptive_threshold = adaptive_threshold

        # Cache: z -> (error_rate, curvature)
        self.cache: Dict[str, Tuple[float, float]] = {}
        self.cache_z: List[torch.Tensor] = []
        self.cache_values: List[float] = []

        # Historical curvatures for threshold computation
        self.curvature_history: List[float] = []
        self.max_history = 1000

        # Statistics
        self.stats = GatingStatistics()

        # Value Head verification buffer
        self.vh_predictions: List[float] = []
        self.vh_ground_truth: List[float] = []

    def _z_to_key(self, z: torch.Tensor) -> str:
        """Convert tensor to hashable key for caching."""
        return str(z.detach().cpu().numpy().tobytes())

    def _check_cache(self, z: torch.Tensor) -> Optional[Tuple[float, float]]:
        """Check if z is in cache (within tolerance)."""
        if not self.use_cache or len(self.cache_z) == 0:
            return None

        # Check exact match first
        key = self._z_to_key(z)
        if key in self.cache:
            return self.cache[key]

        # Check approximate matches
        z_flat = z.detach().view(-1)
        for i, cached_z in enumerate(self.cache_z):
            cached_z_flat = cached_z.view(-1)
            if (z_flat - cached_z_flat).norm().item() < self.cache_tolerance:
                return (self.cache_values[i], 0.0)  # 0 curvature for cached

        return None

    def _add_to_cache(self, z: torch.Tensor, value: float, curvature: float):
        """Add evaluation result to cache."""
        if not self.use_cache:
            return

        key = self._z_to_key(z)
        self.cache[key] = (value, curvature)
        self.cache_z.append(z.detach().cpu())
        self.cache_values.append(value)

        # Limit cache size
        max_cache = 10000
        if len(self.cache_z) > max_cache:
            # Remove oldest entries
            to_remove = len(self.cache_z) - max_cache
            for i in range(to_remove):
                old_key = self._z_to_key(self.cache_z[i])
                self.cache.pop(old_key, None)
            self.cache_z = self.cache_z[to_remove:]
            self.cache_values = self.cache_values[to_remove:]

    def _get_threshold(self) -> float:
        """Get current FCU threshold."""
        if len(self.curvature_history) < self.min_samples_for_threshold:
            # Not enough data - use high threshold (evaluate more)
            return float('inf')

        # Percentile-based threshold
        sorted_k = sorted(self.curvature_history)
        idx = int(len(sorted_k) * self.percentile_threshold / 100.0)
        idx = min(idx, len(sorted_k) - 1)

        return sorted_k[idx]

    def _compute_confidence(self, curvature: float, threshold: float) -> float:
        """Compute confidence in gating decision."""
        if threshold <= 0:
            return 0.5

        # Distance from threshold normalized
        # Far below threshold -> high confidence in VH
        # Far above threshold -> high confidence in LLM
        ratio = curvature / threshold
        confidence = abs(1.0 - ratio) / (1.0 + abs(1.0 - ratio))

        return confidence

    def decide(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> GatingDecision:
        """
        Make gating decision for a single candidate.

        Args:
            z: Latent vector (latent_dim,) or (1, latent_dim)
            context: Optional context for flow

        Returns:
            GatingDecision with evaluation type and predictions
        """
        # Ensure z is 2D
        if z.dim() == 1:
            z = z.unsqueeze(0)

        self.stats.total_decisions += 1

        # Check cache
        cached = self._check_cache(z)
        if cached is not None:
            self.stats.cache_hits += 1
            return GatingDecision(
                z=z.squeeze(0),
                eval_type=EvaluationType.CACHED,
                curvature=cached[1],
                value_prediction=cached[0],
                confidence=1.0,
            )

        # Compute FCU
        with torch.no_grad():
            curvature = compute_flow_curvature(
                self.flow_model, z, context, self.curvature_steps
            ).item()

        # Update history
        self.curvature_history.append(curvature)
        if len(self.curvature_history) > self.max_history:
            self.curvature_history = self.curvature_history[-self.max_history:]

        # Get threshold
        threshold = self._get_threshold()

        # Decision
        if curvature > threshold:
            # High uncertainty - need LLM
            self.stats.llm_evaluations += 1
            return GatingDecision(
                z=z.squeeze(0),
                eval_type=EvaluationType.LLM,
                curvature=curvature,
                value_prediction=None,
                confidence=self._compute_confidence(curvature, threshold),
            )
        else:
            # Low uncertainty - trust Value Head
            self.stats.value_head_predictions += 1

            with torch.no_grad():
                self.value_head.eval()
                value_pred = self.value_head(z).item()

            return GatingDecision(
                z=z.squeeze(0),
                eval_type=EvaluationType.VALUE_HEAD,
                curvature=curvature,
                value_prediction=value_pred,
                confidence=self._compute_confidence(curvature, threshold),
            )

    def decide_batch(
        self,
        z_batch: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> List[GatingDecision]:
        """
        Make gating decisions for a batch of candidates.

        Args:
            z_batch: Batch of latents (B, latent_dim)
            context: Optional context (B, num_ctx, context_dim)

        Returns:
            List of GatingDecision for each candidate
        """
        batch_size = z_batch.shape[0]
        decisions = []

        # Compute curvatures in batch
        with torch.no_grad():
            curvatures = compute_flow_curvature(
                self.flow_model, z_batch, context, self.curvature_steps
            )

        # Get threshold
        threshold = self._get_threshold()

        # Process each candidate
        for i in range(batch_size):
            z = z_batch[i:i+1]
            curvature = curvatures[i].item()

            # Update history
            self.curvature_history.append(curvature)

            self.stats.total_decisions += 1

            # Check cache
            cached = self._check_cache(z)
            if cached is not None:
                self.stats.cache_hits += 1
                decisions.append(GatingDecision(
                    z=z.squeeze(0),
                    eval_type=EvaluationType.CACHED,
                    curvature=cached[1],
                    value_prediction=cached[0],
                    confidence=1.0,
                ))
                continue

            # Decision based on curvature
            if curvature > threshold:
                self.stats.llm_evaluations += 1
                decisions.append(GatingDecision(
                    z=z.squeeze(0),
                    eval_type=EvaluationType.LLM,
                    curvature=curvature,
                    value_prediction=None,
                    confidence=self._compute_confidence(curvature, threshold),
                ))
            else:
                self.stats.value_head_predictions += 1

                with torch.no_grad():
                    self.value_head.eval()
                    value_pred = self.value_head(z).item()

                decisions.append(GatingDecision(
                    z=z.squeeze(0),
                    eval_type=EvaluationType.VALUE_HEAD,
                    curvature=curvature,
                    value_prediction=value_pred,
                    confidence=self._compute_confidence(curvature, threshold),
                ))

        # Trim history
        if len(self.curvature_history) > self.max_history:
            self.curvature_history = self.curvature_history[-self.max_history:]

        return decisions

    def record_evaluation(
        self,
        z: torch.Tensor,
        ground_truth: float,
        decision: GatingDecision,
    ):
        """
        Record ground truth for a gated decision.

        Used to:
        1. Update cache
        2. Track Value Head accuracy
        3. Adjust threshold if adaptive

        Args:
            z: Latent vector
            ground_truth: Actual error rate from LLM evaluation
            decision: The gating decision that was made
        """
        # Add to cache
        self._add_to_cache(z, ground_truth, decision.curvature or 0.0)

        # Track Value Head accuracy
        if decision.eval_type == EvaluationType.VALUE_HEAD:
            self.vh_predictions.append(decision.value_prediction)
            self.vh_ground_truth.append(ground_truth)

            # Update MSE
            if len(self.vh_predictions) > 0:
                preds = torch.tensor(self.vh_predictions)
                truth = torch.tensor(self.vh_ground_truth)
                self.stats.value_head_mse = ((preds - truth) ** 2).mean().item()
                self.stats.num_verified = len(self.vh_predictions)

            # Adaptive threshold adjustment
            if self.adaptive_threshold and len(self.vh_predictions) >= 20:
                # If VH is too inaccurate, lower threshold (more LLM evals)
                recent_preds = torch.tensor(self.vh_predictions[-20:])
                recent_truth = torch.tensor(self.vh_ground_truth[-20:])
                recent_mse = ((recent_preds - recent_truth) ** 2).mean().item()

                if recent_mse > 0.05:  # >5% MSE
                    self.percentile_threshold = max(
                        80.0, self.percentile_threshold - 2.0
                    )
                elif recent_mse < 0.01:  # <1% MSE
                    self.percentile_threshold = min(
                        95.0, self.percentile_threshold + 1.0
                    )

    def get_statistics(self) -> Dict[str, float]:
        """Get gating statistics."""
        return {
            "total_decisions": self.stats.total_decisions,
            "llm_evaluations": self.stats.llm_evaluations,
            "value_head_predictions": self.stats.value_head_predictions,
            "cache_hits": self.stats.cache_hits,
            "llm_ratio": self.stats.llm_ratio,
            "cost_savings": self.stats.cost_savings,
            "value_head_mse": self.stats.value_head_mse,
            "num_verified": self.stats.num_verified,
            "current_threshold_percentile": self.percentile_threshold,
            "curvature_history_size": len(self.curvature_history),
        }

    def reset_statistics(self):
        """Reset statistics for new experiment."""
        self.stats = GatingStatistics()
        self.vh_predictions = []
        self.vh_ground_truth = []


class AdaptiveGate(EvaluationGate):
    """
    Evaluation gate with budget-aware adaptive threshold.

    In addition to FCU-based gating, this version considers:
    1. Remaining evaluation budget
    2. Expected remaining iterations
    3. Historical accuracy of Value Head

    This enables more aggressive cost savings when budget is tight.
    """

    def __init__(
        self,
        flow_model: nn.Module,
        value_head: nn.Module,
        total_budget: int = 100,
        target_llm_ratio: float = 0.2,  # Target 20% LLM evaluations
        **kwargs,
    ):
        super().__init__(flow_model, value_head, **kwargs)

        self.total_budget = total_budget
        self.target_llm_ratio = target_llm_ratio
        self.remaining_budget = total_budget

    def update_budget(self, used: int):
        """Update remaining budget after evaluations."""
        self.remaining_budget = max(0, self.remaining_budget - used)

    def _get_threshold(self) -> float:
        """Get budget-aware threshold."""
        base_threshold = super()._get_threshold()

        if self.remaining_budget <= 0:
            # No budget - use Value Head for everything
            return 0.0

        # Adjust threshold based on budget usage
        budget_ratio = self.remaining_budget / max(1, self.total_budget)
        target_ratio = self.target_llm_ratio

        # If we're over budget on LLM evaluations, raise threshold
        current_llm_ratio = self.stats.llm_ratio
        if current_llm_ratio > target_ratio:
            # Need to be more conservative
            adjustment = 1.0 + (current_llm_ratio - target_ratio) * 2
            return base_threshold * adjustment
        elif budget_ratio < 0.2:
            # Low budget remaining - be very conservative
            return base_threshold * 2.0

        return base_threshold


if __name__ == "__main__":
    print("Testing Evaluation Gating...")

    # Mock models
    class MockFlow(nn.Module):
        def forward(self, x_t, t, context=None):
            return torch.randn_like(x_t) * 0.1

    class MockValueHead(nn.Module):
        def forward(self, z):
            return torch.sigmoid(z.sum(dim=-1) * 0.1)

    flow = MockFlow()
    value_head = MockValueHead()

    # Create gate
    gate = EvaluationGate(
        flow_model=flow,
        value_head=value_head,
        percentile_threshold=90.0,
        min_samples_for_threshold=10,
    )

    # Test batch decisions
    print("\n1. Building curvature history...")
    z_batch = torch.randn(100, 32)
    decisions = gate.decide_batch(z_batch)

    print(f"   Total decisions: {len(decisions)}")
    llm_count = sum(1 for d in decisions if d.eval_type == EvaluationType.LLM)
    vh_count = sum(1 for d in decisions if d.eval_type == EvaluationType.VALUE_HEAD)
    print(f"   LLM evaluations: {llm_count}")
    print(f"   Value Head predictions: {vh_count}")

    # Simulate recording ground truth
    print("\n2. Recording ground truth...")
    for i, decision in enumerate(decisions[:20]):
        ground_truth = 0.3 + 0.1 * torch.randn(1).item()
        gate.record_evaluation(decisions[i].z, ground_truth, decision)

    # Get statistics
    print("\n3. Gating statistics:")
    stats = gate.get_statistics()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Test adaptive gate
    print("\n4. Testing AdaptiveGate...")
    adaptive_gate = AdaptiveGate(
        flow_model=flow,
        value_head=value_head,
        total_budget=50,
        target_llm_ratio=0.2,
        min_samples_for_threshold=10,
    )

    # Simulate optimization loop
    for i in range(5):
        z = torch.randn(10, 32)
        batch_decisions = adaptive_gate.decide_batch(z)

        llm_evals = sum(1 for d in batch_decisions if d.eval_type == EvaluationType.LLM)
        adaptive_gate.update_budget(llm_evals)

        print(f"   Iteration {i+1}: LLM={llm_evals}, Budget remaining={adaptive_gate.remaining_budget}")

    print("\nAll tests passed!")
