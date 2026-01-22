"""Warm-start data loading for FlowPO-HD.

Loads pre-evaluated HbBoPs results to initialize GP with real data
instead of starting from scratch. Uses medium_600 strategy (fidelity >= 600)
which showed best results in GP benchmark (Spearman 0.87).

Data flow:
    HbBoPs JSON → filter(fidelity >= 600) → SONAR encode → Beta posterior smoothing
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class WarmStartPoint:
    """Single evaluated instruction for warm-start."""

    instruction: str
    accuracy: float  # 0-1, raw accuracy
    fidelity: int  # Number of examples evaluated
    embedding: Optional[torch.Tensor] = None  # 1024D SONAR embedding

    @property
    def error_rate(self) -> float:
        """Convert accuracy to error rate."""
        return 1.0 - self.accuracy

    def get_smoothed_error_rate(self, alpha: float, beta: float) -> float:
        """Compute Beta posterior mean for smoothed error rate.

        Beta posterior smoothing accounts for evaluation uncertainty:
        - High fidelity (many samples): closer to raw error rate
        - Low fidelity (few samples): regularized towards prior mean

        Args:
            alpha: Beta prior alpha (from Empirical Bayes)
            beta: Beta prior beta

        Returns:
            Smoothed error rate: (errors + alpha) / (n + alpha + beta)
        """
        num_errors = self.error_rate * self.fidelity
        return (num_errors + alpha) / (self.fidelity + alpha + beta)

    def get_variance(self, alpha: float, beta: float) -> float:
        """Compute Beta posterior variance for heteroscedastic noise.

        Higher fidelity → lower variance (more confident).

        Args:
            alpha: Beta prior alpha
            beta: Beta prior beta

        Returns:
            Variance: p(1-p) / (n + alpha + beta + 1)
        """
        p = self.get_smoothed_error_rate(alpha, beta)
        return (p * (1 - p)) / (self.fidelity + alpha + beta + 1)


@dataclass
class WarmStartData:
    """Container for warm-start tensors ready for GP training."""

    X: torch.Tensor  # (N, 1024) SONAR embeddings
    y: torch.Tensor  # (N,) smoothed error rates
    variances: torch.Tensor  # (N,) observation variances
    fidelities: torch.Tensor  # (N,) fidelity values
    instructions: List[str]  # Original instruction texts

    # Empirical Bayes prior parameters
    beta_alpha: float
    beta_beta: float

    def __len__(self) -> int:
        return self.X.shape[0]

    def to(self, device: str) -> "WarmStartData":
        """Move tensors to device."""
        return WarmStartData(
            X=self.X.to(device),
            y=self.y.to(device),
            variances=self.variances.to(device),
            fidelities=self.fidelities.to(device),
            instructions=self.instructions,
            beta_alpha=self.beta_alpha,
            beta_beta=self.beta_beta,
        )

    @property
    def best_instruction(self) -> Tuple[str, float]:
        """Get instruction with lowest error rate."""
        best_idx = self.y.argmin().item()
        return self.instructions[best_idx], self.y[best_idx].item()


def fit_beta_prior(error_rates: List[float], fidelities: List[int]) -> Tuple[float, float]:
    """Fit Empirical Bayes Beta prior using Method of Moments.

    Estimates prior parameters from observed error rates, weighting
    by fidelity (higher fidelity = more reliable estimate).

    Args:
        error_rates: List of raw error rates
        fidelities: List of fidelity values

    Returns:
        (alpha, beta) for Beta prior
    """
    # Convert to numpy
    p = np.array(error_rates)
    n = np.array(fidelities)

    # Weighted mean and variance (weight by fidelity)
    weights = n / n.sum()
    mean_p = np.average(p, weights=weights)
    var_p = np.average((p - mean_p) ** 2, weights=weights)

    # Method of moments for Beta distribution
    # mean = α / (α + β)
    # var = αβ / ((α + β)² (α + β + 1))

    # Prevent division issues
    mean_p = np.clip(mean_p, 0.01, 0.99)
    var_p = np.clip(var_p, 1e-6, mean_p * (1 - mean_p) - 1e-6)

    # Solve for α and β
    common = mean_p * (1 - mean_p) / var_p - 1
    alpha = mean_p * common
    beta = (1 - mean_p) * common

    # Clamp to reasonable range
    alpha = np.clip(alpha, 0.5, 100)
    beta = np.clip(beta, 0.5, 100)

    return float(alpha), float(beta)


class WarmStartLoader:
    """Loads HbBoPs evaluation results for warm-start."""

    def __init__(
        self,
        hbbops_path: str,
        min_fidelity: int = 600,
        sonar_normalize: bool = False,
        device: str = "cuda",
        cache_path: Optional[str] = None,
    ):
        """Initialize loader.

        Args:
            hbbops_path: Path to HbBoPs results JSON
            min_fidelity: Minimum fidelity threshold (default 600 = medium strategy)
            sonar_normalize: Whether to L2-normalize SONAR embeddings
            device: Device for SONAR encoder
            cache_path: Optional path to cache embeddings
        """
        self.hbbops_path = Path(hbbops_path)
        self.min_fidelity = min_fidelity
        self.sonar_normalize = sonar_normalize
        self.device = device
        self.cache_path = Path(cache_path) if cache_path else None

    def load(self) -> WarmStartData:
        """Load and process HbBoPs data.

        Returns:
            WarmStartData ready for GP training
        """
        logger.info(f"Loading HbBoPs results from {self.hbbops_path}")

        # Load JSON
        with open(self.hbbops_path) as f:
            data = json.load(f)

        # Parse based on format
        points = self._parse_hbbops(data)
        logger.info(f"  Loaded {len(points)} total evaluations")

        # Filter by fidelity
        filtered = [p for p in points if p.fidelity >= self.min_fidelity]
        logger.info(f"  Filtered to {len(filtered)} points with fidelity >= {self.min_fidelity}")

        if not filtered:
            raise ValueError(
                f"No points with fidelity >= {self.min_fidelity}. "
                f"Available: {sorted(set(p.fidelity for p in points))}"
            )

        # Fit Empirical Bayes prior
        alpha, beta = fit_beta_prior(
            [p.error_rate for p in filtered],
            [p.fidelity for p in filtered],
        )
        logger.info(f"  Empirical Bayes prior: α={alpha:.2f}, β={beta:.2f}")
        logger.info(f"  Prior mean error rate: {alpha / (alpha + beta):.2%}")

        # Load or compute SONAR embeddings
        embeddings = self._get_embeddings([p.instruction for p in filtered])

        # Assign embeddings to points
        for p, emb in zip(filtered, embeddings):
            p.embedding = emb

        # Build tensors
        X = torch.stack([p.embedding for p in filtered])
        y = torch.tensor([p.get_smoothed_error_rate(alpha, beta) for p in filtered])
        variances = torch.tensor([p.get_variance(alpha, beta) for p in filtered])
        fidelities = torch.tensor([float(p.fidelity) for p in filtered])
        instructions = [p.instruction for p in filtered]

        logger.info(f"  Error rate range: [{y.min():.3f}, {y.max():.3f}]")
        logger.info(f"  Variance range: [{variances.min():.6f}, {variances.max():.6f}]")

        return WarmStartData(
            X=X,
            y=y,
            variances=variances,
            fidelities=fidelities,
            instructions=instructions,
            beta_alpha=alpha,
            beta_beta=beta,
        )

    def _parse_hbbops(self, data: Dict) -> List[WarmStartPoint]:
        """Parse HbBoPs JSON format.

        Handles multiple formats:
        - {"results": {"id": {...}, ...}, "metadata": {...}} (current HbBoPs format)
        - {"evaluations": [...]} (list format)
        - [{"instruction": ..., ...}, ...] (simple list)
        """
        points = []

        # Current HbBoPs format: {"results": {"id": {...}, ...}, "metadata": {...}}
        if "results" in data and isinstance(data["results"], dict):
            for item_id, item in data["results"].items():
                if isinstance(item, dict) and "instruction" in item:
                    points.append(WarmStartPoint(
                        instruction=item["instruction"],
                        accuracy=item.get("accuracy", 1 - item.get("error_rate", 0)),
                        fidelity=item.get("fidelity", item.get("n_samples", 100)),
                    ))

        # List format: {"evaluations": [...]}
        elif "evaluations" in data:
            for eval_item in data["evaluations"]:
                points.append(WarmStartPoint(
                    instruction=eval_item["instruction"],
                    accuracy=eval_item["accuracy"],
                    fidelity=eval_item["fidelity"],
                ))

        # Simple list format: [{"instruction": ..., ...}, ...]
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "instruction" in item:
                    points.append(WarmStartPoint(
                        instruction=item["instruction"],
                        accuracy=item.get("accuracy", 1 - item.get("error_rate", 0)),
                        fidelity=item.get("fidelity", item.get("n_samples", 100)),
                    ))

        # Results as list: {"results": [...]}
        elif "results" in data and isinstance(data["results"], list):
            for item in data["results"]:
                if isinstance(item, dict) and "instruction" in item:
                    points.append(WarmStartPoint(
                        instruction=item["instruction"],
                        accuracy=item.get("accuracy", 1 - item.get("error_rate", 0)),
                        fidelity=item.get("fidelity", item.get("n_samples", 100)),
                    ))

        # HbBoPs format with brackets
        elif "brackets" in data or any(k.startswith("bracket") for k in data.keys()):
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "instruction" in item:
                            points.append(WarmStartPoint(
                                instruction=item["instruction"],
                                accuracy=item.get("accuracy", 1 - item.get("error_rate", 0)),
                                fidelity=item.get("fidelity", item.get("n_samples", 100)),
                            ))

        return points

    def _get_embeddings(self, instructions: List[str]) -> List[torch.Tensor]:
        """Get SONAR embeddings, using cache if available."""

        # Check cache
        if self.cache_path and self.cache_path.exists():
            logger.info(f"  Loading cached embeddings from {self.cache_path}")
            cache = torch.load(self.cache_path, weights_only=False)

            # Verify cache matches
            if cache.get("instructions") == instructions:
                return [emb for emb in cache["embeddings"]]
            else:
                logger.warning("  Cache mismatch, recomputing embeddings")

        # Compute embeddings
        logger.info(f"  Computing SONAR embeddings for {len(instructions)} instructions...")

        from lido_pp.backbone.sonar_encoder import SONAREncoder

        encoder = SONAREncoder(
            device=self.device,
            normalize=self.sonar_normalize,
        )

        embeddings = []
        batch_size = 32

        for i in range(0, len(instructions), batch_size):
            batch = instructions[i:i + batch_size]
            batch_emb = encoder.encode(batch)  # (B, 1024)
            embeddings.extend([emb for emb in batch_emb])

        # Save cache
        if self.cache_path:
            logger.info(f"  Saving embeddings cache to {self.cache_path}")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "instructions": instructions,
                "embeddings": torch.stack(embeddings),
                "normalize": self.sonar_normalize,
            }, self.cache_path)

        return embeddings


def load_warm_start(
    hbbops_path: str = "lipo/data/hbbops_results_20260102.json",
    min_fidelity: int = 600,
    device: str = "cuda",
    cache_path: Optional[str] = "flowpo_hd/data/warm_start_embeddings.pt",
) -> WarmStartData:
    """Convenience function to load warm-start data.

    Args:
        hbbops_path: Path to HbBoPs results
        min_fidelity: Minimum fidelity (default 600 = medium strategy)
        device: Device for encoding
        cache_path: Path to cache embeddings

    Returns:
        WarmStartData ready for GP
    """
    loader = WarmStartLoader(
        hbbops_path=hbbops_path,
        min_fidelity=min_fidelity,
        sonar_normalize=False,  # FlowPO-HD uses unnormalized
        device=device,
        cache_path=cache_path,
    )
    return loader.load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing WarmStartLoader...")

    # Test loading
    data = load_warm_start(
        hbbops_path="lipo/data/hbbops_results_20260102.json",
        min_fidelity=600,
    )

    print(f"\nLoaded {len(data)} warm-start points:")
    print(f"  X shape: {data.X.shape}")
    print(f"  y shape: {data.y.shape}")
    print(f"  Error rate: [{data.y.min():.3f}, {data.y.max():.3f}]")
    print(f"  Beta prior: α={data.beta_alpha:.2f}, β={data.beta_beta:.2f}")

    best_instr, best_err = data.best_instruction
    print(f"\nBest instruction (error={best_err:.3f}):")
    print(f"  {best_instr[:100]}...")

    print("\n[OK] WarmStartLoader test passed!")
