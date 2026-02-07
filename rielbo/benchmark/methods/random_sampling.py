"""Random sampling baseline for benchmark framework.

Samples random latent vectors from N(0,1) — the VAE prior — decodes them,
and evaluates. No surrogate model, no optimization — pure random exploration.
This is the minimal baseline: any method that doesn't beat random sampling
has no value as an optimizer.
"""

import logging

import numpy as np
import torch

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult

logger = logging.getLogger(__name__)


class RandomSamplingBenchmark(BaseBenchmarkMethod):
    """Random sampling baseline.

    Each step samples z ~ N(0,1) in R^256, decodes to SMILES, and evaluates.
    Uses the VAE prior as the sampling distribution — no learning, no surrogate.
    """

    method_name = "random"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
        latent_dim: int = 256,
    ):
        super().__init__(codec, oracle, seed, device, verbose)
        self.latent_dim = latent_dim

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize with cold start data (same as other methods for fairness)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores

        best_idx = int(np.argmax(scores_np))
        self.best_score = float(scores_np[best_idx])
        self.best_smiles = smiles_list[best_idx]
        self.n_evaluated = len(smiles_list)
        self.smiles_set = set(smiles_list)

        logger.info(
            f"Random sampling: dim={self.latent_dim}, "
            f"best_cold={self.best_score:.4f}"
        )

    def step(self) -> StepResult:
        """Sample a random latent vector, decode, and evaluate."""
        z = torch.randn(1, self.latent_dim, device=self.device)

        # Decode
        try:
            smiles = self.codec.decode(z)[0]
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            logger.warning(f"Random decode failed: {e}")
            return StepResult(
                score=0.0, best_score=self.best_score, smiles="",
                is_duplicate=True, is_valid=False,
            )

        # Check duplicate
        if not smiles or smiles in self.smiles_set:
            return StepResult(
                score=0.0, best_score=self.best_score, smiles=smiles or "",
                is_duplicate=True,
            )

        # Score
        score = self.oracle.score(smiles)
        self.smiles_set.add(smiles)
        self.n_evaluated += 1

        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles

        return StepResult(
            score=score,
            best_score=self.best_score,
            smiles=smiles,
            is_duplicate=False,
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "sampling": "N(0,1)",
        })
        return config
