"""CMA-ES wrapper for benchmark framework.

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a
derivative-free evolutionary optimization algorithm. It serves
as a non-BO baseline that operates directly in the 256D VAE latent space.

Reference:
    Hansen, Ostermeier. "Completely Derandomized Self-Adaptation in
    Evolution Strategies." Evolutionary Computation, 2001.

Uses the pycma library: https://github.com/CMA-ES/pycma
"""

import logging

import cma
import numpy as np
import torch

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult

logger = logging.getLogger(__name__)


class CMAESBenchmark(BaseBenchmarkMethod):
    """CMA-ES benchmark wrapper.

    Covariance Matrix Adaptation Evolution Strategy in full R^256.
    Non-BO baseline — no GP surrogate, pure evolutionary optimization.

    Key characteristics:
    - Population-based search (mu/lambda selection)
    - Adapts covariance matrix to learn search directions
    - No surrogate model — evaluates objective directly
    - Fast per-iteration (no GP fitting)
    """

    method_name = "cmaes"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
        sigma0: float = 0.5,
        popsize: int | None = None,
    ):
        super().__init__(codec, oracle, seed, device, verbose)
        self.sigma0 = sigma0
        self.popsize = popsize

        # CMA-ES internals
        self._es: cma.CMAEvolutionStrategy | None = None
        self._pending_solutions: list | None = None
        self._pending_idx: int = 0
        self._pending_fitnesses: list = []

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize CMA-ES at best cold-start point."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Encode to latent
        Z = self.codec.encode(smiles_list).cpu().numpy()  # [N, 256]
        scores_np = scores.cpu().numpy()

        # Find best cold-start point
        best_idx = int(scores_np.argmax())
        self.best_score = float(scores_np[best_idx])
        self.best_smiles = smiles_list[best_idx]
        self.n_evaluated = len(smiles_list)
        self.smiles_set = set(smiles_list)

        # Initialize CMA-ES at best point
        x0 = Z[best_idx]
        opts = {
            "seed": self.seed,
            "verbose": -9,  # Suppress CMA-ES output
            "maxiter": 10000,
            "tolx": 1e-12,
            "tolfun": 1e-12,
        }
        if self.popsize is not None:
            opts["popsize"] = self.popsize

        # CMA-ES minimizes, so we negate scores
        self._es = cma.CMAEvolutionStrategy(x0.tolist(), self.sigma0, opts)

        # Pre-ask first batch
        self._pending_solutions = self._es.ask()
        self._pending_idx = 0

        logger.info(
            f"CMA-ES: dim={len(x0)}, sigma0={self.sigma0}, "
            f"popsize={self._es.popsize}, best_cold={self.best_score:.4f}"
        )

    def step(self) -> StepResult:
        """Execute one CMA-ES step (evaluate one member of current population)."""
        if self._pending_idx >= len(self._pending_solutions):
            # All members evaluated — tell CMA-ES and ask new batch
            self._es.tell(
                self._pending_solutions,
                self._pending_fitnesses,
            )
            self._pending_solutions = self._es.ask()
            self._pending_idx = 0

        # Initialize fitness tracking at the start of each new population batch
        if self._pending_idx == 0:
            self._pending_fitnesses = []

        # Get current candidate
        z = self._pending_solutions[self._pending_idx]
        self._pending_idx += 1

        # Decode
        z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        try:
            smiles = self.codec.decode(z_tensor)[0]
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            logger.warning(f"CMA-ES decode failed: {e}")
            self._pending_fitnesses.append(0.0)  # Zero score for decode failure
            return StepResult(
                score=0.0, best_score=self.best_score, smiles="",
                is_duplicate=True, is_valid=False,
            )

        # Check duplicate
        if not smiles or smiles in self.smiles_set:
            self._pending_fitnesses.append(0.0)
            return StepResult(
                score=0.0, best_score=self.best_score, smiles=smiles or "",
                is_duplicate=True,
            )

        # Score
        score = self.oracle.score(smiles)
        self.smiles_set.add(smiles)
        self.n_evaluated += 1

        # CMA-ES minimizes — negate score
        self._pending_fitnesses.append(-score)

        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles

        return StepResult(
            score=score,
            best_score=self.best_score,
            smiles=smiles,
            is_duplicate=False,
            extra={
                "sigma": self._es.sigma,
                "popsize": self._es.popsize,
            },
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "sigma0": self.sigma0,
            "popsize": self._es.popsize if self._es else self.popsize,
            "input_dim": 256,
        })
        return config
