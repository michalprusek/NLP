"""LOLBO wrapper for benchmark framework.

Wraps LOLBOState to the BaseBenchmarkMethod interface.

Important: For fair comparison, we:
1. Use the same SELFIES VAE codec as other methods
2. Set num_update_epochs=0 to freeze VAE (no end-to-end updates)
3. Use single-step interface (bsz=1) to match other methods
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Fix gpytorch compatibility: LazyTensor -> LinearOperator
try:
    from gpytorch.lazy import LazyTensor  # noqa: F401
except ImportError:
    # Newer gpytorch versions moved LazyTensor to linear_operator
    import gpytorch.lazy
    try:
        from linear_operator.operators import LinearOperator
        gpytorch.lazy.LazyTensor = LinearOperator
    except ImportError:
        logging.getLogger(__name__).warning("Could not patch gpytorch.lazy.LazyTensor — LOLBO may fail")

# Add lolbo_ref to path for imports
lolbo_path = Path(__file__).parent.parent.parent.parent / "lolbo_ref"
if str(lolbo_path) not in sys.path:
    sys.path.insert(0, str(lolbo_path))

# Monkey-patch ppgpr: old API used GPyTorchPosterior(mvn=dist), new uses distribution=dist
import lolbo.utils.bo_utils.ppgpr as ppgpr
from botorch.posteriors.gpytorch import GPyTorchPosterior

def _fixed_posterior(self, X, output_indices=None, observation_noise=False, *args, **kwargs):
    self.eval()
    self.likelihood.eval()
    dist = self.likelihood(self(X))
    return GPyTorchPosterior(distribution=dist)

ppgpr.GPModel.posterior = _fixed_posterior
ppgpr.GPModelDKL.posterior = _fixed_posterior

from lolbo.lolbo import LOLBOState
from lolbo.latent_space_objective import LatentSpaceObjective

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult


class SharedCodecObjective(LatentSpaceObjective):
    """Custom objective that uses shared codec and oracle.

    This allows LOLBO to use the same infrastructure as other methods
    for fair comparison.
    """

    def __init__(
        self,
        codec,
        oracle,
        task_id: str,
        xs_to_scores_dict: dict | None = None,
        num_calls: int = 0,
    ):
        self.codec = codec
        self.oracle = oracle
        self.dim = 256

        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict or {},
            num_calls=num_calls,
            task_id=task_id,
        )

    def initialize_vae(self):
        """Use shared codec as VAE (LOLBO expects self.vae)."""
        self.vae = self.codec

    def vae_decode(self, z: torch.Tensor) -> list[str]:
        """Decode latent codes to SMILES."""
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        z = z.to(self.codec.device)
        return self.codec.decode(z)

    def query_oracle(self, x: str) -> float:
        """Query oracle for SMILES score."""
        if not x:  # Empty or invalid SMILES
            return np.nan
        try:
            score = self.oracle.score(x)
            return float(score)
        except Exception as e:
            logger.warning(f"Oracle scoring failed for '{x[:50]}...': {e}")
            return np.nan

    def vae_forward(self, xs_batch: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode SMILES to latent codes with zero VAE loss (frozen)."""
        z = self.codec.encode(xs_batch)
        vae_loss = torch.tensor(0.0, device=z.device)
        return z, vae_loss


class LOLBOBenchmark(BaseBenchmarkMethod):
    """LOLBO benchmark wrapper.

    Deep Kernel Learning GP with optional VAE fine-tuning.
    For fair comparison, VAE is frozen (num_update_epochs=0).

    Key characteristics:
    - GPModelDKL: RBF kernel + 2-layer neural network feature extractor
    - TuRBO-style trust region management
    - Thompson Sampling acquisition (default)
    - Batch-based evaluation (bsz=1 for fair comparison)
    """

    method_name = "lolbo"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
        # Method-specific parameters
        bsz: int = 1,  # Batch size per step (1 for fair comparison)
        acq_func: str = "ts",  # "ts" or "ei"
        learning_rte: float = 0.01,
        init_n_epochs: int = 20,
        k: int = 1000,  # Track top k points
        freeze_vae: bool = True,  # Freeze VAE for fair comparison
    ):
        super().__init__(codec, oracle, seed, device, verbose)

        self.bsz = bsz
        self.acq_func = acq_func
        self.learning_rte = learning_rte
        self.init_n_epochs = init_n_epochs
        self.k = k
        self.freeze_vae = freeze_vae

        self.lolbo_state: LOLBOState | None = None
        self.objective: SharedCodecObjective | None = None
        self.task_id = getattr(oracle, "task_id", "unknown")

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize with cold start data."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.objective = SharedCodecObjective(
            codec=self.codec,
            oracle=self.oracle,
            task_id=self.task_id,
        )

        train_x = list(smiles_list)
        train_y = scores.unsqueeze(-1).float().cpu()  # LOLBO expects CPU tensors
        train_z = self.codec.encode(smiles_list).cpu()

        self.lolbo_state = LOLBOState(
            objective=self.objective,
            train_x=train_x,
            train_y=train_y,
            train_z=train_z,
            k=self.k,
            minimize=False,
            num_update_epochs=0 if self.freeze_vae else 2,  # Freeze VAE
            init_n_epochs=self.init_n_epochs,
            learning_rte=self.learning_rte,
            bsz=self.bsz,
            acq_func=self.acq_func,
            verbose=self.verbose,
        )

        self.best_score = self.lolbo_state.best_score_seen.item() if isinstance(
            self.lolbo_state.best_score_seen, torch.Tensor
        ) else self.lolbo_state.best_score_seen
        self.best_smiles = self.lolbo_state.best_x_seen
        self.n_evaluated = len(train_x)
        self.smiles_set = set(train_x)

    def step(self) -> StepResult:
        """Execute a single optimization step.

        LOLBO uses batch acquisition, so we call acquisition() once
        and report the results.
        """
        if self.lolbo_state is None:
            raise RuntimeError("Must call cold_start() before step()")

        prev_n = len(self.lolbo_state.train_x)

        try:
            self.lolbo_state.update_surrogate_model()
            self.lolbo_state.acquisition()
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            logger.error(f"LOLBO step failed: {e}")
            return StepResult(
                score=0.0,
                best_score=self.best_score,
                smiles="",
                is_duplicate=True,
                is_valid=False,
            )

        if self.lolbo_state.tr_state.restart_triggered:
            self.lolbo_state.initialize_tr_state()

        n_new = len(self.lolbo_state.train_x) - prev_n
        current_best = self.lolbo_state.best_score_seen
        self.best_score = current_best.item() if isinstance(current_best, torch.Tensor) else current_best
        self.best_smiles = self.lolbo_state.best_x_seen
        self.n_evaluated = len(self.lolbo_state.train_x)
        self.smiles_set = set(self.lolbo_state.train_x)

        if n_new > 0:
            current_score = self.lolbo_state.train_y[-1].item()
            current_smiles = self.lolbo_state.train_x[-1]
            is_duplicate = False
        else:
            current_score = 0.0
            current_smiles = ""
            is_duplicate = True

        return StepResult(
            score=current_score,
            best_score=self.best_score,
            smiles=current_smiles,
            is_duplicate=is_duplicate,
            is_valid=n_new > 0,
            trust_region_length=self.lolbo_state.tr_state.length,
            extra={
                "n_new_this_step": n_new,
                "oracle_calls": self.objective.num_calls,
            },
        )

    def get_config(self) -> dict:
        """Return method-specific configuration."""
        config = super().get_config()
        config.update({
            "bsz": self.bsz,
            "acq_func": self.acq_func,
            "learning_rte": self.learning_rte,
            "init_n_epochs": self.init_n_epochs,
            "k": self.k,
            "freeze_vae": self.freeze_vae,
        })
        return config
