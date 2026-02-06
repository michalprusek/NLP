"""InvBO wrapper for benchmark framework.

InvBO (Inversion-based Latent Bayesian Optimization) extends LOLBO with:
1. Latent inversion: Optimizes z codes to perfectly reconstruct target molecules
2. Potential-aware trust region anchor selection

Reference:
    Chu, Park, Lee, Kim. "Inversion-based Latent Bayesian Optimization."
    NeurIPS 2024. https://github.com/mlvlab/InvBO

For fair comparison, VAE is frozen (no end-to-end updates).
Inversion method works with frozen VAE (only optimizes z, not VAE params).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Add invbo_ref to path for imports
invbo_path = Path(__file__).parent.parent.parent.parent / "invbo_ref"
if str(invbo_path) not in sys.path:
    sys.path.insert(0, str(invbo_path))

# Monkey-patch InvBO's ppgpr to fix botorch API (mvn= → distribution=)
import invbo.utils.bo_utils.ppgpr as invbo_ppgpr
from botorch.posteriors.gpytorch import GPyTorchPosterior


def _fixed_posterior_gpmodel(self, X, output_indices=None, observation_noise=False, *args, **kwargs):
    self.eval()
    self.likelihood.eval()
    dist = self.likelihood(self(X))
    return GPyTorchPosterior(distribution=dist)


def _fixed_posterior_gpmodeldkl(self, X, output_indices=None, observation_noise=False, *args, **kwargs):
    self.eval()
    self.likelihood.eval()
    dist = self.likelihood(self(X))
    return GPyTorchPosterior(distribution=dist)


# Apply patches
invbo_ppgpr.GPModel.posterior = _fixed_posterior_gpmodel
invbo_ppgpr.GPModelDKL.posterior = _fixed_posterior_gpmodeldkl

from invbo.invbo import InvBOState
from invbo.latent_space_objective import LatentSpaceObjective

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult


class SharedCodecObjective(LatentSpaceObjective):
    """Custom objective that uses shared codec and oracle.

    Provides the interface InvBOState expects, including access to
    the raw VAE model for inversion.
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

        # InvBO inversion() needs these:
        self.smiles_to_selfies: dict[str, str] = {}
        self.dataobj = codec.dataset  # SELFIESDataset for tokenization

        # Initialize parent (calls initialize_vae)
        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict or {},
            num_calls=num_calls,
            task_id=task_id,
        )

    def initialize_vae(self):
        """Use shared codec's raw VAE model."""
        # InvBO's inversion() accesses self.vae.encode() and self.vae.decode()
        # with token-level interface, so we expose the raw model
        self.vae = self.codec.model

    def vae_decode(self, z: torch.Tensor) -> list[str]:
        """Decode latent codes to SMILES via codec."""
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        z = z.to(self.codec.device)
        return self.codec.decode(z)

    def query_oracle(self, x: str) -> float:
        """Query oracle for SMILES score."""
        if not x:
            return np.nan
        try:
            score = self.oracle.score(x)
            return float(score)
        except Exception as e:
            logger.warning(f"Oracle scoring failed for '{x[:50]}...': {e}")
            return np.nan

    def vae_forward(self, xs_batch: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode SMILES to latent codes.

        Returns (z, sigma, recon_loss, kl_div) — for frozen VAE,
        we return zeros for loss terms.
        """
        z = self.codec.encode(xs_batch)
        zero = torch.tensor(0.0, device=z.device)
        # Return (z, sigma, recon_loss_per_sample, kl_div)
        recon_loss = torch.zeros(len(xs_batch), device=z.device)
        return z, zero, recon_loss, zero


class InvBOBenchmark(BaseBenchmarkMethod):
    """InvBO benchmark wrapper.

    Inversion-based Latent Bayesian Optimization with:
    - Deep Kernel Learning GP (same as LOLBO)
    - Latent inversion for better z-code alignment
    - Potential-aware trust region anchor selection
    - Frozen VAE (no end-to-end updates) for fair comparison

    Key difference from LOLBO:
    - Periodically runs inversion() to refine latent codes
    - Anchor selection considers both score and acquisition potential
    """

    method_name = "invbo"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
        bsz: int = 1,
        acq_func: str = "ts",
        learning_rte: float = 0.01,
        init_n_epochs: int = 20,
        k: int = 50,
        e2e_freq: int = 10,
        alpha: float = 100.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 0.1,
    ):
        super().__init__(codec, oracle, seed, device, verbose)

        self.bsz = bsz
        self.acq_func = acq_func
        self.learning_rte = learning_rte
        self.init_n_epochs = init_n_epochs
        self.k = k
        self.e2e_freq = e2e_freq
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # InvBO state (initialized in cold_start)
        self.invbo_state: InvBOState | None = None
        self.objective: SharedCodecObjective | None = None
        self.task_id = getattr(oracle, "task_id", "unknown")

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize with cold start data."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Create objective
        self.objective = SharedCodecObjective(
            codec=self.codec,
            oracle=self.oracle,
            task_id=self.task_id,
        )

        # Pre-populate smiles_to_selfies for inversion
        import selfies as sf
        n_sf_failures = 0
        for smi in smiles_list:
            try:
                selfies_str = sf.encoder(smi)
                if selfies_str:
                    self.objective.smiles_to_selfies[smi] = selfies_str
            except Exception as e:
                n_sf_failures += 1
                logger.debug(f"SELFIES encoding failed for '{smi[:50]}': {e}")
        if n_sf_failures > 0:
            logger.warning(f"InvBO: {n_sf_failures}/{len(smiles_list)} SELFIES encodings failed in cold start")

        # Encode initial data
        train_x = list(smiles_list)
        train_y = scores.unsqueeze(-1).float().cpu()
        train_z = self.codec.encode(smiles_list).cpu()

        # Create InvBO state
        self.invbo_state = InvBOState(
            objective=self.objective,
            train_x=train_x,
            train_y=train_y,
            train_z=train_z,
            k=self.k,
            minimize=False,
            num_update_epochs=2,  # For GP updates only (VAE is frozen)
            init_n_epochs=self.init_n_epochs,
            learning_rte=self.learning_rte,
            bsz=self.bsz,
            acq_func=self.acq_func,
            verbose=self.verbose,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
        )

        # Sync state
        self.best_score = self.invbo_state.best_score_seen
        if isinstance(self.best_score, torch.Tensor):
            self.best_score = self.best_score.item()
        self.best_smiles = self.invbo_state.best_x_seen
        self.n_evaluated = len(train_x)
        self.smiles_set = set(train_x)

    def step(self) -> StepResult:
        """Execute a single optimization step.

        InvBO loop:
        1. If progress stalled → run inversion (skip e2e VAE training)
        2. Update GP surrogate
        3. Acquisition (potential-aware TS)
        """
        if self.invbo_state is None:
            raise RuntimeError("Must call cold_start() before step()")

        prev_best = self.invbo_state.best_score_seen
        if isinstance(prev_best, torch.Tensor):
            prev_best = prev_best.item()
        prev_n = len(self.invbo_state.train_x)

        # Run inversion when progress stalls (skip e2e VAE training)
        if self.invbo_state.progress_fails_since_last_e2e >= self.e2e_freq:
            try:
                self.invbo_state.inversion()
                self.invbo_state.progress_fails_since_last_e2e = 0
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:
                logger.warning(f"InvBO inversion failed: {e}")
                # Continue incrementing counter to detect systematic inversion failures
                self.invbo_state.progress_fails_since_last_e2e += 1
                if self.invbo_state.progress_fails_since_last_e2e == self.e2e_freq * 3:
                    logger.error(
                        f"InvBO inversion has failed {self.invbo_state.progress_fails_since_last_e2e} "
                        f"consecutive times — inversion may be systematically broken"
                    )

        # Update surrogate model (GP only)
        try:
            self.invbo_state.update_surrogate_model()
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            logger.warning(f"InvBO surrogate update failed: {e}, skipping acquisition")
            return StepResult(
                score=0.0, best_score=self.best_score, smiles="",
                is_duplicate=True, is_valid=False,
            )

        # Acquisition (potential-aware TS with trust region)
        try:
            self.invbo_state.acquisition()
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            logger.warning(f"InvBO acquisition failed: {e}")
            return StepResult(
                score=0.0, best_score=self.best_score, smiles="",
                is_duplicate=True, is_valid=False,
            )

        # Handle trust region restart
        if self.invbo_state.tr_state.restart_triggered:
            self.invbo_state.initialize_tr_state()

        # Get results
        n_new = len(self.invbo_state.train_x) - prev_n
        current_best = self.invbo_state.best_score_seen
        if isinstance(current_best, torch.Tensor):
            current_best = current_best.item()

        self.best_score = current_best
        self.best_smiles = self.invbo_state.best_x_seen
        self.n_evaluated = len(self.invbo_state.train_x)
        self.smiles_set = set(self.invbo_state.train_x)

        if n_new > 0:
            current_score = self.invbo_state.train_y[-1].item()
            current_smiles = self.invbo_state.train_x[-1]
            # Update smiles_to_selfies for new molecules
            import selfies as sf
            try:
                s = sf.encoder(current_smiles)
                if s:
                    self.objective.smiles_to_selfies[current_smiles] = s
            except Exception as e:
                logger.debug(f"SELFIES encoding failed for new molecule: {e}")
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
            trust_region_length=self.invbo_state.tr_state.length,
            extra={
                "n_new_this_step": n_new,
                "oracle_calls": self.objective.num_calls,
                "progress_fails": self.invbo_state.progress_fails_since_last_e2e,
                "n_e2e_updates": self.invbo_state.tot_num_e2e_updates,  # Vestigial: always 0 (VAE frozen)
            },
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "bsz": self.bsz,
            "acq_func": self.acq_func,
            "learning_rte": self.learning_rte,
            "init_n_epochs": self.init_n_epochs,
            "k": self.k,
            "e2e_freq": self.e2e_freq,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "freeze_vae": True,
        })
        return config
