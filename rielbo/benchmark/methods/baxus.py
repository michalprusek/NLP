"""BAxUS wrapper for benchmark framework.

BAxUS (Bayesian Optimization in Adaptively Expanding Subspaces) starts
optimization in a low-dimensional random embedding and expands the
subspace dimensionality when the trust region is exhausted.

Reference:
    Papenmeier, Nardi, Poloczek. "Increasing the Scope as You Learn:
    Adaptive Bayesian Optimization in Nested Subspaces." NeurIPS 2022.

Implementation follows the official BoTorch tutorial:
    https://botorch.org/docs/tutorials/baxus/
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BAxUS state and helper functions (from BoTorch tutorial)
# ---------------------------------------------------------------------------

@dataclass
class BaxusState:
    """State for BAxUS trust region and subspace management."""
    dim: int                    # Original input dimension (256)
    eval_budget: int            # Total evaluation budget
    new_bins_on_split: int = 3  # New bins per split
    d_init: int = 0
    target_dim: int = 0
    n_splits: int = 0
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        n_splits = round(math.log(self.dim, self.new_bins_on_split + 1))
        self.d_init = int(1 + np.argmin(
            np.abs(
                (1 + np.arange(self.new_bins_on_split))
                * (1 + self.new_bins_on_split) ** n_splits
                - self.dim
            )
        ))
        self.target_dim = self.d_init
        self.n_splits = n_splits

    @property
    def split_budget(self) -> int:
        return round(
            -1
            * (self.new_bins_on_split * self.eval_budget * self.target_dim)
            / (self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1)))
        )

    @property
    def failure_tolerance(self) -> int:
        if self.target_dim == self.dim:
            return self.target_dim
        k = math.floor(math.log(self.length_min / self.length_init, 0.5))
        sb = self.split_budget
        return min(self.target_dim, max(1, math.floor(sb / k)))


def _embedding_matrix(input_dim: int, target_dim: int, device, dtype) -> torch.Tensor:
    """Create random hash-based embedding S: target_dim x input_dim."""
    if target_dim >= input_dim:
        return torch.eye(input_dim, device=device, dtype=dtype)

    input_dims_perm = torch.randperm(input_dim, device=device) + 1
    bins = torch.tensor_split(input_dims_perm, target_dim)
    bins = torch.nn.utils.rnn.pad_sequence(bins, batch_first=True)

    mtrx = torch.zeros((target_dim, input_dim + 1), dtype=dtype, device=device)
    mtrx = mtrx.scatter_(
        1, bins,
        2 * torch.randint(2, (target_dim, input_dim), dtype=dtype, device=device) - 1,
    )
    return mtrx[:, 1:]


def _increase_embedding(
    S: torch.Tensor, X_target: torch.Tensor, n_new_bins: int, device, dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split bins to increase target dimensionality."""
    S_upd = S.clone()
    X_upd = X_target.clone()

    for row_idx in range(len(S)):
        row = S[row_idx]
        nz = torch.nonzero(row).reshape(-1)
        nz = nz[torch.randperm(len(nz), device=device)]

        if len(nz) <= 1:
            continue

        elems = row[nz]
        n_row_bins = min(n_new_bins, len(nz))
        split_nz = torch.tensor_split(nz, n_row_bins)[1:]
        split_el = torch.tensor_split(elems, n_row_bins)[1:]

        for new_nz, new_el in zip(split_nz, split_el):
            new_row = torch.zeros(S_upd.shape[1], dtype=dtype, device=device)
            new_row[new_nz] = new_el
            S_upd[row_idx, new_nz] = 0
            S_upd = torch.vstack((S_upd, new_row.unsqueeze(0)))
            X_upd = torch.hstack((X_upd, X_target[:, row_idx:row_idx + 1]))

    return S_upd, X_upd


def _update_baxus_state(state: BaxusState, Y_next: torch.Tensor) -> BaxusState:
    """Update trust region based on new observation."""
    y_val = Y_next.max().item()

    if y_val > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, y_val)

    if state.length < state.length_min:
        state.restart_triggered = True

    return state


def _baxus_candidate(
    state: BaxusState,
    model,
    X: torch.Tensor,
    Y: torch.Tensor,
    n_candidates: int = 5000,
    device=None,
    dtype=torch.double,
) -> torch.Tensor:
    """Generate a candidate in the target (low-D) space using Thompson Sampling."""
    dim = X.shape[-1]
    x_center = X[Y.argmax(), :].clone()

    # Lengthscale-weighted trust region (handle ScaleKernel(RBF) or bare RBF)
    covar = model.covar_module
    if hasattr(covar, 'base_kernel') and hasattr(covar.base_kernel, 'lengthscale'):
        ls = covar.base_kernel.lengthscale.detach().view(-1)
    elif hasattr(covar, 'lengthscale'):
        ls = covar.lengthscale.detach().view(-1)
    else:
        ls = torch.ones(dim, dtype=dtype, device=device)
    weights = ls / ls.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    tr_lb = torch.clamp(x_center - weights * state.length, -1.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length, -1.0, 1.0)

    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Probabilistic perturbation (like TuRBO)
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = 1

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    ts = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():
        X_next = ts(X_cand, num_samples=1)

    return X_next


# ---------------------------------------------------------------------------
# Benchmark adapter
# ---------------------------------------------------------------------------

class BAxUSBenchmark(BaseBenchmarkMethod):
    """BAxUS benchmark wrapper.

    Bayesian Optimization in Adaptively Expanding Subspaces.
    Starts in a low-D embedding (~2D) and expands as needed.

    Key characteristics:
    - Random hash-based embedding from input to target space
    - GP with Matern kernel in target space (adaptive dimensionality)
    - Trust region with expansion on exhaustion (bin splitting)
    - Thompson Sampling acquisition
    """

    method_name = "baxus"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
        n_candidates: int = 5000,
        eval_budget: int = 500,
    ):
        super().__init__(codec, oracle, seed, device, verbose)
        self.n_candidates = n_candidates
        self.eval_budget = eval_budget
        self.dtype = torch.double

        # BAxUS internals (initialized in cold_start)
        self.state: BaxusState | None = None
        self.S: torch.Tensor | None = None  # Embedding matrix
        self.X_input: torch.Tensor | None = None  # Full-D normalized points
        self.X_target: torch.Tensor | None = None  # Low-D embedded points
        self.Y: torch.Tensor | None = None  # Scores

        # Normalization bounds
        self._z_min: torch.Tensor | None = None
        self._z_max: torch.Tensor | None = None
        self._t_min: torch.Tensor | None = None  # Target space bounds
        self._t_max: torch.Tensor | None = None

        self._iteration = 0

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Min-max normalize to [-1, 1]^D."""
        return 2 * (z - self._z_min) / (self._z_max - self._z_min + 1e-8) - 1

    def _denormalize(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize from [-1, 1]^D back to original scale."""
        return (z_norm + 1) / 2 * (self._z_max - self._z_min + 1e-8) + self._z_min

    def _normalize_target(self, X_t: torch.Tensor) -> torch.Tensor:
        """Normalize target space to [-1, 1]^d_target."""
        return 2 * (X_t - self._t_min) / (self._t_max - self._t_min + 1e-8) - 1

    def _denormalize_target(self, X_t_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize target space from [-1, 1]^d_target."""
        return (X_t_norm + 1) / 2 * (self._t_max - self._t_min + 1e-8) + self._t_min

    def _update_target_bounds(self):
        """Recompute target space normalization bounds."""
        X_t_raw = self.X_input @ self.S.T  # [N, target_dim]
        self._t_min = X_t_raw.min(dim=0).values
        self._t_max = X_t_raw.max(dim=0).values

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize BAxUS with cold start data."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Encode SMILES to latent vectors
        Z = self.codec.encode(smiles_list).to(self.device)  # [N, 256]
        Y = scores.to(self.device).unsqueeze(-1).to(self.dtype)  # [N, 1]

        # Compute normalization bounds for input space
        self._z_min = Z.min(dim=0).values
        self._z_max = Z.max(dim=0).values

        # Normalize to [-1, 1]^D
        Z_norm = self._normalize(Z).to(self.dtype)

        # Initialize BAxUS state
        input_dim = Z.shape[1]  # 256
        self.state = BaxusState(dim=input_dim, eval_budget=self.eval_budget)
        logger.info(
            f"BAxUS: input_dim={input_dim}, d_init={self.state.d_init}, "
            f"n_splits={self.state.n_splits}"
        )

        # Create initial embedding matrix
        self.S = _embedding_matrix(
            input_dim, self.state.target_dim, self.device, self.dtype
        )

        # Store full-D normalized input
        self.X_input = Z_norm

        # Project to target-D and normalize target space to [-1, 1]
        self._update_target_bounds()
        self.X_target = self._normalize_target(Z_norm @ self.S.T)
        self.Y = Y

        # Sync state
        best_idx = Y.argmax()
        self.best_score = Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]
        self.n_evaluated = len(smiles_list)
        self.smiles_set = set(smiles_list)
        self.state.best_value = self.best_score

    def step(self) -> StepResult:
        """Execute one BAxUS optimization step."""
        self._iteration += 1

        # Fit GP in normalized target space (SingleTaskGP handles Y standardization)
        try:
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            model = SingleTaskGP(
                self.X_target, self.Y, likelihood=likelihood,
            ).to(self.device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        except Exception as e:
            logger.warning(f"BAxUS GP fit failed: {e}, using fallback")
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 1.0))
            model = SingleTaskGP(
                self.X_target, self.Y, likelihood=likelihood,
            ).to(self.device)

        model.eval()

        # Generate candidate in normalized target space [-1, 1]^d_target
        X_next_target_norm = _baxus_candidate(
            state=self.state,
            model=model,
            X=self.X_target,
            Y=self.Y,
            n_candidates=self.n_candidates,
            device=self.device,
            dtype=self.dtype,
        )

        # Denormalize target → raw target → input space
        X_next_target_raw = self._denormalize_target(X_next_target_norm)
        # Back-project: raw target → normalized input space via pseudoinverse
        # For hash embedding S: x_input ≈ S.pinverse() @ x_target, but simpler:
        # x_input = x_target @ S (since S is sparse with ±1)
        X_next_input = X_next_target_raw @ self.S

        # Clamp to [-1, 1] input space
        X_next_input = X_next_input.clamp(-1.0, 1.0)

        # Denormalize to original latent space
        z_full = self._denormalize(X_next_input.float())

        # Decode to SMILES
        try:
            smiles = self.codec.decode(z_full)[0]
        except Exception as e:
            logger.info(f"BAxUS decode failed: {e}")
            self._handle_failure()
            return StepResult(
                score=0.0, best_score=self.best_score, smiles="",
                is_duplicate=True, is_valid=False,
            )

        # Check duplicate
        if not smiles or smiles in self.smiles_set:
            self._handle_failure()
            return StepResult(
                score=0.0, best_score=self.best_score, smiles=smiles or "",
                is_duplicate=True,
            )

        # Score
        score = self.oracle.score(smiles)
        self.smiles_set.add(smiles)
        self.n_evaluated += 1

        # Update BAxUS state
        Y_next = torch.tensor([[score]], dtype=self.dtype, device=self.device)
        self.state = _update_baxus_state(self.state, Y_next)

        # Add to training data (input space + normalized target space)
        self.X_input = torch.cat([self.X_input, X_next_input], dim=0)
        # Re-normalize the new target point using current bounds
        X_next_target_renorm = self._normalize_target(X_next_target_raw)
        self.X_target = torch.cat([self.X_target, X_next_target_renorm], dim=0)
        self.Y = torch.cat([self.Y, Y_next], dim=0)

        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles

        # Handle subspace expansion
        if self.state.restart_triggered:
            self._expand_subspace()

        return StepResult(
            score=score,
            best_score=self.best_score,
            smiles=smiles,
            is_duplicate=False,
            trust_region_length=self.state.length,
            extra={
                "target_dim": self.state.target_dim,
                "n_splits_done": self.state.n_splits,
            },
        )

    def _handle_failure(self):
        """Handle a failed/duplicate step as a failure for TR."""
        self.state.failure_counter += 1
        if self.state.failure_counter >= self.state.failure_tolerance:
            self.state.length /= 2.0
            self.state.failure_counter = 0
        if self.state.length < self.state.length_min:
            self.state.restart_triggered = True
            self._expand_subspace()

    def _expand_subspace(self):
        """Expand the target subspace by splitting bins."""
        self.state.restart_triggered = False

        if self.state.target_dim >= self.state.dim:
            # Already full dimensional — just reset TR
            self.state.length = self.state.length_init
            self.state.failure_counter = 0
            self.state.success_counter = 0
            return

        old_dim = self.state.target_dim

        # _increase_embedding expects raw (unnormalized) target coords
        X_target_raw = self.X_input @ self.S.T
        self.S, X_target_raw_expanded = _increase_embedding(
            self.S, X_target_raw, self.state.new_bins_on_split,
            self.device, self.dtype,
        )
        self.state.target_dim = len(self.S)

        # Re-normalize expanded target space to [-1, 1]
        self._update_target_bounds()
        self.X_target = self._normalize_target(X_target_raw_expanded)

        self.state.length = self.state.length_init
        self.state.failure_counter = 0
        self.state.success_counter = 0

        logger.info(
            f"BAxUS expanded: {old_dim}D → {self.state.target_dim}D "
            f"(of {self.state.dim}D)"
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "n_candidates": self.n_candidates,
            "eval_budget": self.eval_budget,
            "input_dim": 256,
            "d_init": self.state.d_init if self.state else None,
        })
        return config
