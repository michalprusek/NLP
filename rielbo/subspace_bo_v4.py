"""Spherical Subspace BO v4: Windowed GP with Geodesic Novelty Bonus.

Proposal B from deep research report. Builds on v3 with two key additions:
1. Geodesic novelty bonus in acquisition: steers search away from observed points
2. Global novelty: computed against ALL observed points, not just the GP window

The acquisition function becomes:
    acq(v) = GP_acquisition(v) + novelty_weight * geodesic_novelty(v)

where geodesic_novelty(v) = min_i arccos(v · v_i) / π ∈ [0, 1]
for all observed points v_i projected into the current subspace.

This directly addresses:
- 41% duplicate rate in v1 → novelty pushes away from explored regions
- Posterior collapse → even when GP std→0, novelty drives exploration

Pipeline:
    x [N, D] -> normalize -> u [N, D] on S^(D-1) (directions)
    u [N, D] -> project_k -> v [N, d] on S^(d-1)  (d=16, using A[k])
    Window: select K_local nearest + K_random from training data
    Y-normalize: Y_norm = (Y - mean) / std
    GP operates on S^(d-1) with ArcCosine kernel
    Acquisition: GP_acq(v) + γ * min_geodesic_dist(v, V_all)
    v* -> lift_k -> u* on S^(D-1) -> x* = u* * mean_norm -> decode

Usage:
    from rielbo.subspace_bo_v4 import SphericalSubspaceBOv4

    optimizer = SphericalSubspaceBOv4(
        codec=codec,
        oracle=oracle,
        n_projections=3,
        window_local=50,
        window_random=30,
        novelty_weight=0.1,
    )
"""

import logging
import math

import gpytorch
import torch
import torch.nn.functional as F
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from rielbo.gp_diagnostics import GPDiagnostics
from rielbo.kernels import create_kernel

logger = logging.getLogger(__name__)


def geodesic_novelty(
    v_candidates: torch.Tensor,
    v_observed: torch.Tensor,
) -> torch.Tensor:
    """Compute geodesic novelty: min geodesic distance to any observed point.

    For points on the unit sphere S^(d-1):
        novelty(v) = min_i arccos(v · v_i) / π

    Returns values in [0, 1] where:
        0 = candidate is at an observed point (no novelty)
        1 = candidate is antipodal to all observed points (max novelty)

    Args:
        v_candidates: [M, d] candidates on S^(d-1)
        v_observed: [N, d] observed points on S^(d-1)

    Returns:
        [M] geodesic novelty scores in [0, 1]
    """
    # Cosine similarity: [M, N]
    cos_sim = v_candidates @ v_observed.T
    cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)

    # Geodesic distance: arccos(cos_sim) / π → [0, 1]
    geodesic_dist = torch.arccos(cos_sim) / math.pi  # [M, N]

    # Minimum distance to ANY observed point
    min_dist, _ = geodesic_dist.min(dim=-1)  # [M]

    return min_dist


class SphericalSubspaceBOv4:
    """Spherical Subspace BO v4 with windowed GP and geodesic novelty bonus.

    Key improvements over v3:
    1. Geodesic novelty bonus: explicit exploration via min geodesic distance
       to ALL observed points → directly reduces duplicate rate
    2. Novelty-aware acquisition: acq(v) = GP_acq(v) + γ * novelty(v)
    3. Maintains v3's windowed GP and multi-projection ensemble

    The novelty weight γ controls exploration-exploitation:
    - γ = 0: pure GP-driven search (equivalent to v3)
    - γ = 0.1: mild exploration bonus (recommended)
    - γ = 0.5: strong exploration, good for diverse generation
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        subspace_dim: int = 16,
        device: str = "cuda",
        n_candidates: int = 2000,
        ucb_beta: float = 2.0,
        acqf: str = "ts",
        trust_region: float = 0.8,
        seed: int = 42,
        verbose: bool = True,
        kernel: str = "arccosine",
        # V3 inherited: windowed GP + multi-projection
        n_projections: int = 3,
        window_local: int = 50,
        window_random: int = 30,
        # V4 new: novelty bonus
        novelty_weight: float = 0.1,
    ):
        if subspace_dim >= input_dim:
            raise ValueError(
                f"subspace_dim ({subspace_dim}) must be < input_dim ({input_dim})"
            )
        if subspace_dim < 2:
            raise ValueError(f"subspace_dim must be >= 2, got {subspace_dim}")
        if not (0 < trust_region <= 2.0):
            raise ValueError(f"trust_region must be in (0, 2], got {trust_region}")
        if n_projections < 1:
            raise ValueError(f"n_projections must be >= 1, got {n_projections}")
        if window_local < 1:
            raise ValueError(f"window_local must be >= 1, got {window_local}")
        if window_random < 0:
            raise ValueError(f"window_random must be >= 0, got {window_random}")
        if novelty_weight < 0:
            raise ValueError(f"novelty_weight must be >= 0, got {novelty_weight}")

        self.device = device
        self.codec = codec
        self.oracle = oracle
        self.input_dim = input_dim
        self.subspace_dim = subspace_dim
        self.n_candidates = n_candidates
        self.ucb_beta = ucb_beta
        self.acqf = acqf
        self.trust_region = trust_region
        self.verbose = verbose
        self.seed = seed
        self.kernel = kernel
        self.fallback_count = 0

        # V3 parameters
        self.n_projections = n_projections
        self.window_local = window_local
        self.window_random = window_random
        self.window_size = window_local + window_random

        # V4 parameters
        self.novelty_weight = novelty_weight

        # Initialize K orthonormal projection matrices
        torch.manual_seed(seed)
        self.projections = []
        for k in range(n_projections):
            A_raw = torch.randn(input_dim, subspace_dim, device=device)
            A, _ = torch.linalg.qr(A_raw)
            self.projections.append(A)

        logger.info(
            f"SubspaceBOv4: S^{input_dim-1} -> S^{subspace_dim-1}, "
            f"kernel={kernel}, n_proj={n_projections}, "
            f"window={window_local}+{window_random}, "
            f"novelty_weight={novelty_weight}"
        )

        # GP
        self.gp = None
        self.likelihood = None

        # Training data
        self.train_X = None  # Original embeddings [N, D]
        self.train_U = None  # Directions [N, D] on S^(D-1)
        self.train_Y = None  # Scores [N]
        self.mean_norm = None
        self.smiles_observed = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.iteration = 0

        # Windowed GP data (recomputed each step)
        self._window_V = None  # Windowed subspace points [W, d]
        self._window_Y = None  # Windowed scores [W]
        self._window_indices = None
        self._y_mean = None
        self._y_std = None

        # ALL observed points in current subspace (for novelty)
        self._all_V = None  # [N, d] — all points projected, updated each step

        self.history = {
            "iteration": [],
            "best_score": [],
            "current_score": [],
            "n_evaluated": [],
            "gp_mean": [],
            "gp_std": [],
            "nearest_train_cos": [],
            "embedding_norm": [],
            "projection_idx": [],
            "window_size": [],
            "novelty_mean": [],
            "novelty_selected": [],
        }

        # GP diagnostics
        self.gp_diagnostics = GPDiagnostics(verbose=True)
        self.diagnostic_history = []

    def _current_projection_idx(self) -> int:
        """Get current projection index (round-robin)."""
        return self.iteration % self.n_projections

    def _current_A(self) -> torch.Tensor:
        """Get current projection matrix."""
        return self.projections[self._current_projection_idx()]

    def project_to_subspace(self, u: torch.Tensor, A: torch.Tensor | None = None) -> torch.Tensor:
        """Project from S^(D-1) to S^(d-1)."""
        if A is None:
            A = self._current_A()
        v = u @ A
        return F.normalize(v, p=2, dim=-1)

    def lift_to_original(self, v: torch.Tensor, A: torch.Tensor | None = None) -> torch.Tensor:
        """Lift from S^(d-1) to S^(D-1)."""
        if A is None:
            A = self._current_A()
        u = v @ A.T
        return F.normalize(u, p=2, dim=-1)

    def _create_kernel(self):
        """Create covariance kernel."""
        if self.kernel == "arccosine":
            return create_kernel(kernel_type="arccosine", kernel_order=0, use_scale=True)
        elif self.kernel == "matern":
            return create_kernel(kernel_type="matern", use_scale=True)
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'. Valid: arccosine, matern")

    def _select_window(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select windowed training subset: K_local nearest + K_random random."""
        N = len(self.train_Y)
        total_window = self.window_local + self.window_random

        if N <= total_window:
            V = self.project_to_subspace(self.train_U, A)
            self._window_indices = torch.arange(N, device=self.device)
            return V, self.train_Y

        # Find best point
        best_idx = self.train_Y.argmax()
        u_best = self.train_U[best_idx:best_idx + 1]

        # Cosine similarity to best in original space
        cos_sims = (self.train_U @ u_best.T).squeeze()

        # Top-K_local nearest
        _, local_indices = cos_sims.topk(self.window_local)

        # K_random from remaining
        mask = torch.ones(N, dtype=torch.bool, device=self.device)
        mask[local_indices] = False
        remaining_indices = torch.where(mask)[0]

        n_random = min(self.window_random, len(remaining_indices))
        if n_random > 0:
            perm = torch.randperm(len(remaining_indices), device=self.device)[:n_random]
            random_indices = remaining_indices[perm]
            window_indices = torch.cat([local_indices, random_indices])
        else:
            window_indices = local_indices

        self._window_indices = window_indices

        V = self.project_to_subspace(self.train_U[window_indices], A)
        Y = self.train_Y[window_indices]

        return V, Y

    def _fit_gp(self, A: torch.Tensor | None = None):
        """Fit GP on windowed subspace data with Y-normalization."""
        if A is None:
            A = self._current_A()

        # Select window
        V_window, Y_window = self._select_window(A)
        self._window_V = V_window
        self._window_Y = Y_window

        # Project ALL observed points for novelty computation
        self._all_V = self.project_to_subspace(self.train_U, A)

        X = V_window.double()

        # Y-normalization
        y_mean = Y_window.mean()
        y_std = Y_window.std()
        if y_std < 1e-8:
            y_std = torch.tensor(1.0, device=self.device)
        Y_norm = ((Y_window - y_mean) / y_std).double().unsqueeze(-1)

        self._y_mean = y_mean
        self._y_std = y_std

        try:
            covar_module = self._create_kernel()
            self.gp = SingleTaskGP(X, Y_norm, covar_module=covar_module).to(self.device)
            self.likelihood = self.gp.likelihood
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()

            if self.verbose and self.iteration % 10 == 0:
                metrics = self.gp_diagnostics.analyze(
                    self.gp, X.float(), Y_norm.squeeze(-1).float()
                )
                self.gp_diagnostics.log_summary(metrics, prefix=f"[Iter {self.iteration}]")
                self.diagnostic_history.append(
                    self.gp_diagnostics.get_summary_dict(metrics)
                )
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            self.fallback_count += 1
            logger.error(f"GP fit failed (fallback #{self.fallback_count}): {e}")
            self.gp = SingleTaskGP(
                X, Y_norm,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.likelihood = self.gp.likelihood
            self.likelihood.noise = 0.1
            self.gp.eval()

    def _generate_sobol_candidates(self, n_candidates: int, A: torch.Tensor) -> torch.Tensor:
        """Generate candidates in trust region around best point."""
        best_idx = self._window_Y.argmax()
        v_best = self._window_V[best_idx:best_idx + 1]

        half_length = self.trust_region / 2
        tr_lb = v_best - half_length
        tr_ub = v_best + half_length

        sobol = SobolEngine(self.subspace_dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=torch.float32, device=self.device)

        v_cand = tr_lb + (tr_ub - tr_lb) * pert
        v_cand = F.normalize(v_cand, p=2, dim=-1)

        return v_cand

    def _compute_novelty_bonus(self, v_candidates: torch.Tensor) -> torch.Tensor:
        """Compute geodesic novelty for candidates against ALL observed points.

        Args:
            v_candidates: [M, d] candidate points on S^(d-1)

        Returns:
            [M] novelty scores in [0, 1]
        """
        return geodesic_novelty(v_candidates, self._all_V)

    def _fit_gp_for_projection(self, A: torch.Tensor) -> tuple:
        """Fit a GP for a specific projection. Returns (gp, V_window, Y_window, y_mean, y_std, all_V).

        Used by ensemble mode to fit K independent GPs.
        """
        N = len(self.train_Y)
        total_window = self.window_local + self.window_random

        # Window selection
        if N <= total_window:
            V_all = self.project_to_subspace(self.train_U, A)
            V_window = V_all
            Y_window = self.train_Y
        else:
            best_idx = self.train_Y.argmax()
            u_best = self.train_U[best_idx:best_idx + 1]
            cos_sims = (self.train_U @ u_best.T).squeeze()
            _, local_indices = cos_sims.topk(self.window_local)
            mask = torch.ones(N, dtype=torch.bool, device=self.device)
            mask[local_indices] = False
            remaining = torch.where(mask)[0]
            n_random = min(self.window_random, len(remaining))
            if n_random > 0:
                perm = torch.randperm(len(remaining), device=self.device)[:n_random]
                window_indices = torch.cat([local_indices, remaining[perm]])
            else:
                window_indices = local_indices
            V_window = self.project_to_subspace(self.train_U[window_indices], A)
            Y_window = self.train_Y[window_indices]
            V_all = self.project_to_subspace(self.train_U, A)

        X = V_window.double()
        y_mean = Y_window.mean()
        y_std = Y_window.std()
        if y_std < 1e-8:
            y_std = torch.tensor(1.0, device=self.device)
        Y_norm = ((Y_window - y_mean) / y_std).double().unsqueeze(-1)

        try:
            covar_module = self._create_kernel()
            gp = SingleTaskGP(X, Y_norm, covar_module=covar_module).to(self.device)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            self.fallback_count += 1
            logger.error(f"GP fit failed for projection (fallback #{self.fallback_count}): {e}")
            gp = SingleTaskGP(
                X, Y_norm,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            gp.likelihood.noise = 0.1
            gp.eval()

        return gp, V_window, Y_window, y_mean, y_std, V_all

    def _optimize_ensemble_acquisition(self) -> tuple[torch.Tensor, dict]:
        """Ensemble TS: fit K GPs, draw K posterior samples, average, maximize.

        Candidates are generated in the original S^(D-1) space. Each GP
        evaluates them through its own projection, and scores are averaged.
        Novelty is computed as the average geodesic novelty across projections.
        """
        diag = {}

        try:
            # Generate candidates in the original space using best point
            best_idx = self.train_Y.argmax()
            u_best = self.train_U[best_idx:best_idx + 1]

            # Sobol perturbation around best in original space
            half_length = self.trust_region / 2
            sobol = SobolEngine(self.input_dim, scramble=True)
            pert = sobol.draw(self.n_candidates).to(dtype=torch.float32, device=self.device)
            u_cand = u_best - half_length + (2 * half_length) * pert
            u_cand = F.normalize(u_cand, p=2, dim=-1)  # [M, D] on S^(D-1)

            # Fit GP for each projection and score candidates
            f_samples = []
            novelty_scores = []

            for A in self.projections:
                gp_k, V_win_k, Y_win_k, y_mean_k, y_std_k, V_all_k = \
                    self._fit_gp_for_projection(A)

                # Project candidates into this subspace
                v_cand_k = self.project_to_subspace(u_cand, A)  # [M, d]

                # Draw posterior sample
                with torch.no_grad():
                    post = gp_k.posterior(v_cand_k.double())
                    f_sample = post.rsample().squeeze()  # [M]
                    f_samples.append(f_sample)

                # Novelty in this subspace
                nov_k = geodesic_novelty(v_cand_k, V_all_k)
                novelty_scores.append(nov_k)

            # Ensemble: average posterior samples across projections
            f_ensemble = torch.stack(f_samples).mean(dim=0)  # [M]
            novelty_ensemble = torch.stack(novelty_scores).mean(dim=0)  # [M]

            diag["novelty_mean"] = novelty_ensemble.mean().item()

            # Combined acquisition
            combined = f_ensemble + self.novelty_weight * novelty_ensemble.double()
            best_idx = combined.argmax()
            u_opt = u_cand[best_idx:best_idx + 1]
            diag["novelty_selected"] = novelty_ensemble[best_idx].item()

            # Diagnostics from ensemble
            with torch.no_grad():
                diag["gp_mean"] = f_ensemble[best_idx].item()
                diag["gp_std"] = torch.stack(f_samples).std(dim=0)[best_idx].item()
                diag["nearest_train_cos"] = (u_opt @ self.train_U.T).squeeze().max().item()

            return u_opt, diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            logger.error(f"Ensemble acquisition failed: {e}")
            u_opt = F.normalize(torch.randn(1, self.input_dim, device=self.device), dim=-1)
            return u_opt, {
                "gp_mean": 0, "gp_std": 1, "nearest_train_cos": 0,
                "novelty_mean": 0, "novelty_selected": 0, "is_fallback": True,
            }

    def _optimize_acquisition(self, A: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Find optimal v* using acquisition + novelty bonus."""
        diag = {}

        try:
            v_cand = self._generate_sobol_candidates(self.n_candidates, A)

            # Compute novelty bonus for all candidates
            novelty = self._compute_novelty_bonus(v_cand)
            diag["novelty_mean"] = novelty.mean().item()

            if self.acqf == "ts":
                # Thompson Sampling + novelty:
                # Draw posterior sample, add novelty bonus, maximize
                with torch.no_grad():
                    post = self.gp.posterior(v_cand.double())
                    # Sample from posterior
                    f_sample = post.rsample().squeeze()  # [M]
                    # Add novelty bonus (in normalized Y space)
                    combined = f_sample + self.novelty_weight * novelty.double()
                best_idx = combined.argmax()
                v_opt = v_cand[best_idx:best_idx + 1]
                diag["novelty_selected"] = novelty[best_idx].item()

            elif self.acqf == "ei":
                # EI + novelty
                best_f_norm = ((self.train_Y.max() - self._y_mean) / self._y_std).double()
                ei = qExpectedImprovement(self.gp, best_f=best_f_norm)
                with torch.no_grad():
                    ei_vals = ei(v_cand.double().unsqueeze(-2)).squeeze()
                    combined = ei_vals + self.novelty_weight * novelty.double()
                best_idx = combined.argmax()
                v_opt = v_cand[best_idx:best_idx + 1]
                diag["novelty_selected"] = novelty[best_idx].item()

            elif self.acqf == "ucb":
                # UCB + novelty
                with torch.no_grad():
                    post = self.gp.posterior(v_cand.double())
                    ucb_vals = post.mean.squeeze() + self.ucb_beta * post.variance.sqrt().squeeze()
                    combined = ucb_vals + self.novelty_weight * novelty.double()
                best_idx = combined.argmax()
                v_opt = v_cand[best_idx:best_idx + 1]
                diag["novelty_selected"] = novelty[best_idx].item()

            else:
                raise ValueError(f"Unknown acquisition function: {self.acqf}")

            # Diagnostics
            with torch.no_grad():
                post = self.gp.posterior(v_opt.double())
                gp_mean_norm = post.mean.item()
                gp_std_norm = post.variance.sqrt().item()
                diag["gp_mean"] = gp_mean_norm * self._y_std.item() + self._y_mean.item()
                diag["gp_std"] = gp_std_norm * self._y_std.item()
                cos_sims = (v_opt @ self._window_V.T).squeeze()
                diag["nearest_train_cos"] = cos_sims.max().item()

            u_opt = self.lift_to_original(v_opt, A)
            return u_opt, diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            logger.error(f"Acquisition failed: {e}")
            u_opt = F.normalize(torch.randn(1, self.input_dim, device=self.device), dim=-1)
            return u_opt, {
                "gp_mean": 0, "gp_std": 1, "nearest_train_cos": 0,
                "novelty_mean": 0, "novelty_selected": 0, "is_fallback": True,
            }

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules."""
        logger.info(f"Cold start: {len(smiles_list)} molecules")

        from tqdm import tqdm
        embeddings = []
        for i in tqdm(range(0, len(smiles_list), 64), desc="Encoding"):
            batch = smiles_list[i:i + 64]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0).to(self.device)

        self.mean_norm = embeddings.norm(dim=-1).mean().item()
        logger.info(f"Mean embedding norm: {self.mean_norm:.2f}")

        self.train_X = embeddings
        self.train_U = F.normalize(embeddings, p=2, dim=-1)
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]

        self._fit_gp(self.projections[0])

        logger.info(f"Cold start done. Best: {self.best_score:.4f} (n={len(self.train_Y)})")
        logger.info(f"Best SMILES: {self.best_smiles}")

    def step(self) -> dict:
        """One BO iteration with windowed GP, novelty bonus, and projection selection."""
        self.iteration += 1

        if self.acqf == "ets":
            # Ensemble Thompson Sampling: use ALL projections simultaneously
            u_opt, diag = self._optimize_ensemble_acquisition()
            diag["projection_idx"] = -1  # All projections used
            diag["window_size"] = self.window_local + self.window_random
        else:
            # Standard: round-robin single projection
            proj_idx = self._current_projection_idx()
            A = self.projections[proj_idx]

            # Refit GP (cheap with windowed data)
            self._fit_gp(A)

            # Optimize with novelty-augmented acquisition
            u_opt, diag = self._optimize_acquisition(A)
            diag["projection_idx"] = proj_idx
            diag["window_size"] = len(self._window_Y)

        # Reconstruct with mean norm
        x_opt = u_opt * self.mean_norm
        diag["embedding_norm"] = self.mean_norm

        # Decode
        smiles_list = self.codec.decode(x_opt)
        smiles = smiles_list[0] if smiles_list else ""

        if not smiles:
            logger.debug(f"Decode failed at iter {self.iteration}")
            return {"score": 0.0, "best_score": self.best_score, "smiles": "",
                    "is_duplicate": True, "is_decode_failure": True, **diag}

        if smiles in self.smiles_observed:
            return {"score": 0.0, "best_score": self.best_score, "smiles": smiles,
                    "is_duplicate": True, **diag}

        # Evaluate
        score = self.oracle.score(smiles)

        # Update training data
        self.train_X = torch.cat([self.train_X, x_opt], dim=0)
        self.train_U = torch.cat([self.train_U, u_opt], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device, dtype=torch.float32)])
        self.smiles_observed.append(smiles)

        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles
            logger.info(f"New best! {score:.4f}: {smiles}")

        return {"score": score, "best_score": self.best_score, "smiles": smiles,
                "is_duplicate": False, **diag}

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        from tqdm import tqdm

        logger.info(f"SubspaceBOv4: {n_iterations} iterations")
        logger.info(
            f"S^{self.input_dim-1} -> S^{self.subspace_dim-1}, "
            f"{self.n_projections} projections, "
            f"window={self.window_local}+{self.window_random}, "
            f"novelty_weight={self.novelty_weight}"
        )

        pbar = tqdm(range(n_iterations), desc="Optimizing")
        n_dup = 0

        for i in pbar:
            result = self.step()

            self.history["iteration"].append(i)
            self.history["best_score"].append(self.best_score)
            self.history["current_score"].append(result["score"])
            self.history["n_evaluated"].append(len(self.smiles_observed))
            self.history["gp_mean"].append(result.get("gp_mean", 0))
            self.history["gp_std"].append(result.get("gp_std", 0))
            self.history["nearest_train_cos"].append(result.get("nearest_train_cos", 0))
            self.history["embedding_norm"].append(result.get("embedding_norm", 0))
            self.history["projection_idx"].append(result.get("projection_idx", 0))
            self.history["window_size"].append(result.get("window_size", 0))
            self.history["novelty_mean"].append(result.get("novelty_mean", 0))
            self.history["novelty_selected"].append(result.get("novelty_selected", 0))

            if result["is_duplicate"]:
                n_dup += 1

            pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "proj": result.get("projection_idx", 0),
                "gp_s": f"{result.get('gp_std', 0):.4f}",
                "nov": f"{result.get('novelty_selected', 0):.3f}",
                "dup": n_dup,
            })

            if (i + 1) % log_interval == 0 and self.verbose:
                logger.info(
                    f"Iter {i+1}/{n_iterations} | Best: {self.best_score:.4f} | "
                    f"Curr: {result['score']:.4f} | "
                    f"GP: {result.get('gp_mean', 0):.2f}+/-{result.get('gp_std', 0):.4f} | "
                    f"Novelty: {result.get('novelty_selected', 0):.3f} (avg={result.get('novelty_mean', 0):.3f}) | "
                    f"proj: A[{result.get('projection_idx', 0)}] | "
                    f"dup: {n_dup}"
                )

        dup_rate = n_dup / n_iterations if n_iterations > 0 else 0
        logger.info(f"Done. Best: {self.best_score:.4f} | Duplicates: {n_dup}/{n_iterations} ({dup_rate:.1%})")
        logger.info(f"Best SMILES: {self.best_smiles}")
