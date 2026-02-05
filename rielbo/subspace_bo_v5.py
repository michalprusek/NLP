"""Spherical Subspace BO v5: Best-of-All-Worlds Combination.

Combines the proven best features from v2, v3, and TuRBO:
1. Spherical Whitening (v2): Householder rotation centering data at north pole
2. Geodesic Trust Region (v2): Proper Riemannian candidate sampling on S^(d-1)
3. Windowed GP (v3): 50 nearest + 30 random points, prevents posterior collapse
4. Every-step GP refit with Y-normalization (v3): Cheap with 80-point window
5. Adaptive Trust Region (TuRBO-style): Grow on success, shrink on failure, restart

What's NOT included (evidence they don't help):
- Multi-projection ensemble (adds variance, not mean — v3 analysis)
- Geodesic novelty bonus (too aggressive even at γ=0.1 — v4 scored 0.5315 vs 0.5440)
- Gradient-based acquisition (3.5x slower, no improvement — v3 grad_ucb tests)
- Order-2 ArcCosine kernel (only +0.3%, not significant — v2 ablation)

Pipeline:
    x [N, D] → normalize → u [N, D] on S^(D-1)
    u → whitening (Householder) → u_w on S^(D-1)
    u_w → project (A) → v [N, d] on S^(d-1)
    Window: 50 nearest + 30 random → 80 points for GP
    Y-normalize: Y_norm = (Y - mean) / std
    GP on S^(d-1) with ArcCosine kernel
    Candidates: geodesic disk around v_best with adaptive radius
    v* → lift (A^T) → u_w* → inverse whitening → u* → x* = u* * mean_norm → decode
    Adaptive TR: grow on improvement, shrink on stagnation, restart on collapse

Usage:
    from rielbo.subspace_bo_v5 import SphericalSubspaceBOv5

    optimizer = SphericalSubspaceBOv5(
        codec=codec,
        oracle=oracle,
        subspace_dim=16,
    )
"""

import logging

import gpytorch
import torch
import torch.nn.functional as F
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from rielbo.gp_diagnostics import GPDiagnostics
from rielbo.kernels import create_kernel
from rielbo.spherical_transforms import GeodesicTrustRegion, SphericalWhitening

logger = logging.getLogger(__name__)


class SphericalSubspaceBOv5:
    """Spherical Subspace BO v5: combines geodesic TR + windowed GP + whitening + adaptive TR.

    Evidence-based design:
    - Geodesic TR: +2.6% mean over baseline (v2 ablation, 10 seeds)
    - Whitening: +2.0% mean over baseline (v2 ablation, 10 seeds)
    - Windowed GP: prevents posterior collapse (v3 analysis)
    - Adaptive TR: prevents wasted evaluations when stuck (TuRBO principle)
    - Subspace restart: escape local optima by rotating projection (new idea)
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
        seed: int = 42,
        verbose: bool = True,
        kernel: str = "arccosine",
        # Windowed GP (from v3)
        window_local: int = 50,
        window_random: int = 30,
        # Geodesic trust region (from v2)
        geodesic_max_angle: float = 0.5,
        geodesic_global_fraction: float = 0.2,
        # Adaptive trust region (TuRBO-style)
        tr_init: float = 0.4,
        tr_min: float = 0.02,
        tr_max: float = 0.8,
        tr_success_tol: int = 3,
        tr_fail_tol: int = 10,
        tr_grow_factor: float = 1.5,
        tr_shrink_factor: float = 0.5,
        # Subspace restart
        max_restarts: int = 5,
    ):
        # Validation
        if subspace_dim >= input_dim:
            raise ValueError(f"subspace_dim ({subspace_dim}) must be < input_dim ({input_dim})")
        if subspace_dim < 2:
            raise ValueError(f"subspace_dim must be >= 2, got {subspace_dim}")

        self.device = device
        self.codec = codec
        self.oracle = oracle
        self.input_dim = input_dim
        self.subspace_dim = subspace_dim
        self.n_candidates = n_candidates
        self.ucb_beta = ucb_beta
        self.acqf = acqf
        self.verbose = verbose
        self.seed = seed
        self.kernel = kernel
        self.fallback_count = 0

        # Windowed GP params
        self.window_local = window_local
        self.window_random = window_random

        # Geodesic trust region
        self.geodesic_max_angle = geodesic_max_angle
        self.geodesic_tr = GeodesicTrustRegion(
            max_angle=geodesic_max_angle,
            global_fraction=geodesic_global_fraction,
            device=device,
        )

        # Adaptive trust region state (TuRBO-style)
        self.tr_length = tr_init
        self.tr_init = tr_init
        self.tr_min = tr_min
        self.tr_max = tr_max
        self.tr_success_tol = tr_success_tol
        self.tr_fail_tol = tr_fail_tol
        self.tr_grow_factor = tr_grow_factor
        self.tr_shrink_factor = tr_shrink_factor
        self._success_count = 0
        self._fail_count = 0

        # Subspace restart
        self.max_restarts = max_restarts
        self.n_restarts = 0

        # Spherical whitening
        self.whitening = SphericalWhitening(device=device)

        # Initialize projection matrix
        torch.manual_seed(seed)
        self._init_projection()

        # GP
        self.gp = None
        self.likelihood = None

        # Training data
        self.train_X = None
        self.train_U = None  # Directions on S^(D-1)
        self.train_Y = None
        self.mean_norm = None
        self.smiles_observed = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.iteration = 0

        # Windowed GP data
        self._window_V = None
        self._window_Y = None
        self._y_mean = None
        self._y_std = None

        self.history = {
            "iteration": [],
            "best_score": [],
            "current_score": [],
            "n_evaluated": [],
            "gp_mean": [],
            "gp_std": [],
            "nearest_train_cos": [],
            "embedding_norm": [],
            "tr_length": [],
            "n_restarts": [],
        }

        self.gp_diagnostics = GPDiagnostics(verbose=True)
        self.diagnostic_history = []

        logger.info(
            f"SubspaceBOv5: S^{input_dim-1} -> S^{subspace_dim-1}, "
            f"kernel={kernel}, acqf={acqf}, "
            f"window={window_local}+{window_random}, "
            f"geodesic_tr(max_angle={geodesic_max_angle}), "
            f"whitening=True, adaptive_tr(init={tr_init})"
        )

    def _init_projection(self):
        """Initialize orthonormal projection matrix."""
        A_raw = torch.randn(self.input_dim, self.subspace_dim, device=self.device)
        self.A, _ = torch.linalg.qr(A_raw)

    def project_to_subspace(self, u: torch.Tensor) -> torch.Tensor:
        """Project from S^(D-1) to S^(d-1) through whitening."""
        if self.whitening.H is not None:
            u = self.whitening.transform(u)
        v = u @ self.A
        return F.normalize(v, p=2, dim=-1)

    def lift_to_original(self, v: torch.Tensor) -> torch.Tensor:
        """Lift from S^(d-1) to S^(D-1) through inverse whitening."""
        u = v @ self.A.T
        u = F.normalize(u, p=2, dim=-1)
        if self.whitening.H is not None:
            u = self.whitening.inverse_transform(u)
        return u

    def _create_kernel(self):
        """Create covariance kernel."""
        if self.kernel == "arccosine":
            return create_kernel(kernel_type="arccosine", kernel_order=0, use_scale=True)
        elif self.kernel == "matern":
            return create_kernel(kernel_type="matern", use_scale=True)
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'. Valid: arccosine, matern")

    def _select_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Select windowed training subset: K_local nearest + K_random random."""
        N = len(self.train_Y)
        total_window = self.window_local + self.window_random

        if N <= total_window:
            V = self.project_to_subspace(self.train_U)
            return V, self.train_Y

        # Find best point — use cosine similarity in original D-dim space
        best_idx = self.train_Y.argmax()
        u_best = self.train_U[best_idx:best_idx + 1]
        cos_sims = (self.train_U @ u_best.T).squeeze()

        # Top-K nearest
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

        V = self.project_to_subspace(self.train_U[window_indices])
        Y = self.train_Y[window_indices]
        return V, Y

    def _fit_gp(self):
        """Fit GP on windowed subspace data with Y-normalization."""
        V_window, Y_window = self._select_window()
        self._window_V = V_window
        self._window_Y = Y_window

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

    def _generate_candidates(self, n_candidates: int) -> torch.Tensor:
        """Generate candidates using geodesic trust region with adaptive radius."""
        best_idx = self._window_Y.argmax()
        v_best = self._window_V[best_idx:best_idx + 1]

        # Adaptive radius = base max_angle * current TR length
        adaptive_radius = self.geodesic_max_angle * self.tr_length

        v_cand = self.geodesic_tr.sample(
            center=v_best,
            n_samples=n_candidates,
            adaptive_radius=adaptive_radius,
        )
        return v_cand

    def _optimize_acquisition(self) -> tuple[torch.Tensor, dict]:
        """Find optimal v* using acquisition function."""
        diag = {}

        try:
            v_cand = self._generate_candidates(self.n_candidates)

            if self.acqf == "ts":
                thompson = MaxPosteriorSampling(model=self.gp, replacement=False)
                v_opt = thompson(v_cand.double().unsqueeze(0), num_samples=1)
                v_opt = v_opt.squeeze(0).float()
                v_opt = F.normalize(v_opt, p=2, dim=-1)

            elif self.acqf == "ei":
                best_f_norm = ((self.train_Y.max() - self._y_mean) / self._y_std).double()
                ei = qExpectedImprovement(self.gp, best_f=best_f_norm)
                with torch.no_grad():
                    ei_vals = ei(v_cand.double().unsqueeze(-2))
                best_idx = ei_vals.argmax()
                v_opt = v_cand[best_idx:best_idx + 1]

            elif self.acqf == "ucb":
                with torch.no_grad():
                    post = self.gp.posterior(v_cand.double())
                    ucb_vals = post.mean.squeeze() + self.ucb_beta * post.variance.sqrt().squeeze()
                best_idx = ucb_vals.argmax()
                v_opt = v_cand[best_idx:best_idx + 1]

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

            u_opt = self.lift_to_original(v_opt)
            return u_opt, diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            logger.error(f"Acquisition failed: {e}")
            u_opt = F.normalize(torch.randn(1, self.input_dim, device=self.device), dim=-1)
            return u_opt, {"gp_mean": 0, "gp_std": 1, "nearest_train_cos": 0, "is_fallback": True}

    def _update_trust_region(self, improved: bool):
        """Update trust region length based on success/failure (TuRBO-style).

        - 3 consecutive successes → grow TR by 1.5x
        - 10 consecutive failures → shrink TR by 0.5x
        - TR below minimum → restart with new projection
        """
        if improved:
            self._success_count += 1
            self._fail_count = 0
        else:
            self._success_count = 0
            self._fail_count += 1

        if self._success_count >= self.tr_success_tol:
            self.tr_length = min(self.tr_length * self.tr_grow_factor, self.tr_max)
            self._success_count = 0
            if self.verbose:
                logger.info(f"TR grow → {self.tr_length:.4f}")

        elif self._fail_count >= self.tr_fail_tol:
            self.tr_length *= self.tr_shrink_factor
            self._fail_count = 0

            if self.tr_length < self.tr_min:
                self._restart_subspace()
            elif self.verbose:
                logger.info(f"TR shrink → {self.tr_length:.4f}")

    def _restart_subspace(self):
        """Restart with new random projection when TR collapses."""
        if self.n_restarts >= self.max_restarts:
            # No more restarts — reset TR to initial and continue
            self.tr_length = self.tr_init
            logger.info(f"Max restarts ({self.max_restarts}) reached, resetting TR to {self.tr_init}")
            return

        self.n_restarts += 1
        self.tr_length = self.tr_init
        self._success_count = 0
        self._fail_count = 0

        # New random projection
        torch.manual_seed(self.seed + self.n_restarts * 1000)
        self._init_projection()

        logger.info(
            f"Subspace restart #{self.n_restarts}: new projection, "
            f"TR reset to {self.tr_init}"
        )

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

        # Fit whitening on directions
        self.whitening.fit(self.train_U)
        logger.info("Spherical whitening fitted")

        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]

        self._fit_gp()

        logger.info(f"Cold start done. Best: {self.best_score:.4f} (n={len(self.train_Y)})")
        logger.info(f"Best SMILES: {self.best_smiles}")

    def step(self) -> dict:
        """One BO iteration with geodesic TR + windowed GP + adaptive TR."""
        self.iteration += 1

        # Refit GP every step (cheap with 80-point window)
        self._fit_gp()

        # Optimize acquisition
        u_opt, diag = self._optimize_acquisition()
        diag["tr_length"] = self.tr_length
        diag["n_restarts"] = self.n_restarts

        # Reconstruct embedding
        x_opt = u_opt * self.mean_norm
        diag["embedding_norm"] = self.mean_norm

        # Decode
        smiles_list = self.codec.decode(x_opt)
        smiles = smiles_list[0] if smiles_list else ""

        if not smiles:
            logger.debug(f"Decode failed at iter {self.iteration}")
            self._update_trust_region(improved=False)
            return {"score": 0.0, "best_score": self.best_score, "smiles": "",
                    "is_duplicate": True, "is_decode_failure": True, **diag}

        if smiles in self.smiles_observed:
            self._update_trust_region(improved=False)
            return {"score": 0.0, "best_score": self.best_score, "smiles": smiles,
                    "is_duplicate": True, **diag}

        # Evaluate
        score = self.oracle.score(smiles)

        # Update training data
        self.train_X = torch.cat([self.train_X, x_opt], dim=0)
        self.train_U = torch.cat([self.train_U, u_opt], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device, dtype=torch.float32)])
        self.smiles_observed.append(smiles)

        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_smiles = smiles
            logger.info(f"New best! {score:.4f}: {smiles}")

        self._update_trust_region(improved=improved)

        return {"score": score, "best_score": self.best_score, "smiles": smiles,
                "is_duplicate": False, **diag}

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        from tqdm import tqdm

        logger.info(f"SubspaceBOv5: {n_iterations} iterations")
        logger.info(
            f"S^{self.input_dim-1} -> S^{self.subspace_dim-1}, "
            f"geodesic_tr + whitening + windowed_gp + adaptive_tr"
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
            self.history["embedding_norm"].append(result.get("embedding_norm", self.mean_norm))
            self.history["tr_length"].append(result.get("tr_length", self.tr_length))
            self.history["n_restarts"].append(result.get("n_restarts", self.n_restarts))

            if result["is_duplicate"]:
                n_dup += 1

            pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "tr": f"{self.tr_length:.3f}",
                "rst": self.n_restarts,
                "dup": n_dup,
            })

            if (i + 1) % log_interval == 0 and self.verbose:
                logger.info(
                    f"Iter {i+1}/{n_iterations} | Best: {self.best_score:.4f} | "
                    f"Curr: {result['score']:.4f} | "
                    f"GP: {result.get('gp_mean', 0):.2f}+/-{result.get('gp_std', 0):.4f} | "
                    f"TR: {self.tr_length:.4f} | restarts: {self.n_restarts} | "
                    f"dup: {n_dup}"
                )

        dup_rate = n_dup / n_iterations if n_iterations > 0 else 0
        logger.info(f"Done. Best: {self.best_score:.4f} | Duplicates: {n_dup}/{n_iterations} ({dup_rate:.1%})")
        logger.info(f"Best SMILES: {self.best_smiles}")
        logger.info(f"Restarts: {self.n_restarts}/{self.max_restarts}")
