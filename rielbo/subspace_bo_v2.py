"""Spherical Subspace BO v2: Geodesic Trust Region + Adaptive TR.

Extends the base SphericalSubspaceBO with configurable features via V2Config.
Recommended preset: "geodesic" (geodesic TR + adaptive TR with random restart).

Usage:
    from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config

    config = V2Config.from_preset("geodesic")  # Recommended
    optimizer = SphericalSubspaceBOv2(
        codec=codec,
        oracle=oracle,
        config=config,
    )

Available features (configurable via V2Config):
1. ArcCosine kernel (order 0 recommended, order 2 available)
2. Spherical Whitening (center data at north pole)
3. Geodesic Trust Region (proper Riemannian sampling)
4. Adaptive Trust Region (TuRBO-style grow/shrink + random restart)
5. Adaptive Dimension (BAxUS-style d=8→16)
6. Probabilistic Norm reconstruction
7. Product Space geometry (S^3 × S^3 × S^3 × S^3)
"""

import logging
from dataclasses import dataclass

import gpytorch
import torch
import torch.nn.functional as F
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from rielbo.gp_diagnostics import GPDiagnostics
from rielbo.kernels import create_kernel
from rielbo.norm_distribution import NormDistribution, ProbabilisticReconstructor
from rielbo.spherical_transforms import GeodesicTrustRegion, SphericalWhitening

logger = logging.getLogger(__name__)


@dataclass
class V2Config:
    """Configuration for v2 improvements."""

    # Kernel
    kernel_type: str = "arccosine"  # "arccosine", "geodesic_matern", "matern"
    kernel_order: int = 0  # 0 or 2 for arccosine; maps to nu for geodesic_matern
    kernel_ard: bool = False  # Per-dimension lengthscales (for geodesic_matern/matern)
    product_space: bool = False
    n_spheres: int = 4

    # Transforms
    whitening: bool = False
    geodesic_tr: bool = False
    geodesic_max_angle: float = 0.5  # radians
    geodesic_global_fraction: float = 0.2

    # Adaptive dimension
    adaptive_dim: bool = False
    adaptive_start_dim: int = 8
    adaptive_end_dim: int = 16
    adaptive_switch_frac: float = 0.5  # Switch at 50% of iterations

    # Probabilistic norm
    prob_norm: bool = False
    norm_method: str = "gaussian"  # "gaussian", "histogram", "gmm"
    norm_n_candidates: int = 5

    # Adaptive trust region (TuRBO-style)
    adaptive_tr: bool = False
    tr_init: float = 0.4
    tr_min: float = 0.02
    tr_max: float = 0.8
    tr_success_tol: int = 3
    tr_fail_tol: int = 10
    tr_grow_factor: float = 1.5
    tr_shrink_factor: float = 0.5
    max_restarts: int = 5

    # Uncertainty-Responsive Trust Region (UR-TR)
    # Adapts geodesic radius based on GP posterior std, not success/failure.
    ur_tr: bool = False
    ur_relative: bool = True  # Thresholds relative to GP noise_std (scale-invariant)
    ur_std_high: float = 0.15  # GP informative → shrink TR
    ur_std_low: float = 0.05  # GP collapsing → expand TR
    ur_std_collapse: float = 0.005  # GP dead → rotate subspace
    ur_expand_factor: float = 1.5
    ur_shrink_factor: float = 0.8
    ur_collapse_patience: int = 15  # consecutive low-std iters before rotation
    ur_tr_min: float = 0.1  # min geodesic radius (radians)
    ur_tr_max: float = 1.2  # max geodesic radius (radians)
    ur_max_rotations: int = 10

    # Look-Ahead Subspace Selection (LASS)
    # Evaluates K random projections at cold start, picks most informative.
    lass: bool = False
    lass_n_candidates: int = 50

    # Projection type: "random" (QR) or "pca" (PCA on training data).
    # PCA is critical for high-D codecs (e.g. SONAR 1024D) where random
    # QR captures only d/D fraction of variance.
    projection_type: str = "random"

    # Acquisition function schedule
    # Switches to UCB with high beta when GP std is low (exploration boost).
    acqf_schedule: bool = False
    acqf_ucb_beta_high: float = 4.0  # UCB beta when GP is collapsing
    acqf_ucb_beta_low: float = 0.5  # UCB beta when GP is informative

    # Multi-subspace portfolio (TuRBO-M style)
    # Maintains K independent subspaces with UCB bandit allocation.
    multi_subspace: bool = False
    n_subspaces: int = 5
    subspace_ucb_beta: float = 2.0  # UCB exploration parameter for bandit
    subspace_stale_patience: int = 50  # Replace subspace after this many evals without improvement

    @classmethod
    def from_preset(cls, preset: str) -> "V2Config":
        """Create config from preset name."""
        presets = {
            "baseline": cls(),
            "order2": cls(kernel_order=2),
            "whitening": cls(whitening=True),
            "geodesic": cls(geodesic_tr=True, adaptive_tr=True),
            "adaptive": cls(adaptive_dim=True),
            "prob_norm": cls(prob_norm=True),
            "product": cls(product_space=True),
            "smooth": cls(kernel_order=2, whitening=True),
            "geometric": cls(geodesic_tr=True, whitening=True),
            "full": cls(
                kernel_order=2,
                whitening=True,
                geodesic_tr=True,
                adaptive_dim=True,
                prob_norm=True,
            ),
            # New presets for UR-TR + LASS
            "ur_tr": cls(geodesic_tr=True, ur_tr=True),
            "lass": cls(geodesic_tr=True, lass=True),
            "lass_ur": cls(geodesic_tr=True, ur_tr=True, lass=True),
            "explore": cls(
                geodesic_tr=True, ur_tr=True, lass=True, acqf_schedule=True,
            ),
            "portfolio": cls(
                geodesic_tr=True, ur_tr=True, lass=True, acqf_schedule=True,
                multi_subspace=True, n_subspaces=5,
            ),
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return presets[preset]


class SphericalSubspaceBOv2:
    """Spherical Subspace BO with configurable geometric features.

    Recommended configuration (geodesic preset):
    - ArcCosine order 0 kernel (rougher, matches chemical landscape)
    - Geodesic Trust Region (proper Riemannian sampling on S^d)
    - Adaptive TR with random restart (TuRBO-style grow/shrink)

    Additional features (retained for ablation, not recommended by default):
    - ArcCosine Order 2 kernel, Spherical Whitening, Adaptive dimension,
      Probabilistic norm reconstruction, Product space geometry
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        subspace_dim: int = 16,
        config: V2Config | None = None,
        device: str = "cuda",
        n_candidates: int = 2000,
        ucb_beta: float = 2.0,
        acqf: str = "ts",
        trust_region: float = 0.8,
        seed: int = 42,
        verbose: bool = True,
    ):
        if config is None:
            config = V2Config()

        self.config = config
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

        # Adaptive dimension state
        if config.adaptive_dim:
            self._current_dim = config.adaptive_start_dim
            self._dim_switched = False
        else:
            self._current_dim = subspace_dim
            self._dim_switched = True  # No switching needed

        # Initialize projection matrix
        torch.manual_seed(seed)
        self._init_projection()

        # Spherical whitening
        self.whitening = None
        if config.whitening:
            self.whitening = SphericalWhitening(device=device)

        # Geodesic trust region
        self.geodesic_tr = None
        if config.geodesic_tr:
            self.geodesic_tr = GeodesicTrustRegion(
                max_angle=config.geodesic_max_angle,
                global_fraction=config.geodesic_global_fraction,
                device=device,
            )

        # Probabilistic norm
        self.norm_dist = None
        self.prob_reconstructor = None
        if config.prob_norm:
            self.norm_dist = NormDistribution(
                method=config.norm_method,
                device=device,
            )

        # Adaptive trust region state
        self.tr_length = config.tr_init if config.adaptive_tr else None
        self._tr_success_count = 0
        self._tr_fail_count = 0
        self.n_restarts = 0

        # Uncertainty-responsive trust region state
        self._ur_radius = (config.geodesic_max_angle * trust_region) if config.ur_tr else None
        self._ur_collapse_count = 0
        self._ur_n_rotations = 0
        self._prev_gp_std = 1.0

        # Multi-subspace portfolio state
        self._subspaces = []
        self._active_subspace = 0
        self._total_bandit_steps = 0

        # GP
        self.gp = None
        self.likelihood = None

        # Training data
        self.train_X = None
        self.train_U = None
        self.train_V = None
        self.train_Y = None
        self.mean_norm = None
        self.smiles_observed = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.iteration = 0
        self.total_iterations = 0

        self.history = {
            "iteration": [],
            "best_score": [],
            "current_score": [],
            "n_evaluated": [],
            "gp_mean": [],
            "gp_std": [],
            "nearest_train_cos": [],
            "embedding_norm": [],
            "subspace_dim": [],
            "tr_length": [],
            "n_restarts": [],
            "ur_radius": [],
            "ur_rotations": [],
            "acqf_used": [],
            "active_subspace": [],
        }

        self.gp_diagnostics = GPDiagnostics(verbose=True)
        self.diagnostic_history = []
        self.fallback_count = 0
        self.last_mll_value = None  # Log marginal likelihood from last GP fit

        self._log_config()

    def _log_config(self):
        """Log configuration summary."""
        cfg = self.config
        features = []
        if cfg.kernel_type != "arccosine":
            features.append(cfg.kernel_type)
        if cfg.kernel_ard:
            features.append("ard")
        if cfg.kernel_order == 2:
            features.append("order2")
        if cfg.whitening:
            features.append("whitening")
        if cfg.geodesic_tr:
            features.append("geodesic_tr")
        if cfg.adaptive_dim:
            features.append(f"adaptive({cfg.adaptive_start_dim}->{cfg.adaptive_end_dim})")
        if cfg.prob_norm:
            features.append(f"prob_norm({cfg.norm_method})")
        if cfg.product_space:
            features.append(f"product({cfg.n_spheres} spheres)")
        if cfg.adaptive_tr:
            features.append("adaptive_tr")
        if cfg.ur_tr:
            features.append("ur_tr")
        if cfg.lass:
            features.append(f"lass({cfg.lass_n_candidates})")
        if cfg.acqf_schedule:
            features.append("acqf_schedule")
        if cfg.multi_subspace:
            features.append(f"portfolio(K={cfg.n_subspaces})")

        logger.info(f"SubspaceBOv2: S^{self.input_dim-1} -> S^{self._current_dim-1}")
        logger.info(f"Features: {', '.join(features or ['baseline'])}")

    def _init_projection(self):
        """Initialize orthonormal projection matrix.

        Uses PCA on training data when projection_type="pca" and data is available,
        otherwise falls back to random QR.
        """
        if (self.config.projection_type == "pca"
                and getattr(self, "train_U", None) is not None
                and self.train_U.shape[0] > self._current_dim):
            self._init_pca_projection()
        else:
            A_raw = torch.randn(self.input_dim, self._current_dim, device=self.device)
            self.A, _ = torch.linalg.qr(A_raw)

    def _init_pca_projection(self):
        """Compute PCA projection from training data on the unit sphere.

        Stores _pca_mean so project/lift can center/uncenter correctly.
        """
        self._pca_mean = self.train_U.mean(dim=0, keepdim=True).to(self.device)
        centered = self.train_U - self._pca_mean
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        d = min(self._current_dim, Vt.shape[0])
        self.A = Vt[:d].T.contiguous().to(self.device)
        var_explained = (S[:d]**2).sum() / (S**2).sum()
        logger.info(f"PCA projection: {self.input_dim}D → {d}D, variance explained: {var_explained:.3f}")

    def _maybe_switch_dimension(self):
        """Switch to higher dimension if conditions met."""
        if self._dim_switched:
            return

        cfg = self.config
        switch_iter = int(self.total_iterations * cfg.adaptive_switch_frac)

        if self.iteration >= switch_iter:
            old_dim = self._current_dim
            self._current_dim = cfg.adaptive_end_dim
            self._dim_switched = True

            # Create new projection
            torch.manual_seed(self.seed + 1000)  # Different seed for new projection
            self._init_projection()

            # Re-project training data
            if self.train_U is not None:
                self.train_V = self.project_to_subspace(self.train_U)
                self._fit_gp()

            logger.info(f"Adaptive dimension: S^{old_dim-1} → S^{self._current_dim-1} at iter {self.iteration}")

    def project_to_subspace(self, u: torch.Tensor) -> torch.Tensor:
        """Project from ambient space to subspace.

        For random QR: S^(D-1) → S^(d-1) (sphere to sphere).
        For PCA: centered PCA coordinates in R^d (Euclidean).
        """
        if self.config.projection_type == "pca" and hasattr(self, "_pca_mean"):
            return (u - self._pca_mean) @ self.A
        if self.whitening is not None and self.whitening.H is not None:
            u = self.whitening.transform(u)
        v = u @ self.A
        return F.normalize(v, p=2, dim=-1)

    def lift_to_original(self, v: torch.Tensor) -> torch.Tensor:
        """Lift from subspace back to ambient space.

        For random QR: S^(d-1) → S^(D-1).
        For PCA: PCA coordinates → unit sphere (approximately).
        """
        if self.config.projection_type == "pca" and hasattr(self, "_pca_mean"):
            return v @ self.A.T + self._pca_mean
        u = v @ self.A.T
        u = F.normalize(u, p=2, dim=-1)
        if self.whitening is not None and self.whitening.H is not None:
            u = self.whitening.inverse_transform(u)
        return u

    def _select_best_projection(self):
        """LASS: Evaluate K random projections, pick the one with best GP log ML.

        NOTE: Maximizing posterior std is WRONG -- picks projections where
        the GP cannot model the landscape (flat/noisy). Log marginal likelihood
        picks projections where the kernel captures meaningful score structure.
        """
        K = self.config.lass_n_candidates
        best_score = float("-inf")
        best_A = None
        best_k = -1
        all_scores = []
        Y = self.train_Y.double().unsqueeze(-1)

        logger.info(f"LASS: evaluating {K} candidate projections (criterion=log_ml)...")

        for k in range(K):
            torch.manual_seed(self.seed + k * 137)
            A_raw = torch.randn(self.input_dim, self._current_dim, device=self.device)
            A_k, _ = torch.linalg.qr(A_raw)
            X = F.normalize(self.train_U @ A_k, p=2, dim=-1).double()

            try:
                gp_k = SingleTaskGP(X, Y, covar_module=self._create_kernel()).to(self.device)
                mll_fn = ExactMarginalLogLikelihood(gp_k.likelihood, gp_k)
                fit_gpytorch_mll(mll_fn)
                gp_k.eval()

                with torch.no_grad():
                    log_ml = mll_fn(gp_k(X), Y.squeeze(-1)).item()

                all_scores.append(log_ml)
                if log_ml > best_score:
                    best_score = log_ml
                    best_A = A_k.clone()
                    best_k = k
            except Exception as e:
                logger.debug(f"LASS candidate {k}: GP fit failed: {e}")
                all_scores.append(float("-inf"))

        if best_A is not None:
            self.A = best_A
            valid_scores = [s for s in all_scores if s > float("-inf")]
            logger.info(
                f"LASS: selected projection {best_k}/{K} "
                f"with log_ml = {best_score:.4f} "
                f"(range: [{min(valid_scores):.4f}, {max(valid_scores):.4f}])"
            )
        else:
            logger.warning("LASS: all candidates failed, keeping default projection")

    def _create_kernel(self):
        """Create covariance kernel based on config."""
        cfg = self.config
        if cfg.product_space:
            return create_kernel(
                kernel_type="product",
                kernel_order=cfg.kernel_order,
                n_spheres=cfg.n_spheres,
                use_scale=True,
            )
        ard_num_dims = self._current_dim if cfg.kernel_ard else None
        return create_kernel(
            kernel_type=cfg.kernel_type,
            kernel_order=cfg.kernel_order,
            use_scale=True,
            ard_num_dims=ard_num_dims,
        )

    def _fit_gp(self):
        """Fit GP on subspace sphere."""
        self.train_V = self.project_to_subspace(self.train_U)
        X = self.train_V.double()
        Y = self.train_Y.double().unsqueeze(-1)

        try:
            if self.config.projection_type == "pca":
                # PCA coordinates are Euclidean -- use Hvarfner priors with input normalization
                self.gp = SingleTaskGP(
                    X, Y,
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
                    ),
                    input_transform=Normalize(d=X.shape[-1]),
                ).to(self.device)
            else:
                self.gp = SingleTaskGP(
                    X, Y, covar_module=self._create_kernel()
                ).to(self.device)

            self.likelihood = self.gp.likelihood
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()

            try:
                with torch.no_grad():
                    self.last_mll_value = mll(self.gp(X), Y.squeeze(-1)).item()
            except Exception as e:
                logger.debug(f"MLL computation failed: {e}")
                self.last_mll_value = None

            if self.verbose:
                metrics = self.gp_diagnostics.analyze(
                    self.gp, X.float(), Y.squeeze(-1).float()
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
                X, Y,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.likelihood = self.gp.likelihood
            self.likelihood.noise = 0.1
            self.gp.eval()

    def _update_trust_region(self, improved: bool):
        """Update adaptive trust region length (TuRBO-style)."""
        if not self.config.adaptive_tr:
            return

        cfg = self.config
        if improved:
            self._tr_success_count += 1
            self._tr_fail_count = 0
        else:
            self._tr_success_count = 0
            self._tr_fail_count += 1

        if self._tr_success_count >= cfg.tr_success_tol:
            self.tr_length = min(self.tr_length * cfg.tr_grow_factor, cfg.tr_max)
            self._tr_success_count = 0
            if self.verbose:
                logger.info(f"TR grow → {self.tr_length:.4f}")

        elif self._tr_fail_count >= cfg.tr_fail_tol:
            self.tr_length *= cfg.tr_shrink_factor
            self._tr_fail_count = 0

            if self.tr_length < cfg.tr_min:
                self._restart_subspace()
            elif self.verbose:
                logger.info(f"TR shrink → {self.tr_length:.4f}")

    def _restart_subspace(self):
        """Restart with fresh random projection when TR collapses."""
        cfg = self.config
        if self.n_restarts >= cfg.max_restarts:
            self.tr_length = cfg.tr_init
            logger.info(f"Max restarts ({cfg.max_restarts}) reached, resetting TR to {cfg.tr_init}")
            return

        self.n_restarts += 1
        self.tr_length = cfg.tr_init
        self._tr_success_count = 0
        self._tr_fail_count = 0

        torch.manual_seed(self.seed + self.n_restarts * 1000)
        self._init_projection()

        if self.train_U is not None:
            self.train_V = self.project_to_subspace(self.train_U)
            self._fit_gp()

        logger.info(
            f"Subspace restart #{self.n_restarts}: "
            f"{cfg.projection_type} projection, TR reset to {cfg.tr_init}"
        )

    def _get_gp_noise_std(self) -> float:
        """Get GP noise std for scale-invariant thresholds. Returns 1.0 as fallback."""
        if not self.config.ur_relative or not hasattr(self, 'gp') or self.gp is None:
            return 1.0
        try:
            return self.gp.likelihood.noise.item() ** 0.5
        except Exception as e:
            logger.debug(f"Failed to get GP noise std: {e}")
            return 1.0

    def _update_ur_tr(self, gp_std: float):
        """Update UR-TR radius based on GP posterior uncertainty.

        Counter-intuitive: EXPAND when GP is collapsing (need exploration),
        SHRINK when GP is informative (can exploit locally).
        """
        if not self.config.ur_tr:
            return

        cfg = self.config
        self._prev_gp_std = gp_std

        noise_std = self._get_gp_noise_std()
        eff_std_high = cfg.ur_std_high * noise_std
        eff_std_low = cfg.ur_std_low * noise_std

        if gp_std > eff_std_high:
            # GP is informative → shrink TR for local exploitation
            old = self._ur_radius
            self._ur_radius = max(self._ur_radius * cfg.ur_shrink_factor, cfg.ur_tr_min)
            self._ur_collapse_count = 0
            if self.verbose and abs(old - self._ur_radius) > 1e-6:
                logger.debug(f"UR-TR shrink: {old:.3f} → {self._ur_radius:.3f} (std={gp_std:.4f})")

        elif gp_std < eff_std_low:
            # GP is collapsing → expand TR for broader exploration
            old = self._ur_radius
            self._ur_radius = min(self._ur_radius * cfg.ur_expand_factor, cfg.ur_tr_max)
            if self.verbose and abs(old - self._ur_radius) > 1e-6:
                logger.debug(f"UR-TR expand: {old:.3f} → {self._ur_radius:.3f} (std={gp_std:.4f})")

            # Track consecutive collapse (scaled by noise_std like other thresholds)
            eff_std_collapse = cfg.ur_std_collapse * noise_std
            if gp_std < eff_std_collapse:
                self._ur_collapse_count += 1
            else:
                self._ur_collapse_count = 0

            # Rotate subspace on sustained collapse
            if self._ur_collapse_count >= cfg.ur_collapse_patience:
                self._ur_rotate_subspace()
        else:
            # GP in normal range → keep radius
            self._ur_collapse_count = 0

    def _ur_rotate_subspace(self):
        """Rotate to fresh random projection when GP collapses under UR-TR."""
        cfg = self.config
        init_radius = cfg.geodesic_max_angle * self.trust_region

        if self._ur_n_rotations >= cfg.ur_max_rotations:
            logger.info(f"UR-TR: max rotations ({cfg.ur_max_rotations}) reached, resetting radius only")
            self._ur_radius = init_radius
            self._ur_collapse_count = 0
            return

        self._ur_n_rotations += 1
        self._ur_collapse_count = 0
        self._ur_radius = init_radius

        torch.manual_seed(self.seed + self._ur_n_rotations * 2000)
        self._init_projection()

        if self.train_U is not None:
            self.train_V = self.project_to_subspace(self.train_U)
            self._fit_gp()

        logger.info(
            f"UR-TR rotation #{self._ur_n_rotations}: "
            f"fresh projection, radius reset to {self._ur_radius:.3f}"
        )

    def _make_fresh_subspace_state(self, A: torch.Tensor) -> dict:
        """Create initial state dict for a portfolio subspace."""
        return {
            "A": A,
            "n_evals": 0,
            "n_success": 0,
            "n_consec_fail": 0,
            "total_reward": 0.0,
            "ur_radius": self.config.geodesic_max_angle * self.trust_region,
            "ur_collapse_count": 0,
            "ur_n_rotations": 0,
        }

    def _init_subspaces(self):
        """Create K independent subspaces. Subspace 0 uses current (LASS-selected) projection."""
        K = self.config.n_subspaces
        self._subspaces = [self._make_fresh_subspace_state(self.A.clone())]
        for k in range(1, K):
            torch.manual_seed(self.seed + 5000 + k * 137)
            A_raw = torch.randn(self.input_dim, self._current_dim, device=self.device)
            A_k, _ = torch.linalg.qr(A_raw)
            self._subspaces.append(self._make_fresh_subspace_state(A_k))

        self._active_subspace = 0
        self._total_bandit_steps = 0
        logger.info(f"Portfolio: initialized {K} subspaces")

    def _select_subspace_bandit(self) -> int:
        """Select subspace via UCB1: argmax_k (avg_reward_k + beta * sqrt(log(T)/n_k))."""
        import math as _math

        T = max(self._total_bandit_steps, 1)
        log_T = _math.log(T)
        beta = self.config.subspace_ucb_beta

        best_ucb = float("-inf")
        best_k = 0
        for k, s in enumerate(self._subspaces):
            n_k = max(s["n_evals"], 1)
            ucb = s["total_reward"] / n_k + beta * (log_T / n_k) ** 0.5
            if ucb > best_ucb:
                best_ucb = ucb
                best_k = k
        return best_k

    def _switch_to_subspace(self, k: int):
        """Switch active subspace: swap projection, UR-TR state, refit GP."""
        if k == self._active_subspace and self._total_bandit_steps > 0:
            return

        old_k = self._active_subspace
        self._active_subspace = k
        s = self._subspaces[k]
        self.A = s["A"]

        if self.config.ur_tr:
            self._ur_radius = s["ur_radius"]
            self._ur_collapse_count = s["ur_collapse_count"]
            self._ur_n_rotations = s["ur_n_rotations"]

        if self.train_U is not None:
            self.train_V = self.project_to_subspace(self.train_U)
            self._fit_gp()

        if self.verbose and self._total_bandit_steps > 0:
            logger.debug(f"Portfolio: switched {old_k} -> {k}")

    def _update_subspace_stats(self, k: int, improved: bool):
        """Update bandit stats for subspace k after evaluation."""
        s = self._subspaces[k]
        s["n_evals"] += 1
        self._total_bandit_steps += 1

        if improved:
            s["n_success"] += 1
            s["n_consec_fail"] = 0
            s["total_reward"] += 1.0
        else:
            s["n_consec_fail"] += 1

        if self.config.ur_tr:
            s["ur_radius"] = self._ur_radius
            s["ur_collapse_count"] = self._ur_collapse_count
            s["ur_n_rotations"] = self._ur_n_rotations

        if s["n_consec_fail"] >= self.config.subspace_stale_patience:
            self._replace_subspace(k)

    def _replace_subspace(self, k: int):
        """Replace a stale subspace with a fresh random projection."""
        torch.manual_seed(self.seed + 9000 + self._total_bandit_steps * 31 + k)
        A_raw = torch.randn(self.input_dim, self._current_dim, device=self.device)
        A_k, _ = torch.linalg.qr(A_raw)

        old_evals = self._subspaces[k]["n_evals"]
        self._subspaces[k] = self._make_fresh_subspace_state(A_k)

        if k == self._active_subspace:
            self.A = A_k
            if self.config.ur_tr:
                state = self._subspaces[k]
                self._ur_radius = state["ur_radius"]
                self._ur_collapse_count = state["ur_collapse_count"]
                self._ur_n_rotations = state["ur_n_rotations"]
            if self.train_U is not None:
                self.train_V = self.project_to_subspace(self.train_U)
                self._fit_gp()

        logger.info(
            f"Portfolio: replaced stale subspace {k} "
            f"(was {old_evals} evals, 0 successes in last {self.config.subspace_stale_patience})"
        )

    def _get_geodesic_radius(self) -> float:
        """Get effective geodesic radius: UR-TR > adaptive_tr > static."""
        if self.config.ur_tr:
            return self._ur_radius
        if self.config.adaptive_tr:
            return self.config.geodesic_max_angle * self.tr_length
        return self.config.geodesic_max_angle * self.trust_region

    def _generate_candidates(self, n_candidates: int) -> torch.Tensor:
        """Generate candidates in subspace around best observed point."""
        best_idx = self.train_Y.argmax()
        v_best = self.project_to_subspace(self.train_U[best_idx:best_idx+1])

        if self.config.projection_type == "pca":
            from torch.quasirandom import SobolEngine
            train_v = self.project_to_subspace(self.train_U)
            v_std = train_v.std(dim=0, keepdim=True).clamp(min=1e-6)
            if self.config.ur_tr and self._ur_radius is not None:
                init_radius = self.config.geodesic_max_angle * self.trust_region
                tr_scale = self._ur_radius / init_radius
            else:
                tr_scale = self.trust_region
            half_length = v_std * tr_scale * 2
            sobol = SobolEngine(self._current_dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=torch.float32, device=self.device)
            return v_best - half_length + 2 * half_length * pert

        if self.geodesic_tr is not None:
            return self.geodesic_tr.sample(
                center=v_best,
                n_samples=n_candidates,
                adaptive_radius=self._get_geodesic_radius(),
            )

        # Fallback: Sobol + box trust region
        from torch.quasirandom import SobolEngine
        half_length = self.trust_region / 2
        sobol = SobolEngine(self._current_dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=torch.float32, device=self.device)
        return F.normalize(
            (v_best - half_length) + self.trust_region * pert, p=2, dim=-1
        )

    def _get_effective_acqf(self) -> tuple[str, float]:
        """Return (acqf_name, ucb_beta) based on acquisition schedule.

        Switches to UCB when GP std is outside normal range (exploration boost).
        """
        if not self.config.acqf_schedule:
            return self.acqf, self.ucb_beta

        cfg = self.config
        noise_std = self._get_gp_noise_std()

        if self._prev_gp_std < cfg.ur_std_low * noise_std:
            return "ucb", cfg.acqf_ucb_beta_high
        if self._prev_gp_std > cfg.ur_std_high * noise_std:
            return "ucb", cfg.acqf_ucb_beta_low
        return self.acqf, self.ucb_beta

    def _optimize_acquisition(self) -> tuple[torch.Tensor, dict]:
        """Find optimal v* using acquisition function."""
        diag = {}

        try:
            v_cand = self._generate_candidates(self.n_candidates)
            effective_acqf, effective_beta = self._get_effective_acqf()
            diag["acqf_used"] = effective_acqf

            if effective_acqf == "ts":
                thompson = MaxPosteriorSampling(model=self.gp, replacement=False)
                v_opt = thompson(v_cand.double().unsqueeze(0), num_samples=1).squeeze(0).float()
                if self.config.projection_type != "pca":
                    v_opt = F.normalize(v_opt, p=2, dim=-1)

            elif effective_acqf == "ei":
                ei = qExpectedImprovement(self.gp, best_f=self.train_Y.max().double())
                with torch.no_grad():
                    ei_vals = ei(v_cand.double().unsqueeze(-2))
                v_opt = v_cand[ei_vals.argmax():ei_vals.argmax()+1]

            elif effective_acqf == "ucb":
                with torch.no_grad():
                    post = self.gp.posterior(v_cand.double())
                    ucb_vals = post.mean.squeeze() + effective_beta * post.variance.sqrt().squeeze()
                v_opt = v_cand[ucb_vals.argmax():ucb_vals.argmax()+1]

            else:
                raise ValueError(f"Unknown acquisition function: {effective_acqf}")

            with torch.no_grad():
                post = self.gp.posterior(v_opt.double())
                diag["gp_mean"] = post.mean.item()
                diag["gp_std"] = post.variance.sqrt().item()
                train_v_current = self.project_to_subspace(self.train_U)
                if self.config.projection_type == "pca":
                    cos_sims = F.cosine_similarity(
                        v_opt.expand_as(train_v_current), train_v_current, dim=-1
                    )
                else:
                    cos_sims = (v_opt @ train_v_current.T).squeeze()
                diag["nearest_train_cos"] = cos_sims.max().item()

            return self.lift_to_original(v_opt), diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            logger.error(f"Acquisition failed: {e}")
            u_opt = F.normalize(torch.randn(1, self.input_dim, device=self.device), dim=-1)
            return u_opt, {"gp_mean": 0, "gp_std": 1, "nearest_train_cos": 0, "is_fallback": True}

    def _reconstruct_embedding(self, u_opt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Reconstruct embedding from direction."""
        if self.prob_reconstructor is not None and self.gp is not None:
            return self.prob_reconstructor.reconstruct(
                u_opt, gp=self.gp, project_fn=self.project_to_subspace,
            )
        return u_opt * self.mean_norm, {"embedding_norm": self.mean_norm}

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules."""
        logger.info(f"Cold start: {len(smiles_list)} molecules")

        from tqdm import tqdm
        embeddings = []
        for i in tqdm(range(0, len(smiles_list), 64), desc="Encoding"):
            batch = smiles_list[i:i+64]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0).to(self.device)

        norms = embeddings.norm(dim=-1)
        self.mean_norm = norms.mean().item()
        logger.info(f"Mean embedding norm: {self.mean_norm:.2f}")

        if self.norm_dist is not None:
            self.norm_dist.fit(norms)
            self.prob_reconstructor = ProbabilisticReconstructor(
                self.norm_dist,
                n_candidates=self.config.norm_n_candidates,
                selection="gp_mean",
                device=self.device,
            )

        self.train_X = embeddings
        self.train_U = F.normalize(embeddings, p=2, dim=-1)
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        if self.whitening is not None:
            self.whitening.fit(self.train_U)
            logger.info("Spherical whitening fitted")

        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]

        if self.config.projection_type == "pca":
            self._init_projection()
        if self.config.lass and self.config.projection_type != "pca":
            self._select_best_projection()

        self._fit_gp()

        if self.config.multi_subspace:
            self._init_subspaces()

        logger.info(f"Cold start done. Best: {self.best_score:.4f} (n={len(self.train_Y)})")
        logger.info(f"Best SMILES: {self.best_smiles}")

    def _update_all_tr(self, improved: bool, gp_std: float):
        """Update all trust region mechanisms and bandit stats."""
        self._update_ur_tr(gp_std=gp_std)
        self._update_trust_region(improved=improved)
        if self.config.multi_subspace and self._subspaces:
            self._update_subspace_stats(self._active_subspace, improved=improved)

    def step(self) -> dict:
        """One BO iteration."""
        self.iteration += 1
        self._maybe_switch_dimension()

        if self.config.multi_subspace and self._subspaces:
            self._switch_to_subspace(self._select_subspace_bandit())

        u_opt, acq_diag = self._optimize_acquisition()
        x_opt, norm_diag = self._reconstruct_embedding(u_opt)
        diag = {**acq_diag, **norm_diag}
        diag["subspace_dim"] = self._current_dim
        diag["active_subspace"] = self._active_subspace

        smiles_list = self.codec.decode(x_opt)
        smiles = smiles_list[0] if smiles_list else ""
        base_result = {"best_score": self.best_score, "x_opt": x_opt,
                       "mll": self.last_mll_value, **diag}

        if not smiles:
            logger.debug(f"Decode failed at iter {self.iteration}")
            self._update_all_tr(improved=False, gp_std=diag.get("gp_std", 0))
            return {"score": 0.0, "smiles": "", "is_duplicate": True,
                    "is_decode_failure": True, **base_result}

        if smiles in self.smiles_observed:
            self._update_all_tr(improved=False, gp_std=diag.get("gp_std", 0))
            return {"score": 0.0, "smiles": smiles, "is_duplicate": True, **base_result}

        score = self.oracle.score(smiles)

        self.train_X = torch.cat([self.train_X, x_opt], dim=0)
        self.train_U = torch.cat([self.train_U, u_opt], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device, dtype=torch.float32)])
        self.smiles_observed.append(smiles)

        refit_interval = 1 if self.config.projection_type == "pca" else 10
        if self.iteration % refit_interval == 0:
            self._fit_gp()

        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_smiles = smiles
            logger.info(f"New best! {score:.4f}: {smiles}")

        self._update_all_tr(improved=improved, gp_std=diag.get("gp_std", 0))

        return {"score": score, "best_score": self.best_score, "smiles": smiles,
                "is_duplicate": False, "x_opt": x_opt, "mll": self.last_mll_value, **diag}

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        from tqdm import tqdm

        self.total_iterations = n_iterations

        logger.info(f"SubspaceBOv2: {n_iterations} iterations")
        logger.info(f"S^{self.input_dim-1} -> S^{self._current_dim-1}")

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
            self.history["subspace_dim"].append(result.get("subspace_dim", self._current_dim))
            self.history["tr_length"].append(self.tr_length if self.tr_length is not None else self.trust_region)
            self.history["n_restarts"].append(self.n_restarts)
            self.history["ur_radius"].append(self._ur_radius if self._ur_radius is not None else 0)
            self.history["ur_rotations"].append(self._ur_n_rotations)
            self.history["acqf_used"].append(result.get("acqf_used", self.acqf))
            self.history["active_subspace"].append(result.get("active_subspace", 0))

            if result["is_duplicate"]:
                n_dup += 1

            postfix = {
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "dim": self._current_dim,
                "dup": n_dup,
            }
            if self.config.adaptive_tr:
                postfix["tr"] = f"{self.tr_length:.3f}"
                postfix["rst"] = self.n_restarts
            if self.config.ur_tr:
                postfix["ur"] = f"{self._ur_radius:.3f}"
                postfix["rot"] = self._ur_n_rotations
            if self.config.multi_subspace:
                postfix["sub"] = self._active_subspace
            pbar.set_postfix(postfix)

            if (i + 1) % log_interval == 0 and self.verbose:
                ur_info = f" | UR: {self._ur_radius:.3f} rot={self._ur_n_rotations}" if self.config.ur_tr else ""
                port_info = f" | sub={self._active_subspace}" if self.config.multi_subspace else ""
                logger.info(
                    f"Iter {i+1}/{n_iterations} | Best: {self.best_score:.4f} | "
                    f"Curr: {result['score']:.4f} | "
                    f"GP: {result.get('gp_mean', 0):.2f}±{result.get('gp_std', 0):.2f} | "
                    f"dim: S^{self._current_dim-1}{ur_info}{port_info}"
                )

        logger.info(f"Done. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")
        if self.fallback_count > 0:
            logger.warning(f"GP fallbacks: {self.fallback_count} iterations used untrained fallback GP")
