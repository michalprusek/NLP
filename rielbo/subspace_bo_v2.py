"""Spherical Subspace BO v2: Theoretical Improvements.

Extends the base SphericalSubspaceBO with:
1. ArcCosine Order 2 kernel (smoother GP)
2. Spherical Whitening (center data at north pole)
3. Geodesic Trust Region (proper Riemannian sampling)
4. Adaptive Dimension (BAxUS-style d=8→16)
5. Probabilistic Norm reconstruction
6. Product Space geometry (S^3 × S^3 × S^3 × S^3)

Usage:
    from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2

    optimizer = SphericalSubspaceBOv2(
        codec=codec,
        oracle=oracle,
        kernel_order=2,
        whitening=True,
        geodesic_tr=True,
        adaptive_dim=True,
        prob_norm=True,
    )
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
    kernel_order: int = 0  # 0 or 2
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

    @classmethod
    def from_preset(cls, preset: str) -> "V2Config":
        """Create config from preset name."""
        presets = {
            "baseline": cls(),
            "order2": cls(kernel_order=2),
            "whitening": cls(whitening=True),
            "geodesic": cls(geodesic_tr=True),
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
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return presets[preset]


class SphericalSubspaceBOv2:
    """Spherical Subspace BO with theoretical improvements.

    Key innovations over v1:
    1. Smoother GP with ArcCosine Order 2 kernel
    2. Data centering with Spherical Whitening
    3. Proper Riemannian sampling with Geodesic Trust Region
    4. BAxUS-style adaptive dimension for exploration→exploitation
    5. Probabilistic norm reconstruction for diversity
    6. Product space geometry to avoid high-D sphere issues
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

        # GP
        self.gp = None
        self.likelihood = None

        # Training data
        self.train_X = None  # Original embeddings [N, D]
        self.train_U = None  # Directions [N, D] on S^(D-1)
        self.train_V = None  # Subspace [N, d] on S^(d-1)
        self.train_Y = None  # Scores [N]
        self.mean_norm = None
        self.smiles_observed = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.iteration = 0
        self.total_iterations = 0  # Set during optimize()

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
        }

        self.gp_diagnostics = GPDiagnostics(verbose=True)
        self.diagnostic_history = []
        self.fallback_count = 0

        self._log_config()

    def _log_config(self):
        """Log configuration summary."""
        cfg = self.config
        features = []
        if cfg.kernel_order == 2:
            features.append("order2")
        if cfg.whitening:
            features.append("whitening")
        if cfg.geodesic_tr:
            features.append("geodesic_tr")
        if cfg.adaptive_dim:
            features.append(f"adaptive({cfg.adaptive_start_dim}→{cfg.adaptive_end_dim})")
        if cfg.prob_norm:
            features.append(f"prob_norm({cfg.norm_method})")
        if cfg.product_space:
            features.append(f"product({cfg.n_spheres} spheres)")

        if not features:
            features = ["baseline"]

        logger.info(f"SubspaceBOv2: S^{self.input_dim-1} → S^{self._current_dim-1}")
        logger.info(f"Features: {', '.join(features)}")

    def _init_projection(self):
        """Initialize orthonormal projection matrix."""
        A_raw = torch.randn(self.input_dim, self._current_dim, device=self.device)
        self.A, _ = torch.linalg.qr(A_raw)

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
        """Project from S^(D-1) to S^(d-1), optionally through whitening."""
        if self.whitening is not None and self.whitening.H is not None:
            u = self.whitening.transform(u)
        v = u @ self.A
        return F.normalize(v, p=2, dim=-1)

    def lift_to_original(self, v: torch.Tensor) -> torch.Tensor:
        """Lift from S^(d-1) to S^(D-1), optionally through inverse whitening."""
        u = v @ self.A.T
        u = F.normalize(u, p=2, dim=-1)
        if self.whitening is not None and self.whitening.H is not None:
            u = self.whitening.inverse_transform(u)
        return u

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
        else:
            return create_kernel(
                kernel_type="arccosine",
                kernel_order=cfg.kernel_order,
                use_scale=True,
            )

    def _fit_gp(self):
        """Fit GP on subspace sphere."""
        self.train_V = self.project_to_subspace(self.train_U)

        X = self.train_V.double()
        Y = self.train_Y.double().unsqueeze(-1)

        try:
            covar_module = self._create_kernel()
            self.gp = SingleTaskGP(X, Y, covar_module=covar_module).to(self.device)
            self.likelihood = self.gp.likelihood
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()

            # Diagnostics
            if self.verbose and self.iteration % 10 == 0:
                metrics = self.gp_diagnostics.analyze(
                    self.gp, X.float(), Y.squeeze(-1).float()
                )
                self.gp_diagnostics.log_summary(metrics, prefix=f"[Iter {self.iteration}]")
                self.diagnostic_history.append(
                    self.gp_diagnostics.get_summary_dict(metrics)
                )
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            self.fallback_count += 1
            logger.warning(f"GP fit failed (fallback #{self.fallback_count}): {e}")
            self.gp = SingleTaskGP(
                X, Y,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.likelihood = self.gp.likelihood
            self.likelihood.noise = 0.1
            self.gp.eval()

    def _generate_candidates(self, n_candidates: int) -> torch.Tensor:
        """Generate candidates in subspace."""
        best_idx = self.train_Y.argmax()
        v_best = self.train_V[best_idx:best_idx+1]

        if self.geodesic_tr is not None:
            # Use geodesic trust region
            v_cand = self.geodesic_tr.sample(
                center=v_best,
                n_samples=n_candidates,
                adaptive_radius=self.config.geodesic_max_angle * self.trust_region,
            )
        else:
            # Original Sobol + box trust region
            from torch.quasirandom import SobolEngine

            half_length = self.trust_region / 2
            tr_lb = v_best - half_length
            tr_ub = v_best + half_length

            sobol = SobolEngine(self._current_dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=torch.float32, device=self.device)

            v_cand = tr_lb + (tr_ub - tr_lb) * pert
            v_cand = F.normalize(v_cand, p=2, dim=-1)

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
                ei = qExpectedImprovement(self.gp, best_f=self.train_Y.max().double())
                with torch.no_grad():
                    ei_vals = ei(v_cand.double().unsqueeze(-2))
                best_idx = ei_vals.argmax()
                v_opt = v_cand[best_idx:best_idx+1]

            elif self.acqf == "ucb":
                with torch.no_grad():
                    post = self.gp.posterior(v_cand.double())
                    ucb_vals = post.mean.squeeze() + self.ucb_beta * post.variance.sqrt().squeeze()
                best_idx = ucb_vals.argmax()
                v_opt = v_cand[best_idx:best_idx+1]

            else:
                raise ValueError(f"Unknown acquisition function: {self.acqf}")

            # Diagnostics
            with torch.no_grad():
                post = self.gp.posterior(v_opt.double())
                diag["gp_mean"] = post.mean.item()
                diag["gp_std"] = post.variance.sqrt().item()
                cos_sims = (v_opt @ self.train_V.T).squeeze()
                diag["nearest_train_cos"] = cos_sims.max().item()

            u_opt = self.lift_to_original(v_opt)
            return u_opt, diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            logger.warning(f"Acquisition failed: {e}")
            u_opt = F.normalize(torch.randn(1, self.input_dim, device=self.device), dim=-1)
            return u_opt, {"gp_mean": 0, "gp_std": 1, "nearest_train_cos": 0, "is_fallback": True}

    def _reconstruct_embedding(self, u_opt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Reconstruct embedding from direction."""
        if self.prob_reconstructor is not None and self.gp is not None:
            # Probabilistic reconstruction
            x_opt, norm_diag = self.prob_reconstructor.reconstruct(
                u_opt,
                gp=self.gp,
                project_fn=self.project_to_subspace,
            )
            return x_opt, norm_diag
        else:
            # Simple mean norm
            x_opt = u_opt * self.mean_norm
            return x_opt, {"embedding_norm": self.mean_norm}

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

        # Compute statistics
        norms = embeddings.norm(dim=-1)
        self.mean_norm = norms.mean().item()
        logger.info(f"Mean embedding norm: {self.mean_norm:.2f}")

        # Fit norm distribution if using probabilistic reconstruction
        if self.norm_dist is not None:
            self.norm_dist.fit(norms)
            self.prob_reconstructor = ProbabilisticReconstructor(
                self.norm_dist,
                n_candidates=self.config.norm_n_candidates,
                selection="gp_mean",
                device=self.device,
            )

        # Store data
        self.train_X = embeddings
        self.train_U = F.normalize(embeddings, p=2, dim=-1)
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        # Fit whitening transform
        if self.whitening is not None:
            self.whitening.fit(self.train_U)
            logger.info("Spherical whitening fitted")

        # Track best
        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]

        # Fit GP
        self._fit_gp()

        logger.info(f"Cold start done. Best: {self.best_score:.4f} (n={len(self.train_Y)})")
        logger.info(f"Best SMILES: {self.best_smiles}")

    def step(self) -> dict:
        """One BO iteration."""
        self.iteration += 1

        # Check for dimension switch
        self._maybe_switch_dimension()

        # Optimize acquisition
        u_opt, acq_diag = self._optimize_acquisition()

        # Reconstruct embedding
        x_opt, norm_diag = self._reconstruct_embedding(u_opt)
        diag = {**acq_diag, **norm_diag}
        diag["subspace_dim"] = self._current_dim

        # Decode
        smiles_list = self.codec.decode(x_opt)
        smiles = smiles_list[0] if smiles_list else ""

        if not smiles:
            return {"score": 0.0, "best_score": self.best_score, "smiles": "",
                    "is_duplicate": True, **diag}

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

        # Refit GP periodically
        if self.iteration % 10 == 0:
            self._fit_gp()

        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles
            logger.info(f"New best! {score:.4f}: {smiles}")

        return {"score": score, "best_score": self.best_score, "smiles": smiles,
                "is_duplicate": False, **diag}

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        from tqdm import tqdm

        self.total_iterations = n_iterations

        logger.info(f"SubspaceBOv2: {n_iterations} iterations")
        logger.info(f"S^{self.input_dim-1} → S^{self._current_dim-1}")

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

            if result["is_duplicate"]:
                n_dup += 1

            pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "dim": self._current_dim,
                "dup": n_dup,
            })

            if (i + 1) % log_interval == 0 and self.verbose:
                logger.info(
                    f"Iter {i+1}/{n_iterations} | Best: {self.best_score:.4f} | "
                    f"Curr: {result['score']:.4f} | "
                    f"GP: {result.get('gp_mean', 0):.2f}±{result.get('gp_std', 0):.2f} | "
                    f"dim: S^{self._current_dim-1}"
                )

        logger.info(f"Done. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")
