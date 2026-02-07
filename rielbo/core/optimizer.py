"""BaseOptimizer: unified BO loop composing all components.

Provides an alternative to the monolithic optimizer classes (V1, V2, TuRBO,
VanillaBO, Ensemble, AdaS) as a single orchestrator that delegates to
composable components via the Protocol interfaces.

Usage:
    config = OptimizerConfig.from_preset("explore")
    optimizer = BaseOptimizer(codec, oracle, config)
    optimizer.cold_start(smiles_list, scores)
    for i in range(500):
        result = optimizer.step()
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn.functional as F

from rielbo.core.config import OptimizerConfig
from rielbo.core.types import BOHistory, StepResult, TrainingData
from rielbo.components.acquisition import create_acquisition
from rielbo.components.candidate_gen import (
    SobolBoxGenerator,
    create_candidate_generator,
)
from rielbo.components.norm_reconstruction import (
    ProbabilisticNormReconstructor,
    create_norm_reconstructor,
)
from rielbo.components.projection import (
    LASSSelector,
    PCAProjection,
    QRProjection,
    create_projection,
)
from rielbo.components.surrogate import EuclideanGPSurrogate, SphericalGPSurrogate
from rielbo.components.trust_region import URTR, create_trust_region

logger = logging.getLogger(__name__)


class BaseOptimizer:
    """Unified Bayesian optimization loop composing all components.

    The optimizer orchestrates:
    1. Projection (ambient → subspace)
    2. GP surrogate fitting
    3. Trust region adaptation
    4. Candidate generation
    5. Acquisition function selection
    6. Norm reconstruction and decoding
    """

    def __init__(
        self,
        codec,
        oracle,
        config: OptimizerConfig,
        input_dim: int = 256,
        subspace_dim: int | None = None,
        n_candidates: int | None = None,
        acqf: str | None = None,
        ucb_beta: float | None = None,
        trust_region: float = 0.8,
    ):
        self.config = copy.deepcopy(config)
        self.codec = codec
        self.oracle = oracle
        self.device = self.config.device
        self.seed = self.config.seed
        self.verbose = self.config.verbose

        # Dimensions
        self.input_dim = input_dim
        cfg_proj = self.config.projection
        if subspace_dim is not None:
            cfg_proj.subspace_dim = subspace_dim
        cfg_proj.input_dim = input_dim
        self._trust_region_scale = trust_region

        # Override acquisition config from constructor args
        if acqf is not None:
            self.config.acquisition.acqf = acqf
        if ucb_beta is not None:
            self.config.acquisition.ucb_beta = ucb_beta
        if n_candidates is not None:
            self.config.acquisition.n_candidates = n_candidates

        # Adaptive dimension state
        self._current_dim = cfg_proj.subspace_dim
        self._dim_switched = True
        if cfg_proj.adaptive_dim:
            self._current_dim = cfg_proj.adaptive_start_dim
            self._dim_switched = False

        # ── Create components ────────────────────────────────────────

        # Projection
        self.projection = create_projection(
            cfg_proj, device=self.device, seed=self.seed,
        )

        # Surrogate
        is_pca = cfg_proj.projection_type == "pca"
        is_identity = cfg_proj.projection_type == "identity"

        if is_identity:
            self.surrogate = EuclideanGPSurrogate(
                self.config.kernel, input_dim, device=self.device,
                verbose=self.verbose,
            )
        else:
            self.surrogate = SphericalGPSurrogate(
                self.config.kernel, self._current_dim, device=self.device,
                verbose=self.verbose, pca_mode=is_pca,
            )

        # Trust region
        gp_getter = lambda: self.surrogate.gp  # noqa: E731
        self.trust_region = create_trust_region(
            self.config.trust_region,
            trust_region=trust_region,
            dim=input_dim,
            gp_getter=gp_getter,
        )

        # UR-TR (separate from adaptive TR — both can be active)
        self.ur_tr: URTR | None = None
        if self.config.trust_region.ur_tr:
            init_radius = self.config.trust_region.geodesic_max_angle * trust_region
            self.ur_tr = URTR(
                self.config.trust_region,
                initial_radius=init_radius,
                gp_getter=gp_getter,
            )

        # Candidate generator
        self.candidate_gen = create_candidate_generator(
            self.config.candidate_gen,
            dim=self._current_dim,
            device=self.device,
            geodesic_max_angle=self.config.trust_region.geodesic_max_angle,
            geodesic_global_fraction=self.config.trust_region.geodesic_global_fraction,
        )

        # Acquisition
        self.acq_selector, self.acq_schedule = create_acquisition(
            self.config.acquisition, self.config.trust_region,
        )

        # Norm reconstruction
        self.norm_reconstructor = create_norm_reconstructor(
            self.config.norm_reconstruction, device=self.device,
        )

        # LASS selector (created if lass enabled, used at cold_start)
        self.lass_selector: LASSSelector | None = None
        if cfg_proj.lass and cfg_proj.projection_type == "random":
            self.lass_selector = LASSSelector(
                cfg_proj, self.config.kernel, device=self.device, seed=self.seed,
            )

        # ── State ────────────────────────────────────────────────────

        self.data = TrainingData()
        self.history = BOHistory()
        self.iteration = 0
        self.total_iterations = 0

        self._log_config()

    def _log_config(self) -> None:
        """Log configuration summary."""
        cfg = self.config
        features = []
        kc = cfg.kernel
        if kc.kernel_type != "arccosine":
            features.append(kc.kernel_type)
        if kc.kernel_ard:
            features.append("ard")
        if kc.kernel_order == 2:
            features.append("order2")
        pc = cfg.projection
        if pc.whitening:
            features.append("whitening")
        if pc.projection_type != "random":
            features.append(pc.projection_type)
        if pc.adaptive_dim:
            features.append(
                f"adaptive({pc.adaptive_start_dim}→{pc.adaptive_end_dim})"
            )
        if pc.lass:
            features.append(f"lass({pc.lass_n_candidates})")
        tc = cfg.trust_region
        if tc.geodesic:
            features.append("geodesic_tr")
        if tc.strategy == "adaptive":
            features.append("adaptive_tr")
        if tc.ur_tr:
            features.append("ur_tr")
        if tc.strategy == "turbo":
            features.append("turbo_tr")
        ac = cfg.acquisition
        if ac.schedule:
            features.append("acqf_schedule")
        nc = cfg.norm_reconstruction
        if nc.method == "probabilistic":
            features.append(f"prob_norm({nc.prob_method})")
        if not features:
            features = ["baseline"]

        logger.info(
            f"BaseOptimizer: {self.input_dim}D → {self._current_dim}D "
            f"({cfg.projection.projection_type})"
        )
        logger.info(f"Features: {', '.join(features)}")

    # ── Cold start ───────────────────────────────────────────────────

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize with pre-scored molecules."""
        logger.info(f"Cold start: {len(smiles_list)} molecules")

        from tqdm import tqdm

        embeddings = []
        for i in tqdm(range(0, len(smiles_list), 64), desc="Encoding"):
            batch = smiles_list[i : i + 64]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0).to(self.device)

        norms = embeddings.norm(dim=-1)
        self.data.mean_norm = norms.mean().item()
        logger.info(f"Mean embedding norm: {self.data.mean_norm:.2f}")

        # Fit norm reconstruction
        if isinstance(self.norm_reconstructor, ProbabilisticNormReconstructor):
            self.norm_reconstructor.fit(norms)
        else:
            self.norm_reconstructor.mean_norm = self.data.mean_norm

        # Store training data
        self.data.train_X = embeddings
        self.data.train_U = F.normalize(embeddings, p=2, dim=-1)
        self.data.train_Y = scores.to(self.device).float()
        self.data.smiles_observed = smiles_list.copy()
        self.data.update_best()

        # PCA needs data to compute projection
        if isinstance(self.projection, PCAProjection):
            self.projection.fit(self.data.train_U)

        # Fit whitening
        if isinstance(self.projection, QRProjection):
            self.projection.fit_whitening(self.data.train_U)

        # LASS: select best projection before first GP fit
        if self.lass_selector is not None and isinstance(
            self.projection, QRProjection,
        ):
            self.lass_selector.select_best_projection(
                self.projection, self.data.train_U, self.data.train_Y,
            )

        # Set normalization bounds for Euclidean surrogates
        if isinstance(self.surrogate, EuclideanGPSurrogate):
            self.surrogate.set_normalization_bounds(embeddings)

        # Fit GP
        self._fit_gp()

        logger.info(
            f"Cold start done. Best: {self.data.best_score:.4f} "
            f"(n={self.data.n_observed})"
        )
        logger.info(f"Best SMILES: {self.data.best_smiles}")

    # ── GP fitting ───────────────────────────────────────────────────

    def _fit_gp(self) -> None:
        """Re-project training data and fit GP."""
        if isinstance(self.surrogate, EuclideanGPSurrogate):
            # Full-dim: fit on raw embeddings
            mode = "turbo" if self.config.kernel.kernel_type == "matern" else "hvarfner"
            self.surrogate.fit(
                self.data.train_X, self.data.train_Y,
                iteration=self.iteration, mode=mode,
            )
        else:
            # Subspace: project then fit
            self.data.train_V = self.projection.project(self.data.train_U)
            self.surrogate.fit(
                self.data.train_V, self.data.train_Y,
                iteration=self.iteration,
            )

    # ── Adaptive dimension ───────────────────────────────────────────

    def _maybe_switch_dimension(self) -> None:
        if self._dim_switched:
            return
        cfg = self.config.projection
        switch_iter = int(self.total_iterations * cfg.adaptive_switch_frac)
        if self.iteration >= switch_iter:
            old_dim = self._current_dim
            self._current_dim = cfg.adaptive_end_dim
            self._dim_switched = True
            self.projection.reinitialize(
                self.seed + 1000, train_U=self.data.train_U,
            )
            self._fit_gp()
            logger.info(
                f"Adaptive dimension: {old_dim}D → {self._current_dim}D "
                f"at iter {self.iteration}"
            )

    # ── Candidate generation ─────────────────────────────────────────

    def _get_effective_radius(self) -> float:
        """Compute the current effective trust region radius."""
        cfg = self.config.trust_region

        if self.ur_tr is not None:
            return self.ur_tr.radius

        if cfg.geodesic:
            return cfg.geodesic_max_angle * self.trust_region.radius
        else:
            return self.trust_region.radius

    def _generate_candidates(self) -> torch.Tensor:
        """Generate candidates in the appropriate space."""
        n_cand = self.config.acquisition.n_candidates
        best_idx = self.data.train_Y.argmax()

        if isinstance(self.surrogate, EuclideanGPSurrogate):
            # Full-dim Euclidean: Sobol box around best point
            center = self.data.train_X[best_idx : best_idx + 1]
            radius = self.trust_region.radius
            gen = self.candidate_gen
            if isinstance(gen, SobolBoxGenerator):
                return gen.generate(
                    center, n_cand, radius,
                    normalize_to_sphere=False,
                    clip_bounds=None,
                )
            return gen.generate(center, n_cand, radius)

        # Subspace: project best point fresh from train_U
        v_best = self.projection.project(
            self.data.train_U[best_idx : best_idx + 1],
        )

        # PCA mode: Sobol box in Euclidean PCA space
        if isinstance(self.projection, PCAProjection):
            gen = self.candidate_gen
            if isinstance(gen, SobolBoxGenerator):
                train_v = self.projection.project(self.data.train_U)
                v_std = train_v.std(dim=0, keepdim=True).clamp(min=1e-6)
                if self.ur_tr is not None:
                    init_radius = (
                        self.config.trust_region.geodesic_max_angle
                        * self._trust_region_scale
                    )
                    tr_scale = self.ur_tr.radius / init_radius
                else:
                    tr_scale = self._trust_region_scale
                return gen.generate_pca(v_best, v_std, n_cand, tr_scale)
            return gen.generate(v_best, n_cand, self._get_effective_radius())

        # Spherical subspace: geodesic or Sobol
        radius = self._get_effective_radius()
        return self.candidate_gen.generate(v_best, n_cand, radius)

    # ── Acquisition ──────────────────────────────────────────────────

    def _optimize_acquisition(self) -> tuple[torch.Tensor, dict]:
        """Find optimal point using acquisition function."""
        gp = self.surrogate.gp
        if gp is None:
            u_opt = F.normalize(
                torch.randn(1, self.input_dim, device=self.device), dim=-1,
            )
            return u_opt, {"gp_mean": 0, "gp_std": 1, "is_fallback": True}

        try:
            candidates = self._generate_candidates()

            # Get effective acqf from schedule
            acqf_name = None
            ucb_beta = None
            if self.acq_schedule is not None:
                prev_std = 1.0
                if self.ur_tr is not None:
                    prev_std = self.ur_tr.get_state().prev_gp_std
                acqf_name, ucb_beta = self.acq_schedule.get_effective_acqf(
                    prev_std, gp=gp,
                )

            is_pca = isinstance(self.projection, PCAProjection)
            v_opt, diag = self.acq_selector.select(
                gp, candidates, self.data.train_Y,
                acqf=acqf_name, ucb_beta=ucb_beta,
                pca_mode=is_pca,
            )

            # Nearest-train cosine similarity diagnostic
            with torch.no_grad():
                train_v = self.projection.project(self.data.train_U)
                if is_pca:
                    cos_sims = F.cosine_similarity(
                        v_opt.expand_as(train_v), train_v, dim=-1,
                    )
                else:
                    cos_sims = (v_opt @ train_v.T).squeeze()
                diag["nearest_train_cos"] = cos_sims.max().item()

            # Lift back to ambient space
            u_opt = self.projection.lift(v_opt)
            return u_opt, diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            logger.error(f"Acquisition failed: {e}")
            u_opt = F.normalize(
                torch.randn(1, self.input_dim, device=self.device), dim=-1,
            )
            return u_opt, {
                "gp_mean": 0, "gp_std": 1,
                "nearest_train_cos": 0, "is_fallback": True,
            }

    # ── Reconstruction ───────────────────────────────────────────────

    def _reconstruct_embedding(
        self, u_opt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Reconstruct full embedding from unit direction."""
        return self.norm_reconstructor.reconstruct(
            u_opt, gp=self.surrogate.gp,
            project_fn=self.projection.project,
        )

    # ── Subspace restart / rotation ──────────────────────────────────

    def _handle_restart(self) -> None:
        """Handle trust region collapse → fresh projection + refit GP."""
        self.trust_region.reset()
        seed = self.seed + self.trust_region.get_state().n_restarts * 1000
        self.projection.reinitialize(seed, train_U=self.data.train_U)
        self._fit_gp()
        logger.info(
            f"Subspace restart #{self.trust_region.get_state().n_restarts}: "
            f"fresh projection, TR reset"
        )

    def _handle_rotation(self) -> None:
        """Handle UR-TR sustained collapse → fresh projection + refit GP."""
        if self.ur_tr is None:
            return
        self.ur_tr.reset()
        seed = self.seed + self.ur_tr.n_rotations * 2000
        self.projection.reinitialize(seed, train_U=self.data.train_U)
        self._fit_gp()
        logger.info(
            f"UR-TR rotation #{self.ur_tr.n_rotations}: "
            f"fresh projection, radius reset to {self.ur_tr.radius:.3f}"
        )

    # ── Step ─────────────────────────────────────────────────────────

    def step(self) -> StepResult:
        """Execute one BO iteration."""
        self.iteration += 1

        # Check for adaptive dimension switch
        self._maybe_switch_dimension()

        # Optimize acquisition → get direction on S^(D-1)
        u_opt, acq_diag = self._optimize_acquisition()

        # Reconstruct embedding with norm
        x_opt, norm_diag = self._reconstruct_embedding(u_opt)
        diag = {**acq_diag, **norm_diag}

        # Decode
        smiles_list = self.codec.decode(x_opt)
        smiles = smiles_list[0] if smiles_list else ""

        gp_std = diag.get("gp_std", 0)

        # Handle decode failure
        if not smiles:
            logger.debug(f"Decode failed at iter {self.iteration}")
            self._update_tr(improved=False, gp_std=gp_std)
            return self._make_result(
                score=0.0, smiles="", diag=diag,
                x_opt=x_opt, is_duplicate=True, is_decode_failure=True,
            )

        # Handle duplicate
        if smiles in self.data.smiles_observed:
            self._update_tr(improved=False, gp_std=gp_std)
            return self._make_result(
                score=0.0, smiles=smiles, diag=diag,
                x_opt=x_opt, is_duplicate=True,
            )

        # Evaluate
        score = self.oracle.score(smiles)

        # Update training data
        improved = self.data.add_observation(x_opt, u_opt, score, smiles)

        # Update normalization bounds for Euclidean surrogates
        if isinstance(self.surrogate, EuclideanGPSurrogate):
            self.surrogate.update_normalization_bounds(x_opt)

        # Refit GP periodically (every iter for PCA, every 10 for spherical)
        is_pca = isinstance(self.projection, PCAProjection)
        refit_interval = 1 if is_pca else 10
        if self.iteration % refit_interval == 0:
            self._fit_gp()

        if improved:
            logger.info(f"New best! {score:.4f}: {smiles}")

        # Update trust regions
        self._update_tr(improved=improved, gp_std=gp_std)

        return self._make_result(
            score=score, smiles=smiles, diag=diag, x_opt=x_opt,
        )

    def _update_tr(self, improved: bool, gp_std: float) -> None:
        """Update all active trust region strategies."""
        # UR-TR (based on GP std)
        if self.ur_tr is not None:
            self.ur_tr.update(improved, gp_std=gp_std)
            if self.ur_tr.needs_rotation:
                self._handle_rotation()

        # Adaptive / TuRBO TR (based on improvement)
        self.trust_region.update(improved, gp_std=gp_std)
        if self.trust_region.needs_restart:
            self._handle_restart()

    def _make_result(
        self,
        score: float,
        smiles: str,
        diag: dict,
        x_opt: torch.Tensor,
        is_duplicate: bool = False,
        is_decode_failure: bool = False,
    ) -> StepResult:
        """Build StepResult from step outputs."""
        tr_state = self.trust_region.get_state()
        ur_radius = self.ur_tr.radius if self.ur_tr is not None else 0.0
        ur_rotations = self.ur_tr.n_rotations if self.ur_tr is not None else 0

        return StepResult(
            score=score,
            best_score=self.data.best_score,
            smiles=smiles,
            is_duplicate=is_duplicate,
            is_decode_failure=is_decode_failure,
            gp_mean=diag.get("gp_mean", 0),
            gp_std=diag.get("gp_std", 0),
            nearest_train_cos=diag.get("nearest_train_cos", 0),
            embedding_norm=diag.get("embedding_norm", self.data.mean_norm),
            subspace_dim=self._current_dim,
            tr_length=tr_state.radius,
            n_restarts=tr_state.n_restarts,
            ur_radius=ur_radius,
            ur_rotations=ur_rotations,
            acqf_used=diag.get("acqf_used", self.config.acquisition.acqf),
            mll=self.surrogate.last_mll_value,
            x_opt=x_opt,
        )

    # ── Optimize loop ────────────────────────────────────────────────

    def optimize(
        self,
        n_iterations: int,
        log_interval: int = 10,
    ) -> None:
        """Run optimization loop."""
        from tqdm import tqdm

        self.total_iterations = n_iterations
        logger.info(f"BaseOptimizer: {n_iterations} iterations")

        pbar = tqdm(range(n_iterations), desc="Optimizing")
        n_dup = 0

        for i in pbar:
            result = self.step()
            self.history.append_from_result(result, self.data.n_observed)

            if result.is_duplicate:
                n_dup += 1

            postfix: dict = {
                "best": f"{self.data.best_score:.4f}",
                "curr": f"{result.score:.4f}",
                "dim": self._current_dim,
                "dup": n_dup,
            }
            tc = self.config.trust_region
            if tc.strategy == "adaptive":
                postfix["tr"] = f"{self.trust_region.radius:.3f}"
                postfix["rst"] = self.trust_region.get_state().n_restarts
            if self.ur_tr is not None:
                postfix["ur"] = f"{self.ur_tr.radius:.3f}"
                postfix["rot"] = self.ur_tr.n_rotations
            pbar.set_postfix(postfix)

            if (i + 1) % log_interval == 0 and self.verbose:
                ur_info = ""
                if self.ur_tr is not None:
                    ur_info = (
                        f" | UR: {self.ur_tr.radius:.3f} "
                        f"rot={self.ur_tr.n_rotations}"
                    )
                logger.info(
                    f"Iter {i+1}/{n_iterations} | "
                    f"Best: {self.data.best_score:.4f} | "
                    f"Curr: {result.score:.4f} | "
                    f"GP: {result.gp_mean:.2f}±{result.gp_std:.2f} | "
                    f"dim: {self._current_dim}D{ur_info}"
                )

        logger.info(f"Done. Best: {self.data.best_score:.4f}")
        logger.info(f"Best SMILES: {self.data.best_smiles}")
        if isinstance(self.surrogate, SphericalGPSurrogate):
            if self.surrogate.fallback_count > 0:
                logger.warning(
                    f"GP fallbacks: {self.surrogate.fallback_count} "
                    f"iterations used untrained fallback GP"
                )

    # ── Convenience properties ───────────────────────────────────────

    @property
    def best_score(self) -> float:
        return self.data.best_score

    @property
    def best_smiles(self) -> str:
        return self.data.best_smiles

    @property
    def gp(self):
        return self.surrogate.gp
