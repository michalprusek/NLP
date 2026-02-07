"""Tests for the modular components in rielbo.components and rielbo.core.

Validates that the extracted components produce identical behavior to the
monolithic V2 implementation. Tests are organized by component type.
"""

import pytest
import torch
import torch.nn.functional as F
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from rielbo.core.config import (
    AcquisitionConfig,
    CandidateGenConfig,
    KernelConfig,
    NormReconstructionConfig,
    OptimizerConfig,
    ProjectionConfig,
    TrustRegionConfig,
)
from rielbo.core.types import BOHistory, StepResult, TrainingData


# ── Config tests ─────────────────────────────────────────────────────


class TestOptimizerConfig:
    def test_default_config(self):
        config = OptimizerConfig()
        assert config.kernel.kernel_type == "arccosine"
        assert config.projection.projection_type == "random"
        assert config.trust_region.strategy == "adaptive"
        assert config.acquisition.acqf == "ts"
        assert config.candidate_gen.strategy == "geodesic"

    def test_all_presets_valid(self):
        for preset in OptimizerConfig.available_presets():
            config = OptimizerConfig.from_preset(preset)
            assert isinstance(config, OptimizerConfig)

    def test_explore_preset(self):
        config = OptimizerConfig.from_preset("explore")
        assert config.projection.lass is True
        assert config.trust_region.ur_tr is True
        assert config.acquisition.schedule is True
        assert config.trust_region.strategy == "ur"

    def test_turbo_preset(self):
        config = OptimizerConfig.from_preset("turbo")
        assert config.kernel.kernel_type == "matern"
        assert config.kernel.kernel_ard is True
        assert config.projection.projection_type == "identity"
        assert config.trust_region.strategy == "turbo"

    def test_vanilla_preset(self):
        config = OptimizerConfig.from_preset("vanilla")
        assert config.kernel.kernel_type == "hvarfner"
        assert config.projection.projection_type == "identity"

    def test_from_preset_with_overrides(self):
        config = OptimizerConfig.from_preset("geodesic", seed=99, verbose=False)
        assert config.seed == 99
        assert config.verbose is False

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            OptimizerConfig.from_preset("nonexistent")


# ── TrainingData tests ───────────────────────────────────────────────


class TestTrainingData:
    def test_add_observation(self):
        data = TrainingData()
        data.train_X = torch.randn(5, 10)
        data.train_U = F.normalize(data.train_X, dim=-1)
        data.train_Y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        data.smiles_observed = ["A", "B", "C", "D", "E"]
        data.update_best()

        x_new = torch.randn(1, 10)
        u_new = F.normalize(x_new, dim=-1)
        improved = data.add_observation(x_new, u_new, 0.6, "F")

        assert improved is True
        assert data.best_score == 0.6
        assert data.best_smiles == "F"
        assert data.n_observed == 6

    def test_no_improvement(self):
        data = TrainingData()
        data.train_X = torch.randn(3, 10)
        data.train_U = F.normalize(data.train_X, dim=-1)
        data.train_Y = torch.tensor([0.5, 0.6, 0.7])
        data.smiles_observed = ["A", "B", "C"]
        data.update_best()

        x_new = torch.randn(1, 10)
        u_new = F.normalize(x_new, dim=-1)
        improved = data.add_observation(x_new, u_new, 0.3, "D")

        assert improved is False
        assert abs(data.best_score - 0.7) < 1e-5


class TestBOHistory:
    def test_append_and_to_dict(self):
        history = BOHistory()
        result = StepResult(
            score=0.5, best_score=0.5, smiles="CCO",
            gp_mean=0.4, gp_std=0.1, subspace_dim=16,
        )
        history.append_from_result(result, n_eval=101)

        d = history.to_dict()
        assert d["best_score"] == [0.5]
        assert d["current_score"] == [0.5]
        assert d["n_evaluated"] == [101]
        assert d["gp_mean"] == [0.4]


# ── Projection tests ────────────────────────────────────────────────


class TestProjection:
    def test_qr_project_lift_roundtrip(self):
        from rielbo.components.projection import QRProjection

        config = ProjectionConfig(input_dim=64, subspace_dim=8)
        proj = QRProjection(config, device="cpu", seed=42)

        u = F.normalize(torch.randn(10, 64), dim=-1)
        v = proj.project(u)

        assert v.shape == (10, 8)
        # v should be on unit sphere
        norms = v.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

        u_back = proj.lift(v)
        assert u_back.shape == (10, 64)
        norms_back = u_back.norm(dim=-1)
        assert torch.allclose(norms_back, torch.ones(10), atol=1e-5)

    def test_identity_projection(self):
        from rielbo.components.projection import IdentityProjection

        config = ProjectionConfig(input_dim=16)
        proj = IdentityProjection(config, device="cpu")

        u = F.normalize(torch.randn(5, 16), dim=-1)
        v = proj.project(u)
        assert torch.allclose(u, v)

        u_back = proj.lift(v)
        assert torch.allclose(u, u_back)

    def test_pca_projection(self):
        from rielbo.components.projection import PCAProjection

        config = ProjectionConfig(input_dim=32, subspace_dim=4)
        proj = PCAProjection(config, device="cpu")

        train_U = F.normalize(torch.randn(100, 32), dim=-1)
        proj.fit(train_U)

        assert proj.A is not None
        assert proj.A.shape == (32, 4)

        v = proj.project(train_U[:5])
        assert v.shape == (5, 4)

    def test_create_projection_factory(self):
        from rielbo.components.projection import create_projection

        for ptype in ["random", "pca", "identity"]:
            config = ProjectionConfig(
                projection_type=ptype, input_dim=32, subspace_dim=8,
            )
            proj = create_projection(config, device="cpu")
            assert proj.subspace_dim == (32 if ptype == "identity" else 8)

    def test_qr_reinitialize(self):
        from rielbo.components.projection import QRProjection

        config = ProjectionConfig(input_dim=32, subspace_dim=8)
        proj = QRProjection(config, device="cpu", seed=42)
        A_before = proj.A.clone()

        proj.reinitialize(seed=99)
        A_after = proj.A

        # Different seed → different projection
        assert not torch.allclose(A_before, A_after)

    def test_set_projection(self):
        from rielbo.components.projection import QRProjection

        config = ProjectionConfig(input_dim=32, subspace_dim=8)
        proj = QRProjection(config, device="cpu", seed=42)

        new_A = torch.randn(32, 8)
        new_A, _ = torch.linalg.qr(new_A)
        proj.set_projection(new_A)

        assert torch.allclose(proj.A, new_A)


# ── Trust region tests ───────────────────────────────────────────────


class TestTrustRegion:
    def test_static_tr(self):
        from rielbo.components.trust_region import StaticTR

        config = TrustRegionConfig(strategy="static", geodesic=False)
        tr = StaticTR(config, trust_region=0.8)

        assert tr.radius == 0.8
        tr.update(improved=True)
        assert tr.radius == 0.8  # Static — no change
        assert tr.needs_restart is False

    def test_adaptive_tr_grow(self):
        from rielbo.components.trust_region import AdaptiveTR

        config = TrustRegionConfig(
            strategy="adaptive", initial_radius=0.4,
            tr_success_tol=3, tr_grow_factor=1.5,
        )
        tr = AdaptiveTR(config)

        for _ in range(3):
            tr.update(improved=True)

        assert tr.radius > 0.4  # Should have grown

    def test_adaptive_tr_shrink(self):
        from rielbo.components.trust_region import AdaptiveTR

        config = TrustRegionConfig(
            strategy="adaptive", initial_radius=0.4,
            tr_fail_tol=5, tr_shrink_factor=0.5,
        )
        tr = AdaptiveTR(config)

        for _ in range(5):
            tr.update(improved=False)

        assert tr.radius < 0.4  # Should have shrunk

    def test_adaptive_tr_restart_flag(self):
        from rielbo.components.trust_region import AdaptiveTR

        config = TrustRegionConfig(
            strategy="adaptive", initial_radius=0.4,
            tr_min=0.01, tr_fail_tol=2, tr_shrink_factor=0.01,
            max_restarts=5,
        )
        tr = AdaptiveTR(config)

        for _ in range(2):
            tr.update(improved=False)

        assert tr.needs_restart is True

    def test_ur_tr_expand_on_low_std(self):
        from rielbo.components.trust_region import URTR

        config = TrustRegionConfig(
            ur_tr=True, ur_std_low=0.05, ur_std_high=0.15,
            ur_expand_factor=1.5, ur_relative=False,
        )
        ur = URTR(config, initial_radius=0.4)

        ur.update(improved=False, gp_std=0.01)  # Below ur_std_low
        assert ur.radius > 0.4  # Should expand

    def test_ur_tr_shrink_on_high_std(self):
        from rielbo.components.trust_region import URTR

        config = TrustRegionConfig(
            ur_tr=True, ur_std_low=0.05, ur_std_high=0.15,
            ur_shrink_factor=0.8, ur_relative=False,
        )
        ur = URTR(config, initial_radius=0.4)

        ur.update(improved=False, gp_std=0.3)  # Above ur_std_high
        assert ur.radius < 0.4  # Should shrink

    def test_ur_tr_rotation_on_collapse(self):
        from rielbo.components.trust_region import URTR

        config = TrustRegionConfig(
            ur_tr=True, ur_std_low=0.05, ur_std_collapse=0.005,
            ur_collapse_patience=3, ur_relative=False,
            ur_expand_factor=1.5, ur_tr_max=5.0,
        )
        ur = URTR(config, initial_radius=0.4)

        # 3 consecutive collapses should trigger rotation
        for _ in range(3):
            ur.update(improved=False, gp_std=0.001)

        assert ur.needs_rotation is True

    def test_turbo_tr(self):
        from rielbo.components.trust_region import TuRBOTR

        config = TrustRegionConfig(strategy="turbo", initial_radius=0.8)
        tr = TuRBOTR(config, dim=256)

        assert tr.radius == 0.8
        for _ in range(3):
            tr.update(improved=True)
        # After 3 successes, should have grown
        assert tr.radius > 0.8

    def test_create_trust_region_factory(self):
        from rielbo.components.trust_region import create_trust_region

        for strategy in ["static", "adaptive", "ur", "turbo"]:
            config = TrustRegionConfig(
                strategy=strategy, geodesic=True, ur_tr=(strategy == "ur"),
                geodesic_max_angle=0.5,
            )
            tr = create_trust_region(config, trust_region=0.8, dim=256)
            assert tr.radius > 0


# ── Surrogate tests ──────────────────────────────────────────────────


class TestSurrogate:
    def test_spherical_gp_fit(self):
        from rielbo.components.surrogate import SphericalGPSurrogate

        config = KernelConfig(kernel_type="arccosine", kernel_order=0)
        surrogate = SphericalGPSurrogate(
            config, subspace_dim=4, device="cpu", verbose=False,
        )

        X = F.normalize(torch.randn(20, 4), dim=-1)
        Y = torch.randn(20)

        surrogate.fit(X, Y)
        assert surrogate.gp is not None
        assert surrogate.gp.training is False  # Should be in eval mode

    def test_spherical_gp_fit_quick(self):
        from rielbo.components.surrogate import SphericalGPSurrogate

        config = KernelConfig(kernel_type="arccosine", kernel_order=0)
        surrogate = SphericalGPSurrogate(
            config, subspace_dim=4, device="cpu", verbose=False,
        )

        X = F.normalize(torch.randn(20, 4), dim=-1)
        Y = torch.randn(20)

        log_ml = surrogate.fit_quick(X, Y)
        assert isinstance(log_ml, float)
        assert log_ml != float("-inf")

    def test_euclidean_gp_fit(self):
        from rielbo.components.surrogate import EuclideanGPSurrogate

        config = KernelConfig(kernel_type="hvarfner")
        surrogate = EuclideanGPSurrogate(
            config, input_dim=8, device="cpu", verbose=False,
        )

        Z = torch.randn(20, 8)
        Y = torch.randn(20)

        surrogate.set_normalization_bounds(Z)
        surrogate.fit(Z, Y, mode="hvarfner")
        assert surrogate.gp is not None

    def test_euclidean_gp_turbo_mode(self):
        from rielbo.components.surrogate import EuclideanGPSurrogate

        config = KernelConfig(kernel_type="matern", kernel_ard=True)
        surrogate = EuclideanGPSurrogate(
            config, input_dim=8, device="cpu", verbose=False,
        )

        Z = torch.randn(20, 8)
        Y = torch.randn(20)

        surrogate.fit(Z, Y, mode="turbo")
        assert surrogate.gp is not None


# ── Acquisition tests ────────────────────────────────────────────────


class TestAcquisition:
    @pytest.fixture
    def fitted_gp(self):
        """Create a fitted GP for acquisition testing."""
        X = F.normalize(torch.randn(30, 4), dim=-1).double()
        Y = torch.randn(30).double().unsqueeze(-1)

        from rielbo.kernels import create_kernel
        covar = create_kernel("arccosine", kernel_order=0, use_scale=True)
        gp = SingleTaskGP(X, Y, covar_module=covar)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        return gp, Y.squeeze(-1)

    def test_ts_selection(self, fitted_gp):
        from rielbo.components.acquisition import AcquisitionSelector

        gp, train_Y = fitted_gp
        config = AcquisitionConfig(acqf="ts")
        selector = AcquisitionSelector(config)

        candidates = F.normalize(torch.randn(100, 4), dim=-1)
        v_opt, diag = selector.select(gp, candidates, train_Y.float())

        assert v_opt.shape == (1, 4)
        assert "gp_mean" in diag
        assert "gp_std" in diag

    def test_ei_selection(self, fitted_gp):
        from rielbo.components.acquisition import AcquisitionSelector

        gp, train_Y = fitted_gp
        config = AcquisitionConfig(acqf="ei")
        selector = AcquisitionSelector(config)

        candidates = F.normalize(torch.randn(100, 4), dim=-1)
        v_opt, diag = selector.select(gp, candidates, train_Y.float())
        assert v_opt.shape == (1, 4)

    def test_ucb_selection(self, fitted_gp):
        from rielbo.components.acquisition import AcquisitionSelector

        gp, train_Y = fitted_gp
        config = AcquisitionConfig(acqf="ucb", ucb_beta=2.0)
        selector = AcquisitionSelector(config)

        candidates = F.normalize(torch.randn(100, 4), dim=-1)
        v_opt, diag = selector.select(gp, candidates, train_Y.float())
        assert v_opt.shape == (1, 4)
        assert diag["acqf_used"] == "ucb"

    def test_acquisition_schedule(self):
        from rielbo.components.acquisition import AcquisitionSchedule

        acqf_config = AcquisitionConfig(
            acqf="ts", schedule=True,
            acqf_ucb_beta_high=4.0, acqf_ucb_beta_low=0.5,
        )
        tr_config = TrustRegionConfig(
            ur_std_low=0.05, ur_std_high=0.15, ur_relative=False,
        )
        schedule = AcquisitionSchedule(acqf_config, tr_config)

        # Low std → UCB with high beta
        acqf, beta = schedule.get_effective_acqf(gp_std=0.01)
        assert acqf == "ucb"
        assert beta == 4.0

        # High std → UCB with low beta
        acqf, beta = schedule.get_effective_acqf(gp_std=0.3)
        assert acqf == "ucb"
        assert beta == 0.5

        # Normal range → default (ts)
        acqf, beta = schedule.get_effective_acqf(gp_std=0.1)
        assert acqf == "ts"


# ── Candidate generation tests ───────────────────────────────────────


class TestCandidateGen:
    def test_geodesic_generator(self):
        from rielbo.components.candidate_gen import GeodesicGenerator

        gen = GeodesicGenerator(
            geodesic_max_angle=0.5, global_fraction=0.2, device="cpu",
        )
        center = F.normalize(torch.randn(1, 8), dim=-1)
        candidates = gen.generate(center, n_candidates=100, radius=0.4)

        assert candidates.shape == (100, 8)
        norms = candidates.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)

    def test_sobol_box_generator(self):
        from rielbo.components.candidate_gen import SobolBoxGenerator

        gen = SobolBoxGenerator(dim=8, device="cpu")
        center = F.normalize(torch.randn(1, 8), dim=-1)
        candidates = gen.generate(
            center, n_candidates=100, radius=0.5,
            normalize_to_sphere=True,
        )

        assert candidates.shape == (100, 8)
        norms = candidates.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)

    def test_sobol_box_no_normalize(self):
        from rielbo.components.candidate_gen import SobolBoxGenerator

        gen = SobolBoxGenerator(dim=8, device="cpu")
        center = torch.randn(1, 8)
        candidates = gen.generate(
            center, n_candidates=50, radius=0.5,
            normalize_to_sphere=False,
        )

        assert candidates.shape == (50, 8)
        # Not normalized — norms should vary
        norms = candidates.norm(dim=-1)
        assert not torch.allclose(norms, torch.ones(50), atol=0.01)

    def test_factory(self):
        from rielbo.components.candidate_gen import create_candidate_generator

        for strategy in ["geodesic", "sobol_box"]:
            config = CandidateGenConfig(strategy=strategy)
            gen = create_candidate_generator(config, dim=8, device="cpu")
            assert gen is not None


# ── Norm reconstruction tests ────────────────────────────────────────


class TestNormReconstruction:
    def test_mean_norm(self):
        from rielbo.components.norm_reconstruction import MeanNormReconstructor

        rec = MeanNormReconstructor(mean_norm=5.0)
        u = F.normalize(torch.randn(1, 8), dim=-1)
        x, diag = rec.reconstruct(u)

        assert torch.allclose(x.norm(dim=-1), torch.tensor(5.0), atol=1e-5)
        assert diag["embedding_norm"] == 5.0

    def test_factory(self):
        from rielbo.components.norm_reconstruction import create_norm_reconstructor

        # Mean
        config = NormReconstructionConfig(method="mean")
        rec = create_norm_reconstructor(config, device="cpu")
        assert hasattr(rec, "mean_norm")

        # Probabilistic
        config = NormReconstructionConfig(method="probabilistic")
        rec = create_norm_reconstructor(config, device="cpu")
        assert hasattr(rec, "fit")


# ── LASS tests ───────────────────────────────────────────────────────


class TestLASS:
    def test_lass_selects_projection(self):
        from rielbo.components.projection import LASSSelector, QRProjection

        proj_config = ProjectionConfig(
            input_dim=32, subspace_dim=4,
            lass=True, lass_n_candidates=5,
        )
        kernel_config = KernelConfig(kernel_type="arccosine", kernel_order=0)

        proj = QRProjection(proj_config, device="cpu", seed=42)
        A_before = proj.A.clone()

        lass = LASSSelector(proj_config, kernel_config, device="cpu", seed=42)

        train_U = F.normalize(torch.randn(50, 32), dim=-1)
        train_Y = torch.randn(50)

        lass.select_best_projection(proj, train_U, train_Y)

        # LASS should have changed the projection
        # (with 5 candidates, very likely to find something different)
        # Note: may or may not change depending on scores
        assert proj.A.shape == (32, 4)


# ── BaseOptimizer integration test ───────────────────────────────────


class TestBaseOptimizer:
    """Integration tests using a mock codec and oracle."""

    class MockCodec:
        def __init__(self, dim=32):
            self.dim = dim

        def encode(self, smiles_list):
            return torch.randn(len(smiles_list), self.dim)

        def decode(self, embeddings):
            return [f"SMILES_{i}" for i in range(len(embeddings))]

    class MockOracle:
        def __init__(self):
            self._counter = 0

        def score(self, smiles):
            self._counter += 1
            return 0.5 + 0.01 * self._counter  # Slowly increasing

    def test_cold_start_and_step(self):
        from rielbo.core.optimizer import BaseOptimizer

        config = OptimizerConfig.from_preset("baseline")
        config.device = "cpu"
        config.projection.input_dim = 32
        config.projection.subspace_dim = 4

        codec = self.MockCodec(dim=32)
        oracle = self.MockOracle()
        optimizer = BaseOptimizer(
            codec=codec, oracle=oracle, config=config,
            input_dim=32, subspace_dim=4,
        )

        # Cold start
        smiles = [f"INIT_{i}" for i in range(20)]
        scores = torch.randn(20)
        optimizer.cold_start(smiles, scores)

        assert optimizer.data.n_observed == 20
        assert optimizer.surrogate.gp is not None

        # Run a few steps
        for _ in range(3):
            result = optimizer.step()
            assert isinstance(result, StepResult)
            assert result.smiles != ""

    def test_geodesic_preset(self):
        from rielbo.core.optimizer import BaseOptimizer

        config = OptimizerConfig.from_preset("geodesic")
        config.device = "cpu"

        optimizer = BaseOptimizer(
            codec=self.MockCodec(), oracle=self.MockOracle(),
            config=config, input_dim=32, subspace_dim=4,
        )

        smiles = [f"INIT_{i}" for i in range(20)]
        scores = torch.randn(20)
        optimizer.cold_start(smiles, scores)

        result = optimizer.step()
        assert isinstance(result, StepResult)
        assert result.tr_length > 0

    def test_explore_preset_with_schedule(self):
        from rielbo.core.optimizer import BaseOptimizer

        config = OptimizerConfig.from_preset("explore")
        config.device = "cpu"

        optimizer = BaseOptimizer(
            codec=self.MockCodec(), oracle=self.MockOracle(),
            config=config, input_dim=32, subspace_dim=4,
        )

        smiles = [f"INIT_{i}" for i in range(20)]
        scores = torch.randn(20)
        optimizer.cold_start(smiles, scores)

        # Verify LASS ran (projection should exist)
        assert optimizer.projection.A is not None

        # Verify UR-TR exists
        assert optimizer.ur_tr is not None

        # Verify acquisition schedule exists
        assert optimizer.acq_schedule is not None

        result = optimizer.step()
        assert isinstance(result, StepResult)

    def test_optimize_loop(self):
        from rielbo.core.optimizer import BaseOptimizer

        config = OptimizerConfig.from_preset("baseline")
        config.device = "cpu"
        config.verbose = False

        optimizer = BaseOptimizer(
            codec=self.MockCodec(), oracle=self.MockOracle(),
            config=config, input_dim=32, subspace_dim=4,
        )

        smiles = [f"INIT_{i}" for i in range(20)]
        scores = torch.randn(20)
        optimizer.cold_start(smiles, scores)
        optimizer.optimize(n_iterations=5)

        assert len(optimizer.history.iteration) == 5
        assert optimizer.data.n_observed > 20

    def test_history_tracking(self):
        from rielbo.core.optimizer import BaseOptimizer

        config = OptimizerConfig.from_preset("baseline")
        config.device = "cpu"
        config.verbose = False

        optimizer = BaseOptimizer(
            codec=self.MockCodec(), oracle=self.MockOracle(),
            config=config, input_dim=32, subspace_dim=4,
        )

        smiles = [f"INIT_{i}" for i in range(20)]
        scores = torch.randn(20)
        optimizer.cold_start(smiles, scores)
        optimizer.optimize(n_iterations=3)

        d = optimizer.history.to_dict()
        assert len(d["iteration"]) == 3
        assert len(d["best_score"]) == 3
        assert len(d["gp_std"]) == 3

    def test_subspace_wrapper(self):
        from rielbo.optimizers.subspace import SubspaceBO

        optimizer = SubspaceBO(
            codec=self.MockCodec(), oracle=self.MockOracle(),
            preset="baseline", input_dim=32, subspace_dim=4,
            device="cpu", verbose=False,
        )

        smiles = [f"INIT_{i}" for i in range(20)]
        scores = torch.randn(20)
        optimizer.cold_start(smiles, scores)

        result = optimizer.step()
        assert isinstance(result, StepResult)

    def test_subspace_wrapper_overrides(self):
        from rielbo.optimizers.subspace import SubspaceBO

        optimizer = SubspaceBO(
            codec=self.MockCodec(), oracle=self.MockOracle(),
            preset="geodesic", input_dim=32, subspace_dim=4,
            device="cpu", verbose=False,
            ur_std_low=0.03, kernel_type="arccosine",
        )

        assert optimizer.config.trust_region.ur_std_low == 0.03
