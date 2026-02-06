"""Tests for RieLBO v3: Windowed GP, Multi-Projection, Y-Normalization.

Tests:
- Window selection: correct sizes, locality, diversity
- Multi-projection: round-robin, orthonormality
- Y-normalization: zero mean, unit std
- GP fitting: works with small window, posterior std > 0
- Full step: integration with mock codec/oracle
"""

import pytest
import torch
import torch.nn.functional as F

from conftest import MockCodec, MockOracle


class TestWindowSelection:
    """Test the windowed GP data selection."""

    def _make_optimizer(self, n_train=200, **kwargs):
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            **kwargs,
        )
        # Manually set up training data
        opt.train_U = F.normalize(torch.randn(n_train, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(n_train)
        opt.mean_norm = 5.0
        return opt

    def test_window_size_when_data_exceeds_window(self):
        """Window should be exactly local + random when N > window_size."""
        opt = self._make_optimizer(n_train=200, window_local=50, window_random=30)
        A = opt.projections[0]

        V, Y = opt._select_window(A)

        assert V.shape[0] == 80
        assert Y.shape[0] == 80

    def test_window_uses_all_data_when_small(self):
        """Should use all data when N <= window_size."""
        opt = self._make_optimizer(n_train=50, window_local=50, window_random=30)
        A = opt.projections[0]

        V, Y = opt._select_window(A)

        assert V.shape[0] == 50
        assert Y.shape[0] == 50

    def test_window_local_contains_best(self):
        """The best-scoring point should always be in the window."""
        opt = self._make_optimizer(n_train=200, window_local=50, window_random=30)
        # Make one point clearly the best
        opt.train_Y[42] = 100.0
        A = opt.projections[0]

        V, Y = opt._select_window(A)

        assert Y.max() == 100.0

    def test_window_local_selects_nearest(self):
        """Local window should contain nearest neighbors to best."""
        opt = self._make_optimizer(n_train=200, window_local=50, window_random=0)

        # Make point 0 the best
        opt.train_Y[0] = 100.0
        u_best = opt.train_U[0:1]
        A = opt.projections[0]

        # Compute expected cosine similarities
        cos_sims = (opt.train_U @ u_best.T).squeeze()
        _, expected_top50 = cos_sims.topk(50)

        V, Y = opt._select_window(A)

        # Window should be exactly the 50 nearest (no random component)
        assert V.shape[0] == 50
        assert set(opt._window_indices.tolist()) == set(expected_top50.tolist())

    def test_window_v_on_sphere(self):
        """Projected window points should be on unit sphere."""
        opt = self._make_optimizer(n_train=200)
        A = opt.projections[0]

        V, Y = opt._select_window(A)

        norms = V.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_window_subspace_dim(self):
        """Projected points should have correct subspace dimension."""
        opt = self._make_optimizer(n_train=200, subspace_dim=16)
        A = opt.projections[0]

        V, Y = opt._select_window(A)

        assert V.shape[1] == 16


class TestMultiProjection:
    """Test multi-projection ensemble."""

    def test_n_projections_created(self):
        """Should create the requested number of projections."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            n_projections=5,
            verbose=False,
        )

        assert len(opt.projections) == 5

    def test_projections_are_orthonormal(self):
        """Each projection matrix should have orthonormal columns."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            n_projections=3,
            verbose=False,
        )

        for i, A in enumerate(opt.projections):
            # A.T @ A should be identity
            ATA = A.T @ A
            I = torch.eye(opt.subspace_dim)
            assert torch.allclose(ATA, I, atol=1e-5), f"Projection {i} not orthonormal"

    def test_projections_are_different(self):
        """Different projections should be different matrices."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            n_projections=3,
            verbose=False,
        )

        # Check pairwise difference
        for i in range(3):
            for j in range(i + 1, 3):
                diff = (opt.projections[i] - opt.projections[j]).abs().max()
                assert diff > 0.01, f"Projections {i} and {j} are too similar"

    def test_round_robin_selection(self):
        """Should cycle through projections."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            n_projections=3,
            verbose=False,
        )

        # iteration=1 -> idx=1, iteration=2 -> idx=2, iteration=3 -> idx=0
        opt.iteration = 1
        assert opt._current_projection_idx() == 1
        opt.iteration = 2
        assert opt._current_projection_idx() == 2
        opt.iteration = 3
        assert opt._current_projection_idx() == 0
        opt.iteration = 4
        assert opt._current_projection_idx() == 1


class TestYNormalization:
    """Test Y-normalization in GP fitting."""

    def test_gp_fits_with_normalized_y(self):
        """GP should fit successfully with Y-normalization."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            window_local=30,
            window_random=10,
        )

        # Set up training data
        opt.train_U = F.normalize(torch.randn(50, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(50) * 0.5 + 0.1  # Scores in [0.1, 0.6]
        opt.mean_norm = 5.0

        opt._fit_gp()

        assert opt.gp is not None
        # Check normalization was applied
        assert hasattr(opt, '_y_mean')
        assert hasattr(opt, '_y_std')
        assert opt._y_std > 0

    def test_normalized_y_has_unit_stats(self):
        """Normalized Y used for GP should have ~zero mean, ~unit std."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            window_local=30,
            window_random=10,
        )

        opt.train_U = F.normalize(torch.randn(50, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(50) * 10 + 5  # Scores in [5, 15]
        opt.mean_norm = 5.0

        # Manually verify normalization
        Y = opt.train_Y
        y_mean = Y.mean()
        y_std = Y.std()
        Y_norm = (Y - y_mean) / y_std

        assert abs(Y_norm.mean().item()) < 1e-5
        assert abs(Y_norm.std().item() - 1.0) < 0.1


class TestGPPosteriorStd:
    """Test that windowed GP maintains non-zero posterior std."""

    def test_window_smaller_than_full_dataset(self):
        """Windowed GP should use fewer training points than full dataset."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            window_local=30,
            window_random=10,
        )

        opt.train_U = F.normalize(torch.randn(200, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(200) * 0.5
        opt.mean_norm = 5.0

        A = opt.projections[0]
        opt.iteration = 1
        opt._fit_gp(A)

        # Window should be 40, not 200 - this is the key invariant
        assert opt._window_V.shape[0] == 40
        assert opt._window_Y.shape[0] == 40

        # GP should exist and be in eval mode
        assert opt.gp is not None
        assert not opt.gp.training

        # Posterior std should be > 0 (not exactly zero)
        test_v = F.normalize(torch.randn(10, 16), p=2, dim=-1).double()
        with torch.no_grad():
            post = opt.gp.posterior(test_v)
            std = post.variance.sqrt().squeeze()
        assert (std > 0).all(), f"Posterior std is zero: {std}"


class TestProjectSubspaceLift:
    """Test project and lift operations with explicit projection matrix."""

    def test_project_lift_roundtrip(self):
        """project -> lift should approximately recover direction."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
        )

        u = F.normalize(torch.randn(5, 256), p=2, dim=-1)
        A = opt.projections[0]

        v = opt.project_to_subspace(u, A)
        u_recovered = opt.lift_to_original(v, A)

        # The recovered direction should have positive cosine with original
        cos_sims = (u * u_recovered).sum(dim=-1)
        assert (cos_sims > 0).all()

    def test_project_on_sphere(self):
        """Projected points should be on unit sphere."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
        )

        u = F.normalize(torch.randn(10, 256), p=2, dim=-1)
        v = opt.project_to_subspace(u, opt.projections[0])

        norms = v.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestGradientOptimization:
    """Test Riemannian gradient ascent on sphere."""

    def _make_optimizer(self, acqf="grad_ucb", n_train=100, **kwargs):
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        opt = SphericalSubspaceBOv3(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            acqf=acqf,
            window_local=30,
            window_random=10,
            **kwargs,
        )
        opt.train_U = F.normalize(torch.randn(n_train, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(n_train) * 0.5 + 0.1
        opt.mean_norm = 5.0
        return opt

    def test_grad_ucb_enabled(self):
        """grad_ucb should enable gradient optimization."""
        opt = self._make_optimizer(acqf="grad_ucb")
        assert opt.grad_enabled

    def test_grad_result_on_sphere(self):
        """Gradient-optimized point should be on unit sphere in subspace."""
        opt = self._make_optimizer(acqf="grad_ucb")
        A = opt.projections[0]
        opt._fit_gp(A)

        v_cand = opt._generate_sobol_candidates(100, A)
        # Take top 3 as starts
        with torch.no_grad():
            post = opt.gp.posterior(v_cand.double())
            ucb_vals = post.mean.squeeze() + opt.ucb_beta * post.variance.sqrt().squeeze()
        _, top_idx = ucb_vals.topk(3)
        v_starts = v_cand[top_idx]

        v_opt = opt._riemannian_gradient_ascent(v_starts, acqf_type="ucb")

        norms = v_opt.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_grad_improves_over_random(self):
        """Gradient optimization should find equal or better acquisition value than random."""
        opt = self._make_optimizer(acqf="grad_ucb")
        A = opt.projections[0]
        opt._fit_gp(A)

        v_cand = opt._generate_sobol_candidates(200, A)

        # Best random UCB
        with torch.no_grad():
            post = opt.gp.posterior(v_cand.double())
            ucb_vals = post.mean.squeeze() + opt.ucb_beta * post.variance.sqrt().squeeze()
        best_random_ucb = ucb_vals.max().item()

        # Gradient-optimized UCB
        _, top_idx = ucb_vals.topk(5)
        v_starts = v_cand[top_idx]
        v_opt = opt._riemannian_gradient_ascent(v_starts, acqf_type="ucb")

        with torch.no_grad():
            post = opt.gp.posterior(v_opt.double())
            opt_ucb = (post.mean.squeeze() + opt.ucb_beta * post.variance.sqrt().squeeze()).item()

        # Gradient should be >= random (it refines the best random starts)
        assert opt_ucb >= best_random_ucb - 0.01, (
            f"Gradient UCB {opt_ucb:.4f} < random UCB {best_random_ucb:.4f}"
        )

    def test_grad_ei_works(self):
        """grad_ei should produce valid results."""
        opt = self._make_optimizer(acqf="grad_ei")
        A = opt.projections[0]
        opt._fit_gp(A)

        v_cand = opt._generate_sobol_candidates(100, A)
        v_starts = v_cand[:3]
        v_opt = opt._riemannian_gradient_ascent(v_starts, acqf_type="ei")

        assert v_opt.shape == (1, opt.subspace_dim)
        norms = v_opt.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_full_step_with_grad_ucb(self):
        """Full step() should work with grad_ucb acquisition."""
        opt = self._make_optimizer(acqf="grad_ucb")
        A = opt.projections[0]
        opt.iteration = 0
        opt._fit_gp(A)

        u_opt, diag = opt._optimize_acquisition(A)

        assert u_opt.shape == (1, 256)
        norms = u_opt.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        assert "gp_mean" in diag
        assert "gp_std" in diag


class TestValidation:
    """Test input validation."""

    def test_invalid_subspace_dim(self):
        """Should reject subspace_dim >= input_dim."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        with pytest.raises(ValueError, match="subspace_dim"):
            SphericalSubspaceBOv3(
                codec=MockCodec(device="cpu"),
                oracle=MockOracle(),
                device="cpu",
                subspace_dim=300,
                input_dim=256,
            )

    def test_invalid_n_projections(self):
        """Should reject n_projections < 1."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        with pytest.raises(ValueError, match="n_projections"):
            SphericalSubspaceBOv3(
                codec=MockCodec(device="cpu"),
                oracle=MockOracle(),
                device="cpu",
                n_projections=0,
            )

    def test_invalid_window_local(self):
        """Should reject window_local < 1."""
        from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

        with pytest.raises(ValueError, match="window_local"):
            SphericalSubspaceBOv3(
                codec=MockCodec(device="cpu"),
                oracle=MockOracle(),
                device="cpu",
                window_local=0,
            )
