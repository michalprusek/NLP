"""Tests for RieLBO v4: Geodesic Novelty Bonus.

Tests:
- Geodesic novelty computation: correctness, bounds, edge cases
- Novelty integration with acquisition: TS, EI, UCB
- Novelty reduces duplicate selection
- Backward compatibility: novelty_weight=0 behaves like v3
- Full step integration with mock codec/oracle
"""

import math

import pytest
import torch
import torch.nn.functional as F

from rielbo.subspace_bo_v4 import SphericalSubspaceBOv4, geodesic_novelty


class MockCodec:
    """Mock codec for testing."""

    def __init__(self, dim=256, device="cpu"):
        self.dim = dim
        self.device = device

    def encode(self, smiles_list):
        n = len(smiles_list)
        return torch.randn(n, self.dim, device=self.device)

    def decode(self, embeddings):
        n = embeddings.shape[0]
        return [f"SMILES_{i}_{torch.randint(0, 10000, (1,)).item()}" for i in range(n)]


class MockOracle:
    """Mock oracle that returns random scores."""

    def score(self, smiles):
        return torch.rand(1).item()


# ─────────────────────────────────────────────────
# Geodesic novelty function tests
# ─────────────────────────────────────────────────

class TestGeodesicNovelty:
    """Test the geodesic_novelty standalone function."""

    def test_identical_point_has_zero_novelty(self):
        """A candidate at an observed point should have ~0 novelty."""
        v_obs = F.normalize(torch.randn(10, 16), p=2, dim=-1)
        v_cand = v_obs[0:1]  # Exact copy of first observed point

        novelty = geodesic_novelty(v_cand, v_obs)

        assert novelty.shape == (1,)
        assert novelty[0].item() < 1e-3

    def test_antipodal_point_has_max_novelty(self):
        """A point antipodal to all observed should have novelty ~1."""
        # Single observed point
        v_obs = torch.zeros(1, 16)
        v_obs[0, 0] = 1.0  # North pole

        v_cand = torch.zeros(1, 16)
        v_cand[0, 0] = -1.0  # South pole

        novelty = geodesic_novelty(v_cand, v_obs)

        assert abs(novelty[0].item() - 1.0) < 0.01

    def test_novelty_in_unit_range(self):
        """Novelty should always be in [0, 1]."""
        v_obs = F.normalize(torch.randn(50, 16), p=2, dim=-1)
        v_cand = F.normalize(torch.randn(100, 16), p=2, dim=-1)

        novelty = geodesic_novelty(v_cand, v_obs)

        assert (novelty >= -1e-6).all()
        assert (novelty <= 1.0 + 1e-6).all()

    def test_novelty_shape(self):
        """Output shape should be [M] for M candidates."""
        v_obs = F.normalize(torch.randn(20, 8), p=2, dim=-1)
        v_cand = F.normalize(torch.randn(50, 8), p=2, dim=-1)

        novelty = geodesic_novelty(v_cand, v_obs)

        assert novelty.shape == (50,)

    def test_novelty_decreases_near_observed(self):
        """Points closer to observed should have lower novelty."""
        v_obs = F.normalize(torch.randn(1, 16), p=2, dim=-1)

        # Create points at increasing geodesic distances
        v_near = F.normalize(v_obs + 0.01 * torch.randn(1, 16), p=2, dim=-1)
        v_mid = F.normalize(v_obs + 0.5 * torch.randn(1, 16), p=2, dim=-1)
        v_far = F.normalize(-v_obs + 0.1 * torch.randn(1, 16), p=2, dim=-1)

        n_near = geodesic_novelty(v_near, v_obs)[0].item()
        n_mid = geodesic_novelty(v_mid, v_obs)[0].item()
        n_far = geodesic_novelty(v_far, v_obs)[0].item()

        assert n_near < n_mid < n_far

    def test_novelty_is_min_over_observed(self):
        """Novelty should be the MIN distance, not mean."""
        d = 16
        # Two observed points far apart
        v1 = torch.zeros(1, d)
        v1[0, 0] = 1.0
        v2 = torch.zeros(1, d)
        v2[0, 1] = 1.0
        v_obs = torch.cat([v1, v2], dim=0)

        # Candidate near v1
        v_cand = F.normalize(v1 + 0.01 * torch.randn(1, d), p=2, dim=-1)

        novelty = geodesic_novelty(v_cand, v_obs)

        # Should be close to 0 (near v1), not averaging with distance to v2
        assert novelty[0].item() < 0.1

    def test_batch_computation(self):
        """Should handle batches correctly."""
        v_obs = F.normalize(torch.randn(100, 16), p=2, dim=-1)
        v_cand = F.normalize(torch.randn(2000, 16), p=2, dim=-1)

        novelty = geodesic_novelty(v_cand, v_obs)

        assert novelty.shape == (2000,)
        assert (novelty >= 0).all()


# ─────────────────────────────────────────────────
# SphericalSubspaceBOv4 class tests
# ─────────────────────────────────────────────────

class TestNoveltyIntegration:
    """Test novelty integration with the BO optimizer."""

    def _make_optimizer(self, n_train=200, novelty_weight=0.1, acqf="ts", **kwargs):
        opt = SphericalSubspaceBOv4(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            novelty_weight=novelty_weight,
            acqf=acqf,
            **kwargs,
        )
        opt.train_U = F.normalize(torch.randn(n_train, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(n_train)
        opt.train_X = opt.train_U * 5.0
        opt.mean_norm = 5.0
        opt.smiles_observed = [f"MOL_{i}" for i in range(n_train)]
        return opt

    def test_all_v_computed_on_fit(self):
        """_all_V should be set after _fit_gp and contain ALL points."""
        opt = self._make_optimizer(n_train=200)
        A = opt.projections[0]
        opt._fit_gp(A)

        assert opt._all_V is not None
        assert opt._all_V.shape == (200, opt.subspace_dim)

    def test_all_v_on_sphere(self):
        """All projected points should be on unit sphere."""
        opt = self._make_optimizer(n_train=200)
        A = opt.projections[0]
        opt._fit_gp(A)

        norms = opt._all_V.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_novelty_bonus_computed(self):
        """_compute_novelty_bonus should return valid novelty scores."""
        opt = self._make_optimizer(n_train=200)
        A = opt.projections[0]
        opt._fit_gp(A)

        v_cand = opt._generate_sobol_candidates(100, A)
        novelty = opt._compute_novelty_bonus(v_cand)

        assert novelty.shape == (100,)
        assert (novelty >= 0).all()
        assert (novelty <= 1).all()

    def test_novelty_weight_zero_no_effect(self):
        """With novelty_weight=0, should behave like v3."""
        opt = self._make_optimizer(n_train=100, novelty_weight=0.0)
        A = opt.projections[0]
        opt._fit_gp(A)

        # Should still compute novelty but it has zero weight
        u_opt, diag = opt._optimize_acquisition(A)

        assert u_opt.shape == (1, 256)
        assert "novelty_mean" in diag

    def test_acquisition_returns_diagnostics(self):
        """Acquisition should return novelty diagnostics."""
        opt = self._make_optimizer(n_train=100)
        A = opt.projections[0]
        opt._fit_gp(A)

        u_opt, diag = opt._optimize_acquisition(A)

        assert "novelty_mean" in diag
        assert "novelty_selected" in diag
        assert "gp_mean" in diag
        assert "gp_std" in diag

    def test_ts_with_novelty(self):
        """Thompson Sampling + novelty should work."""
        opt = self._make_optimizer(acqf="ts", novelty_weight=0.1)
        A = opt.projections[0]
        opt._fit_gp(A)

        u_opt, diag = opt._optimize_acquisition(A)

        norms = u_opt.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_ei_with_novelty(self):
        """EI + novelty should work."""
        opt = self._make_optimizer(acqf="ei", novelty_weight=0.1)
        A = opt.projections[0]
        opt._fit_gp(A)

        u_opt, diag = opt._optimize_acquisition(A)

        norms = u_opt.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_ucb_with_novelty(self):
        """UCB + novelty should work."""
        opt = self._make_optimizer(acqf="ucb", novelty_weight=0.1)
        A = opt.projections[0]
        opt._fit_gp(A)

        u_opt, diag = opt._optimize_acquisition(A)

        norms = u_opt.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestNoveltyReducesDuplicates:
    """Test that novelty bonus steers away from observed points."""

    def test_novelty_changes_candidate_ranking(self):
        """Adding novelty bonus should change which candidate is selected.

        Test directly: for a set of candidates, the one nearest to observed
        data should have lower combined score when novelty_weight > 0.
        """
        torch.manual_seed(42)
        v_obs = F.normalize(torch.randn(50, 16), p=2, dim=-1)

        # Create two candidates: one near observed, one far
        v_near = F.normalize(v_obs[0:1] + 0.01 * torch.randn(1, 16), p=2, dim=-1)
        v_far = F.normalize(-v_obs[0:1] + 0.3 * torch.randn(1, 16), p=2, dim=-1)

        n_near = geodesic_novelty(v_near, v_obs)[0].item()
        n_far = geodesic_novelty(v_far, v_obs)[0].item()

        # Far point should have much more novelty
        assert n_far > n_near + 0.1

    def test_novelty_bonus_adds_to_acquisition(self):
        """Novelty should be additive with GP acquisition value."""
        v_obs = F.normalize(torch.randn(20, 8), p=2, dim=-1)
        v_cand = F.normalize(torch.randn(100, 8), p=2, dim=-1)

        novelty = geodesic_novelty(v_cand, v_obs)

        # Simulate: GP value is constant, novelty breaks the tie
        gp_vals = torch.ones(100) * 0.5
        combined_0 = gp_vals + 0.0 * novelty  # No novelty
        combined_1 = gp_vals + 1.0 * novelty  # Strong novelty

        # Without novelty, all candidates are equally good
        # With novelty, the most novel candidate wins
        best_idx_0 = combined_0.argmax().item()
        best_idx_1 = combined_1.argmax().item()

        # The novelty-selected point should have highest novelty
        assert novelty[best_idx_1] == novelty.max()


class TestV4InheritedFromV3:
    """Test that v3 functionality is preserved."""

    def _make_optimizer(self, **kwargs):
        return SphericalSubspaceBOv4(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            **kwargs,
        )

    def test_window_selection(self):
        """Window should work same as v3."""
        opt = self._make_optimizer(window_local=50, window_random=30)
        opt.train_U = F.normalize(torch.randn(200, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(200)
        opt.mean_norm = 5.0

        A = opt.projections[0]
        V, Y = opt._select_window(A)

        assert V.shape[0] == 80
        assert Y.shape[0] == 80

    def test_multi_projection_round_robin(self):
        """Round-robin should cycle through projections."""
        opt = self._make_optimizer(n_projections=3)

        opt.iteration = 1
        assert opt._current_projection_idx() == 1
        opt.iteration = 3
        assert opt._current_projection_idx() == 0

    def test_projections_orthonormal(self):
        """Projection matrices should be orthonormal."""
        opt = self._make_optimizer(n_projections=3)

        for A in opt.projections:
            ATA = A.T @ A
            I = torch.eye(opt.subspace_dim)
            assert torch.allclose(ATA, I, atol=1e-5)

    def test_project_lift_roundtrip(self):
        """Project -> lift should approximately recover direction."""
        opt = self._make_optimizer()
        u = F.normalize(torch.randn(5, 256), p=2, dim=-1)
        A = opt.projections[0]

        v = opt.project_to_subspace(u, A)
        u_rec = opt.lift_to_original(v, A)

        cos_sims = (u * u_rec).sum(dim=-1)
        assert (cos_sims > 0).all()


class TestV4Validation:
    """Test input validation."""

    def test_negative_novelty_weight(self):
        """Should reject negative novelty_weight."""
        with pytest.raises(ValueError, match="novelty_weight"):
            SphericalSubspaceBOv4(
                codec=MockCodec(device="cpu"),
                oracle=MockOracle(),
                device="cpu",
                novelty_weight=-0.1,
            )

    def test_zero_novelty_weight_ok(self):
        """Zero novelty_weight should be accepted (disables novelty)."""
        opt = SphericalSubspaceBOv4(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            novelty_weight=0.0,
            verbose=False,
        )
        assert opt.novelty_weight == 0.0


class TestFullStep:
    """Integration test for full step cycle."""

    def test_step_returns_expected_keys(self):
        """Step should return all expected diagnostic keys."""
        opt = SphericalSubspaceBOv4(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            novelty_weight=0.1,
        )
        opt.train_U = F.normalize(torch.randn(50, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(50)
        opt.train_X = opt.train_U * 5.0
        opt.mean_norm = 5.0
        opt.smiles_observed = [f"MOL_{i}" for i in range(50)]

        result = opt.step()

        assert "score" in result or "best_score" in result
        assert "novelty_mean" in result or "is_duplicate" in result

    def test_history_tracks_novelty(self):
        """History should include novelty metrics."""
        opt = SphericalSubspaceBOv4(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            novelty_weight=0.1,
        )
        opt.train_U = F.normalize(torch.randn(50, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(50)
        opt.train_X = opt.train_U * 5.0
        opt.mean_norm = 5.0
        opt.smiles_observed = [f"MOL_{i}" for i in range(50)]

        opt.optimize(n_iterations=3, log_interval=10)

        assert "novelty_mean" in opt.history
        assert "novelty_selected" in opt.history
        assert len(opt.history["novelty_mean"]) == 3


# ─────────────────────────────────────────────────
# Proposal E: Ensemble Thompson Sampling tests
# ─────────────────────────────────────────────────

class TestEnsembleTS:
    """Test Ensemble Thompson Sampling (Proposal E)."""

    def _make_ets_optimizer(self, n_train=100, n_projections=3, **kwargs):
        opt = SphericalSubspaceBOv4(
            codec=MockCodec(device="cpu"),
            oracle=MockOracle(),
            device="cpu",
            verbose=False,
            acqf="ets",
            n_projections=n_projections,
            **kwargs,
        )
        opt.train_U = F.normalize(torch.randn(n_train, 256), p=2, dim=-1)
        opt.train_Y = torch.rand(n_train)
        opt.train_X = opt.train_U * 5.0
        opt.mean_norm = 5.0
        opt.smiles_observed = [f"MOL_{i}" for i in range(n_train)]
        return opt

    def test_ensemble_returns_valid_point(self):
        """Ensemble should return a point on S^(D-1)."""
        opt = self._make_ets_optimizer()
        u_opt, diag = opt._optimize_ensemble_acquisition()

        assert u_opt.shape == (1, 256)
        norms = u_opt.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_ensemble_returns_diagnostics(self):
        """Ensemble should return novelty and GP diagnostics."""
        opt = self._make_ets_optimizer()
        u_opt, diag = opt._optimize_ensemble_acquisition()

        assert "novelty_mean" in diag
        assert "novelty_selected" in diag
        assert "gp_mean" in diag
        assert "gp_std" in diag

    def test_ensemble_uses_all_projections(self):
        """Ensemble should use all K projections (not round-robin)."""
        opt = self._make_ets_optimizer(n_projections=5)

        # The step should use all projections
        result = opt.step()

        # projection_idx=-1 indicates ensemble mode
        assert result.get("projection_idx") == -1

    def test_ensemble_step_full_cycle(self):
        """Full step cycle with ensemble TS."""
        opt = self._make_ets_optimizer()
        result = opt.step()

        assert "score" in result or "best_score" in result
        assert "novelty_mean" in result

    def test_ensemble_with_novelty(self):
        """Ensemble + novelty should work together."""
        opt = self._make_ets_optimizer(novelty_weight=0.2)
        u_opt, diag = opt._optimize_ensemble_acquisition()

        assert diag["novelty_selected"] >= 0
        assert diag["novelty_mean"] >= 0

    def test_fit_gp_for_projection(self):
        """_fit_gp_for_projection should return valid GP and data."""
        opt = self._make_ets_optimizer()
        A = opt.projections[0]

        gp, V_win, Y_win, y_mean, y_std, V_all = opt._fit_gp_for_projection(A)

        assert gp is not None
        assert V_win.shape[1] == opt.subspace_dim
        assert V_all.shape == (100, opt.subspace_dim)
        assert y_std > 0

    def test_ensemble_multiple_steps(self):
        """Ensemble should work across multiple steps."""
        opt = self._make_ets_optimizer()
        opt.optimize(n_iterations=3, log_interval=10)

        assert len(opt.history["novelty_mean"]) == 3
        assert opt.iteration == 3
