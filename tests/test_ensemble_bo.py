"""Tests for Spherical Ensemble BO.

Tests:
- EnsembleConfig presets and multi-scale dims
- SubspaceMember projection/lift roundtrip
- SubspaceMember independent projections
- SubspaceMember adaptive TR logic
- SubspaceMember retirement
- SphericalEnsembleBO multi-scale member creation
- SphericalEnsembleBO candidate selection (max-std)
- SphericalEnsembleBO retirement scheduling
"""

import pytest
import torch
import torch.nn.functional as F


class TestEnsembleConfig:
    """Test EnsembleConfig creation and presets."""

    def test_default_config(self):
        from rielbo.ensemble_bo import EnsembleConfig

        config = EnsembleConfig()
        assert config.member_dims == [4, 8, 12, 16, 20, 24]
        assert config.n_subspaces == 6
        assert config.retirement_interval == 100
        assert config.geodesic_tr is True
        assert config.adaptive_tr is True

    def test_custom_dims(self):
        from rielbo.ensemble_bo import EnsembleConfig

        config = EnsembleConfig(member_dims=[4, 16, 32])
        assert config.member_dims == [4, 16, 32]
        assert config.n_subspaces == 3

    def test_preset_default(self):
        from rielbo.ensemble_bo import EnsembleConfig

        config = EnsembleConfig.from_preset("default")
        assert config.member_dims == [4, 8, 12, 16, 20, 24]
        assert config.n_subspaces == 6

    def test_preset_small(self):
        from rielbo.ensemble_bo import EnsembleConfig

        config = EnsembleConfig.from_preset("small")
        assert config.member_dims == [4, 8, 12]
        assert config.n_subspaces == 3

    def test_preset_medium(self):
        from rielbo.ensemble_bo import EnsembleConfig

        config = EnsembleConfig.from_preset("medium")
        assert config.member_dims == [8, 12, 16, 20]
        assert config.n_subspaces == 4

    def test_preset_aggressive(self):
        from rielbo.ensemble_bo import EnsembleConfig

        config = EnsembleConfig.from_preset("aggressive")
        assert config.retirement_interval == 75

    def test_preset_conservative(self):
        from rielbo.ensemble_bo import EnsembleConfig

        config = EnsembleConfig.from_preset("conservative")
        assert config.retirement_interval == 150

    def test_unknown_preset_raises(self):
        from rielbo.ensemble_bo import EnsembleConfig

        with pytest.raises(ValueError):
            EnsembleConfig.from_preset("nonexistent")

    def test_n_subspaces_is_property(self):
        """n_subspaces should be derived from member_dims length."""
        from rielbo.ensemble_bo import EnsembleConfig

        config = EnsembleConfig(member_dims=[2, 4, 6, 8, 10])
        assert config.n_subspaces == 5


class TestSubspaceMember:
    """Test SubspaceMember projection, lifting, and TR logic."""

    def _make_member(self, member_id=0, seed=42, subspace_dim=8, device="cpu"):
        from rielbo.ensemble_bo import EnsembleConfig, SubspaceMember

        config = EnsembleConfig(n_candidates=50)  # Small for tests
        return SubspaceMember(
            member_id=member_id,
            input_dim=64,
            subspace_dim=subspace_dim,
            config=config,
            device=device,
            seed=seed,
        )

    def test_projection_on_sphere(self):
        """Projected points should lie on unit sphere."""
        member = self._make_member()
        u = F.normalize(torch.randn(20, 64), p=2, dim=-1)

        v = member.project(u)

        norms = v.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        assert v.shape == (20, 8)

    def test_projection_respects_dim(self):
        """Different subspace_dim should produce different output sizes."""
        m4 = self._make_member(subspace_dim=4)
        m16 = self._make_member(subspace_dim=16)
        u = F.normalize(torch.randn(5, 64), p=2, dim=-1)

        assert m4.project(u).shape == (5, 4)
        assert m16.project(u).shape == (5, 16)

    def test_lift_on_sphere(self):
        """Lifted points should lie on unit sphere."""
        member = self._make_member()
        v = F.normalize(torch.randn(20, 8), p=2, dim=-1)

        u = member.lift(v)

        norms = u.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        assert u.shape == (20, 64)

    def test_project_lift_approximate_identity(self):
        """project(lift(v)) should approximately recover v (up to subspace projection)."""
        member = self._make_member()
        v = F.normalize(torch.randn(10, 8), p=2, dim=-1)

        v_recovered = member.project(member.lift(v))

        # Cosine similarity should be high (≈1) since lift→project is close to identity
        cos_sim = (v * v_recovered).sum(dim=-1)
        assert (cos_sim > 0.99).all()

    def test_different_seeds_different_projections(self):
        """Members with different seeds should have different projections."""
        m1 = self._make_member(seed=42)
        m2 = self._make_member(seed=142)

        # Frobenius distance between projection matrices
        dist = (m1.A - m2.A).norm()
        assert dist > 1.0  # Should be substantially different

    def test_same_seed_same_projection(self):
        """Members with same seed should have identical projections."""
        m1 = self._make_member(seed=42)
        m2 = self._make_member(seed=42)

        assert torch.allclose(m1.A, m2.A)

    def test_projection_is_orthonormal(self):
        """A should have orthonormal columns (from QR)."""
        member = self._make_member()

        # A^T A should be identity
        AtA = member.A.T @ member.A
        I = torch.eye(8)
        assert torch.allclose(AtA, I, atol=1e-5)

    def test_fit_gp_runs(self):
        """GP fitting should complete without error."""
        member = self._make_member()
        train_U = F.normalize(torch.randn(30, 64), p=2, dim=-1)
        train_Y = torch.randn(30)

        member.fit_gp(train_U, train_Y)

        assert member.gp is not None

    def test_fit_gp_different_dims(self):
        """GP fitting should work for various subspace dimensions."""
        for d in [4, 8, 16, 24]:
            member = self._make_member(subspace_dim=d)
            train_U = F.normalize(torch.randn(30, 64), p=2, dim=-1)
            train_Y = torch.randn(30)
            member.fit_gp(train_U, train_Y)
            assert member.gp is not None

    def test_generate_candidate_shape(self):
        """generate_candidate should return (u_opt[1, D], std)."""
        member = self._make_member()
        train_U = F.normalize(torch.randn(30, 64), p=2, dim=-1)
        train_Y = torch.randn(30)
        member.fit_gp(train_U, train_Y)

        u_opt, std = member.generate_candidate(train_U, train_Y)

        assert u_opt.shape == (1, 64)
        assert isinstance(std, float)
        assert std >= 0

    def test_generate_candidate_on_sphere(self):
        """Generated candidate should lie on unit sphere."""
        member = self._make_member()
        train_U = F.normalize(torch.randn(30, 64), p=2, dim=-1)
        train_Y = torch.randn(30)
        member.fit_gp(train_U, train_Y)

        u_opt, _ = member.generate_candidate(train_U, train_Y)

        norm = u_opt.norm(dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


class TestSubspaceMemberAdaptiveTR:
    """Test per-member adaptive trust region logic."""

    def _make_member(self):
        from rielbo.ensemble_bo import EnsembleConfig, SubspaceMember

        config = EnsembleConfig(
            adaptive_tr=True,
            tr_init=0.4,
            tr_min=0.02,
            tr_max=0.8,
            tr_success_tol=3,
            tr_fail_tol=10,
            tr_grow_factor=1.5,
            tr_shrink_factor=0.5,
            max_restarts=5,
        )
        return SubspaceMember(
            member_id=0,
            input_dim=64,
            subspace_dim=8,
            config=config,
            device="cpu",
            seed=42,
        )

    def test_tr_grows_on_successes(self):
        member = self._make_member()
        initial = member.tr_length

        for _ in range(3):  # tr_success_tol
            member.update_tr(improved=True)

        assert member.tr_length == initial * 1.5

    def test_tr_shrinks_on_failures(self):
        member = self._make_member()
        initial = member.tr_length

        for _ in range(10):  # tr_fail_tol
            member.update_tr(improved=False)

        assert abs(member.tr_length - initial * 0.5) < 1e-8

    def test_restart_on_collapse(self):
        member = self._make_member()
        # Set TR just above min, so one shrink triggers restart
        member.tr_length = 0.03  # 0.03 * 0.5 = 0.015 < tr_min=0.02

        for _ in range(10):  # tr_fail_tol
            member.update_tr(improved=False)

        assert member.n_restarts == 1
        assert abs(member.tr_length - 0.4) < 1e-8  # Reset to tr_init

    def test_max_restarts_caps(self):
        member = self._make_member()
        member.n_restarts = 5  # max_restarts

        old_A = member.A.clone()
        member._restart()

        # Should NOT increment restarts beyond max
        assert member.n_restarts == 5
        # Projection should NOT change
        assert torch.allclose(member.A, old_A)
        # TR should reset
        assert abs(member.tr_length - 0.4) < 1e-8

    def test_success_resets_failure_counter(self):
        member = self._make_member()
        member._tr_fail_count = 7

        member.update_tr(improved=True)

        assert member._tr_fail_count == 0
        assert member._tr_success_count == 1


class TestSubspaceMemberRetirement:
    """Test member retirement (replacement with fresh projection)."""

    def _make_member(self):
        from rielbo.ensemble_bo import EnsembleConfig, SubspaceMember

        config = EnsembleConfig()
        return SubspaceMember(
            member_id=0,
            input_dim=64,
            subspace_dim=8,
            config=config,
            device="cpu",
            seed=42,
        )

    def test_retire_changes_projection(self):
        member = self._make_member()
        old_A = member.A.clone()

        train_U = F.normalize(torch.randn(30, 64), p=2, dim=-1)
        train_Y = torch.randn(30)
        member.retire(new_seed=999, train_U=train_U, train_Y=train_Y)

        # Projection should be different
        dist = (member.A - old_A).norm()
        assert dist > 1.0

    def test_retire_preserves_dim(self):
        """Retirement should keep the same subspace dimension."""
        member = self._make_member()
        old_dim = member.subspace_dim

        train_U = F.normalize(torch.randn(30, 64), p=2, dim=-1)
        train_Y = torch.randn(30)
        member.retire(new_seed=999, train_U=train_U, train_Y=train_Y)

        assert member.subspace_dim == old_dim
        assert member.A.shape == (64, old_dim)

    def test_retire_resets_stats(self):
        member = self._make_member()
        member.n_selected = 50
        member.n_improved = 10
        member.n_restarts = 3

        train_U = F.normalize(torch.randn(30, 64), p=2, dim=-1)
        train_Y = torch.randn(30)
        member.retire(new_seed=999, train_U=train_U, train_Y=train_Y)

        assert member.n_selected == 0
        assert member.n_improved == 0
        assert member.n_restarts == 0
        assert member.seed == 999

    def test_retire_refits_gp(self):
        member = self._make_member()

        train_U = F.normalize(torch.randn(30, 64), p=2, dim=-1)
        train_Y = torch.randn(30)
        member.retire(new_seed=999, train_U=train_U, train_Y=train_Y)

        assert member.gp is not None


class TestSphericalEnsembleBO:
    """Test the full SphericalEnsembleBO orchestrator."""

    def _make_ensemble(self, member_dims=None, device="cpu"):
        from rielbo.ensemble_bo import EnsembleConfig, SphericalEnsembleBO
        from unittest.mock import MagicMock

        if member_dims is None:
            member_dims = [4, 8, 12]  # Small for tests

        config = EnsembleConfig(
            member_dims=member_dims,
            retirement_interval=50,
            n_candidates=50,
        )

        codec = MagicMock()
        oracle = MagicMock()

        ensemble = SphericalEnsembleBO(
            codec=codec,
            oracle=oracle,
            input_dim=64,
            config=config,
            device=device,
            seed=42,
        )
        return ensemble

    def test_creates_correct_number_of_members(self):
        ensemble = self._make_ensemble(member_dims=[4, 8, 12, 16, 20])
        assert len(ensemble.members) == 5

    def test_members_have_correct_dims(self):
        """Each member should have its specified subspace dimension."""
        dims = [4, 8, 12, 16, 20, 24]
        ensemble = self._make_ensemble(member_dims=dims)

        for i, member in enumerate(ensemble.members):
            assert member.subspace_dim == dims[i]
            assert member.A.shape == (64, dims[i])

    def test_members_have_different_projections(self):
        ensemble = self._make_ensemble(member_dims=[8, 8, 8])

        A0 = ensemble.members[0].A
        A1 = ensemble.members[1].A
        A2 = ensemble.members[2].A

        # Even with same dim, different seeds → different projections
        assert (A0 - A1).norm() > 1.0
        assert (A1 - A2).norm() > 1.0

    def test_select_candidate_picks_max_std(self):
        """_select_candidate should pick the member with highest posterior std."""
        ensemble = self._make_ensemble(member_dims=[4, 8, 12])

        # Mock generate_candidate to return controlled stds
        def mock_gen_0(train_U, train_Y):
            return torch.randn(1, 64), 0.1

        def mock_gen_1(train_U, train_Y):
            return torch.randn(1, 64), 0.5  # Highest std

        def mock_gen_2(train_U, train_Y):
            return torch.randn(1, 64), 0.3

        ensemble.members[0].generate_candidate = mock_gen_0
        ensemble.members[1].generate_candidate = mock_gen_1
        ensemble.members[2].generate_candidate = mock_gen_2

        ensemble.train_U = torch.randn(10, 64)
        ensemble.train_Y = torch.randn(10)

        _, selected_id, diag = ensemble._select_candidate()

        assert selected_id == 1  # Member 1 has highest std
        assert diag["selected_std"] == 0.5
        assert diag["member_stds"] == [0.1, 0.5, 0.3]

    def test_retirement_not_triggered_at_wrong_interval(self):
        ensemble = self._make_ensemble()
        ensemble.train_U = torch.randn(10, 64)
        ensemble.train_Y = torch.randn(10)

        # Iteration not at retirement_interval
        ensemble.iteration = 37
        ensemble._maybe_retire()
        assert ensemble.n_retirements == 0

    def test_retirement_triggered_at_interval(self):
        ensemble = self._make_ensemble()
        ensemble.train_U = F.normalize(torch.randn(30, 64), p=2, dim=-1)
        ensemble.train_Y = torch.randn(30)

        # Fit GPs first
        for m in ensemble.members:
            m.fit_gp(ensemble.train_U, ensemble.train_Y)

        # Set different stds to control which gets retired
        ensemble.members[0].last_std = 0.5
        ensemble.members[1].last_std = 0.01  # Lowest → retired
        ensemble.members[2].last_std = 0.3

        ensemble.iteration = 50  # retirement_interval
        ensemble._maybe_retire()

        assert ensemble.n_retirements == 1

    def test_retirement_not_at_iteration_zero(self):
        ensemble = self._make_ensemble()
        ensemble.iteration = 0
        ensemble._maybe_retire()
        assert ensemble.n_retirements == 0

    def test_history_keys(self):
        ensemble = self._make_ensemble()
        expected_keys = {
            "iteration", "best_score", "current_score", "n_evaluated",
            "selected_member", "selected_std", "member_stds", "n_retirements",
        }
        assert set(ensemble.history.keys()) == expected_keys

    def test_default_dims_six_members(self):
        """Default config should create 6 members with dims [4,8,12,16,20,24]."""
        from rielbo.ensemble_bo import EnsembleConfig, SphericalEnsembleBO
        from unittest.mock import MagicMock

        config = EnsembleConfig()  # default
        ensemble = SphericalEnsembleBO(
            codec=MagicMock(),
            oracle=MagicMock(),
            input_dim=64,
            config=config,
            device="cpu",
        )
        assert len(ensemble.members) == 6
        dims = [m.subspace_dim for m in ensemble.members]
        assert dims == [4, 8, 12, 16, 20, 24]
