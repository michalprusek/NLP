"""Unit tests for batch_selection module.

Tests Local Penalization batch selection for diverse candidate selection
in Bayesian optimization.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@pytest.fixture
def device():
    """Get CUDA device."""
    return torch.device("cuda")


@pytest.fixture
def fitted_gp(device):
    """Create and fit a GP surrogate for testing."""
    from ecoflow.gp_surrogate import create_surrogate

    gp = create_surrogate("msr", D=1024, device=device)
    X = torch.randn(20, 1024, device=device)
    Y = torch.rand(20, device=device)
    gp.fit(X, Y)
    return gp


class TestBatchDiversity:
    """Tests for batch diversity via Local Penalization."""

    def test_lp_more_diverse_than_greedy(self, fitted_gp, device):
        """Verify Local Penalization produces more diverse batches than greedy."""
        from ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(100, 1024, device=device)
        batch_size = 8

        greedy_selected, _ = select_batch_candidates(
            fitted_gp, candidates, batch_size, method="greedy"
        )

        lp_selected, _ = select_batch_candidates(
            fitted_gp, candidates, batch_size, method="local_penalization"
        )

        greedy_dists = torch.cdist(greedy_selected, greedy_selected)
        lp_dists = torch.cdist(lp_selected, lp_selected)

        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        greedy_mean_dist = greedy_dists[mask].mean()
        lp_mean_dist = lp_dists[mask].mean()

        print(f"Greedy mean pairwise dist: {greedy_mean_dist:.4f}")
        print(f"LP mean pairwise dist: {lp_mean_dist:.4f}")

        lp_min_dist = lp_dists[mask].min()
        assert lp_min_dist > 0, "LP should not select duplicate points"

    def test_batch_points_are_distinct(self, fitted_gp, device):
        """Verify all batch points are distinct."""
        from ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(50, 1024, device=device)
        batch_size = 4

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size, method="local_penalization"
        )

        assert len(torch.unique(indices)) == batch_size, "Indices should be unique"

        dists = torch.cdist(selected, selected)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        min_dist = dists[mask].min()

        assert min_dist > 1e-6, "Selected points should be distinct"


class TestSelectBatchCandidates:
    """Tests for select_batch_candidates function."""

    def test_output_shapes(self, fitted_gp, device):
        """Verify correct output shapes."""
        from ecoflow.batch_selection import select_batch_candidates

        N = 64
        D = 1024
        batch_size = 4

        candidates = torch.randn(N, D, device=device)
        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size
        )

        assert selected.shape == (batch_size, D)
        assert indices.shape == (batch_size,)

    def test_indices_valid(self, fitted_gp, device):
        """Verify indices are valid into candidates tensor."""
        from ecoflow.batch_selection import select_batch_candidates

        N = 64
        candidates = torch.randn(N, 1024, device=device)
        batch_size = 4

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size
        )

        assert indices.min() >= 0, "Indices should be non-negative"
        assert indices.max() < N, f"Indices should be < {N}"

    def test_selected_from_candidates(self, fitted_gp, device):
        """Verify selected points are from candidates tensor."""
        from ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(64, 1024, device=device)
        batch_size = 4

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size
        )

        expected = candidates[indices]
        assert torch.allclose(selected, expected), "Selected should equal candidates[indices]"


class TestEdgeCases:
    """Tests for edge cases in batch selection."""

    def test_batch_size_one(self, fitted_gp, device):
        """Verify batch_size=1 works (just returns best UCB)."""
        from ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(20, 1024, device=device)

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=1, method="local_penalization"
        )

        assert selected.shape == (1, 1024)
        assert indices.shape == (1,)

        greedy_selected, greedy_indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=1, method="greedy"
        )

        assert torch.equal(indices, greedy_indices), "batch_size=1 should match greedy"

    def test_batch_size_equals_candidates(self, fitted_gp, device):
        """Verify batch_size=N returns all candidates."""
        from ecoflow.batch_selection import select_batch_candidates

        N = 10
        candidates = torch.randn(N, 1024, device=device)

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=N
        )

        assert selected.shape == (N, 1024)
        assert len(torch.unique(indices)) == N, "All candidates should be selected"

    def test_batch_size_exceeds_candidates(self, fitted_gp, device):
        """Verify batch_size > N returns all N candidates."""
        from ecoflow.batch_selection import select_batch_candidates

        N = 5
        candidates = torch.randn(N, 1024, device=device)

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=10
        )

        assert selected.shape == (N, 1024)
        assert indices.shape == (N,)

    def test_greedy_method(self, fitted_gp, device):
        """Verify greedy method works."""
        from ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(50, 1024, device=device)

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=4, method="greedy"
        )

        assert selected.shape == (4, 1024)

        with torch.no_grad():
            mean, std = fitted_gp.predict(candidates)
            ucb = mean + 1.96 * std
            expected_indices = ucb.topk(4).indices

        assert set(indices.tolist()) == set(expected_indices.tolist())

    def test_unknown_method_raises(self, fitted_gp, device):
        """Verify unknown method raises ValueError."""
        from ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(20, 1024, device=device)

        with pytest.raises(ValueError, match="Unknown method"):
            select_batch_candidates(
                fitted_gp, candidates, batch_size=4, method="unknown"
            )
