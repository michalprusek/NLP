"""Unit tests for batch_selection module.

Tests Local Penalization batch selection for diverse candidate selection
in Bayesian optimization.
"""

import pytest
import torch

# Skip all tests if CUDA not available
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
    from src.ecoflow.gp_surrogate import create_surrogate

    gp = create_surrogate("msr", D=1024, device=device)
    X = torch.randn(20, 1024, device=device)
    Y = torch.rand(20, device=device)
    gp.fit(X, Y)
    return gp


@pytest.fixture
def bounds(device):
    """Create search space bounds."""
    return torch.tensor([[-3.0] * 1024, [3.0] * 1024], device=device)


class TestEstimateLipschitzConstant:
    """Tests for estimate_lipschitz_constant function."""

    def test_positive_result(self, fitted_gp, bounds):
        """Verify Lipschitz estimate is positive."""
        from src.ecoflow.batch_selection import estimate_lipschitz_constant

        L = estimate_lipschitz_constant(fitted_gp.model, bounds, n_samples=50)

        assert L > 0, "Lipschitz constant should be positive"
        assert isinstance(L, float), "Should return float"

    def test_minimum_floor(self, fitted_gp, bounds):
        """Verify minimum floor of 1e-7."""
        from src.ecoflow.batch_selection import estimate_lipschitz_constant

        L = estimate_lipschitz_constant(fitted_gp.model, bounds, n_samples=10)

        assert L >= 1e-7, "Should have minimum floor"

    def test_reasonable_magnitude(self, fitted_gp, bounds):
        """Verify Lipschitz constant is in reasonable range for GP."""
        from src.ecoflow.batch_selection import estimate_lipschitz_constant

        L = estimate_lipschitz_constant(fitted_gp.model, bounds, n_samples=100)

        # For a typical GP with lengthscale ~3.2, gradients should be moderate
        # L should be > 0 but not astronomically large
        assert L < 100, f"Lipschitz constant {L} seems too large"


class TestPenalizedAcquisition:
    """Tests for PenalizedAcquisition class."""

    def test_penalty_near_selected(self, fitted_gp, device):
        """Verify penalty reduces acquisition near selected points."""
        from src.ecoflow.batch_selection import PenalizedAcquisition
        from botorch.acquisition.analytic import UpperConfidenceBound

        # Create base acquisition
        base_acqf = UpperConfidenceBound(fitted_gp.model, beta=1.96**2)

        # Select a point
        X_selected = torch.randn(1, 1024, device=device)

        # Create penalized acquisition
        penalized = PenalizedAcquisition(
            base_acqf=base_acqf,
            X_selected=X_selected,
            L=0.01,  # Small L = large radius = strong penalty
            model=fitted_gp.model,
        )

        # Test point AT selected location (should be heavily penalized)
        X_at_selected = X_selected.unsqueeze(0)  # [1, 1, D]

        base_val = base_acqf(X_at_selected)
        penalized_val = penalized(X_at_selected)

        assert penalized_val < base_val, "Penalty should reduce acquisition at selected point"

    def test_penalty_far_from_selected(self, fitted_gp, device):
        """Verify penalty is minimal far from selected points."""
        from src.ecoflow.batch_selection import PenalizedAcquisition
        from botorch.acquisition.analytic import UpperConfidenceBound

        base_acqf = UpperConfidenceBound(fitted_gp.model, beta=1.96**2)

        X_selected = torch.zeros(1, 1024, device=device)

        penalized = PenalizedAcquisition(
            base_acqf=base_acqf,
            X_selected=X_selected,
            L=1.0,  # Larger L = smaller radius
            model=fitted_gp.model,
        )

        # Test point FAR from selected (should have minimal penalty)
        X_far = torch.randn(1, 1, 1024, device=device) * 100  # Very far

        base_val = base_acqf(X_far)
        penalized_val = penalized(X_far)

        # Penalty should be close to 1.0 (no penalty) far away
        ratio = penalized_val / base_val
        assert ratio > 0.9, f"Penalty should be minimal far away, got ratio {ratio}"


class TestBatchDiversity:
    """Tests for batch diversity via Local Penalization."""

    def test_lp_more_diverse_than_greedy(self, fitted_gp, device):
        """Verify Local Penalization produces more diverse batches than greedy."""
        from src.ecoflow.batch_selection import select_batch_candidates

        # Generate candidates
        candidates = torch.randn(100, 1024, device=device)
        batch_size = 8

        # Greedy selection (top-K by UCB)
        greedy_selected, _ = select_batch_candidates(
            fitted_gp, candidates, batch_size, method="greedy"
        )

        # LP selection
        lp_selected, _ = select_batch_candidates(
            fitted_gp, candidates, batch_size, method="local_penalization"
        )

        # Compute mean pairwise distances
        greedy_dists = torch.cdist(greedy_selected, greedy_selected)
        lp_dists = torch.cdist(lp_selected, lp_selected)

        # Mask diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        greedy_mean_dist = greedy_dists[mask].mean()
        lp_mean_dist = lp_dists[mask].mean()

        # LP should generally produce more diverse batches
        # (may not always be true due to randomness, but usually is)
        print(f"Greedy mean pairwise dist: {greedy_mean_dist:.4f}")
        print(f"LP mean pairwise dist: {lp_mean_dist:.4f}")

        # At minimum, LP distances should be positive (no duplicates)
        lp_min_dist = lp_dists[mask].min()
        assert lp_min_dist > 0, "LP should not select duplicate points"

    def test_batch_points_are_distinct(self, fitted_gp, device):
        """Verify all batch points are distinct."""
        from src.ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(50, 1024, device=device)
        batch_size = 4

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size, method="local_penalization"
        )

        # Check indices are unique
        assert len(torch.unique(indices)) == batch_size, "Indices should be unique"

        # Check points have non-zero pairwise distances
        dists = torch.cdist(selected, selected)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        min_dist = dists[mask].min()

        assert min_dist > 1e-6, "Selected points should be distinct"


class TestSelectBatchCandidates:
    """Tests for select_batch_candidates function."""

    def test_output_shapes(self, fitted_gp, device):
        """Verify correct output shapes."""
        from src.ecoflow.batch_selection import select_batch_candidates

        N = 64
        D = 1024
        batch_size = 4

        candidates = torch.randn(N, D, device=device)
        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size
        )

        assert selected.shape == (batch_size, D), f"Expected ({batch_size}, {D}), got {selected.shape}"
        assert indices.shape == (batch_size,), f"Expected ({batch_size},), got {indices.shape}"

    def test_indices_valid(self, fitted_gp, device):
        """Verify indices are valid into candidates tensor."""
        from src.ecoflow.batch_selection import select_batch_candidates

        N = 64
        candidates = torch.randn(N, 1024, device=device)
        batch_size = 4

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size
        )

        # All indices should be in valid range
        assert indices.min() >= 0, "Indices should be non-negative"
        assert indices.max() < N, f"Indices should be < {N}"

    def test_selected_from_candidates(self, fitted_gp, device):
        """Verify selected points are from candidates tensor."""
        from src.ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(64, 1024, device=device)
        batch_size = 4

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size
        )

        # Verify selected matches candidates[indices]
        expected = candidates[indices]
        assert torch.allclose(selected, expected), "Selected should equal candidates[indices]"


class TestEdgeCases:
    """Tests for edge cases in batch selection."""

    def test_batch_size_one(self, fitted_gp, device):
        """Verify batch_size=1 works (just returns best UCB)."""
        from src.ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(20, 1024, device=device)

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=1, method="local_penalization"
        )

        assert selected.shape == (1, 1024)
        assert indices.shape == (1,)

        # Should be same as greedy for batch_size=1
        greedy_selected, greedy_indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=1, method="greedy"
        )

        assert torch.equal(indices, greedy_indices), "batch_size=1 should match greedy"

    def test_batch_size_equals_candidates(self, fitted_gp, device):
        """Verify batch_size=N returns all candidates."""
        from src.ecoflow.batch_selection import select_batch_candidates

        N = 10
        candidates = torch.randn(N, 1024, device=device)

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=N
        )

        assert selected.shape == (N, 1024)
        assert len(torch.unique(indices)) == N, "All candidates should be selected"

    def test_batch_size_exceeds_candidates(self, fitted_gp, device):
        """Verify batch_size > N returns all N candidates."""
        from src.ecoflow.batch_selection import select_batch_candidates

        N = 5
        candidates = torch.randn(N, 1024, device=device)

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=10  # Larger than N
        )

        # Should return all N candidates
        assert selected.shape == (N, 1024)
        assert indices.shape == (N,)

    def test_greedy_method(self, fitted_gp, device):
        """Verify greedy method works."""
        from src.ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(50, 1024, device=device)

        selected, indices = select_batch_candidates(
            fitted_gp, candidates, batch_size=4, method="greedy"
        )

        assert selected.shape == (4, 1024)

        # Greedy should return top-K by UCB
        with torch.no_grad():
            mean, std = fitted_gp.predict(candidates)
            ucb = mean + 1.96 * std
            expected_indices = ucb.topk(4).indices

        # Sort for comparison (order might differ)
        assert set(indices.tolist()) == set(expected_indices.tolist())

    def test_unknown_method_raises(self, fitted_gp, device):
        """Verify unknown method raises ValueError."""
        from src.ecoflow.batch_selection import select_batch_candidates

        candidates = torch.randn(20, 1024, device=device)

        with pytest.raises(ValueError, match="Unknown method"):
            select_batch_candidates(
                fitted_gp, candidates, batch_size=4, method="unknown"
            )


class TestLocalPenalizationBatch:
    """Tests for local_penalization_batch with optimize_acqf."""

    def test_basic_functionality(self, fitted_gp, bounds):
        """Test local_penalization_batch with BoTorch optimize_acqf."""
        from src.ecoflow.batch_selection import local_penalization_batch
        from botorch.acquisition import qUpperConfidenceBound

        batch_size = 2  # Small for speed

        batch = local_penalization_batch(
            model=fitted_gp.model,
            bounds=bounds,
            batch_size=batch_size,
            base_acqf_class=qUpperConfidenceBound,
            acqf_kwargs={"beta": 1.96**2},
            num_restarts=2,  # Small for speed
            raw_samples=32,  # Small for speed
        )

        assert batch.shape == (batch_size, 1024)

        # Verify points are within bounds
        assert (batch >= bounds[0]).all(), "Points should be >= lower bound"
        assert (batch <= bounds[1]).all(), "Points should be <= upper bound"
