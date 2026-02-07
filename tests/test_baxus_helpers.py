"""Tests for BAxUS helper functions and normalization.

Tests the pure functions that are core to BAxUS operation:
- Embedding matrix construction
- Bin splitting (increase embedding)
- Trust region state management
- Normalization round-trips
"""

import math

import numpy as np
import pytest
import torch

from rielbo.benchmark.methods.baxus import (
    BaxusState,
    _embedding_matrix,
    _increase_embedding,
    _update_baxus_state,
)


class TestEmbeddingMatrix:
    """Test hash-based embedding matrix construction."""

    def test_shape(self):
        """Embedding should have shape [target_dim, input_dim]."""
        S = _embedding_matrix(256, 4, "cpu", torch.double)
        assert S.shape == (4, 256)

    def test_identity_when_target_ge_input(self):
        """When target_dim >= input_dim, should return identity."""
        S = _embedding_matrix(10, 10, "cpu", torch.double)
        assert torch.allclose(S, torch.eye(10, dtype=torch.double))

        S = _embedding_matrix(10, 15, "cpu", torch.double)
        assert torch.allclose(S, torch.eye(10, dtype=torch.double))

    def test_coverage(self):
        """Every input dim should appear in at least one row."""
        S = _embedding_matrix(256, 4, "cpu", torch.double)
        # Each column should have exactly one non-zero entry
        nonzero_per_col = (S != 0).sum(dim=0)
        assert (nonzero_per_col >= 1).all(), "Some input dims have no coverage"

    def test_values_are_plus_minus_one(self):
        """Non-zero entries should be +/-1 only."""
        S = _embedding_matrix(256, 4, "cpu", torch.double)
        nonzero_vals = S[S != 0]
        assert torch.all((nonzero_vals == 1.0) | (nonzero_vals == -1.0))

    def test_reproducible_with_seed(self):
        """Same seed should produce same embedding."""
        torch.manual_seed(42)
        S1 = _embedding_matrix(256, 4, "cpu", torch.double)
        torch.manual_seed(42)
        S2 = _embedding_matrix(256, 4, "cpu", torch.double)
        assert torch.allclose(S1, S2)


class TestIncreaseEmbedding:
    """Test bin splitting to increase target dimensionality."""

    def test_expands_dims(self):
        """After bin split, target_dim should increase."""
        torch.manual_seed(42)
        S = _embedding_matrix(256, 4, "cpu", torch.double)
        X_target = torch.randn(10, 4, dtype=torch.double)

        S_new, X_new = _increase_embedding(S, X_target, 3, "cpu", torch.double)

        assert S_new.shape[0] > S.shape[0], "Target dim should increase"
        assert S_new.shape[1] == 256, "Input dim should stay the same"
        assert X_new.shape[0] == 10, "N samples should stay the same"
        assert X_new.shape[1] == S_new.shape[0], "X cols should match S rows"

    def test_preserves_input_coverage(self):
        """After splitting, every input dim should still be covered."""
        torch.manual_seed(42)
        S = _embedding_matrix(256, 4, "cpu", torch.double)
        X_target = torch.randn(10, 4, dtype=torch.double)

        S_new, _ = _increase_embedding(S, X_target, 3, "cpu", torch.double)

        nonzero_per_col = (S_new != 0).sum(dim=0)
        assert (nonzero_per_col >= 1).all(), "Some input dims lost coverage after split"


class TestUpdateBaxusState:
    """Test trust region state management."""

    def _make_state(self):
        return BaxusState(dim=256, eval_budget=500)

    def test_success_doubles_length(self):
        """After success_tolerance successes, length should double."""
        state = self._make_state()
        state.best_value = 0.0  # Start with finite value
        initial_length = state.length

        for i in range(state.success_tolerance):
            # Each observation must beat the previous best by > 1e-3 * |best|
            state = _update_baxus_state(state, torch.tensor([[state.best_value + 1.0]]))

        assert abs(state.length - min(2.0 * initial_length, state.length_max)) < 1e-8

    def test_failure_halves_length(self):
        """After failure_tolerance failures, length should halve."""
        state = self._make_state()
        initial_length = state.length
        state.best_value = 100.0  # High so new values are always failures

        for _ in range(state.failure_tolerance):
            state = _update_baxus_state(state, torch.tensor([[0.0]]))

        assert abs(state.length - initial_length / 2.0) < 1e-8

    def test_restart_on_min_length(self):
        """restart_triggered should be True when length drops below min."""
        state = self._make_state()
        state.length = state.length_min * 2.5
        state.best_value = 100.0

        # Keep failing to shrink
        for _ in range(100):
            state = _update_baxus_state(state, torch.tensor([[0.0]]))
            if state.restart_triggered:
                break

        assert state.restart_triggered

    def test_baxus_state_init_dim256(self):
        """d_init and n_splits should be reasonable for dim=256."""
        state = BaxusState(dim=256, eval_budget=500)
        assert state.d_init >= 1
        assert state.target_dim == state.d_init
        assert state.n_splits > 0
        # Verify: (1 + new_bins_on_split)^n_splits * d_init ≈ dim
        product = state.d_init * (1 + state.new_bins_on_split) ** state.n_splits
        assert abs(product - 256) < 256 * 0.5  # Within 50%


class TestBAxUSNormalization:
    """Test normalization round-trips."""

    def test_normalize_roundtrip(self):
        """normalize then denormalize should recover original values."""
        from rielbo.benchmark.methods.baxus import BAxUSBenchmark
        adapter = object.__new__(BAxUSBenchmark)
        adapter._z_min = torch.tensor([-2.0, 0.0, 1.0])
        adapter._z_max = torch.tensor([2.0, 4.0, 5.0])

        z = torch.tensor([[0.0, 2.0, 3.0]])
        z_norm = adapter._normalize(z)
        z_back = adapter._denormalize(z_norm)
        assert torch.allclose(z, z_back, atol=1e-6)

    def test_target_normalize_roundtrip(self):
        """Target space normalize/denormalize should be inverse operations."""
        from rielbo.benchmark.methods.baxus import BAxUSBenchmark
        adapter = object.__new__(BAxUSBenchmark)
        adapter._t_min = torch.tensor([-5.0, -3.0])
        adapter._t_max = torch.tensor([5.0, 7.0])

        x = torch.tensor([[0.0, 2.0]])
        x_norm = adapter._normalize_target(x)
        x_back = adapter._denormalize_target(x_norm)
        assert torch.allclose(x, x_back, atol=1e-6)

    def test_normalize_produces_unit_range(self):
        """Normalization should map [min, max] to [-1, 1]."""
        from rielbo.benchmark.methods.baxus import BAxUSBenchmark
        adapter = object.__new__(BAxUSBenchmark)
        adapter._z_min = torch.tensor([0.0, 0.0])
        adapter._z_max = torch.tensor([10.0, 10.0])

        z_min_norm = adapter._normalize(adapter._z_min.unsqueeze(0))
        z_max_norm = adapter._normalize(adapter._z_max.unsqueeze(0))

        assert torch.allclose(z_min_norm, torch.tensor([[-1.0, -1.0]]), atol=1e-6)
        assert torch.allclose(z_max_norm, torch.tensor([[1.0, 1.0]]), atol=1e-6)
