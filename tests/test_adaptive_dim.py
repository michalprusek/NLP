"""Tests for adaptive dimensionality estimation module.

Tests:
- Intrinsic dimensionality estimation (TwoNN + MLE, N//6 cap)
- GP health assessment (overfit, stagnation, dead GP detection)
"""

import numpy as np
import pytest
import torch


class TestEstimateIntrinsicDim:
    """Test estimate_intrinsic_dim function."""

    def test_estimate_id_synthetic_sphere(self):
        """Points on a 16D sphere in 256D should yield estimate in [8, 24]."""
        from rielbo.adaptive_dim import estimate_intrinsic_dim

        rng = np.random.RandomState(42)

        Q, _ = torch.linalg.qr(torch.from_numpy(rng.randn(256, 16)).float())
        Q_np = Q.numpy()

        raw = rng.randn(500, 16).astype(np.float32)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        points = raw @ Q_np.T
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        d_est, diagnostics = estimate_intrinsic_dim(points, n_points=500)

        assert isinstance(d_est, int)
        assert 8 <= d_est <= 24, f"Expected d in [8, 24], got {d_est}"
        assert "d_twonn" in diagnostics
        assert "d_mle" in diagnostics
        assert "d_raw" in diagnostics
        assert "upper_cap" in diagnostics
        assert diagnostics["upper_cap"] == 4 * (min(500 // 6, 48) // 4)

    def test_estimate_id_n100_capped_at_16(self):
        """With N=100, N//6=16 should cap the estimate."""
        from rielbo.adaptive_dim import estimate_intrinsic_dim

        rng = np.random.RandomState(42)

        Q, _ = torch.linalg.qr(torch.from_numpy(rng.randn(256, 32)).float())
        Q_np = Q.numpy()

        raw = rng.randn(100, 32).astype(np.float32)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        points = raw @ Q_np.T
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        d_est, diag = estimate_intrinsic_dim(points, n_points=100)

        assert d_est <= 16, f"Expected d <= 16 (N//6=16), got {d_est}"
        assert d_est >= 8, f"Expected d >= 8 (d_min), got {d_est}"

    def test_estimate_id_multiple_of_4(self):
        """Estimate should always be a multiple of 4."""
        from rielbo.adaptive_dim import estimate_intrinsic_dim

        rng = np.random.RandomState(42)
        points = rng.randn(200, 256).astype(np.float32)
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        d_est, _ = estimate_intrinsic_dim(points, n_points=200)
        assert d_est % 4 == 0, f"Expected multiple of 4, got {d_est}"

    def test_estimate_id_respects_d_min_d_max(self):
        """Estimate should respect d_min and d_max bounds."""
        from rielbo.adaptive_dim import estimate_intrinsic_dim

        rng = np.random.RandomState(42)
        points = rng.randn(300, 256).astype(np.float32)
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        d_est, _ = estimate_intrinsic_dim(points, n_points=300, d_min=12, d_max=20)
        assert 12 <= d_est <= 20, f"Expected d in [12, 20], got {d_est}"


class TestAssessGPHealth:
    """Test assess_gp_health function."""

    def test_assess_gp_health_overfit(self):
        """3+ entries with train_correlation > 0.995 should flag persistent_overfit."""
        from rielbo.adaptive_dim import assess_gp_health

        diagnostic_history = [
            {"train_correlation": 0.999} for _ in range(5)
        ]
        best_score_history = [0.40 + i * 0.01 for i in range(200)]

        result = assess_gp_health(
            diagnostic_history=diagnostic_history,
            iteration=200,
            last_restart_iter=100,
            best_score_history=best_score_history,
        )

        assert result["persistent_overfit"] is True

    def test_assess_gp_health_stagnation(self):
        """< 0.001 improvement over last 100 entries should flag stagnation."""
        from rielbo.adaptive_dim import assess_gp_health

        diagnostic_history = [
            {"train_correlation": 0.90} for _ in range(150)
        ]
        best_score_history = [0.40] * 50 + [0.40 + i * 0.000005 for i in range(100)]

        result = assess_gp_health(
            diagnostic_history=diagnostic_history,
            iteration=150,
            last_restart_iter=0,
            best_score_history=best_score_history,
        )

        assert result["stagnation"] is True

    def test_assess_gp_health_healthy(self):
        """Normal metrics should yield all-False flags."""
        from rielbo.adaptive_dim import assess_gp_health

        diagnostic_history = [
            {"train_correlation": 0.85, "train_std_ratio": 0.1} for _ in range(50)
        ]
        best_score_history = [0.30 + i * 0.005 for i in range(200)]

        result = assess_gp_health(
            diagnostic_history=diagnostic_history,
            iteration=200,
            last_restart_iter=150,
            best_score_history=best_score_history,
        )

        assert result["persistent_overfit"] is False
        assert result["stagnation"] is False
        assert result["dead_gp"] is False

    def test_assess_gp_health_dead_gp(self):
        """GP with near-zero std_ratio for 3+ fits should flag dead_gp."""
        from rielbo.adaptive_dim import assess_gp_health

        diagnostic_history = [
            {"train_correlation": 0.2, "train_std_ratio": 0.0001} for _ in range(5)
        ]
        best_score_history = [0.40 + i * 0.01 for i in range(200)]

        result = assess_gp_health(
            diagnostic_history=diagnostic_history,
            iteration=200,
            last_restart_iter=100,
            best_score_history=best_score_history,
        )

        assert result["dead_gp"] is True

    def test_assess_gp_health_empty_history(self):
        """Empty diagnostic history should return all-healthy."""
        from rielbo.adaptive_dim import assess_gp_health

        result = assess_gp_health(
            diagnostic_history=[],
            iteration=0,
            last_restart_iter=0,
            best_score_history=[],
        )

        assert result["persistent_overfit"] is False
        assert result["stagnation"] is False
        assert result["dead_gp"] is False
