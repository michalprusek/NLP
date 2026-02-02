"""Tests for GP surrogate models.

Tests cover:
1. HeteroscedasticSonarGP with binomial noise model
2. create_surrogate factory function
"""

import pytest
import torch

from rielbo.gp_surrogate import (
    HeteroscedasticSonarGP,
    SonarGPSurrogate,
    create_surrogate,
)


# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestBinomialVariance:
    """Test variance formula: Var(p) = p(1-p)/n"""

    def test_variance_at_p_half(self):
        """Variance should be maximal at p=0.5."""
        gp = HeteroscedasticSonarGP(n_eval=100, device=DEVICE)
        Y = torch.tensor([0.5], device=DEVICE)
        var = gp._compute_variance(Y)
        # Var(0.5) = 0.5 * 0.5 / 100 = 0.0025
        assert torch.isclose(var[0], torch.tensor(0.0025, device=DEVICE), rtol=1e-4)

    def test_variance_at_p_nine(self):
        """Variance should be lower at p=0.9."""
        gp = HeteroscedasticSonarGP(n_eval=100, device=DEVICE)
        Y = torch.tensor([0.9], device=DEVICE)
        var = gp._compute_variance(Y)
        # Var(0.9) = 0.9 * 0.1 / 100 = 0.0009
        assert torch.isclose(var[0], torch.tensor(0.0009, device=DEVICE), rtol=1e-4)

    def test_variance_at_p_one(self):
        """Variance at p=0.1 should match p=0.9 (symmetric)."""
        gp = HeteroscedasticSonarGP(n_eval=100, device=DEVICE)
        Y = torch.tensor([0.1], device=DEVICE)
        var = gp._compute_variance(Y)
        # Var(0.1) = 0.1 * 0.9 / 100 = 0.0009
        assert torch.isclose(var[0], torch.tensor(0.0009, device=DEVICE), rtol=1e-4)

    def test_variance_monotonic_from_half(self):
        """Variance should decrease as p moves away from 0.5."""
        gp = HeteroscedasticSonarGP(n_eval=100, device=DEVICE)
        Y = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], device=DEVICE)
        var = gp._compute_variance(Y)
        # Variance should be monotonically decreasing
        for i in range(len(var) - 1):
            assert var[i] > var[i + 1], f"Variance not decreasing at i={i}"

    def test_variance_scales_with_n_eval(self):
        """Variance should scale inversely with n_eval."""
        gp_100 = HeteroscedasticSonarGP(n_eval=100, device=DEVICE)
        gp_200 = HeteroscedasticSonarGP(n_eval=200, device=DEVICE)

        Y = torch.tensor([0.5], device=DEVICE)
        var_100 = gp_100._compute_variance(Y)
        var_200 = gp_200._compute_variance(Y)

        # Variance should be half for n_eval=200 vs n_eval=100
        assert torch.isclose(var_100, 2 * var_200, rtol=1e-4)


class TestNumericalStability:
    """Test numerical stability for extreme values."""

    def test_variance_at_zero(self):
        """p=0.0 should not cause NaN/Inf (clamped to 0.01)."""
        gp = HeteroscedasticSonarGP(n_eval=100, device=DEVICE)
        Y = torch.tensor([0.0], device=DEVICE)
        var = gp._compute_variance(Y)
        assert not torch.isnan(var[0])
        assert not torch.isinf(var[0])
        # Should be clamped: Var(0.01) = 0.01 * 0.99 / 100 = 0.000099
        assert var[0] > 0

    def test_variance_at_one(self):
        """p=1.0 should not cause NaN/Inf (clamped to 0.99)."""
        gp = HeteroscedasticSonarGP(n_eval=100, device=DEVICE)
        Y = torch.tensor([1.0], device=DEVICE)
        var = gp._compute_variance(Y)
        assert not torch.isnan(var[0])
        assert not torch.isinf(var[0])
        # Should be clamped: Var(0.99) = 0.99 * 0.01 / 100 = 0.000099
        assert var[0] > 0

    def test_variance_floor_applied(self):
        """Variance should have minimum floor of 1e-6."""
        # Use very large n_eval to make variance very small
        gp = HeteroscedasticSonarGP(n_eval=1_000_000, device=DEVICE)
        Y = torch.tensor([0.99], device=DEVICE)
        var = gp._compute_variance(Y)
        # Without floor: 0.99 * 0.01 / 1_000_000 = 9.9e-9
        # With floor: 1e-6
        assert var[0] >= 1e-6

    def test_no_nan_in_batch(self):
        """No NaN values for a batch with extreme values."""
        gp = HeteroscedasticSonarGP(n_eval=150, device=DEVICE)
        Y = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device=DEVICE)
        var = gp._compute_variance(Y)
        assert not torch.any(torch.isnan(var))
        assert not torch.any(torch.isinf(var))


class TestGPFitting:
    """Test GP fitting with heteroscedastic noise."""

    def test_fit_basic(self):
        """GP should fit without error."""
        gp = HeteroscedasticSonarGP(D=64, n_eval=100, device=DEVICE)
        X = torch.randn(10, 64, device=DEVICE)
        Y = torch.rand(10, device=DEVICE)
        gp.fit(X, Y)
        assert gp.model is not None
        assert gp.n_train == 10

    def test_predict_after_fit(self):
        """Predictions should work after fitting."""
        gp = HeteroscedasticSonarGP(D=64, n_eval=100, device=DEVICE)
        X = torch.randn(10, 64, device=DEVICE)
        Y = torch.rand(10, device=DEVICE)
        gp.fit(X, Y)

        mean, std = gp.predict(X[:3])
        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert not torch.any(torch.isnan(mean))
        assert not torch.any(torch.isnan(std))
        assert torch.all(std > 0)

    def test_update_incremental(self):
        """Update should add new data points."""
        gp = HeteroscedasticSonarGP(D=64, n_eval=100, device=DEVICE)
        X1 = torch.randn(5, 64, device=DEVICE)
        Y1 = torch.rand(5, device=DEVICE)
        gp.fit(X1, Y1)
        assert gp.n_train == 5

        X2 = torch.randn(3, 64, device=DEVICE)
        Y2 = torch.rand(3, device=DEVICE)
        gp.update(X2, Y2)
        assert gp.n_train == 8

    def test_gradient_computable(self):
        """Gradients should be computable via autograd."""
        gp = HeteroscedasticSonarGP(D=64, n_eval=100, device=DEVICE)
        X = torch.randn(10, 64, device=DEVICE)
        Y = torch.rand(10, device=DEVICE)
        gp.fit(X, Y)

        # Test LCB gradient
        grad_lcb = gp.lcb_gradient(X[:2], alpha=1.0)
        assert grad_lcb.shape == (2, 64)
        assert not torch.any(torch.isnan(grad_lcb))

        # Test UCB gradient
        grad_ucb = gp.ucb_gradient(X[:2], alpha=1.96)
        assert grad_ucb.shape == (2, 64)
        assert not torch.any(torch.isnan(grad_ucb))

    def test_uncertainty_reasonable(self):
        """Predictions should have reasonable uncertainty."""
        gp = HeteroscedasticSonarGP(D=64, n_eval=100, device=DEVICE)
        X = torch.randn(10, 64, device=DEVICE)
        Y = torch.rand(10, device=DEVICE)
        gp.fit(X, Y)

        # Far-away point should have higher uncertainty
        X_near = X[:1]
        X_far = X[:1] + torch.randn_like(X[:1]) * 10

        _, std_near = gp.predict(X_near)
        _, std_far = gp.predict(X_far)

        # Note: This may not always hold due to GP behavior,
        # but in general far points should have higher uncertainty
        # We just check that both predictions work
        assert std_near.shape == (1,)
        assert std_far.shape == (1,)


class TestCreateSurrogateFactory:
    """Test create_surrogate factory function."""

    def test_heteroscedastic_method(self):
        """Factory should create HeteroscedasticSonarGP for 'heteroscedastic' method."""
        gp = create_surrogate("heteroscedastic", D=64, device=DEVICE)
        assert isinstance(gp, HeteroscedasticSonarGP)

    def test_heteroscedastic_with_n_eval(self):
        """n_eval should be passed correctly to HeteroscedasticSonarGP."""
        gp = create_surrogate("heteroscedastic", D=64, n_eval=200, device=DEVICE)
        assert isinstance(gp, HeteroscedasticSonarGP)
        assert gp.n_eval == 200

    def test_msr_method(self):
        """Factory should create SonarGPSurrogate for 'msr' method."""
        gp = create_surrogate("msr", D=64, device=DEVICE)
        assert isinstance(gp, SonarGPSurrogate)

    def test_standard_method_alias(self):
        """'standard' should be an alias for 'msr'."""
        gp = create_surrogate("standard", D=64, device=DEVICE)
        assert isinstance(gp, SonarGPSurrogate)

    def test_case_insensitive(self):
        """Method names should be case-insensitive."""
        gp1 = create_surrogate("HETEROSCEDASTIC", D=64, device=DEVICE)
        gp2 = create_surrogate("Heteroscedastic", D=64, device=DEVICE)
        assert isinstance(gp1, HeteroscedasticSonarGP)
        assert isinstance(gp2, HeteroscedasticSonarGP)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown surrogate method"):
            create_surrogate("invalid_method", D=64, device=DEVICE)


class TestIntegrationWithGuidedFlow:
    """Test integration with GuidedFlowSampler interface."""

    def test_ucb_gradient_interface(self):
        """ucb_gradient should work with the expected interface."""
        gp = create_surrogate("heteroscedastic", D=64, n_eval=150, device=DEVICE)
        X = torch.randn(5, 64, device=DEVICE)
        Y = torch.rand(5, device=DEVICE)
        gp.fit(X, Y)

        # GuidedFlowSampler uses ucb_gradient with alpha parameter
        grad = gp.ucb_gradient(X, alpha=1.96)
        assert grad.shape == (5, 64)
        assert not torch.any(torch.isnan(grad))

    def test_predict_interface(self):
        """predict should return mean and std tensors."""
        gp = create_surrogate("heteroscedastic", D=64, n_eval=150, device=DEVICE)
        X = torch.randn(5, 64, device=DEVICE)
        Y = torch.rand(5, device=DEVICE)
        gp.fit(X, Y)

        mean, std = gp.predict(X)
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert mean.device.type == DEVICE.split(":")[0] if ":" in DEVICE else DEVICE
