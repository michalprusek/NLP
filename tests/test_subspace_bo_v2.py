"""Tests for RieLBO v2 theoretical improvements.

Tests:
- Kernels: positive definite, k(x,x)=1, symmetry
- Spherical Whitening: mean at north pole, inverse works
- Geodesic Trust Region: samples on sphere, within angular bound
- Norm Distribution: samples in observed range
"""

import math

import pytest
import torch
import torch.nn.functional as F


class TestArcCosineKernels:
    """Test ArcCosine kernel implementations."""

    def test_arccosine_order0_self_similarity(self):
        """k(x, x) should be close to 1 (within clamping tolerance)."""
        from rielbo.kernels import ArcCosineKernel

        kernel = ArcCosineKernel()
        x = F.normalize(torch.randn(10, 16), p=2, dim=-1)

        k_diag = kernel(x, x, diag=True)

        # Due to clamping at 1-1e-6, diagonal is slightly below 1
        assert torch.allclose(k_diag, torch.ones_like(k_diag), atol=1e-3)

    def test_arccosine_order2_self_similarity(self):
        """k(x, x) should equal 1."""
        from rielbo.kernels import ArcCosineKernelOrder2

        kernel = ArcCosineKernelOrder2()
        x = F.normalize(torch.randn(10, 16), p=2, dim=-1)

        k_diag = kernel(x, x, diag=True)

        assert torch.allclose(k_diag, torch.ones_like(k_diag), atol=1e-3)

    def test_arccosine_symmetry(self):
        """K(x, y) should be symmetric."""
        from rielbo.kernels import ArcCosineKernel

        kernel = ArcCosineKernel()
        x = F.normalize(torch.randn(5, 16), p=2, dim=-1)
        y = F.normalize(torch.randn(7, 16), p=2, dim=-1)

        K_xy = kernel(x, y).evaluate()  # Convert from lazy tensor
        K_yx = kernel(y, x).evaluate()

        assert torch.allclose(K_xy, K_yx.T, atol=1e-5)

    def test_arccosine_order2_symmetry(self):
        """K(x, y) should be symmetric."""
        from rielbo.kernels import ArcCosineKernelOrder2

        kernel = ArcCosineKernelOrder2()
        x = F.normalize(torch.randn(5, 16), p=2, dim=-1)
        y = F.normalize(torch.randn(7, 16), p=2, dim=-1)

        K_xy = kernel(x, y).evaluate()
        K_yx = kernel(y, x).evaluate()

        assert torch.allclose(K_xy, K_yx.T, atol=1e-5)

    def test_arccosine_positive_definite(self):
        """Kernel matrix should be positive semi-definite."""
        from rielbo.kernels import ArcCosineKernel

        kernel = ArcCosineKernel()
        x = F.normalize(torch.randn(20, 16), p=2, dim=-1)

        K = kernel(x, x).evaluate()
        eigenvalues = torch.linalg.eigvalsh(K)

        # All eigenvalues should be >= -1e-5 (numerical tolerance)
        assert (eigenvalues >= -1e-5).all()

    def test_arccosine_order2_positive_definite(self):
        """Kernel matrix should be positive semi-definite."""
        from rielbo.kernels import ArcCosineKernelOrder2

        kernel = ArcCosineKernelOrder2()
        x = F.normalize(torch.randn(20, 16), p=2, dim=-1)

        K = kernel(x, x).evaluate()
        eigenvalues = torch.linalg.eigvalsh(K)

        assert (eigenvalues >= -1e-5).all()

    def test_order2_smoother_than_order0(self):
        """Order 2 should produce higher similarity for nearby points."""
        from rielbo.kernels import ArcCosineKernel, ArcCosineKernelOrder2

        kernel0 = ArcCosineKernel()
        kernel2 = ArcCosineKernelOrder2()

        # Create two nearby points
        x = F.normalize(torch.randn(1, 16), p=2, dim=-1)
        noise = F.normalize(torch.randn(1, 16), p=2, dim=-1) * 0.1
        y = F.normalize(x + noise, p=2, dim=-1)

        k0 = kernel0(x, y).evaluate().item()
        k2 = kernel2(x, y).evaluate().item()

        # Order 2 should give higher similarity for nearby points
        # (smoother means slower decay)
        assert k2 >= k0 - 0.01  # Order 2 should be at least as smooth


class TestProductSphereKernel:
    """Test Product Sphere kernel."""

    def test_product_self_similarity(self):
        """k(x, x) should be close to 1 (within clamping tolerance)."""
        from rielbo.kernels import ProductSphereKernel

        kernel = ProductSphereKernel(n_spheres=4, order=0)
        x = F.normalize(torch.randn(10, 16), p=2, dim=-1)

        k_diag = kernel(x, x, diag=True)

        # Due to clamping, product of 4 values each ~0.9995 is ~0.998
        assert torch.allclose(k_diag, torch.ones_like(k_diag), atol=5e-3)

    def test_product_symmetry(self):
        """K(x, y) should be symmetric."""
        from rielbo.kernels import ProductSphereKernel

        kernel = ProductSphereKernel(n_spheres=4, order=0)
        x = F.normalize(torch.randn(5, 16), p=2, dim=-1)
        y = F.normalize(torch.randn(7, 16), p=2, dim=-1)

        K_xy = kernel(x, y).evaluate()
        K_yx = kernel(y, x).evaluate()

        assert torch.allclose(K_xy, K_yx.T, atol=1e-5)

    def test_product_dimension_check(self):
        """Should raise error if dimension not divisible by n_spheres."""
        from rielbo.kernels import ProductSphereKernel

        kernel = ProductSphereKernel(n_spheres=4, order=0)
        x = F.normalize(torch.randn(5, 15), p=2, dim=-1)  # 15 not divisible by 4

        with pytest.raises(ValueError):
            kernel(x, x).evaluate()  # Force evaluation to trigger the check


class TestSphericalWhitening:
    """Test Spherical Whitening transform."""

    def test_mean_at_north_pole(self):
        """After whitening, mean direction should be at north pole."""
        from rielbo.spherical_transforms import SphericalWhitening

        # Create clustered data
        center = F.normalize(torch.randn(16), p=2, dim=-1)
        noise = torch.randn(100, 16) * 0.3
        X = F.normalize(center + noise, p=2, dim=-1)

        whitening = SphericalWhitening(device="cpu")
        X_white = whitening.fit_transform(X)

        # Mean of whitened data should be close to [1, 0, 0, ...]
        mean_dir = F.normalize(X_white.mean(dim=0), p=2, dim=-1)
        north_pole = torch.zeros(16)
        north_pole[0] = 1.0

        # Cosine similarity should be close to 1
        cos_sim = (mean_dir * north_pole).sum()
        assert cos_sim > 0.95

    def test_inverse_recovers_original(self):
        """inverse_transform(transform(x)) should recover x."""
        from rielbo.spherical_transforms import SphericalWhitening

        X = F.normalize(torch.randn(50, 16), p=2, dim=-1)

        whitening = SphericalWhitening(device="cpu")
        whitening.fit(X)

        X_white = whitening.transform(X)
        X_recovered = whitening.inverse_transform(X_white)

        # Should recover original (up to numerical precision)
        assert torch.allclose(X, X_recovered, atol=1e-5)

    def test_preserves_sphere(self):
        """Transform should keep points on unit sphere."""
        from rielbo.spherical_transforms import SphericalWhitening

        X = F.normalize(torch.randn(50, 16), p=2, dim=-1)

        whitening = SphericalWhitening(device="cpu")
        X_white = whitening.fit_transform(X)

        norms = X_white.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestGeodesicTrustRegion:
    """Test Geodesic Trust Region sampling."""

    def test_samples_on_sphere(self):
        """All samples should be on unit sphere."""
        from rielbo.spherical_transforms import GeodesicTrustRegion

        tr = GeodesicTrustRegion(max_angle=0.5, device="cpu")
        center = F.normalize(torch.randn(16), p=2, dim=-1)

        samples = tr.sample(center, n_samples=100)

        norms = samples.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_samples_within_angle(self):
        """Local samples should be within max_angle of center."""
        from rielbo.spherical_transforms import GeodesicTrustRegion, geodesic_distance

        max_angle = 0.3
        tr = GeodesicTrustRegion(max_angle=max_angle, global_fraction=0.0, device="cpu")
        center = F.normalize(torch.randn(16), p=2, dim=-1)

        samples = tr.sample(center, n_samples=100)

        distances = geodesic_distance(samples, center.unsqueeze(0).expand_as(samples))

        # All samples should be within max_angle (with small tolerance)
        assert (distances <= max_angle + 1e-5).all()

    def test_global_samples_distributed(self):
        """With global_fraction=1.0, samples should be uniformly distributed."""
        torch.manual_seed(42)
        from rielbo.spherical_transforms import GeodesicTrustRegion

        tr = GeodesicTrustRegion(max_angle=0.5, global_fraction=1.0, device="cpu")
        center = F.normalize(torch.randn(16), p=2, dim=-1)

        samples = tr.sample(center, n_samples=1000)

        # Mean should be close to zero (uniform distribution)
        mean_dir = samples.mean(dim=0)
        assert mean_dir.norm() < 0.2  # Should be small for uniform

    def test_concentrated_samples(self):
        """Concentrated sampling should cluster near center."""
        torch.manual_seed(42)
        from rielbo.spherical_transforms import GeodesicTrustRegion, geodesic_distance

        tr = GeodesicTrustRegion(max_angle=0.5, device="cpu")
        center = F.normalize(torch.randn(16), p=2, dim=-1)

        samples_low = tr.sample_concentrated(center, n_samples=100, concentration=0.5)
        samples_high = tr.sample_concentrated(center, n_samples=100, concentration=5.0)

        dist_low = geodesic_distance(samples_low, center.unsqueeze(0).expand_as(samples_low)).mean()
        dist_high = geodesic_distance(samples_high, center.unsqueeze(0).expand_as(samples_high)).mean()

        # Higher concentration should mean smaller distances
        assert dist_high < dist_low


class TestNormDistribution:
    """Test Norm Distribution modeling."""

    def test_gaussian_fit(self):
        """Gaussian should capture mean and std."""
        torch.manual_seed(42)
        from rielbo.norm_distribution import NormDistribution

        norms = torch.randn(1000) * 2 + 10  # mean=10, std=2

        dist = NormDistribution(method="gaussian", device="cpu")
        dist.fit(norms)

        assert abs(dist.mean - 10) < 0.2
        assert abs(dist.std - 2) < 0.2

    def test_samples_in_range(self):
        """Samples should be in reasonable range."""
        torch.manual_seed(42)
        from rielbo.norm_distribution import NormDistribution

        norms = torch.rand(1000) * 5 + 5  # range [5, 10]

        dist = NormDistribution(method="gaussian", device="cpu")
        dist.fit(norms)

        samples = dist.sample(1000)

        # Most samples should be in [3, 12] (allowing some spread)
        assert (samples > 3).float().mean() > 0.9
        assert (samples < 12).float().mean() > 0.9

    def test_histogram_samples(self):
        """Histogram should produce samples matching distribution."""
        torch.manual_seed(42)
        from rielbo.norm_distribution import NormDistribution

        # Bimodal distribution
        norms = torch.cat([
            torch.randn(500) * 0.5 + 5,
            torch.randn(500) * 0.5 + 10,
        ])

        dist = NormDistribution(method="histogram", n_bins=50, device="cpu")
        dist.fit(norms)

        samples = dist.sample(1000)

        # Should have samples around both modes
        near_5 = ((samples > 4) & (samples < 6)).float().mean()
        near_10 = ((samples > 9) & (samples < 11)).float().mean()

        assert near_5 > 0.2
        assert near_10 > 0.2


class TestV2Config:
    """Test V2Config presets."""

    def test_baseline_preset(self):
        """Baseline should have no improvements."""
        from rielbo.subspace_bo_v2 import V2Config

        config = V2Config.from_preset("baseline")

        assert config.kernel_order == 0
        assert not config.whitening
        assert not config.geodesic_tr
        assert not config.adaptive_dim
        assert not config.prob_norm
        assert not config.product_space

    def test_full_preset(self):
        """Full should have all improvements."""
        from rielbo.subspace_bo_v2 import V2Config

        config = V2Config.from_preset("full")

        assert config.kernel_order == 2
        assert config.whitening
        assert config.geodesic_tr
        assert config.adaptive_dim
        assert config.prob_norm

    def test_geodesic_preset(self):
        """Geodesic preset should have geodesic_tr AND adaptive_tr enabled."""
        from rielbo.subspace_bo_v2 import V2Config

        config = V2Config.from_preset("geodesic")
        assert config.geodesic_tr
        assert config.adaptive_tr
        assert not config.whitening
        assert config.kernel_order == 0

    def test_baseline_has_no_adaptive_tr(self):
        """Baseline should not have adaptive_tr."""
        from rielbo.subspace_bo_v2 import V2Config

        config = V2Config.from_preset("baseline")
        assert not config.adaptive_tr

    def test_unknown_preset_raises(self):
        """Unknown preset should raise error."""
        from rielbo.subspace_bo_v2 import V2Config

        with pytest.raises(ValueError):
            V2Config.from_preset("unknown_preset")


class TestAdaptiveTrustRegion:
    """Test adaptive trust region logic (TuRBO-style)."""

    def _make_optimizer_with_adaptive_tr(self):
        """Create a minimal optimizer with adaptive TR for testing."""
        from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config
        from unittest.mock import MagicMock

        config = V2Config(geodesic_tr=True, adaptive_tr=True)
        opt = object.__new__(SphericalSubspaceBOv2)
        opt.config = config
        opt.verbose = False
        opt.seed = 42
        opt.n_restarts = 0
        opt.tr_length = config.tr_init
        opt._tr_success_count = 0
        opt._tr_fail_count = 0
        opt.train_U = None
        opt.train_V = None
        opt._init_projection = MagicMock()
        opt._fit_gp = MagicMock()
        return opt

    def test_tr_grows_on_consecutive_successes(self):
        """TR should grow after tr_success_tol consecutive improvements."""
        opt = self._make_optimizer_with_adaptive_tr()
        initial_tr = opt.tr_length

        for _ in range(opt.config.tr_success_tol):
            opt._update_trust_region(improved=True)

        assert opt.tr_length > initial_tr
        expected = initial_tr * opt.config.tr_grow_factor
        assert abs(opt.tr_length - expected) < 1e-8

    def test_tr_shrinks_on_consecutive_failures(self):
        """TR should shrink after tr_fail_tol consecutive failures."""
        opt = self._make_optimizer_with_adaptive_tr()
        initial_tr = opt.tr_length

        for _ in range(opt.config.tr_fail_tol):
            opt._update_trust_region(improved=False)

        expected = initial_tr * opt.config.tr_shrink_factor
        assert abs(opt.tr_length - expected) < 1e-8

    def test_tr_capped_at_max(self):
        """TR length should never exceed tr_max."""
        opt = self._make_optimizer_with_adaptive_tr()
        opt.tr_length = opt.config.tr_max - 0.01

        for _ in range(opt.config.tr_success_tol):
            opt._update_trust_region(improved=True)

        assert opt.tr_length <= opt.config.tr_max

    def test_restart_triggered_when_tr_below_min(self):
        """_restart_subspace should be called when TR drops below tr_min."""
        opt = self._make_optimizer_with_adaptive_tr()
        # Set TR so that after one shrink (× 0.5) it falls below tr_min
        opt.tr_length = opt.config.tr_min * (1.0 / opt.config.tr_shrink_factor) - 1e-6

        # This many failures should trigger a shrink that goes below tr_min
        for _ in range(opt.config.tr_fail_tol):
            opt._update_trust_region(improved=False)

        # Should have restarted — tr_length reset to tr_init
        assert opt.n_restarts == 1
        assert abs(opt.tr_length - opt.config.tr_init) < 1e-8

    def test_max_restarts_stops_restarting(self):
        """After max_restarts, TR should reset to tr_init without restarting."""
        opt = self._make_optimizer_with_adaptive_tr()
        opt.n_restarts = opt.config.max_restarts
        opt.tr_length = opt.config.tr_min * 0.5  # Below minimum

        opt._restart_subspace()

        # Should NOT have incremented n_restarts
        assert opt.n_restarts == opt.config.max_restarts
        # But TR should be reset
        assert abs(opt.tr_length - opt.config.tr_init) < 1e-8

    def test_success_resets_failure_counter(self):
        """A success should reset the failure counter to 0."""
        opt = self._make_optimizer_with_adaptive_tr()
        opt._tr_fail_count = 5

        opt._update_trust_region(improved=True)

        assert opt._tr_fail_count == 0
        assert opt._tr_success_count == 1

    def test_failure_resets_success_counter(self):
        """A failure should reset the success counter to 0."""
        opt = self._make_optimizer_with_adaptive_tr()
        opt._tr_success_count = 2

        opt._update_trust_region(improved=False)

        assert opt._tr_success_count == 0
        assert opt._tr_fail_count == 1

    def test_no_op_when_adaptive_tr_disabled(self):
        """_update_trust_region should be a no-op when adaptive_tr=False."""
        from rielbo.subspace_bo_v2 import V2Config
        opt = self._make_optimizer_with_adaptive_tr()
        opt.config = V2Config.from_preset("baseline")
        original_tr = opt.tr_length

        opt._update_trust_region(improved=True)
        opt._update_trust_region(improved=False)

        assert opt.tr_length == original_tr


class TestKernelFactory:
    """Test kernel factory function."""

    def test_create_arccosine_order0(self):
        """Should create ArcCosine order 0 kernel."""
        from rielbo.kernels import create_kernel, ArcCosineKernel
        import gpytorch

        kernel = create_kernel(kernel_type="arccosine", kernel_order=0)

        assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(kernel.base_kernel, ArcCosineKernel)

    def test_create_arccosine_order2(self):
        """Should create ArcCosine order 2 kernel."""
        from rielbo.kernels import create_kernel, ArcCosineKernelOrder2
        import gpytorch

        kernel = create_kernel(kernel_type="arccosine", kernel_order=2)

        assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(kernel.base_kernel, ArcCosineKernelOrder2)

    def test_create_product_kernel(self):
        """Should create Product Sphere kernel."""
        from rielbo.kernels import create_kernel, ProductSphereKernel
        import gpytorch

        kernel = create_kernel(kernel_type="product", n_spheres=4)

        assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(kernel.base_kernel, ProductSphereKernel)


class TestURTR:
    """Test Uncertainty-Responsive Trust Region (UR-TR)."""

    def _make_optimizer_with_ur_tr(self, **overrides):
        """Create a minimal optimizer with UR-TR for testing."""
        from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config
        from unittest.mock import MagicMock

        config_kwargs = {"geodesic_tr": True, "ur_tr": True}
        config_kwargs.update(overrides)
        config = V2Config(**config_kwargs)
        opt = object.__new__(SphericalSubspaceBOv2)
        opt.config = config
        opt.verbose = False
        opt.seed = 42
        opt.trust_region = 0.8
        opt.input_dim = 256
        opt._current_dim = 16
        opt.device = "cpu"
        opt._ur_radius = config.geodesic_max_angle * 0.8  # 0.4
        opt._ur_collapse_count = 0
        opt._ur_n_rotations = 0
        opt._prev_gp_std = 1.0
        opt.gp = None  # No GP → falls back to absolute thresholds
        opt.train_U = None
        opt.train_V = None
        opt._init_projection = MagicMock()
        opt._fit_gp = MagicMock()
        return opt

    def test_expand_on_low_std(self):
        """UR-TR should expand radius when GP std is below ur_std_low."""
        opt = self._make_optimizer_with_ur_tr()
        initial_radius = opt._ur_radius

        opt._update_ur_tr(gp_std=0.01)  # Below ur_std_low (0.05)

        assert opt._ur_radius > initial_radius
        expected = initial_radius * opt.config.ur_expand_factor
        assert abs(opt._ur_radius - expected) < 1e-8

    def test_shrink_on_high_std(self):
        """UR-TR should shrink radius when GP std is above ur_std_high."""
        opt = self._make_optimizer_with_ur_tr()
        initial_radius = opt._ur_radius

        opt._update_ur_tr(gp_std=0.3)  # Above ur_std_high (0.15)

        assert opt._ur_radius < initial_radius
        expected = initial_radius * opt.config.ur_shrink_factor
        assert abs(opt._ur_radius - expected) < 1e-8

    def test_no_change_in_normal_range(self):
        """UR-TR should keep radius when GP std is in normal range."""
        opt = self._make_optimizer_with_ur_tr()
        initial_radius = opt._ur_radius

        opt._update_ur_tr(gp_std=0.08)  # Between ur_std_low and ur_std_high

        assert abs(opt._ur_radius - initial_radius) < 1e-8

    def test_radius_capped_at_max(self):
        """Radius should not exceed ur_tr_max."""
        opt = self._make_optimizer_with_ur_tr()
        opt._ur_radius = opt.config.ur_tr_max - 0.01

        opt._update_ur_tr(gp_std=0.01)  # trigger expand

        assert opt._ur_radius <= opt.config.ur_tr_max + 1e-8

    def test_radius_capped_at_min(self):
        """Radius should not go below ur_tr_min."""
        opt = self._make_optimizer_with_ur_tr()
        opt._ur_radius = opt.config.ur_tr_min + 0.01

        opt._update_ur_tr(gp_std=0.5)  # trigger shrink

        assert opt._ur_radius >= opt.config.ur_tr_min - 1e-8

    def test_relative_thresholds_with_gp(self):
        """With ur_relative=True and a GP, thresholds scale by noise_std."""
        from unittest.mock import MagicMock

        opt = self._make_optimizer_with_ur_tr(ur_relative=True)
        initial_radius = opt._ur_radius

        # Mock GP with noise_var = 4.0 → noise_std = 2.0
        mock_gp = MagicMock()
        mock_gp.likelihood.noise.item.return_value = 4.0
        opt.gp = mock_gp

        # Effective ur_std_low = 0.05 * 2.0 = 0.10
        # gp_std=0.09 is below effective threshold → should expand
        opt._update_ur_tr(gp_std=0.09)

        assert opt._ur_radius > initial_radius

    def test_relative_thresholds_no_expand_above_scaled(self):
        """With ur_relative, values above scaled threshold should not expand."""
        from unittest.mock import MagicMock

        opt = self._make_optimizer_with_ur_tr(ur_relative=True)
        initial_radius = opt._ur_radius

        # Mock GP with noise_var = 4.0 → noise_std = 2.0
        mock_gp = MagicMock()
        mock_gp.likelihood.noise.item.return_value = 4.0
        opt.gp = mock_gp

        # Effective ur_std_low = 0.05 * 2.0 = 0.10
        # gp_std=0.12 is above effective low but below effective high (0.15*2=0.30)
        opt._update_ur_tr(gp_std=0.12)

        assert abs(opt._ur_radius - initial_radius) < 1e-8  # No change

    def test_absolute_thresholds_when_relative_disabled(self):
        """With ur_relative=False, thresholds are used directly."""
        opt = self._make_optimizer_with_ur_tr(ur_relative=False)
        initial_radius = opt._ur_radius

        opt._update_ur_tr(gp_std=0.01)  # Below absolute ur_std_low (0.05)

        assert opt._ur_radius > initial_radius

    def test_collapse_count_increments(self):
        """Collapse count should increment when gp_std < ur_std_collapse."""
        opt = self._make_optimizer_with_ur_tr()
        opt._ur_collapse_count = 0

        opt._update_ur_tr(gp_std=0.001)  # Below ur_std_collapse (0.005)

        assert opt._ur_collapse_count == 1

    def test_collapse_count_resets_on_normal_std(self):
        """Collapse count should reset when gp_std is in normal range."""
        opt = self._make_optimizer_with_ur_tr()
        opt._ur_collapse_count = 5

        opt._update_ur_tr(gp_std=0.08)  # Normal range

        assert opt._ur_collapse_count == 0

    def test_rotation_triggers_on_sustained_collapse(self):
        """Rotation should trigger after ur_collapse_patience consecutive collapses."""
        opt = self._make_optimizer_with_ur_tr(ur_collapse_patience=3)
        opt._ur_collapse_count = 0
        opt._ur_n_rotations = 0

        for _ in range(3):
            opt._update_ur_tr(gp_std=0.001)

        assert opt._ur_n_rotations == 1
        assert opt._ur_collapse_count == 0  # Reset after rotation

    def test_rotation_resets_radius(self):
        """Rotation should reset radius to initial value."""
        opt = self._make_optimizer_with_ur_tr(ur_collapse_patience=1)
        opt._ur_radius = 0.01  # Very small
        opt._ur_collapse_count = 0
        opt._ur_n_rotations = 0

        opt._update_ur_tr(gp_std=0.001)  # Trigger collapse + rotation

        expected_radius = opt.config.geodesic_max_angle * opt.trust_region
        assert abs(opt._ur_radius - expected_radius) < 1e-8

    def test_max_rotations_stops_rotating(self):
        """After max_rotations, no more rotations should occur."""
        opt = self._make_optimizer_with_ur_tr(ur_collapse_patience=1, ur_max_rotations=2)
        opt._ur_n_rotations = 2  # Already at max

        opt._update_ur_tr(gp_std=0.001)  # Would trigger rotation

        assert opt._ur_n_rotations == 2  # Didn't increment

    def test_no_op_when_disabled(self):
        """_update_ur_tr should be no-op when ur_tr=False."""
        from rielbo.subspace_bo_v2 import V2Config
        opt = self._make_optimizer_with_ur_tr()
        opt.config = V2Config()  # ur_tr=False by default
        initial_radius = opt._ur_radius

        opt._update_ur_tr(gp_std=0.001)

        assert abs(opt._ur_radius - initial_radius) < 1e-8

    def test_ur_tr_preset(self):
        """ur_tr preset should have geodesic_tr and ur_tr enabled."""
        from rielbo.subspace_bo_v2 import V2Config
        config = V2Config.from_preset("ur_tr")
        assert config.geodesic_tr
        assert config.ur_tr
        assert not config.adaptive_tr
        assert not config.lass

    def test_lass_ur_preset(self):
        """lass_ur preset should have geodesic_tr, ur_tr, and lass."""
        from rielbo.subspace_bo_v2 import V2Config
        config = V2Config.from_preset("lass_ur")
        assert config.geodesic_tr
        assert config.ur_tr
        assert config.lass
        assert not config.adaptive_tr

    def test_explore_preset(self):
        """explore preset should have geodesic_tr, ur_tr, lass, and acqf_schedule."""
        from rielbo.subspace_bo_v2 import V2Config
        config = V2Config.from_preset("explore")
        assert config.geodesic_tr
        assert config.ur_tr
        assert config.lass
        assert config.acqf_schedule


class TestLASS:
    """Test Look-Ahead Subspace Selection (LASS)."""

    def test_lass_selects_projection(self):
        """LASS should pick a projection and set self.A."""
        from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config
        from unittest.mock import MagicMock

        config = V2Config(lass=True, lass_n_candidates=5)
        opt = object.__new__(SphericalSubspaceBOv2)
        opt.config = config
        opt.verbose = False
        opt.seed = 42
        opt.input_dim = 32
        opt._current_dim = 8
        opt.device = "cpu"

        # Create synthetic training data
        opt.train_U = F.normalize(torch.randn(50, 32), p=2, dim=-1)
        opt.train_Y = torch.randn(50)

        # Initialize A first
        torch.manual_seed(42)
        A_raw = torch.randn(32, 8)
        opt.A, _ = torch.linalg.qr(A_raw)
        original_A = opt.A.clone()

        opt._select_best_projection()

        # A should have been updated (different from original with high probability)
        # Note: could be the same by chance, but with 5 candidates it's very unlikely
        assert opt.A.shape == (32, 8)
        # Check it's orthonormal
        ATA = opt.A.T @ opt.A
        assert torch.allclose(ATA, torch.eye(8), atol=1e-5)

    def test_lass_different_from_default(self):
        """LASS with many candidates should likely pick a different projection."""
        from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config
        torch.manual_seed(123)

        config = V2Config(lass=True, lass_n_candidates=20)
        opt = object.__new__(SphericalSubspaceBOv2)
        opt.config = config
        opt.verbose = False
        opt.seed = 42
        opt.input_dim = 32
        opt._current_dim = 8
        opt.device = "cpu"

        opt.train_U = F.normalize(torch.randn(50, 32), p=2, dim=-1)
        opt.train_Y = torch.randn(50)

        # Default projection
        torch.manual_seed(42)
        A_raw = torch.randn(32, 8)
        opt.A, _ = torch.linalg.qr(A_raw)
        default_A = opt.A.clone()

        opt._select_best_projection()

        # With 20 candidates and seed 42 used for default, LASS should pick differently
        # (seed 42 * 137 = different from seed 42)
        diff = (opt.A - default_A).abs().max().item()
        assert diff > 0.01  # Should be different

    def test_lass_config_default(self):
        """LASS should default to 50 candidates."""
        from rielbo.subspace_bo_v2 import V2Config
        config = V2Config(lass=True)
        assert config.lass_n_candidates == 50


class TestAcqfSchedule:
    """Test acquisition function schedule."""

    def _make_optimizer_with_acqf_schedule(self, **overrides):
        """Create a minimal optimizer with acqf_schedule for testing."""
        from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config

        config_kwargs = {
            "geodesic_tr": True, "ur_tr": True, "acqf_schedule": True,
        }
        config_kwargs.update(overrides)
        config = V2Config(**config_kwargs)
        opt = object.__new__(SphericalSubspaceBOv2)
        opt.config = config
        opt.acqf = "ts"
        opt.ucb_beta = 2.0
        opt._prev_gp_std = 1.0
        return opt

    def test_default_acqf_when_normal(self):
        """Should return default acqf when GP std is in normal range."""
        opt = self._make_optimizer_with_acqf_schedule()
        opt._prev_gp_std = 0.08  # Normal range

        acqf, beta = opt._get_effective_acqf()

        assert acqf == "ts"
        assert beta == 2.0

    def test_ucb_high_beta_when_collapsing(self):
        """Should switch to UCB with high beta when GP std is low."""
        opt = self._make_optimizer_with_acqf_schedule()
        opt._prev_gp_std = 0.01  # Below ur_std_low

        acqf, beta = opt._get_effective_acqf()

        assert acqf == "ucb"
        assert beta == opt.config.acqf_ucb_beta_high

    def test_ucb_low_beta_when_informative(self):
        """Should use UCB with low beta when GP std is high."""
        opt = self._make_optimizer_with_acqf_schedule()
        opt._prev_gp_std = 0.3  # Above ur_std_high

        acqf, beta = opt._get_effective_acqf()

        assert acqf == "ucb"
        assert beta == opt.config.acqf_ucb_beta_low

    def test_no_schedule_returns_default(self):
        """With acqf_schedule=False, should always return default acqf."""
        opt = self._make_optimizer_with_acqf_schedule(acqf_schedule=False)
        opt._prev_gp_std = 0.001  # Very low

        acqf, beta = opt._get_effective_acqf()

        assert acqf == "ts"
        assert beta == 2.0


class TestExponentialMap:
    """Test exponential and logarithmic maps."""

    def test_exp_map_stays_on_sphere(self):
        """Exponential map should produce points on sphere."""
        from rielbo.spherical_transforms import exponential_map

        base = F.normalize(torch.randn(10, 16), p=2, dim=-1)
        # Create orthogonal tangent vectors
        tangent = torch.randn(10, 16)
        proj = (tangent * base).sum(dim=-1, keepdim=True) * base
        tangent = tangent - proj

        result = exponential_map(base, tangent, t=0.5)

        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_log_map_inverse_of_exp(self):
        """log_map should be inverse of exp_map for small tangents."""
        from rielbo.spherical_transforms import exponential_map, logarithmic_map

        base = F.normalize(torch.randn(10, 16), p=2, dim=-1)
        tangent = torch.randn(10, 16) * 0.3
        proj = (tangent * base).sum(dim=-1, keepdim=True) * base
        tangent = tangent - proj

        target = exponential_map(base, tangent, t=1.0)
        recovered_tangent = logarithmic_map(base, target)

        # Should recover tangent (direction and magnitude)
        # Note: tangent is orthogonal to base, so we compare directly
        assert torch.allclose(tangent, recovered_tangent, atol=1e-4)


class TestMultiSubspacePortfolio:
    """Test multi-subspace portfolio (TuRBO-M style)."""

    def _make_optimizer_with_portfolio(self, n_subspaces=3, **overrides):
        """Create a minimal optimizer with multi-subspace portfolio."""
        from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config
        from unittest.mock import MagicMock

        config_kwargs = {
            "geodesic_tr": True, "ur_tr": True, "lass": False,
            "multi_subspace": True, "n_subspaces": n_subspaces,
            "subspace_stale_patience": 10,
        }
        config_kwargs.update(overrides)
        config = V2Config(**config_kwargs)
        opt = object.__new__(SphericalSubspaceBOv2)
        opt.config = config
        opt.verbose = False
        opt.seed = 42
        opt.trust_region = 0.8
        opt.input_dim = 32
        opt._current_dim = 8
        opt.device = "cpu"
        opt._ur_radius = config.geodesic_max_angle * 0.8
        opt._ur_collapse_count = 0
        opt._ur_n_rotations = 0
        opt._prev_gp_std = 1.0
        opt.gp = None
        opt.train_U = F.normalize(torch.randn(50, 32), p=2, dim=-1)
        opt.train_V = None
        opt.train_Y = torch.randn(50)
        opt.best_score = opt.train_Y.max().item()
        opt.whitening = None

        # Initialize projection
        torch.manual_seed(42)
        A_raw = torch.randn(32, 8)
        opt.A, _ = torch.linalg.qr(A_raw)

        opt._fit_gp = MagicMock()
        return opt

    def test_portfolio_preset(self):
        """portfolio preset should have multi_subspace enabled."""
        from rielbo.subspace_bo_v2 import V2Config
        config = V2Config.from_preset("portfolio")
        assert config.multi_subspace
        assert config.n_subspaces == 5
        assert config.geodesic_tr
        assert config.ur_tr
        assert config.lass

    def test_init_subspaces_creates_K(self):
        """_init_subspaces should create K subspaces."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()

        assert len(opt._subspaces) == 3
        assert opt._active_subspace == 0
        assert opt._total_bandit_steps == 0

    def test_subspaces_have_distinct_projections(self):
        """Each subspace should have a different projection matrix."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()

        A0 = opt._subspaces[0]["A"]
        A1 = opt._subspaces[1]["A"]
        A2 = opt._subspaces[2]["A"]

        # Projections should differ
        assert (A0 - A1).abs().max() > 0.01
        assert (A1 - A2).abs().max() > 0.01

    def test_subspace0_uses_current_projection(self):
        """Subspace 0 should use the current (possibly LASS-selected) projection."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        original_A = opt.A.clone()
        opt._init_subspaces()

        assert torch.allclose(opt._subspaces[0]["A"], original_A)

    def test_subspaces_are_orthonormal(self):
        """All subspace projections should be orthonormal."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=5)
        opt._init_subspaces()

        for k, s in enumerate(opt._subspaces):
            ATA = s["A"].T @ s["A"]
            assert torch.allclose(ATA, torch.eye(8), atol=1e-5), f"Subspace {k} not orthonormal"

    def test_bandit_selects_unexplored(self):
        """UCB bandit should prefer unexplored subspaces."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()

        # Simulate: subspace 0 explored many times, 1 and 2 never
        opt._subspaces[0]["n_evals"] = 100
        opt._subspaces[0]["total_reward"] = 5.0
        opt._total_bandit_steps = 100

        k = opt._select_subspace_bandit()
        # Should pick 1 or 2 (unexplored, high UCB exploration bonus)
        assert k in [1, 2]

    def test_bandit_selects_best_reward(self):
        """With equal exploration, UCB should prefer higher reward."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()
        opt._total_bandit_steps = 300

        for k in range(3):
            opt._subspaces[k]["n_evals"] = 100

        opt._subspaces[0]["total_reward"] = 10.0
        opt._subspaces[1]["total_reward"] = 2.0
        opt._subspaces[2]["total_reward"] = 5.0

        k = opt._select_subspace_bandit()
        assert k == 0

    def test_switch_subspace_changes_A(self):
        """Switching subspace should swap projection matrix."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()

        A1_expected = opt._subspaces[1]["A"].clone()
        opt._switch_to_subspace(1)

        assert torch.allclose(opt.A, A1_expected)
        assert opt._active_subspace == 1

    def test_switch_subspace_refits_gp(self):
        """Switching to a different subspace should trigger GP refit."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()
        opt._total_bandit_steps = 1  # Not first step
        opt._fit_gp.reset_mock()

        opt._switch_to_subspace(1)
        opt._fit_gp.assert_called_once()

    def test_switch_same_subspace_no_refit(self):
        """Switching to the same subspace should NOT refit GP."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()
        opt._total_bandit_steps = 1
        opt._active_subspace = 1
        opt._fit_gp.reset_mock()

        opt._switch_to_subspace(1)
        opt._fit_gp.assert_not_called()

    def test_update_stats_tracks_improvement(self):
        """_update_subspace_stats should track improvements."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()

        opt._update_subspace_stats(0, improved=True)
        assert opt._subspaces[0]["n_evals"] == 1
        assert opt._subspaces[0]["n_success"] == 1
        assert opt._subspaces[0]["total_reward"] == 1.0
        assert opt._subspaces[0]["n_consec_fail"] == 0

    def test_update_stats_tracks_failure(self):
        """_update_subspace_stats should track consecutive failures."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()

        for _ in range(5):
            opt._update_subspace_stats(0, improved=False)

        assert opt._subspaces[0]["n_evals"] == 5
        assert opt._subspaces[0]["n_success"] == 0
        assert opt._subspaces[0]["n_consec_fail"] == 5

    def test_stale_subspace_replaced(self):
        """Subspace should be replaced after subspace_stale_patience failures."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()
        old_A = opt._subspaces[1]["A"].clone()

        # Simulate stale_patience consecutive failures on subspace 1
        for _ in range(opt.config.subspace_stale_patience):
            opt._update_subspace_stats(1, improved=False)

        # Should have been replaced
        assert opt._subspaces[1]["n_evals"] == 0
        assert opt._subspaces[1]["n_consec_fail"] == 0
        assert (opt._subspaces[1]["A"] - old_A).abs().max() > 0.01

    def test_replacement_resets_ur_tr_state(self):
        """Replaced subspace should have fresh UR-TR state."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()
        init_radius = opt.config.geodesic_max_angle * opt.trust_region

        # Mess up UR-TR state then replace
        opt._subspaces[0]["ur_radius"] = 0.01
        opt._subspaces[0]["ur_collapse_count"] = 5
        opt._subspaces[0]["ur_n_rotations"] = 3
        opt._active_subspace = 0
        opt._replace_subspace(0)

        assert opt._subspaces[0]["ur_radius"] == init_radius
        assert opt._subspaces[0]["ur_collapse_count"] == 0
        assert opt._subspaces[0]["ur_n_rotations"] == 0

    def test_improvement_resets_consec_fail(self):
        """An improvement should reset consecutive failure counter."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()

        for _ in range(5):
            opt._update_subspace_stats(0, improved=False)
        assert opt._subspaces[0]["n_consec_fail"] == 5

        opt._update_subspace_stats(0, improved=True)
        assert opt._subspaces[0]["n_consec_fail"] == 0

    def test_saves_ur_tr_state_per_subspace(self):
        """UR-TR state should be saved per subspace when switching."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()

        # Modify UR-TR state while in subspace 0
        opt._ur_radius = 0.123
        opt._ur_collapse_count = 7
        opt._ur_n_rotations = 2
        opt._update_subspace_stats(0, improved=False)

        # State should be saved back
        assert opt._subspaces[0]["ur_radius"] == 0.123
        assert opt._subspaces[0]["ur_collapse_count"] == 7
        assert opt._subspaces[0]["ur_n_rotations"] == 2

    def test_switch_restores_ur_tr_state(self):
        """Switching to subspace should restore its full UR-TR state."""
        opt = self._make_optimizer_with_portfolio(n_subspaces=3)
        opt._init_subspaces()

        # Set unique UR-TR state for subspace 1
        opt._subspaces[1]["ur_radius"] = 0.777
        opt._subspaces[1]["ur_collapse_count"] = 3
        opt._subspaces[1]["ur_n_rotations"] = 1
        opt._total_bandit_steps = 1

        opt._switch_to_subspace(1)

        assert opt._ur_radius == 0.777
        assert opt._ur_collapse_count == 3
        assert opt._ur_n_rotations == 1
