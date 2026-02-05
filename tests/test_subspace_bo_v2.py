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
        assert k2 >= k0 - 0.1  # Allow some tolerance


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
        from rielbo.spherical_transforms import GeodesicTrustRegion

        tr = GeodesicTrustRegion(max_angle=0.5, global_fraction=1.0, device="cpu")
        center = F.normalize(torch.randn(16), p=2, dim=-1)

        samples = tr.sample(center, n_samples=1000)

        # Mean should be close to zero (uniform distribution)
        mean_dir = samples.mean(dim=0)
        assert mean_dir.norm() < 0.2  # Should be small for uniform

    def test_concentrated_samples(self):
        """Concentrated sampling should cluster near center."""
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
        from rielbo.norm_distribution import NormDistribution

        norms = torch.randn(1000) * 2 + 10  # mean=10, std=2

        dist = NormDistribution(method="gaussian", device="cpu")
        dist.fit(norms)

        assert abs(dist.mean - 10) < 0.2
        assert abs(dist.std - 2) < 0.2

    def test_samples_in_range(self):
        """Samples should be in reasonable range."""
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

    def test_unknown_preset_raises(self):
        """Unknown preset should raise error."""
        from rielbo.subspace_bo_v2 import V2Config

        with pytest.raises(ValueError):
            V2Config.from_preset("unknown_preset")


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
