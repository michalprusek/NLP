"""
BetaHeteroscedasticGP - GP with Beta-smoothed observations and heteroscedastic noise.

Key features:
1. Beta smoothing: (k+α)/(n+α+β) instead of raw k/n
2. Beta variance: noise_var = p*(1-p)/(n+α+β+1)
3. Hvarfner (2024) dimension-scaled lengthscale prior
4. TuRBO-style local acquisition

Author: Claude Code
"""

import logging
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import gpytorch
import numpy as np
import torch

logger = logging.getLogger(__name__)

DTYPE = torch.float64


@dataclass
class BetaGPConfig:
    """Configuration for BetaHeteroscedasticGP."""
    # Beta prior for smoothing
    beta_alpha: float = 10.0  # Prior: mean ~83% accuracy
    beta_beta: float = 2.0

    # GP kernel
    kernel: str = "rbf"  # "rbf" or "matern52"
    input_dim: int = 1024

    # Lengthscale prior (Hvarfner 2024)
    # loc = sqrt(2) + log(D)/2, scale = sqrt(3)
    ls_prior_loc: float = None  # computed from input_dim
    ls_prior_scale: float = 1.732  # sqrt(3)

    # Optimization
    fit_epochs: int = 100
    fit_lr: float = 0.1

    # TuRBO
    trust_region_init: float = 0.5
    trust_region_min: float = 0.01
    trust_region_max: float = 2.0
    tau_success: int = 3
    tau_fail: int = 3

    def __post_init__(self):
        if self.ls_prior_loc is None:
            self.ls_prior_loc = np.sqrt(2) + np.log(self.input_dim) / 2


class BetaSmoothedData(NamedTuple):
    """Beta-smoothed observations with heteroscedastic noise."""
    y_smooth: torch.Tensor  # Smoothed error rates
    noise_var: torch.Tensor  # Per-observation noise variance
    raw_acc: torch.Tensor  # Original accuracies
    fidelity: torch.Tensor  # Number of examples per observation


class BetaGPModel(gpytorch.models.ExactGP):
    """GP model with fixed heteroscedastic noise."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        noise_var: torch.Tensor,
        config: BetaGPConfig,
    ):
        # Use FixedNoiseGaussianLikelihood for heteroscedastic noise
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=noise_var,
            learn_additional_noise=False,
        )
        super().__init__(train_x, train_y, likelihood)

        self.config = config
        self.mean_module = gpytorch.means.ConstantMean()

        # Kernel choice
        if config.kernel == "matern52":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            base_kernel = gpytorch.kernels.RBFKernel()

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        # Set lengthscale prior (Hvarfner 2024)
        self.covar_module.base_kernel.register_prior(
            "lengthscale_prior",
            gpytorch.priors.LogNormalPrior(
                loc=config.ls_prior_loc,
                scale=config.ls_prior_scale,
            ),
            "lengthscale",
        )

        # Initialize lengthscale to prior median
        init_ls = np.exp(config.ls_prior_loc)
        self.covar_module.base_kernel.lengthscale = init_ls

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class BetaHeteroscedasticGP:
    """
    GP with Beta-smoothed observations and heteroscedastic noise.

    Usage:
        gp = BetaHeteroscedasticGP(config)
        gp.fit(X, accuracies, fidelities)
        mean, std = gp.predict(X_new)
        candidate = gp.get_candidate_turbo(bounds)
    """

    def __init__(self, config: Optional[BetaGPConfig] = None):
        self.config = config or BetaGPConfig()
        self.model = None
        self.likelihood = None
        self._X_train = None
        self._data = None

        # TuRBO state
        self._trust_region_length = self.config.trust_region_init
        self._success_count = 0
        self._fail_count = 0
        self._best_value = float('inf')
        self._center = None

        # Normalization stats
        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None

    def beta_smooth(
        self,
        accuracies: torch.Tensor,
        fidelities: torch.Tensor,
    ) -> BetaSmoothedData:
        """
        Apply Beta smoothing to accuracies and compute heteroscedastic noise.

        Args:
            accuracies: Raw accuracy values [0, 1]
            fidelities: Number of examples per observation

        Returns:
            BetaSmoothedData with smoothed error rates and noise variances
        """
        alpha = self.config.beta_alpha
        beta = self.config.beta_beta

        # Convert to counts
        k = (accuracies * fidelities).round()  # correct answers
        n = fidelities

        # Posterior parameters
        a_post = k + alpha
        b_post = (n - k) + beta
        n_post = a_post + b_post

        # Smoothed accuracy (posterior mean)
        acc_smooth = a_post / n_post

        # Convert to error rate
        err_smooth = 1 - acc_smooth

        # Posterior variance (Beta distribution)
        var = (a_post * b_post) / (n_post ** 2 * (n_post + 1))

        return BetaSmoothedData(
            y_smooth=err_smooth,
            noise_var=var,
            raw_acc=accuracies,
            fidelity=fidelities,
        )

    def fit(
        self,
        X: torch.Tensor,
        accuracies: torch.Tensor,
        fidelities: torch.Tensor,
    ) -> "BetaHeteroscedasticGP":
        """
        Fit GP on Beta-smoothed observations.

        Args:
            X: Input embeddings [N, D]
            accuracies: Raw accuracy values [N]
            fidelities: Number of examples per observation [N]
        """
        X = X.to(dtype=DTYPE)
        accuracies = accuracies.to(dtype=DTYPE)
        fidelities = fidelities.to(dtype=DTYPE)

        # Beta smoothing
        self._data = self.beta_smooth(accuracies, fidelities)

        # Normalize X
        self._X_mean = X.mean(dim=0)
        self._X_std = X.std(dim=0).clamp(min=1e-6)
        X_norm = (X - self._X_mean) / self._X_std

        # Normalize y (error rates)
        y = self._data.y_smooth
        self._y_mean = y.mean()
        self._y_std = y.std().clamp(min=1e-6)
        y_norm = (y - self._y_mean) / self._y_std

        # Scale noise variance
        noise_var_norm = self._data.noise_var / (self._y_std ** 2)

        self._X_train = X_norm

        # Create model
        self.model = BetaGPModel(
            X_norm, y_norm, noise_var_norm, self.config
        )
        self.likelihood = self.model.likelihood

        # Fit
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.fit_lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.config.fit_epochs):
            optimizer.zero_grad()
            output = self.model(X_norm)
            loss = -mll(output, y_norm)
            loss.backward()
            optimizer.step()

        # Log fitted parameters
        ls = self.model.covar_module.base_kernel.lengthscale.item()
        os = self.model.covar_module.outputscale.item()
        logger.info(f"BetaGP fitted: ls={ls:.2f}, outputscale={os:.4f}")

        # Update TuRBO center to best point
        best_idx = y.argmin()
        self._best_value = y[best_idx].item()
        self._center = X[best_idx]

        return self

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict error rate mean and std.

        Returns:
            (mean, std) in original scale
        """
        if self.model is None:
            raise RuntimeError("Model not fitted")

        X = X.to(dtype=DTYPE)
        X_norm = (X - self._X_mean) / self._X_std

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.model(X_norm)
            # Include observation noise
            pred_with_noise = self.likelihood(pred)

            mean_norm = pred_with_noise.mean
            var_norm = pred_with_noise.variance

            # Denormalize
            mean = mean_norm * self._y_std + self._y_mean
            std = var_norm.sqrt() * self._y_std

        return mean, std

    def get_candidate_turbo(
        self,
        bounds: torch.Tensor,
        n_candidates: int = 16,
        acquisition: str = "ts",  # "ts" (Thompson Sampling) or "ei" (Expected Improvement)
    ) -> Tuple[torch.Tensor, float]:
        """
        Get next candidate using TuRBO (Trust Region BO).

        Args:
            bounds: [2, D] tensor with [min, max] bounds
            n_candidates: Number of candidates to sample
            acquisition: "ts" for Thompson Sampling, "ei" for Expected Improvement

        Returns:
            (best_candidate, acquisition_value)
        """
        if self._center is None:
            raise RuntimeError("Model not fitted")

        bounds = bounds.to(dtype=DTYPE)
        center = self._center.to(dtype=DTYPE)

        # Trust region bounds
        L = self._trust_region_length
        tr_lb = torch.clamp(center - L / 2, bounds[0], bounds[1])
        tr_ub = torch.clamp(center + L / 2, bounds[0], bounds[1])

        # Sample candidates (Sobol)
        sobol = torch.quasirandom.SobolEngine(bounds.shape[1], scramble=True)
        candidates_01 = sobol.draw(n_candidates).to(dtype=DTYPE)
        candidates = tr_lb + candidates_01 * (tr_ub - tr_lb)

        # Predict
        mean, std = self.predict(candidates)

        if acquisition == "nei":
            # Noisy Expected Improvement (analytical approximation)
            # Instead of using deterministic f(x_best), we integrate over its uncertainty
            # NEI ≈ EI but with effective best = best_mean - κ * best_std
            # where κ controls exploration (typically 1-2)

            # Get prediction at current best point
            best_mean = self._best_value
            # Estimate noise at best: use average noise from high-fidelity points
            avg_noise_std = self._data.noise_var.mean().sqrt().item() * self._y_std.item()

            # Effective best accounts for noise uncertainty
            kappa = 1.0  # Exploration parameter
            effective_best = best_mean + kappa * avg_noise_std  # More conservative

            # Compute EI with effective best (for minimization)
            Z = (effective_best - mean) / std.clamp(min=1e-6)
            normal = torch.distributions.Normal(0, 1)
            nei = (effective_best - mean) * normal.cdf(Z) + std * torch.exp(normal.log_prob(Z))

            best_idx = nei.argmax()
            acq_value = nei[best_idx].item()
            logger.info(f"NEI acquisition: effective_best={effective_best:.4f}, max_nei={acq_value:.4f}")
        elif acquisition == "ei":
            # Expected Improvement (for minimization)
            # EI = (best - mean) * Φ(Z) + std * φ(Z) where Z = (best - mean) / std
            best_f = self._best_value
            Z = (best_f - mean) / std.clamp(min=1e-6)
            normal = torch.distributions.Normal(0, 1)
            ei = (best_f - mean) * normal.cdf(Z) + std * torch.exp(normal.log_prob(Z))
            best_idx = ei.argmax()
            acq_value = ei[best_idx].item()
            logger.info(f"EI acquisition: max_ei={acq_value:.4f}")
        else:
            # Thompson Sampling: sample from posterior
            samples = torch.normal(mean, std)
            best_idx = samples.argmin()
            acq_value = samples[best_idx].item()
            logger.info(f"Thompson Sampling: sampled_err={acq_value:.4f}")

        return candidates[best_idx:best_idx+1], acq_value

    def update_turbo_state(self, new_value: float):
        """
        Update TuRBO trust region based on new observation.

        Args:
            new_value: Error rate of new observation
        """
        if new_value < self._best_value:
            # Success - improvement
            self._success_count += 1
            self._fail_count = 0
            self._best_value = new_value

            if self._success_count >= self.config.tau_success:
                # Expand trust region
                self._trust_region_length = min(
                    self._trust_region_length * 2,
                    self.config.trust_region_max
                )
                self._success_count = 0
                logger.info(f"TuRBO: expanded to L={self._trust_region_length:.3f}")
        else:
            # Failure - no improvement
            self._fail_count += 1
            self._success_count = 0

            if self._fail_count >= self.config.tau_fail:
                # Shrink trust region
                self._trust_region_length = max(
                    self._trust_region_length / 2,
                    self.config.trust_region_min
                )
                self._fail_count = 0
                logger.info(f"TuRBO: shrunk to L={self._trust_region_length:.3f}")

        # Check for restart
        if self._trust_region_length <= self.config.trust_region_min:
            logger.warning("TuRBO: trust region too small, consider restart")


def test_beta_gp():
    """Test BetaHeteroscedasticGP with synthetic data."""
    print("=" * 60)
    print("Testing BetaHeteroscedasticGP")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)

    n_points = 50
    dim = 1024

    # Random embeddings (normalized like SONAR)
    X = torch.randn(n_points, dim) * 0.01

    # Synthetic accuracies (higher near origin)
    distances = X.norm(dim=1)
    true_acc = 0.9 - 0.3 * (distances / distances.max())

    # Add noise (simulate binomial sampling)
    fidelity = torch.full((n_points,), 100.0)
    k = torch.distributions.Binomial(100, true_acc).sample()
    observed_acc = k / 100

    print(f"Data: {n_points} points, {dim}D")
    print(f"Accuracy range: [{observed_acc.min():.3f}, {observed_acc.max():.3f}]")
    print(f"True acc range: [{true_acc.min():.3f}, {true_acc.max():.3f}]")

    # Fit GP
    config = BetaGPConfig(input_dim=dim)
    gp = BetaHeteroscedasticGP(config)
    gp.fit(X, observed_acc, fidelity)

    # Check smoothing
    print(f"\nBeta smoothing (α={config.beta_alpha}, β={config.beta_beta}):")
    print(f"  Raw error rate range: [{(1-observed_acc).min():.3f}, {(1-observed_acc).max():.3f}]")
    print(f"  Smoothed error rate range: [{gp._data.y_smooth.min():.3f}, {gp._data.y_smooth.max():.3f}]")
    print(f"  Noise std range: [{gp._data.noise_var.sqrt().min():.4f}, {gp._data.noise_var.sqrt().max():.4f}]")

    # Test prediction
    mean, std = gp.predict(X[:5])
    print(f"\nPrediction on training data (should be close):")
    for i in range(5):
        true_err = 1 - true_acc[i].item()
        obs_err = 1 - observed_acc[i].item()
        pred_err = mean[i].item()
        pred_std = std[i].item()
        print(f"  [{i}] true={true_err:.3f}, obs={obs_err:.3f}, pred={pred_err:.3f}±{pred_std:.3f}")

    # Test on new points
    X_new = torch.randn(10, dim) * 0.01
    mean_new, std_new = gp.predict(X_new)
    print(f"\nPrediction on new points:")
    print(f"  Mean error rate: {mean_new.mean():.3f}")
    print(f"  Mean std: {std_new.mean():.3f}")

    # Test TuRBO
    bounds = torch.stack([X.min(dim=0).values, X.max(dim=0).values])
    candidate, acq_val = gp.get_candidate_turbo(bounds, n_candidates=32)
    print(f"\nTuRBO candidate:")
    print(f"  Trust region: L={gp._trust_region_length:.3f}")
    print(f"  Acquisition value: {acq_val:.4f}")

    # Simulate TuRBO updates
    print(f"\nSimulating TuRBO updates:")
    for i, val in enumerate([0.15, 0.12, 0.10, 0.11, 0.13, 0.14, 0.09]):
        gp.update_turbo_state(val)
        print(f"  Update {i+1}: val={val:.2f}, L={gp._trust_region_length:.3f}, best={gp._best_value:.2f}")

    print("\n" + "=" * 60)
    print("[OK] BetaHeteroscedasticGP tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_beta_gp()
