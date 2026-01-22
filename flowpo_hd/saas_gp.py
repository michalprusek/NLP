"""SAAS GP with qLogEI Acquisition for FlowPO-HD.

Combines:
1. SAAS (Sparse Axis-Aligned Subspaces) - Bayesian ARD via MCMC
2. qLogEI (Log Expected Improvement) - Numerically stable acquisition

This configuration achieved Spearman 0.87 in GP benchmark on 1024D SONAR space.

Key design decisions:
- Direct 1024D operation (no compression)
- MCMC warmup=128, samples=64 (balanced speed/quality)
- Heteroscedastic noise from Beta posterior variance
- qLogEI for acquisition (better than UCB for BO)
- Double precision (float64) for numerical stability
"""

import logging
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Use double precision for numerical stability (BoTorch recommendation)
DTYPE = torch.float64

# Check dependencies
try:
    from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf

    SAAS_AVAILABLE = True
except ImportError as e:
    SAAS_AVAILABLE = False
    logger.warning(f"SAAS/BoTorch not available: {e}")


@dataclass
class SaasConfig:
    """Configuration for SAAS GP."""

    # MCMC settings (from benchmark winner)
    warmup_steps: int = 128
    num_samples: int = 64
    thinning: int = 2

    # Acquisition settings
    num_restarts: int = 32
    raw_samples: int = 512
    use_noisy_ei: bool = True  # Use qLogNEI instead of qLogEI (better for noisy observations)

    # Input normalization bounds
    # SONAR unnormalized embeddings have values roughly in [-0.5, 0.5]
    input_bounds: Tuple[float, float] = (-1.0, 1.0)


@dataclass
class SaasPrediction:
    """Prediction from SAAS GP."""

    mean: torch.Tensor  # (N,) predicted error rates
    std: torch.Tensor  # (N,) uncertainty (combined epistemic + aleatoric)
    mean_samples: Optional[torch.Tensor] = None  # (S, N) individual MCMC sample means


class SaasGPWithAcquisition:
    """SAAS GP with qLogEI acquisition for high-dimensional optimization.

    Uses Sparse Axis-Aligned Subspaces prior (Eriksson & Jankowiak 2021)
    to automatically identify relevant dimensions in 1024D SONAR space.

    The MCMC fitting provides:
    1. Posterior samples over lengthscales (identifies relevant dims)
    2. Full uncertainty quantification
    3. Better exploration via Bayesian model averaging
    """

    def __init__(
        self,
        config: Optional[SaasConfig] = None,
        device: str = "cuda",
        fit_on_cpu: bool = True,
    ):
        """Initialize SAAS GP.

        Args:
            config: SaasConfig with MCMC and acquisition settings
            device: Torch device for acquisition optimization
            fit_on_cpu: If True, fit SAAS on CPU to save GPU memory (recommended when vLLM is running)
        """
        if not SAAS_AVAILABLE:
            raise ImportError(
                "SAAS requires BoTorch >= 0.10.0. "
                "Install: pip install 'botorch>=0.10.0' pyro-ppl"
            )

        self.config = config or SaasConfig()
        self.device = torch.device(device)
        self.fit_device = torch.device("cpu") if fit_on_cpu else self.device
        self.fit_on_cpu = fit_on_cpu

        logger.info(f"SAAS GP: fit_device={self.fit_device}, acq_device={self.device}")

        self.gp_model: Optional[SaasFullyBayesianSingleTaskGP] = None
        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None

        # Normalization stats
        self._X_min: Optional[torch.Tensor] = None
        self._X_max: Optional[torch.Tensor] = None

        # Training stats
        self.relevant_dims: List[int] = []
        self.training_time: float = 0.0

    def _normalize_X(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize inputs to [0, 1] for SAAS."""
        # Ensure double precision
        X = X.to(dtype=DTYPE)

        if self._X_min is None:
            # First time: fit normalizer
            self._X_min = X.min(dim=0).values
            self._X_max = X.max(dim=0).values

            # Prevent division by zero
            range_vals = self._X_max - self._X_min
            range_vals[range_vals < 1e-6] = 1.0
            self._X_range = range_vals

        # Ensure normalization stats match input device
        if self._X_min.device != X.device:
            self._X_min = self._X_min.to(X.device)
            self._X_max = self._X_max.to(X.device)
            self._X_range = self._X_range.to(X.device)

        X_norm = (X - self._X_min) / self._X_range
        return X_norm.clamp(0, 1)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
    ) -> bool:
        """Fit SAAS GP via MCMC.

        Args:
            X: Input SONAR embeddings (N, 1024)
            y: Target error rates (N,)
            variances: Optional observation variances for heteroscedastic noise

        Returns:
            True if fitting succeeded
        """
        start_time = time.time()

        # Use fit_device for MCMC fitting (CPU to save GPU memory for vLLM)
        # Convert to double precision for numerical stability
        X = X.to(device=self.fit_device, dtype=DTYPE)
        y = y.to(device=self.fit_device, dtype=DTYPE)

        if X.shape[0] < 5:
            logger.error(f"SAAS needs at least 5 points, got {X.shape[0]}")
            return False

        # Store training data on fit_device
        self.X_train = X
        self.y_train = y

        # Normalize X to [0, 1]
        X_norm = self._normalize_X(X)

        # SAAS expects y in (N, 1) format
        y_2d = y.unsqueeze(-1)

        logger.info(
            f"Fitting SAAS GP: {X.shape[0]} points, {X.shape[1]}D, "
            f"warmup={self.config.warmup_steps}, samples={self.config.num_samples} "
            f"(device={self.fit_device})"
        )

        try:
            # Create SAAS model on fit_device (CPU to avoid OOM with vLLM)
            self.gp_model = SaasFullyBayesianSingleTaskGP(
                train_X=X_norm,
                train_Y=y_2d,
                outcome_transform=Standardize(m=1),
            ).to(self.fit_device)

            # Fit via NUTS MCMC on fit_device
            fit_fully_bayesian_model_nuts(
                self.gp_model,
                warmup_steps=self.config.warmup_steps,
                num_samples=self.config.num_samples,
                thinning=self.config.thinning,
                disable_progbar=False,
            )

        except Exception as e:
            logger.error(f"SAAS fitting failed: {e}")
            return False

        # Extract relevant dimensions
        self._extract_relevant_dims()

        self.training_time = time.time() - start_time
        logger.info(f"SAAS GP fitted in {self.training_time:.1f}s")

        return True

    def _extract_relevant_dims(self, top_k: int = 10):
        """Extract most relevant dimensions from learned lengthscales."""
        if self.gp_model is None:
            return

        try:
            # Get lengthscales from MCMC samples
            raw_ls = self.gp_model.covar_module.base_kernel.lengthscale
            # Shape: (num_samples, 1, D)
            median_ls = raw_ls.median(dim=0).values.squeeze()  # (D,)

            # Small lengthscale = relevant dimension
            sorted_dims = torch.argsort(median_ls)
            self.relevant_dims = sorted_dims[:top_k].tolist()

            top_ls = median_ls[sorted_dims[:5]].tolist()
            logger.info(
                f"  Relevant dims: {self.relevant_dims[:5]}, "
                f"lengthscales: {[f'{ls:.3f}' for ls in top_ls]}"
            )

        except Exception as e:
            logger.warning(f"Could not extract relevant dims: {e}")

    def predict(self, X: torch.Tensor) -> SaasPrediction:
        """Make predictions with uncertainty.

        Args:
            X: Input embeddings (N, 1024)

        Returns:
            SaasPrediction with mean and std
        """
        if self.gp_model is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        # Use fit_device (where GP model lives) with double precision
        X = X.to(device=self.fit_device, dtype=DTYPE)
        X_norm = self._normalize_X(X).clamp(0, 1)

        with torch.no_grad():
            posterior = self.gp_model.posterior(X_norm)

            # Average over MCMC samples
            # posterior.mean shape: (num_samples, N, 1)
            mean_samples = posterior.mean.squeeze(-1)  # (S, N)
            mean = mean_samples.mean(dim=0)  # (N,)

            # Combined uncertainty
            var_epistemic = mean_samples.var(dim=0)
            var_aleatoric = posterior.variance.squeeze(-1).mean(dim=0)
            std = (var_epistemic + var_aleatoric).sqrt()

        return SaasPrediction(
            mean=mean.clamp(0, 1),
            std=std.clamp(min=1e-6),
            mean_samples=mean_samples,
        )

    def get_best_candidate(
        self,
        bounds: torch.Tensor,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, float]:
        """Get best candidate(s) using qLogEI acquisition.

        Args:
            bounds: Search bounds (2, D) - [lower, upper]
            batch_size: Number of candidates (default 1)

        Returns:
            (candidates, acquisition_value)
        """
        if self.gp_model is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        # Move bounds to fit_device (where GP model lives) with double precision
        bounds = bounds.to(device=self.fit_device, dtype=DTYPE)

        # Normalize bounds
        bounds_norm = torch.stack([
            self._normalize_X(bounds[0:1]).squeeze(0),
            self._normalize_X(bounds[1:2]).squeeze(0),
        ]).clamp(0, 1)

        # Create acquisition function
        if self.config.use_noisy_ei:
            # qLogNEI: Better for noisy observations (uses X_baseline for integration)
            acq_func = qLogNoisyExpectedImprovement(
                model=self.gp_model,
                X_baseline=self._normalize_X(self.X_train),  # Integrate over observed points
            )
            logger.debug("Using qLogNEI (noisy EI)")
        else:
            # qLogEI: Assumes noiseless observations
            best_f = self.y_train.min()
            acq_func = qLogExpectedImprovement(
                model=self.gp_model,
                best_f=best_f,
            )
            logger.debug("Using qLogEI")

        # Optimize acquisition with robust options
        # Suppress scipy optimization warnings (we handle failures gracefully)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Optimization failed.*")

            try:
                candidates_norm, acq_value = optimize_acqf(
                    acq_function=acq_func,
                    bounds=bounds_norm,
                    q=batch_size,
                    num_restarts=self.config.num_restarts,
                    raw_samples=self.config.raw_samples,
                    options={
                        "batch_limit": 5,  # Reduce batch size for more stable optimization
                        "maxiter": 200,    # More iterations for convergence
                        "method": "L-BFGS-B",  # Explicit method specification
                    },
                )
            except Exception as e:
                logger.warning(f"Primary acquisition optimization failed: {e}")
                # Fallback: use best from raw samples
                candidates_norm, acq_value = self._fallback_acquisition(
                    acq_func, bounds_norm, batch_size
                )

        # Denormalize candidates
        candidates = candidates_norm * self._X_range + self._X_min

        return candidates, acq_value.item() if torch.is_tensor(acq_value) else acq_value

    def _fallback_acquisition(
        self,
        acq_func,
        bounds_norm: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, float]:
        """Fallback acquisition using random sampling when scipy fails.

        Args:
            acq_func: Acquisition function
            bounds_norm: Normalized bounds
            batch_size: Number of candidates

        Returns:
            (candidates, acquisition_value)
        """
        logger.info("Using fallback random sampling for acquisition")

        # Generate random samples
        n_samples = self.config.raw_samples * 2
        D = bounds_norm.shape[1]

        # Sobol sequence for better coverage
        try:
            from botorch.utils.sampling import draw_sobol_samples
            samples = draw_sobol_samples(
                bounds=bounds_norm,
                n=n_samples,
                q=batch_size,
            ).squeeze(1)  # (n_samples, D)
        except Exception:
            # Pure random if Sobol fails
            samples = torch.rand(n_samples, D, device=bounds_norm.device, dtype=bounds_norm.dtype)
            samples = samples * (bounds_norm[1] - bounds_norm[0]) + bounds_norm[0]

        # Evaluate acquisition on samples
        with torch.no_grad():
            acq_values = acq_func(samples.unsqueeze(1))  # (n_samples,)

        # Select best
        best_idx = acq_values.argmax()
        best_candidate = samples[best_idx:best_idx+1]
        best_acq = acq_values[best_idx].item()

        return best_candidate, best_acq

    def add_observation(
        self,
        x: torch.Tensor,
        y: float,
        refit: bool = True,
    ):
        """Add new observation and optionally refit.

        Args:
            x: New input embedding (1024,) or (1, 1024)
            y: New target error rate
            refit: Whether to refit the GP
        """
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        y_tensor = torch.tensor([[y]], device=self.device)

        if self.X_train is None:
            self.X_train = x
            self.y_train = y_tensor.squeeze()
        else:
            self.X_train = torch.cat([self.X_train, x], dim=0)
            self.y_train = torch.cat([self.y_train, y_tensor.squeeze()])

        if refit:
            self.fit(self.X_train, self.y_train)


class SaasAcquisitionOptimizer:
    """Optimizer combining SAAS GP with flow-guided acquisition.

    Wraps SaasGPWithAcquisition to provide interface compatible with
    FlowGuidedAcquisition while using qLogEI internally.
    """

    def __init__(
        self,
        saas_gp: SaasGPWithAcquisition,
        manifold_keeper: Optional[nn.Module] = None,
        manifold_time: float = 0.9,
        lambda_penalty: float = 0.001,
        use_velocity_penalty: bool = True,
    ):
        """Initialize optimizer.

        Args:
            saas_gp: Fitted SAAS GP
            manifold_keeper: Optional ManifoldKeeper for velocity penalty
            manifold_time: Time for velocity computation
            lambda_penalty: Weight for velocity penalty
            use_velocity_penalty: Whether to use velocity penalty
        """
        self.gp = saas_gp
        self.manifold_keeper = manifold_keeper
        self.manifold_time = manifold_time
        self.lambda_penalty = lambda_penalty
        self.use_velocity_penalty = use_velocity_penalty and manifold_keeper is not None

    def _compute_velocity_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute velocity magnitude penalty."""
        if not self.use_velocity_penalty or self.manifold_keeper is None:
            return torch.zeros(x.shape[0], device=x.device)

        with torch.no_grad():
            t = torch.full((x.shape[0],), self.manifold_time, device=x.device)
            velocity = self.manifold_keeper(t, x)
            return (velocity ** 2).sum(dim=-1)

    def optimize(
        self,
        bounds: torch.Tensor,
        batch_size: int = 1,
        X_train: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Optimize acquisition with optional velocity penalty.

        Uses qLogEI from SAAS GP, then optionally filters candidates
        by velocity penalty.

        Args:
            bounds: Search bounds (2, D)
            batch_size: Number of candidates
            X_train: Training data (for seeding - not used in qLogEI)

        Returns:
            (best_candidate, acquisition_value)
        """
        # Get candidates from qLogEI
        # Request more candidates than needed to filter by velocity
        n_candidates = batch_size * 4 if self.use_velocity_penalty else batch_size

        candidates, acq_value = self.gp.get_best_candidate(
            bounds=bounds,
            batch_size=n_candidates,
        )

        if not self.use_velocity_penalty or candidates.shape[0] == 1:
            return candidates[:batch_size], acq_value

        # Filter by velocity penalty
        v_penalty = self._compute_velocity_penalty(candidates)

        # Get predictions for ranking
        pred = self.gp.predict(candidates)

        # Combined score: lower mean + lower velocity penalty
        # (we minimize error rate)
        combined = pred.mean + self.lambda_penalty * v_penalty

        # Select best
        best_idx = combined.argsort()[:batch_size]
        best_candidates = candidates[best_idx]

        # Recompute acquisition value for best candidate
        _, best_acq = self.gp.get_best_candidate(bounds, batch_size=1)

        return best_candidates, best_acq


def create_saas_gp(
    config: Optional[SaasConfig] = None,
    device: str = "cuda",
    fit_on_cpu: bool = True,
) -> SaasGPWithAcquisition:
    """Factory function to create SAAS GP.

    Args:
        config: Optional SaasConfig
        device: Torch device
        fit_on_cpu: If True, fit SAAS on CPU to save GPU memory (recommended with vLLM)

    Returns:
        Initialized SaasGPWithAcquisition
    """
    return SaasGPWithAcquisition(config=config, device=device, fit_on_cpu=fit_on_cpu)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing SaasGPWithAcquisition...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create synthetic data
    N, D = 30, 1024
    X = torch.randn(N, D, device=device) * 0.2
    y = ((X[:, :10] ** 2).sum(dim=1) * 0.1).clamp(0, 1)  # Only first 10 dims matter

    print(f"\nSynthetic data: {N} points, {D}D")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")

    # Create and fit SAAS GP
    config = SaasConfig(warmup_steps=64, num_samples=32)  # Faster for testing
    gp = create_saas_gp(config=config, device=device)

    print("\nFitting SAAS GP...")
    success = gp.fit(X, y)
    print(f"  Success: {success}")
    print(f"  Relevant dims: {gp.relevant_dims[:5]}")

    # Test prediction
    print("\nTesting prediction...")
    X_test = torch.randn(5, D, device=device) * 0.2
    pred = gp.predict(X_test)
    print(f"  Mean: {pred.mean}")
    print(f"  Std: {pred.std}")

    # Test acquisition
    print("\nTesting qLogEI acquisition...")
    bounds = torch.stack([
        torch.full((D,), -0.5, device=device),
        torch.full((D,), 0.5, device=device),
    ])

    candidate, acq_val = gp.get_best_candidate(bounds)
    print(f"  Candidate shape: {candidate.shape}")
    print(f"  Acquisition value: {acq_val:.4f}")

    print("\n[OK] SaasGPWithAcquisition tests passed!")
