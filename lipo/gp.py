"""Deep Kernel Gaussian Process for instruction optimization.

Provides:
- InstructionDeepKernelGP: GP with Matern 5/2 kernel on 10D latent features
- GPWithEI: Wrapper with Expected Improvement computation

Self-contained within the lipo package (no imports from other lipo/ modules).
"""

import torch
import torch.nn as nn
import gpytorch
import math
import numpy as np
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.outcome import Standardize
from scipy.stats import norm
from scipy.special import erfcx, log1p, expm1
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.model_selection import KFold


# =============================================================================
# LogEI: Numerically stable log Expected Improvement
# Based on: "Unexpected Improvements to Expected Improvement" (NeurIPS 2023)
# =============================================================================

# Constants for log_h computation
_C1 = math.log(2 * math.pi) / 2  # log(2π)/2
_C2 = math.log(math.pi / 2) / 2  # log(π/2)/2
_EPS = np.finfo(np.float64).eps  # Machine epsilon
_THRESH = -1 / math.sqrt(_EPS)  # Threshold for asymptotic branch


def _log1mexp(x: float) -> float:
    """Numerically stable computation of log(1 - exp(x)) for x < 0.

    Uses different branches based on threshold to avoid numerical issues.
    Based on Mächler (2012) "Accurately computing log(1 - exp(-|a|))".
    """
    if x > -math.log(2):
        return math.log(-expm1(x))
    else:
        return log1p(-math.exp(x))


def log_h(z: float) -> float:
    """Numerically stable computation of log(h(z)) where h(z) = φ(z) + z·Φ(z).

    This is the core of LogEI from the paper. Uses three branches:
    1. Direct computation for z > -1
    2. erfcx-based computation for -1/√ε < z ≤ -1
    3. Asymptotic approximation for z ≤ -1/√ε

    Args:
        z: Standardized improvement (μ - y*) / σ

    Returns:
        log(h(z)) value, can be negative (no clipping!)
    """
    if z > -1:
        # Direct computation is numerically stable for z > -1
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        h_val = phi_z + z * Phi_z
        if h_val > 0:
            return math.log(h_val)
        else:
            # Fallback to middle branch if numerical issues
            z = -1.0 - 1e-10

    if z > _THRESH:
        # Middle branch: -1/√ε < z ≤ -1
        # log_h(z) = -z²/2 - c1 + log1mexp(log(erfcx(-z/√2)·|z|) + c2)
        z_scaled = -z / math.sqrt(2)
        erfcx_val = erfcx(z_scaled)
        inner = math.log(erfcx_val * abs(z)) + _C2
        return -z * z / 2 - _C1 + _log1mexp(inner)
    else:
        # Asymptotic branch: z ≤ -1/√ε
        # log_h(z) ≈ -z²/2 - c1 - 2·log(|z|)
        return -z * z / 2 - _C1 - 2 * math.log(abs(z))


def log_h_tensor(z: torch.Tensor) -> torch.Tensor:
    """Tensor version of log_h that supports autograd.

    Numerically stable computation of log(h(z)) where h(z) = φ(z) + z·Φ(z).
    Uses piecewise approximation with smooth transitions for gradient stability.

    Args:
        z: Standardized improvement tensor (can be batched)

    Returns:
        log(h(z)) tensor with gradients preserved
    """
    # Constants
    LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
    SQRT_2 = math.sqrt(2)

    # Standard normal PDF and CDF
    phi_z = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    # Use error function for CDF: Φ(z) = 0.5 * (1 + erf(z/√2))
    Phi_z = 0.5 * (1 + torch.erf(z / SQRT_2))

    # h(z) = φ(z) + z * Φ(z)
    h_z = phi_z + z * Phi_z

    # Clamp to avoid log(0) - use small positive value
    h_z_clamped = h_z.clamp(min=1e-40)

    # For very negative z, use asymptotic approximation
    # log_h(z) ≈ -z²/2 - log(2π)/2 - 2*log(|z|) for z << -1
    log_h_asymptotic = -0.5 * z * z - LOG_SQRT_2PI - 2 * torch.log(torch.abs(z).clamp(min=1e-10))

    # Use asymptotic for z < -5, direct for z >= -5
    # Smooth transition using sigmoid
    weight = torch.sigmoid(5 * (z + 5))  # 0 for z << -5, 1 for z >> -5
    log_h_direct = torch.log(h_z_clamped)

    return weight * log_h_direct + (1 - weight) * log_h_asymptotic


class InstructionDeepKernelGP(ExactGP, GPyTorchModel):
    """Gaussian Process with deep kernel for instruction optimization.

    Uses trainable adapter (64D → 10D) + ARD Matern 5/2 kernel.
    Inherits from GPyTorchModel for BoTorch compatibility (enables EI, etc.).

    Architecture:
        64D VAE latent -> Adapter (trainable) -> 10D
                                                  |
                                  Matern 5/2 kernel (ARD)
                                                  |
                                  GP(mean=0, K(latent))

    Training: Adapter and GP kernel are trained jointly on 64D VAE latents.
    VAE encoder is frozen and applied before GP training.
    """

    _num_outputs = 1  # Required for BoTorch

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Union[GaussianLikelihood, FixedNoiseGaussianLikelihood],
        adapter: nn.Module,
    ):
        """Initialize GP.

        Args:
            train_x: Training VAE latents (N, 64)
            train_y: Training targets (N,) - negated error rates (for BoTorch maximization)
            likelihood: Gaussian or FixedNoiseGaussian likelihood
            adapter: Trainable adapter MLP (64D → 10D)
        """
        # Ensure train_y is 1D for ExactGP
        train_y = train_y.squeeze(-1) if train_y.dim() > 1 else train_y
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,  # Matern 5/2 - smooth but flexible
                ard_num_dims=10,  # Per-dimension lengthscales (adapter output dim)
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through GP.

        Args:
            x: VAE latent (batch, 64) or (batch, n, 64)

        Returns:
            MultivariateNormal distribution over function values
        """
        # Apply adapter: 64D → 10D
        latent = self.adapter(x)

        # Handle 3D for BoTorch posterior
        if latent.dim() == 3:
            latent = latent.squeeze(-2) if latent.shape[-2] == 1 else latent

        return MultivariateNormal(
            self.mean_module(latent),
            self.covar_module(latent),
        )


class GPWithEI:
    """Wrapper for GP with Expected Improvement acquisition.

    Manages GP training, prediction, and EI computation for
    instruction optimization.

    Architecture:
        Training: embeddings (768D) → frozen VAE encoder → z (64D) → adapter+GP training
        Inference: z (64D) → adapter → z_gp (10D) → GP → qLogEI
    """

    def __init__(
        self,
        device: str = "cuda",
        latent_dim: int = 10,
    ):
        """Initialize GP wrapper.

        Args:
            device: Device to use
            latent_dim: Adapter output dimension (10)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        # Components (initialized during training)
        self.vae_with_adapter: Optional[nn.Module] = None  # VAEWithAdapter (frozen VAE + trainable adapter)
        self.likelihood: Optional[GaussianLikelihood] = None
        self.gp_model: Optional[InstructionDeepKernelGP] = None

        # Training data - stored as 64D VAE latents
        self.X_train: Optional[torch.Tensor] = None  # (N, 64) VAE latents
        self.y_train: Optional[torch.Tensor] = None  # (N,) negated error rates (internal)
        self.fidelity_train: Optional[torch.Tensor] = None  # (N,) sample counts for each observation
        self._error_rates_original: Optional[torch.Tensor] = None  # (N,) positive error rates for noise computation

        # Normalization parameters (for 64D latents)
        self.X_min: Optional[torch.Tensor] = None
        self.X_max: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        # Best observed value (for EI)
        self.y_best: Optional[float] = None

        # Training stats (populated after train())
        self.training_stats: Dict[str, Any] = {}

    def set_training_data(
        self,
        embeddings: torch.Tensor,
        error_rates: torch.Tensor,
        fidelities: Optional[torch.Tensor] = None,
    ):
        """Set training data for GP.

        Transforms embeddings to 64D VAE latents using frozen VAE encoder.

        IMPORTANT: We negate error_rates for BoTorch compatibility.
        BoTorch's qLogExpectedImprovement MAXIMIZES by default:
            EI(x) = E[max(f(x) - best_f, 0)]
        To minimize error_rate, we store y = -error_rate so that:
            - Low error (0.05) → y = -0.05 (closer to 0, better for maximization)
            - High error (0.30) → y = -0.30 (more negative, worse for maximization)
        The GP learns to predict -error_rate, and EI seeks to maximize it
        (i.e., find the least negative value = lowest error).

        Args:
            embeddings: Instruction embeddings (N, 768)
            error_rates: Error rates (N,) in [0, 1] - will be negated internally
            fidelities: Sample counts for each observation (N,) - used for heteroscedastic noise
        """
        if self.vae_with_adapter is None:
            raise RuntimeError("vae_with_adapter must be set before setting training data.")

        embeddings = embeddings.to(self.device)

        # Validate input dimension
        expected_embed_dim = 768
        if embeddings.shape[-1] != expected_embed_dim:
            raise ValueError(
                f"Expected embeddings with dim {expected_embed_dim}, got {embeddings.shape[-1]}. "
                f"Ensure you're passing GTR embeddings, not VAE latents."
            )

        # Transform to 64D VAE latents using frozen encoder
        self.vae_with_adapter.eval()
        with torch.no_grad():
            vae_latents = self.vae_with_adapter.encode_vae(embeddings)

        # Validate VAE output dimension
        expected_vae_dim = 64
        if vae_latents.shape[-1] != expected_vae_dim:
            raise RuntimeError(
                f"VAE encoder output dimension mismatch: expected {expected_vae_dim}, "
                f"got {vae_latents.shape[-1]}. Check VAE configuration."
            )

        self.X_train = vae_latents  # (N, 64)

        # Store original error rates for noise computation (needs positive values)
        self._error_rates_original = error_rates.to(self.device)

        # CRITICAL: Negate error rates for BoTorch maximization → minimization
        # GP predicts -error_rate, so lower error → higher predicted value
        self.y_train = -error_rates.to(self.device)

        # y_best is the maximum of negated errors = minimum of original errors
        self.y_best = self.y_train.max().item()  # max(-error) = -min(error)

        # Store fidelities for heteroscedastic noise computation
        if fidelities is not None:
            self.fidelity_train = fidelities.to(self.device)
        else:
            # Fidelities are required for heteroscedastic noise computation
            raise ValueError(
                f"fidelities must be provided for {len(self.y_train)} observations. "
                f"Heteroscedastic noise computation requires sample counts. "
                f"If all observations are full-fidelity, pass fidelities=torch.ones(n) * max_fidelity"
            )

    def _compute_observation_noise(self, y: torch.Tensor, fidelity: torch.Tensor) -> torch.Tensor:
        """Compute observation noise variance based on Bernoulli statistics.

        For error_rate measured on n samples, variance is: Var = p(1-p)/n
        where p is error_rate and n is fidelity (sample count).

        This gives GP information about observation reliability:
        - Low fidelity (small n) → high variance → less trust
        - High fidelity (large n) → low variance → more trust

        Args:
            y: Error rates (N,)
            fidelity: Sample counts (N,)

        Returns:
            Observation noise variance for each point (N,)
        """
        # Bernoulli variance: p(1-p)/n
        # Clamp fidelity to minimum of 1 to avoid division by zero
        safe_fidelity = torch.clamp(fidelity, min=1.0)
        variance = (y * (1 - y)) / safe_fidelity

        # Clamp to avoid numerical issues:
        # - min: prevent zero variance (causes numerical issues)
        # - max: prevent extremely high variance for low-fidelity points
        variance = torch.clamp(variance, min=1e-6, max=0.1)

        return variance

    def train(
        self,
        epochs: int = 3000,
        lr: float = 0.01,
        patience: int = 10,
        verbose: bool = True,
    ) -> bool:
        """Train GP on stored 64D VAE latents.

        Trains adapter (64D→10D) and GP kernel jointly.
        VAE encoder is frozen.

        Uses FixedNoiseGaussianLikelihood with heteroscedastic noise based on
        Bernoulli variance: Var = p(1-p)/n, where n is fidelity (sample count).

        Args:
            epochs: Maximum training epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("No training data. Call set_training_data() first.")

        X = self.X_train  # Already 64D VAE latents
        y = self.y_train

        # Unit cube normalization for 64D latents
        self.X_min = X.min(dim=0)[0]
        self.X_max = X.max(dim=0)[0]
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        X_norm = (X - self.X_min) / denom

        # Standardize output transform (BoTorch standard approach)
        self.outcome_transform = Standardize(m=1).to(self.device)
        y_transformed, _ = self.outcome_transform(y.unsqueeze(-1))
        y_norm = y_transformed.squeeze(-1)

        # Keep y_mean/y_std for backwards compatibility with LogEI methods
        self.y_mean = self.outcome_transform.means.squeeze()
        self.y_std = self.outcome_transform.stdvs.squeeze()

        # VAEWithAdapter must be pre-set
        if self.vae_with_adapter is None:
            raise RuntimeError(
                "vae_with_adapter must be set before training. "
                "Use VAEWithAdapter from encoder.py."
            )

        # Get trainable adapter from VAEWithAdapter
        adapter = self.vae_with_adapter.adapter

        # Compute heteroscedastic noise from Bernoulli variance
        # Use original (positive) error rates for variance computation: p(1-p)/n
        # Noise in standardized space: scale by y_std^2
        raw_noise = self._compute_observation_noise(self._error_rates_original, self.fidelity_train)
        # Transform noise to standardized space (divide by y_std^2)
        # Clamp to prevent extreme values when y_std is near zero
        noise_standardized = raw_noise / (self.y_std ** 2 + 1e-8)
        noise_standardized = torch.clamp(noise_standardized, max=1.0)

        if verbose:
            print(f"  Observation noise: min={raw_noise.min():.6f}, max={raw_noise.max():.6f}")
            print(f"  Fidelity range: {self.fidelity_train.min().item():.0f} - {self.fidelity_train.max().item():.0f}")

        # FixedNoiseGaussianLikelihood with heteroscedastic noise
        # learn_additional_noise=True allows GP to learn residual noise beyond Bernoulli variance
        self.likelihood = FixedNoiseGaussianLikelihood(
            noise=noise_standardized,
            learn_additional_noise=True,
        ).to(self.device)
        self.gp_model = InstructionDeepKernelGP(
            X_norm, y_norm, self.likelihood, adapter
        ).to(self.device)

        # Register outcome transform for BoTorch compatibility
        self.gp_model.outcome_transform = self.outcome_transform

        # Training loop
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss = float("inf")
        patience_counter = 0

        if verbose:
            print("Training GP...")

        with gpytorch.settings.cholesky_jitter(1e-4):
            for epoch in range(epochs):
                try:
                    optimizer.zero_grad()
                    output = self.gp_model(X_norm)
                    loss = -mll(output, y_norm)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        break

                    if verbose and (epoch + 1) % 100 == 0:
                        print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")

                except RuntimeError as e:
                    error_msg = str(e)
                    # Only handle specific GPyTorch/PyTorch numerical errors
                    numerical_error_patterns = [
                        "not positive definite",
                        "cholesky_cpu",
                        "cholesky_cuda",
                        "singular matrix",
                        "LinAlgError",
                    ]
                    is_numerical_error = any(pattern in error_msg for pattern in numerical_error_patterns)

                    if is_numerical_error:
                        print(f"ERROR: GP training failed - numerical instability")
                        print(f"  Epoch: {epoch + 1}, Training samples: {len(X)}")
                        print(f"  Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
                        print(f"  Error type: {type(e).__name__}")
                        print(f"  Error: {error_msg}")
                        print(f"  Possible causes: duplicate inputs, near-constant outputs, or ill-conditioned kernel")
                        return False
                    # Re-raise all other RuntimeErrors - they indicate bugs, not expected failures
                    raise

        # Store training stats
        self.training_stats = {
            "epochs_trained": epoch + 1,
            "final_loss": float(best_loss),
            "early_stopped": patience_counter >= patience,
            "num_samples": len(X),
        }

        if verbose:
            print(f"  GP training complete (epochs={epoch + 1}, loss={best_loss:.4f})")

        # Validation: compute MAE on full-fidelity samples
        if verbose and self.fidelity_train is not None:
            self._validate_on_full_fidelity()

        return True

    def _validate_on_full_fidelity(self):
        """Validate GP predictions on full-fidelity samples only.

        Computes MAE between GP predictions and actual error rates for
        samples with fidelity == max_fidelity (most reliable observations).

        Note: Internally uses negated values but reports in positive error rate space.
        """
        if self.X_train is None or self.y_train is None:
            return

        max_fid = self.fidelity_train.max().item()
        full_fid_mask = self.fidelity_train >= max_fid * 0.99  # Allow small tolerance

        if full_fid_mask.sum() < 3:
            print(f"  WARNING: Only {full_fid_mask.sum().item()} full-fidelity samples (need >= 3 for validation)")
            print(f"    Max fidelity: {max_fid:.0f}, threshold: {max_fid * 0.99:.0f}")
            print(f"    Consider using more full-fidelity evaluations for reliable GP quality assessment")
            return

        # Get full-fidelity samples
        X_val = self.X_train[full_fid_mask]
        # y_train is negated (-error_rate), use original for comparison
        y_val_original = self._error_rates_original[full_fid_mask]

        # Normalize X for prediction
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        X_val_norm = (X_val - self.X_min) / denom

        # Get predictions
        self.gp_model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp_model(X_val_norm))
            pred_mean_norm = pred.mean

        # Denormalize predictions (GP predicts -error_rate)
        pred_neg_error = pred_mean_norm * self.y_std + self.y_mean
        # Negate to get positive error rate
        pred_mean = -pred_neg_error

        # Compute MAE against original (positive) error rates
        mae = (pred_mean - y_val_original).abs().mean().item()
        rmse = ((pred_mean - y_val_original) ** 2).mean().sqrt().item()

        # Also compute correlation
        pred_np = pred_mean.cpu().numpy()
        y_np = y_val_original.cpu().numpy()
        correlation = float(np.corrcoef(pred_np, y_np)[0, 1]) if len(pred_np) > 1 else 0.0

        print(f"  GP Validation (n={full_fid_mask.sum().item()} full-fidelity):")
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, Corr: {correlation:.4f}")

    def predict(
        self,
        embedding: torch.Tensor,
    ) -> Tuple[float, float]:
        """Predict error rate for embedding.

        The GP internally predicts -error_rate (for BoTorch maximization).
        This method negates the prediction to return positive error_rate.

        Args:
            embedding: Instruction embedding (768,) or (1, 768)

        Returns:
            (mean, std) predictions as positive error rates in [0, 1]
        """
        if self.gp_model is None:
            raise RuntimeError("GP not trained.")

        # Ensure correct shape
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        # Encode 768D embedding to 64D VAE latent if needed
        with torch.no_grad():
            if embedding.shape[-1] == 768:
                # Encode through VAE to get 64D latent
                z_vae = self.vae_with_adapter.encode_vae(embedding)
            else:
                z_vae = embedding

        # Normalize
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        X_norm = (z_vae - self.X_min) / denom

        # Predict
        self.gp_model.eval()
        self.likelihood.eval()

        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = self.likelihood(self.gp_model(X_norm))
                mean_norm = pred.mean.item()
                std_norm = pred.stddev.item()
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cholesky" in error_msg or "singular" in error_msg or "positive definite" in error_msg:
                raise RuntimeError(
                    f"GP prediction failed due to numerical instability. "
                    f"This may indicate ill-conditioned training data. "
                    f"Original error: {e}"
                ) from e
            raise

        # Denormalize (GP predicts in standardized -error_rate space)
        neg_error = mean_norm * self.y_std.item() + self.y_mean.item()
        std = std_norm * self.y_std.item()

        # CRITICAL: Negate back to positive error rate
        # GP predicts -error_rate, so we return -predicted = error_rate
        mean = -neg_error

        return mean, std

    @property
    def best_error_rate(self) -> Optional[float]:
        """Get best observed error rate (positive).

        Internally y_best stores -min_error for BoTorch compatibility.
        This property returns the positive error rate.
        """
        if self.y_best is None:
            return None
        return -self.y_best  # Negate back: -(-min_error) = min_error

    def expected_improvement(
        self,
        embedding: torch.Tensor,
        xi: float = 0.01,
    ) -> float:
        """Compute Expected Improvement for embedding.

        EI(x) = (best - mu(x)) * Phi(z) + sigma(x) * phi(z)
        where z = (best - mu(x) - xi) / sigma(x)

        We minimize error, so improvement = best_error - predicted_error.
        Uses positive error rates for interpretability.

        Args:
            embedding: Instruction embedding (768,)
            xi: Exploration-exploitation trade-off (default 0.01)

        Returns:
            Expected improvement value (non-negative by definition)
        """
        if self.gp_model is None or self.y_best is None:
            return 0.0

        mean, std = self.predict(embedding)  # Returns positive error rate
        best = self.best_error_rate  # Positive error rate

        if std <= 0:
            return max(0.0, best - mean)

        z = (best - mean - xi) / std
        ei = (best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
        return max(0.0, ei)

    def log_expected_improvement(
        self,
        embedding: torch.Tensor,
        xi: float = 0.01,
    ) -> float:
        """Compute Log Expected Improvement for embedding.

        LogEI(x) = log_h(z) + log(σ(x))
        where z = (best - mu(x) - xi) / sigma(x)

        This is numerically stable even when EI values are extremely small.
        Uses positive error rates for interpretability.

        Args:
            embedding: Instruction embedding (768,)
            xi: Exploration-exploitation trade-off (default 0.01)

        Returns:
            Log expected improvement value (can be very negative!)
        """
        if self.gp_model is None or self.y_best is None:
            return float("-inf")

        mean, std = self.predict(embedding)  # Returns positive error rate
        best = self.best_error_rate  # Positive error rate

        if std <= 1e-10:
            improvement = best - mean
            if improvement > 0:
                return math.log(improvement)
            return float("-inf")

        z = (best - mean - xi) / std

        # LogEI = log_h(z) + log(σ)
        return log_h(z) + math.log(std)

    def log_expected_improvement_tensor(
        self,
        embedding: torch.Tensor,
        xi: float = 0.01,
    ) -> torch.Tensor:
        """Compute Log Expected Improvement as a differentiable tensor.

        This version maintains gradients for optimization.
        Note: This is a standalone method for debugging/analysis.
        The main optimization uses BoTorch's qLogExpectedImprovement.

        Args:
            embedding: Instruction embedding (768,) or (1, 768)
            xi: Exploration-exploitation trade-off (default 0.01)

        Returns:
            LogEI as scalar tensor with gradient support
        """
        if self.gp_model is None or self.y_best is None:
            return torch.tensor(float("-inf"), device=self.device)

        # Ensure correct shape
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        # Normalize (preserving gradients)
        denom = self.X_max - self.X_min
        denom = denom.clone()
        denom[denom == 0] = 1.0
        X_norm = (embedding - self.X_min) / denom

        # Get GP prediction with gradients (predicts -error_rate)
        self.gp_model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred = self.gp_model(X_norm)
            mean_norm = pred.mean
            var_norm = pred.variance.clamp(min=1e-12)
            std_norm = torch.sqrt(var_norm)

        # Denormalize (GP predicts -error_rate)
        neg_error = mean_norm * self.y_std + self.y_mean
        std = std_norm * self.y_std

        # Convert to positive error rate space for EI
        mean = -neg_error  # Positive error rate
        best = -self.y_best  # Positive best error rate (self.y_best = -min_error)

        # Compute z-score: improvement = best - mean (lower error is better)
        z = (best - mean - xi) / std.clamp(min=1e-10)

        # LogEI = log_h(z) + log(σ)
        log_ei = log_h_tensor(z) + torch.log(std.clamp(min=1e-10))

        return log_ei.squeeze()

    def get_latent(self, embedding: torch.Tensor) -> torch.Tensor:
        """Get 10D latent for embedding using trained adapter.

        Pipeline: embedding (768D) → VAE encoder → z (64D) → normalize → adapter → z_gp (10D)

        Args:
            embedding: Instruction embedding (768,)

        Returns:
            Latent (10,)
        """
        if self.vae_with_adapter is None:
            raise RuntimeError("vae_with_adapter not initialized.")

        embedding = embedding.to(self.device)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        # First encode to 64D VAE latent
        self.vae_with_adapter.eval()
        with torch.no_grad():
            z_vae = self.vae_with_adapter.encode_vae(embedding)

        # Normalize 64D latent
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        z_norm = (z_vae - self.X_min) / denom

        # Apply adapter: 64D → 10D
        with torch.no_grad():
            latent = self.vae_with_adapter.adapter(z_norm)

        return latent.squeeze(0)

    def get_training_size(self) -> int:
        """Get number of training samples."""
        return len(self.y_train) if self.y_train is not None else 0

    def add_observation(
        self,
        embedding: torch.Tensor,
        error_rate: float,
        fidelity: int = 1319,
    ):
        """Add a single observation to training data.

        NOTE: Does NOT retrain - call train() after adding observations.

        Error rate is Laplace-smoothed and negated internally for consistency
        with training data and BoTorch compatibility.

        Args:
            embedding: Instruction embedding (768,) or (1, 768)
            error_rate: Observed error rate (positive, will be smoothed and negated internally)
            fidelity: Number of samples used for evaluation (for noise estimation and smoothing)

        Raises:
            ValueError: If error_rate is not in [0, 1] or fidelity < 1
            RuntimeError: If vae_with_adapter is not set
        """
        # Validate inputs
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(
                f"error_rate must be in [0, 1], got {error_rate}. "
                f"This indicates a bug in the evaluation pipeline."
            )
        if fidelity < 1:
            raise ValueError(
                f"fidelity must be >= 1, got {fidelity}. "
                f"Fidelity represents the number of samples used for evaluation."
            )

        if self.vae_with_adapter is None:
            raise RuntimeError("vae_with_adapter must be set before adding observations.")

        # Ensure correct shape
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        # Encode to 64D VAE latent
        self.vae_with_adapter.eval()
        with torch.no_grad():
            new_z = self.vae_with_adapter.encode_vae(embedding)  # (1, 64)

        # Apply Laplace smoothing for consistency with training data
        # Formula: (errors + 1) / (n + 2) - penalizes "lucky guesses" on low-fidelity samples
        # This matches the smoothing applied in training.py:load_from_hyperband_evaluations()
        num_errors = error_rate * fidelity
        smoothed_error = (num_errors + 1) / (fidelity + 2)

        # Store original (smoothed) error rate for noise computation
        new_error_original = torch.tensor([smoothed_error], dtype=torch.float32, device=self.device)
        # Negate for GP training (BoTorch maximization)
        new_y = torch.tensor([-smoothed_error], dtype=torch.float32, device=self.device)
        new_fid = torch.tensor([fidelity], dtype=torch.float32, device=self.device)

        # Append to existing data
        if self.X_train is not None:
            self.X_train = torch.cat([self.X_train, new_z], dim=0)
            self.y_train = torch.cat([self.y_train, new_y], dim=0)
            self._error_rates_original = torch.cat([self._error_rates_original, new_error_original], dim=0)
            self.fidelity_train = torch.cat([self.fidelity_train, new_fid], dim=0)
        else:
            self.X_train = new_z
            self.y_train = new_y
            self._error_rates_original = new_error_original
            self.fidelity_train = new_fid

        # Update best: y_best is max(-error) = -min(error)
        # New observation improves if -error_rate > y_best (i.e., error_rate < -y_best)
        neg_error = -error_rate
        if neg_error > (self.y_best or float('-inf')):
            self.y_best = neg_error

    def validate_cross_validation(
        self,
        n_splits: int = 5,
        cv_epochs: int = 500,
        cv_lr: float = 0.01,
        cv_patience: int = 30,
        full_fidelity_only: bool = False,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """K-Fold Cross-Validation on training data.

        This provides an objective measure of GP generalization without
        evaluating on training data. For each fold:
        1. Train GP on (k-1) folds
        2. Predict on held-out fold (which GP has never seen)
        3. Compute MAE and correlation

        This is computationally expensive (n_splits × training), but gives
        the only unbiased estimate of generalization quality.

        Args:
            n_splits: Number of CV folds (default 5)
            cv_epochs: Training epochs per fold (reduced for speed)
            cv_lr: Learning rate for fold training
            cv_patience: Early stopping patience per fold
            full_fidelity_only: If True, use only full-fidelity samples (default False = use all)
            verbose: Print progress

        Returns:
            Dict with 'cv_mae', 'cv_rmse', 'cv_corr' averaged across folds
        """
        if self.X_train is None or self.y_train is None:
            if verbose:
                print("Skipping CV: No training data")
            return {}

        if self.vae_with_adapter is None:
            if verbose:
                print("Skipping CV: VAE with adapter not set")
            return {}

        # Select samples based on fidelity filter
        if full_fidelity_only:
            max_fid = self.fidelity_train.max().item()
            mask = self.fidelity_train >= max_fid * 0.99
            filter_desc = "full-fidelity"
        else:
            mask = torch.ones(len(self.X_train), dtype=torch.bool, device=self.device)
            filter_desc = "all"

        n_samples = mask.sum().item()
        if n_samples < n_splits * 2:
            if verbose:
                print(f"Skipping CV: Not enough samples ({n_samples} < {n_splits * 2})")
            return {}

        # Extract data for CV
        X_full = self.X_train[mask]  # (N, 64) VAE latents
        y_full = self._error_rates_original[mask]  # (N,) positive error rates
        fid_full = self.fidelity_train[mask]  # (N,) fidelities

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        mae_scores = []
        rmse_scores = []
        correlations = []

        if verbose:
            print(f"\nRunning {n_splits}-Fold Cross-Validation on {len(X_full)} {filter_desc} samples...")

        # Save original model state to restore after CV
        original_gp_state = self.gp_model.state_dict() if self.gp_model is not None else None
        original_likelihood_state = self.likelihood.state_dict() if self.likelihood is not None else None
        original_X_min = self.X_min.clone() if self.X_min is not None else None
        original_X_max = self.X_max.clone() if self.X_max is not None else None
        original_y_mean = self.y_mean.clone() if self.y_mean is not None else None
        original_y_std = self.y_std.clone() if self.y_std is not None else None

        # Save original adapter state to clone for each fold
        # This prevents later folds from starting with weights optimized for earlier folds
        import copy
        original_adapter_state = self.vae_with_adapter.adapter.state_dict()

        # Track skipped folds for reporting
        skipped_folds = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
            # Clone adapter for this fold to avoid bias
            adapter = copy.deepcopy(self.vae_with_adapter.adapter)
            adapter.load_state_dict(original_adapter_state)
            adapter = adapter.to(self.device)
            # Split data for this fold
            X_fold_train = X_full[train_idx]
            X_fold_val = X_full[val_idx]
            y_fold_train = y_full[train_idx]  # Positive error rates
            y_fold_val = y_full[val_idx]  # Positive error rates
            fid_fold_train = fid_full[train_idx]

            # === NORMALIZATION FOR THIS FOLD ===
            # Unit cube normalization based on TRAINING fold only
            X_min_fold = X_fold_train.min(dim=0)[0]
            X_max_fold = X_fold_train.max(dim=0)[0]
            denom = X_max_fold - X_min_fold
            denom[denom == 0] = 1.0

            X_train_norm = (X_fold_train - X_min_fold) / denom

            # Negate error rates for GP (BoTorch maximization)
            y_train_neg = -y_fold_train

            # Z-score standardization based on TRAINING fold
            y_mean_fold = y_train_neg.mean()
            y_std_fold = y_train_neg.std()
            if y_std_fold < 1e-6:
                y_std_fold = torch.tensor(1e-6, device=self.device)
            y_train_norm = (y_train_neg - y_mean_fold) / y_std_fold

            # === COMPUTE HETEROSCEDASTIC NOISE ===
            raw_noise = self._compute_observation_noise(y_fold_train, fid_fold_train)
            noise_standardized = raw_noise / (y_std_fold ** 2 + 1e-8)
            noise_standardized = torch.clamp(noise_standardized, max=1.0)

            # === CREATE AND TRAIN NEW GP FOR THIS FOLD ===
            fold_likelihood = FixedNoiseGaussianLikelihood(
                noise=noise_standardized,
                learn_additional_noise=True,
            ).to(self.device)

            fold_gp = InstructionDeepKernelGP(
                X_train_norm, y_train_norm, fold_likelihood, adapter
            ).to(self.device)

            fold_gp.train()
            fold_likelihood.train()

            optimizer = torch.optim.AdamW(fold_gp.parameters(), lr=cv_lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(fold_likelihood, fold_gp)

            best_loss = float("inf")
            patience_counter = 0

            with gpytorch.settings.cholesky_jitter(1e-4):
                for epoch in range(cv_epochs):
                    try:
                        optimizer.zero_grad()
                        output = fold_gp(X_train_norm)
                        loss = -mll(output, y_train_norm)
                        loss.backward()
                        optimizer.step()

                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= cv_patience:
                            break

                    except RuntimeError as e:
                        if "cholesky" in str(e).lower() or "singular" in str(e).lower():
                            if verbose:
                                print(f"  Fold {fold + 1}: Training failed (numerical)")
                            skipped_folds.append(fold + 1)
                            break
                        raise

            # === VALIDATION ON HELD-OUT FOLD ===
            fold_gp.eval()
            fold_likelihood.eval()

            with torch.no_grad():
                # Normalize validation input using TRAINING fold statistics
                X_val_norm = (X_fold_val - X_min_fold) / denom

                # Predict
                pred_dist = fold_likelihood(fold_gp(X_val_norm))
                pred_mean_norm = pred_dist.mean

                # Denormalize: reverse z-score and negation
                pred_neg = pred_mean_norm * y_std_fold + y_mean_fold
                pred_pos = -pred_neg  # Back to positive error rate

                # === COMPUTE METRICS ===
                mae = (pred_pos - y_fold_val).abs().mean().item()
                rmse = ((pred_pos - y_fold_val) ** 2).mean().sqrt().item()

                if len(pred_pos) > 1:
                    pred_np = pred_pos.cpu().numpy()
                    y_np = y_fold_val.cpu().numpy()
                    corr = np.corrcoef(pred_np, y_np)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0

                mae_scores.append(mae)
                rmse_scores.append(rmse)
                correlations.append(corr)

                if verbose:
                    print(f"  Fold {fold + 1}/{n_splits}: MAE={mae:.4f}, RMSE={rmse:.4f}, Corr={corr:.4f}")

        # === RESTORE ORIGINAL MODEL STATE ===
        if original_gp_state is not None and self.gp_model is not None:
            self.gp_model.load_state_dict(original_gp_state)
        if original_likelihood_state is not None and self.likelihood is not None:
            self.likelihood.load_state_dict(original_likelihood_state)
        if original_X_min is not None:
            self.X_min = original_X_min
        if original_X_max is not None:
            self.X_max = original_X_max
        if original_y_mean is not None:
            self.y_mean = original_y_mean
        if original_y_std is not None:
            self.y_std = original_y_std

        # === AGGREGATE RESULTS ===
        avg_mae = np.mean(mae_scores) if mae_scores else 0.0
        avg_rmse = np.mean(rmse_scores) if rmse_scores else 0.0
        avg_corr = np.mean(correlations) if correlations else 0.0
        std_mae = np.std(mae_scores) if mae_scores else 0.0
        std_corr = np.std(correlations) if correlations else 0.0

        if verbose:
            print(f"\nCross-Validation Results ({n_splits}-fold):")
            print(f"  MAE:  {avg_mae:.4f} ± {std_mae:.4f}")
            print(f"  RMSE: {avg_rmse:.4f}")
            print(f"  Corr: {avg_corr:.4f} ± {std_corr:.4f}")
            if skipped_folds:
                print(f"  WARNING: {len(skipped_folds)} folds skipped due to numerical issues: {skipped_folds}")
                print(f"    CV metrics may be optimistic - consider more training data")

        return {
            "cv_mae": avg_mae,
            "cv_mae_std": std_mae,
            "cv_rmse": avg_rmse,
            "cv_corr": avg_corr,
            "cv_corr_std": std_corr,
            "cv_n_samples": len(X_full),
            "cv_n_folds": n_splits,
            "cv_skipped_folds": len(skipped_folds),
        }
