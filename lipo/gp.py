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
from botorch.models.transforms.input import Warp
from scipy.stats import norm
from scipy.special import erfcx, log1p, expm1
from typing import Tuple, Optional, Dict, Any, Union


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

    Uses optional Kumaraswamy warping + trainable adapter (D → 10D) + ARD Matern 5/2 kernel.
    Inherits from GPyTorchModel for BoTorch compatibility (enables EI, etc.).

    Architecture:
        VAE latent (D, normalized to [0,1]) -> Kumaraswamy Warp -> Adapter (D → 10D) -> Matern 5/2 kernel (ARD)
                                                                                                      |
                                                                                            GP(mean=0, K(latent))

    Kumaraswamy Input Warping:
        Transforms inputs in [0,1] non-linearly using learned concentration parameters (a, b).
        Helps GP handle non-uniform data distributions by concentrating density where data is.
        Applied BEFORE adapter because inputs must be in [0,1] for Kumaraswamy CDF.
        Formula: F(x; a,b) = 1 - (1-x^a)^b

    ARD (Automatic Relevance Determination):
        Per-dimension lengthscales allow the kernel to learn which dimensions
        are most relevant for prediction. Irrelevant dimensions get large
        lengthscales (effectively ignored).

    Training: Warping, adapter, and GP kernel are trained jointly on VAE latents.
    VAE encoder is frozen and applied before GP training.
    """

    _num_outputs = 1  # Required for BoTorch

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Union[GaussianLikelihood, FixedNoiseGaussianLikelihood],
        adapter: nn.Module,
        use_input_warping: bool = True,
    ):
        """Initialize GP.

        Args:
            train_x: Training VAE latents (N, D) where D is VAE latent dimension
            train_y: Training targets (N,) - negated error rates (for BoTorch maximization)
            likelihood: Gaussian or FixedNoiseGaussian likelihood
            adapter: Trainable adapter MLP (D → 10D)
            use_input_warping: Whether to apply Kumaraswamy input warping before adapter
        """
        # Ensure train_y is 1D for ExactGP
        train_y = train_y.squeeze(-1) if train_y.dim() > 1 else train_y
        super().__init__(train_x, train_y, likelihood)

        self.adapter = adapter

        # Kumaraswamy input warping on VAE latents (BEFORE adapter)
        # Learns concentration parameters (a, b) per dimension to handle non-uniform data
        # Applied before adapter because:
        # 1. X_norm is already in [0,1] (unit cube normalized) - required by Kumaraswamy
        # 2. Warping helps GP model non-uniform input distributions
        # Note: Only set attribute if warping is enabled to avoid BoTorch GPyTorchModel issues
        self._use_input_warping = use_input_warping
        self._input_dim = train_x.shape[-1]  # Get actual VAE latent dimension
        if use_input_warping:
            self.input_transform = Warp(
                d=self._input_dim,  # Dynamic: matches actual VAE latent dimension
                indices=list(range(self._input_dim)),  # Warp all dims
            )

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,  # Matern 5/2 - smooth but flexible
                ard_num_dims=10,  # Per-dimension lengthscales (adapter output dim)
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through GP.

        Pipeline: x (D) -> warping (if enabled) -> adapter (10D) -> kernel

        Args:
            x: VAE latent (batch, D) or (batch, n, D), already unit cube normalized to [0,1]

        Returns:
            MultivariateNormal distribution over function values
        """
        # Apply Kumaraswamy input warping BEFORE adapter
        # x is already in [0,1] from unit cube normalization, which is required for warping
        if self._use_input_warping:
            # Clamp to (eps, 1-eps) to avoid numerical issues at boundaries
            # Kumaraswamy CDF has numerical issues at exactly 0 or 1
            eps = 1e-6
            x = x.clamp(min=eps, max=1 - eps)
            x = self.input_transform(x)

        # Apply adapter: D → 10D
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
        Training: embeddings (768D) → frozen VAE encoder → z (D) → normalize [0,1] → warping+adapter+GP training
        Inference: z_norm (D, [0,1]) → Kumaraswamy Warp → adapter (10D) → ARD Matern kernel → GP → qLogEI

    Where D is the VAE latent dimension (configurable, default 16).
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

        # Training data - stored as 32D VAE latents
        self.X_train: Optional[torch.Tensor] = None  # (N, 32) VAE latents
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

        Transforms embeddings to 32D VAE latents using frozen VAE encoder.

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

        # Transform to 32D VAE latents using frozen encoder
        self.vae_with_adapter.eval()
        with torch.no_grad():
            vae_latents = self.vae_with_adapter.encode_vae(embeddings)

        # Validate VAE output dimension
        expected_vae_dim = self.vae_with_adapter.vae_latent_dim
        if vae_latents.shape[-1] != expected_vae_dim:
            raise RuntimeError(
                f"VAE encoder output dimension mismatch: expected {expected_vae_dim}, "
                f"got {vae_latents.shape[-1]}. Check VAE configuration."
            )

        self.X_train = vae_latents  # (N, 32)

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
        use_input_warping: bool = True,
    ) -> bool:
        """Train GP on stored 32D VAE latents.

        Trains adapter (32D→10D), Kumaraswamy warping, and GP kernel jointly.
        VAE encoder is frozen.

        Uses FixedNoiseGaussianLikelihood with heteroscedastic noise based on
        Bernoulli variance: Var = p(1-p)/n, where n is fidelity (sample count).

        Args:
            epochs: Maximum training epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress
            use_input_warping: Whether to apply Kumaraswamy input warping on 10D adapter output

        Returns:
            True if training succeeded
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("No training data. Call set_training_data() first.")

        X = self.X_train  # Already 32D VAE latents
        y = self.y_train

        # Unit cube normalization for 32D latents
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
        # Use .item() to avoid tensor shape issues, add robust fallback
        y_std_val = self.y_std.item() if self.y_std.numel() == 1 else self.y_std.squeeze().item()
        y_std_sq = max(y_std_val ** 2, 1e-6)  # Ensure positive
        noise_standardized = raw_noise / y_std_sq
        # Clamp both min and max to prevent numerical issues:
        # - min: very small noise causes GP numerical instability
        # - max: very large noise makes observations uninformative
        noise_standardized = torch.clamp(noise_standardized, min=1e-4, max=1.0)
        # Ensure noise tensor is contiguous and correct shape
        noise_standardized = noise_standardized.view(-1).contiguous()

        if verbose:
            print(f"  Observation noise: min={raw_noise.min():.6f}, max={raw_noise.max():.6f}")
            print(f"  Fidelity range: {self.fidelity_train.min().item():.0f} - {self.fidelity_train.max().item():.0f}")

        # FixedNoiseGaussianLikelihood with heteroscedastic noise
        # NOTE: learn_additional_noise=False to avoid CUDA errors with second_noise_covar
        # The Bernoulli variance from fidelity is sufficient observation noise
        self.likelihood = FixedNoiseGaussianLikelihood(
            noise=noise_standardized,
            learn_additional_noise=False,
        ).to(self.device)
        self.gp_model = InstructionDeepKernelGP(
            X_norm, y_norm, self.likelihood, adapter,
            use_input_warping=use_input_warping,
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

        return True

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

        # Encode 768D embedding to 32D VAE latent if needed
        with torch.no_grad():
            if embedding.shape[-1] == 768:
                # Encode through VAE to get 32D latent
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
                # Get posterior from GP model directly, NOT through likelihood
                # FixedNoiseGaussianLikelihood stores noise for training points only;
                # passing test points through it without explicit noise causes warnings
                # and incorrect uncertainty estimates
                posterior = self.gp_model(X_norm)
                mean_norm = posterior.mean.item()
                std_norm = posterior.stddev.item()
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

        # Clamp to valid error rate range [0, 1]
        # GP extrapolation can produce values outside this range
        mean = max(0.0, min(1.0, mean))

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

        Pipeline: embedding (768D) → VAE encoder → z (32D) → normalize → adapter → z_gp (10D)

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

        # First encode to 32D VAE latent
        self.vae_with_adapter.eval()
        with torch.no_grad():
            z_vae = self.vae_with_adapter.encode_vae(embedding)

        # Normalize 32D latent
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        z_norm = (z_vae - self.X_min) / denom

        # Apply adapter: 32D → 10D
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

        # Encode to 32D VAE latent
        self.vae_with_adapter.eval()
        with torch.no_grad():
            new_z = self.vae_with_adapter.encode_vae(embedding)  # (1, 32)

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
