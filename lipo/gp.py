"""Deep Kernel Gaussian Process for instruction optimization.

Provides:
- InstructionDeepKernelGP: GP with Matern 5/2 kernel on 32D VAE latent
- GPWithEI: Wrapper with Expected Improvement computation

GP operates directly on 32D VAE latent space (no adapter compression).
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
    """Gaussian Process with ARD Matern 5/2 kernel for instruction optimization.

    Operates directly on VAE latent space with ARD kernel.
    Inherits from GPyTorchModel for BoTorch compatibility (enables EI, etc.).

    Architecture:
        VAE latent (32D, normalized to [0,1]) -> Matern 5/2 kernel (ARD)
                                                        |
                                                GP(mean=0, K(latent))

    ARD (Automatic Relevance Determination):
        Per-dimension lengthscales allow the kernel to learn which
        dimensions are most relevant for prediction. Irrelevant dimensions get large
        lengthscales (effectively ignored).

    Note: No adapter compression - GP works directly on 32D VAE latent.
    This simplifies the architecture and avoids overfitting with limited training data.
    """

    _num_outputs = 1  # Required for BoTorch

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Union[GaussianLikelihood, FixedNoiseGaussianLikelihood],
    ):
        """Initialize GP.

        Args:
            train_x: Training VAE latents (N, D) where D is VAE latent dimension (32)
            train_y: Training targets (N,) - negated error rates (for BoTorch maximization)
            likelihood: Gaussian or FixedNoiseGaussian likelihood
        """
        # Ensure train_y is 1D for ExactGP
        train_y = train_y.squeeze(-1) if train_y.dim() > 1 else train_y
        super().__init__(train_x, train_y, likelihood)

        self._input_dim = train_x.shape[-1]  # Get actual VAE latent dimension (32)

        # === Data-driven prior estimation ===
        # Lengthscale: estimate from median pairwise distances per dimension
        # For [0,1] normalized inputs, typical distances are ~0.2-0.5
        # We use a broad prior centered on median distance to allow flexibility
        with torch.no_grad():
            # Sample subset for efficiency (max 100 pairs)
            n_samples = min(train_x.shape[0], 100)
            idx = torch.randperm(train_x.shape[0])[:n_samples]
            X_sample = train_x[idx]

            # Compute per-dimension distances
            # Shape: (n_samples, n_samples, D)
            diffs = X_sample.unsqueeze(0) - X_sample.unsqueeze(1)
            per_dim_dists = diffs.abs()

            # Median distance per dimension (excluding self-distances)
            mask = ~torch.eye(n_samples, dtype=torch.bool, device=train_x.device)
            median_dists = []
            for d in range(self._input_dim):
                dim_dists = per_dim_dists[:, :, d][mask]
                median_dists.append(dim_dists.median().item())
            median_dist = np.median(median_dists)

            # Clamp to reasonable range for numerical stability
            median_dist = max(0.05, min(0.5, median_dist))

        # Lengthscale prior: Gamma with mean = median_dist, moderate variance
        # Gamma(α, β) has mean = α/β, variance = α/β²
        # We want mean ≈ median_dist, std ≈ median_dist/2
        # α = 4, β = 4/median_dist gives mean=median_dist, std=median_dist/2
        ls_alpha = 4.0
        ls_beta = ls_alpha / median_dist

        # Outputscale prior: Since y is standardized (mean=0, std=1),
        # outputscale should be around 1.0
        # Gamma(2, 2) has mean=1, std=0.71 - reasonable for standardized targets
        os_alpha = 2.0
        os_beta = 2.0

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,  # Matern 5/2 - smooth but flexible
                ard_num_dims=self._input_dim,  # Per-dimension lengthscales (32D)
                lengthscale_prior=gpytorch.priors.GammaPrior(ls_alpha, ls_beta),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(os_alpha, os_beta),
        )

        # Initialize lengthscales to median distance (better starting point)
        self.covar_module.base_kernel.lengthscale = median_dist

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through GP.

        Pipeline: x (32D) -> Matern 5/2 ARD kernel

        Args:
            x: VAE latent (batch, 32) or (batch, n, 32), already unit cube normalized to [0,1]

        Returns:
            MultivariateNormal distribution over function values
        """
        latent = x

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
        Training: embeddings (768D) → frozen VAE encoder → z (32D) → normalize [0,1] → GP training
        Inference: z_norm (32D, [0,1]) → ARD Matern kernel → GP → qLogEI

    No adapter compression - GP works directly on 32D VAE latent space.
    """

    def __init__(
        self,
        device: str = "cuda",
    ):
        """Initialize GP wrapper.

        Args:
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Components (initialized during training)
        self.vae_with_adapter: Optional[nn.Module] = None  # VAEWithAdapter (frozen VAE wrapper)
        self.likelihood: Optional[GaussianLikelihood] = None
        self.gp_model: Optional[InstructionDeepKernelGP] = None

        # Training data - stored as 32D VAE latents
        self.X_train: Optional[torch.Tensor] = None  # (N, 32) VAE latents
        self.y_train: Optional[torch.Tensor] = None  # (N,) negated error rates (internal)
        self.fidelity_train: Optional[torch.Tensor] = None  # (N,) sample counts for each observation
        self._error_rates_original: Optional[torch.Tensor] = None  # (N,) positive error rates for noise computation

        # Normalization parameters (for 32D latents)
        self.X_min: Optional[torch.Tensor] = None
        self.X_max: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        # Best observed value (for EI)
        self.y_best: Optional[float] = None

        # Empirical Bayes prior parameters (set from training data)
        # Default to Beta(1,1) = uniform prior (equivalent to Laplace smoothing)
        self.beta_alpha: float = 1.0
        self.beta_beta: float = 1.0

        # Training stats (populated after train())
        self.training_stats: Dict[str, Any] = {}

    def set_training_data(
        self,
        embeddings: torch.Tensor,
        error_rates: torch.Tensor,
        fidelities: Optional[torch.Tensor] = None,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.0,
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
            beta_alpha: Empirical Bayes prior alpha (default 1.0 = uniform prior)
            beta_beta: Empirical Bayes prior beta (default 1.0 = uniform prior)
        """
        # Store Empirical Bayes prior parameters for noise computation
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
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

        self.X_train = vae_latents  # (N, 32) - VAE latent dimension

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
        """Compute observation noise variance using Beta posterior with Empirical Bayes prior.

        Uses Empirical Bayes prior Beta(α, β) learned from data.
        For error_rate p measured on n samples:
        - Posterior: Beta(α+errors, β+successes)
        - Posterior mean: (errors + α) / (n + α + β)
        - Posterior variance: p(1-p) / (n + α + β + 1)

        With Empirical Bayes:
        - Prior is centered at actual data mean (not 50%)
        - More accurate uncertainty for low-fidelity samples
        - Falls back to Beta(1,1) if prior not set

        Args:
            y: Beta posterior mean of error rates (N,) - already smoothed
            fidelity: Sample counts n (N,)

        Returns:
            Beta posterior variance for each point (N,)
        """
        # Get Empirical Bayes prior parameters
        alpha = self.beta_alpha
        beta = self.beta_beta

        # Beta posterior variance: p(1-p) / (n + α + β + 1)
        # where p = posterior mean, n = fidelity
        # This naturally handles p=0 or p=1 (no zero variance problem)
        variance = (y * (1 - y)) / (fidelity + alpha + beta + 1)

        # Minimal clamp for numerical stability only
        # Beta posterior already provides reasonable variance even for extreme p
        variance = torch.clamp(variance, min=1e-8, max=0.1)

        return variance

    def train(
        self,
        epochs: int = 3000,
        lr: float = 0.01,
        patience: int = 10,
        verbose: bool = True,
    ) -> bool:
        """Train GP on stored 32D VAE latents.

        Trains GP kernel on normalized VAE latents.
        VAE encoder is frozen. No adapter - GP operates directly on 32D latent.

        Uses FixedNoiseGaussianLikelihood with heteroscedastic noise based on
        Beta posterior variance: Var = p(1-p)/(n+3), where n is fidelity.

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

        # VAEWithAdapter must be pre-set (for encoding embeddings to VAE latents)
        if self.vae_with_adapter is None:
            raise RuntimeError(
                "vae_with_adapter must be set before training. "
                "Use VAEWithAdapter from encoder.py."
            )

        # Compute heteroscedastic noise from Beta posterior variance
        # Use posterior mean for variance computation: p(1-p)/(n+3)
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
        # The Beta posterior variance from fidelity is sufficient observation noise
        self.likelihood = FixedNoiseGaussianLikelihood(
            noise=noise_standardized,
            learn_additional_noise=False,
        ).to(self.device)
        # No adapter - GP operates directly on 32D VAE latent
        self.gp_model = InstructionDeepKernelGP(
            X_norm, y_norm, self.likelihood,
        ).to(self.device)

        # Register outcome transform for BoTorch compatibility
        self.gp_model.outcome_transform = self.outcome_transform

        # Log data-driven prior estimation
        if verbose:
            init_ls = self.gp_model.covar_module.base_kernel.lengthscale.mean().item()
            print(f"  Data-driven lengthscale prior: mean={init_ls:.4f} (from median pairwise distances)")
            print(f"  Outputscale prior: mean=1.0 (for standardized targets)")

        # Training loop
        self.gp_model.train()
        self.likelihood.train()

        # Lower learning rate for higher dimensions to prevent gradient explosion
        # With 32D we have ~35 parameters (32 ARD + kernel scale + outputscale)
        dim = self.X_train.shape[-1]
        if dim >= 32:
            actual_lr = lr * 0.25  # 0.0025 for 32D
        else:
            actual_lr = lr
        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=actual_lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss = float("inf")
        patience_counter = 0

        if verbose:
            print(f"Training GP (lr={actual_lr:.4f}, dim={dim})...")

        # Higher jitter for numerical stability with larger dimension
        if dim >= 32:
            jitter = 1e-3
        else:
            jitter = 1e-4
        with gpytorch.settings.cholesky_jitter(jitter):
            for epoch in range(epochs):
                try:
                    optimizer.zero_grad()
                    output = self.gp_model(X_norm)

                    # Early NaN detection - catch before Cholesky fails
                    if torch.isnan(output.mean).any() or torch.isnan(output.lazy_covariance_matrix.diagonal()).any():
                        print(f"ERROR: GP produced NaN at epoch {epoch + 1}")
                        print(f"  This indicates numerical instability in kernel parameters")
                        # Check which parameters went bad
                        for name, param in self.gp_model.named_parameters():
                            if torch.isnan(param).any():
                                print(f"  NaN in parameter: {name}")
                        return False

                    loss = -mll(output, y_norm)

                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"ERROR: NaN loss at epoch {epoch + 1}")
                        return False

                    loss.backward()

                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.gp_model.parameters(), max_norm=1.0)

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

        # Get final lengthscales for analysis
        final_ls = self.gp_model.covar_module.base_kernel.lengthscale.detach().cpu().squeeze()
        final_os = self.gp_model.covar_module.outputscale.detach().cpu().item()

        # Identify most relevant dimensions (smallest lengthscales = most important)
        sorted_dims = torch.argsort(final_ls)
        top_5_dims = sorted_dims[:5].tolist()
        top_5_ls = final_ls[sorted_dims[:5]].tolist()

        # Store training stats
        self.training_stats = {
            "epochs_trained": epoch + 1,
            "final_loss": float(best_loss),
            "early_stopped": patience_counter >= patience,
            "num_samples": len(X),
            "lengthscale_mean": float(final_ls.mean()),
            "lengthscale_min": float(final_ls.min()),
            "lengthscale_max": float(final_ls.max()),
            "outputscale": float(final_os),
            "top_5_relevant_dims": top_5_dims,
        }

        if verbose:
            print(f"  GP training complete (epochs={epoch + 1}, loss={best_loss:.4f})")
            print(f"  Final lengthscales: mean={final_ls.mean():.4f}, min={final_ls.min():.4f}, max={final_ls.max():.4f}")
            print(f"  Final outputscale: {final_os:.4f}")
            print(f"  Top 5 relevant dims (smallest lengthscale): {list(zip(top_5_dims, [f'{ls:.3f}' for ls in top_5_ls]))}")

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

        # Encode 768D embedding to 32D VAE latent if needed
        with torch.no_grad():
            if embedding.shape[-1] == 768:
                # Get the device where VAE actually is (may have been moved to CPU for eval)
                vae_device = self.vae_with_adapter.device
                embedding_for_vae = embedding.to(vae_device)
                # Encode through VAE to get 32D latent
                z_vae = self.vae_with_adapter.encode_vae(embedding_for_vae)
                # Move result back to GP device
                z_vae = z_vae.to(self.device)
            else:
                z_vae = embedding.to(self.device)

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
        """Get 32D VAE latent for embedding.

        Pipeline: embedding (768D) → VAE encoder → z (32D) → normalize

        Args:
            embedding: Instruction embedding (768,)

        Returns:
            Normalized VAE latent (32,)
        """
        if self.vae_with_adapter is None:
            raise RuntimeError("vae_with_adapter not initialized.")

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        # Get the device where VAE actually is (may have been moved to CPU for eval)
        vae_device = self.vae_with_adapter.device
        embedding = embedding.to(vae_device)

        # Encode to 32D VAE latent
        self.vae_with_adapter.eval()
        with torch.no_grad():
            z_vae = self.vae_with_adapter.encode_vae(embedding)

        # Move result back to GP device
        z_vae = z_vae.to(self.device)

        # Normalize 32D latent
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        z_norm = (z_vae - self.X_min) / denom

        # No adapter - return normalized VAE latent directly
        return z_norm.squeeze(0)

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

        Error rate is converted to Empirical Bayes posterior mean and negated internally
        for consistency with training data and BoTorch compatibility.

        Args:
            embedding: Instruction embedding (768,) or (1, 768)
            error_rate: Observed error rate (positive, will be regularized and negated internally)
            fidelity: Number of samples used for evaluation (for Beta posterior variance)

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

        # Get the device where VAE actually is (may have been moved to CPU for eval)
        vae_device = self.vae_with_adapter.device
        embedding = embedding.to(vae_device)

        # Encode to 32D VAE latent
        self.vae_with_adapter.eval()
        with torch.no_grad():
            new_z = self.vae_with_adapter.encode_vae(embedding)  # (1, 32)

        # Move result back to GP device for training data storage
        new_z = new_z.to(self.device)

        # Empirical Bayes posterior mean for consistency with training data
        # Formula: (errors + α) / (n + α + β) - data-driven prior instead of uniform
        # This matches the posterior mean applied in training.py:load_from_hyperband_evaluations()
        num_errors = error_rate * fidelity
        posterior_mean = (num_errors + self.beta_alpha) / (fidelity + self.beta_alpha + self.beta_beta)

        # Store posterior mean for noise computation (Beta posterior variance)
        new_error_original = torch.tensor([posterior_mean], dtype=torch.float32, device=self.device)
        # Negate for GP training (BoTorch maximization)
        new_y = torch.tensor([-posterior_mean], dtype=torch.float32, device=self.device)
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
