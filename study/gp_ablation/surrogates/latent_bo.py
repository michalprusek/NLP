"""Latent Space Bayesian Optimization.

Implements:
- LatentSpaceBO: GP operates in flow's noise space z ~ N(0, I)

This is a PRIORITY NOVEL METHOD - the simplest approach that leverages
the trained flow model. Instead of doing GP in embedding space x,
we do GP in the flow's latent/noise space z.

Key Insight: The noise space z is Gaussian by construction (flow maps
N(0,I) -> data distribution), so GP works naturally with standard
Matern/RBF kernels.

Algorithm:
1. For initial data (x, y): invert to get z = flow.invert(x)
2. Fit GP on (z, y) pairs
3. BO loop:
   a. GP suggests z_new (acquisition optimization in z-space)
   b. Flow forward: x_new = flow(z_new)
   c. Evaluate: y_new = objective(x_new)
   d. Store (z_new, y_new), update GP

No inversion needed during BO loop - only at initialization!
"""

import logging
import math
import warnings
from typing import Optional, Tuple

import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate
from study.gp_ablation.surrogates.standard_gp import create_kernel

logger = logging.getLogger(__name__)


class LatentSpaceBO(BaseGPSurrogate):
    """Bayesian Optimization in flow's latent (noise) space.

    Instead of fitting GP in embedding space x, operates in the flow's
    noise space z where z ~ N(0, I). This has several advantages:

    1. Noise space is Gaussian by design - GP works naturally
    2. Standard lengthscale ~1.0 is reasonable (no MSR needed)
    3. No manifold geometry issues - z-space is Euclidean
    4. Flow model handles the hard part (mapping to valid embeddings)

    Key Insight: We DON'T invert during optimization! We propose z,
    transform to x, evaluate, and store (z, y). Only initial points
    need inversion.
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.flow_checkpoint = config.flow_checkpoint
        self.flow_steps = config.flow_steps
        self.invert_method = config.invert_method

        # In latent space, lengthscale ~1.0 is reasonable
        self.initial_lengthscale = 1.0

        # Flow model (loaded lazily)
        self._flow_model = None

        # Store latent representations
        self._train_Z: Optional[torch.Tensor] = None

    def _load_flow_model(self):
        """Lazily load the trained flow model."""
        if self._flow_model is not None:
            return self._flow_model

        if self.flow_checkpoint is None:
            raise ValueError(
                "LatentSpaceBO requires flow_checkpoint. "
                "Set config.flow_checkpoint to path of trained flow model."
            )

        logger.info(f"Loading flow model from {self.flow_checkpoint}")

        # Load checkpoint
        checkpoint = torch.load(
            self.flow_checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        # Get model architecture from config
        model_config = checkpoint.get("config", {})
        arch = model_config.get("arch", "mlp")
        scale = model_config.get("scale", "small")

        # Create model
        from study.flow_matching.models import create_model
        self._flow_model = create_model(arch, scale=scale)

        # Load weights (prefer EMA if available)
        if "ema_state_dict" in checkpoint:
            self._flow_model.load_state_dict(checkpoint["ema_state_dict"])
            logger.info("Loaded EMA weights")
        else:
            self._flow_model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model weights")

        self._flow_model = self._flow_model.to(self.device)
        self._flow_model.eval()

        # Load normalization stats if available
        self._stats = checkpoint.get("stats", None)
        if self._stats is not None:
            self._stats = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in self._stats.items()
            }
            logger.info("Loaded normalization stats")

        return self._flow_model

    def invert(self, x: torch.Tensor, max_error: float = 0.05) -> torch.Tensor:
        """Invert embedding x to noise z using ODE solver.

        Args:
            x: Embedding [D] or [B, D].
            max_error: Maximum reconstruction error (for validation).

        Returns:
            Noise z [D] or [B, D].

        Note: This is only called for initial data points, not during BO loop.
        """
        flow = self._load_flow_model()

        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        # Normalize if stats available
        if self._stats is not None:
            mean = self._stats.get("mean", torch.zeros(x.shape[-1], device=self.device))
            std = self._stats.get("std", torch.ones(x.shape[-1], device=self.device))
            x_norm = (x - mean) / (std + 1e-8)
        else:
            x_norm = x

        # Invert using ODE solver (backward in time: t=1 -> t=0)
        from torchdiffeq import odeint

        def velocity_backward(t, z):
            """Velocity field backward in time."""
            with torch.no_grad():
                # t goes from 0 to 1, but we integrate backward
                # So actual time is (1 - t)
                actual_t = 1 - t
                v = flow(z, torch.full((z.shape[0],), actual_t, device=z.device))
                return -v  # Negative for backward integration

        # Integrate from x (at t=1) to z (at t=0)
        t_span = torch.linspace(0, 1, self.flow_steps, device=self.device)

        with torch.no_grad():
            trajectory = odeint(
                velocity_backward,
                x_norm,
                t_span,
                method="rk4",
                options={"step_size": 1.0 / self.flow_steps},
            )
            z = trajectory[-1]

        # Validate reconstruction
        x_reconstructed = self.forward(z)

        # Unnormalize for error computation
        if self._stats is not None:
            x_reconstructed = x_reconstructed * (std + 1e-8) + mean

        error = (x - x_reconstructed).norm(dim=-1).mean().item()
        if error > max_error:
            logger.warning(
                f"Flow inversion error: {error:.4f} > {max_error}. "
                f"Results may be unreliable."
            )

        if was_1d:
            z = z.squeeze(0)

        return z

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Sample x from noise z using ODE solver.

        Args:
            z: Noise [D] or [B, D].

        Returns:
            Embedding x [D] or [B, D].
        """
        flow = self._load_flow_model()

        was_1d = z.dim() == 1
        if was_1d:
            z = z.unsqueeze(0)

        z = z.to(self.device)

        # Integrate from z (at t=0) to x (at t=1)
        from torchdiffeq import odeint

        def velocity(t, x):
            with torch.no_grad():
                v = flow(x, torch.full((x.shape[0],), t.item(), device=x.device))
                return v

        t_span = torch.linspace(0, 1, self.flow_steps, device=self.device)

        with torch.no_grad():
            trajectory = odeint(
                velocity,
                z,
                t_span,
                method="rk4",
                options={"step_size": 1.0 / self.flow_steps},
            )
            x = trajectory[-1]

        if was_1d:
            x = x.squeeze(0)

        return x

    def _create_model(
        self, train_Z: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create GP model in latent space."""
        covar_module = create_kernel(
            self.config.kernel,
            self.D,
            self.device,
            use_msr_prior=False,  # z-space doesn't need MSR
        ).to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_Z,
                train_Y=train_Y,
                covar_module=covar_module,
                input_transform=Normalize(d=self.D),
                outcome_transform=Standardize(m=1),
            )

        # In latent space, lengthscale ~1.0 is natural
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.D,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data.

        First inverts all embeddings X to latent Z, then fits GP on (Z, Y).
        This inversion is only done once at initialization.

        Args:
            train_X: Training embeddings [N, D].
            train_Y: Training targets [N] or [N, 1].
        """
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        # Invert all embeddings to latent space (one-time cost)
        logger.info(f"Inverting {len(train_X)} embeddings to latent space...")
        self._train_Z = self.invert(self._train_X)
        logger.info("Inversion complete")

        self.model = self._create_model(self._train_Z, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def update_from_latent(
        self, new_Z: torch.Tensor, new_Y: torch.Tensor
    ) -> None:
        """Update GP with new (z, y) pairs from BO loop.

        During BO, we propose z directly (no inversion needed).

        Args:
            new_Z: New latent points [B, D].
            new_Y: New target values [B] or [B, 1].
        """
        new_Z = new_Z.to(self.device)
        new_Y = new_Y.to(self.device)
        if new_Y.dim() == 1:
            new_Y = new_Y.unsqueeze(-1)

        if self._train_Z is None:
            self._train_Z = new_Z
            self._train_Y = new_Y
        else:
            self._train_Z = torch.cat([self._train_Z, new_Z], dim=0)
            self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        # Also update X for compatibility
        new_X = self.forward(new_Z)
        if self._train_X is None:
            self._train_X = new_X
        else:
            self._train_X = torch.cat([self._train_X, new_X], dim=0)

        self.model = self._create_model(self._train_Z, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and std for embeddings X.

        Note: This requires inverting X to Z first.
        For efficiency, use predict_latent() if you have Z directly.

        Args:
            X: Test embeddings [M, D].

        Returns:
            Tuple of (mean [M], std [M]).
        """
        Z = self.invert(X)
        return self.predict_latent(Z)

    def predict_latent(
        self, Z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and std for latent points Z.

        Args:
            Z: Test latent points [M, D].

        Returns:
            Tuple of (mean [M], std [M]).
        """
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            Z = Z.to(self.device)
            posterior = self.model.posterior(Z)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def suggest_latent(
        self,
        n_candidates: int = 1,
        n_samples: int = 512,
    ) -> torch.Tensor:
        """Suggest next latent points to evaluate.

        Args:
            n_candidates: Number of candidates to return.
            n_samples: Number of random samples for acquisition optimization.

        Returns:
            Suggested latent points [n_candidates, D].
        """
        from botorch.acquisition import LogExpectedImprovement

        self._ensure_fitted("suggestion")

        # Sample candidates from N(0, I) - the prior in latent space
        candidates = torch.randn(n_samples, self.D, device=self.device)

        # Evaluate acquisition function
        best_f = self._train_Y.max().item()
        ei = LogExpectedImprovement(model=self.model, best_f=best_f)

        with torch.no_grad():
            ei_values = ei(candidates.unsqueeze(-2))

        # Select top candidates
        top_indices = ei_values.argsort(descending=True)[:n_candidates]
        return candidates[top_indices]

    def suggest(
        self,
        n_candidates: int = 1,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_samples: int = 512,
    ) -> torch.Tensor:
        """Suggest next embeddings to evaluate.

        This suggests in latent space and transforms to embeddings.

        Args:
            n_candidates: Number of candidates to return.
            bounds: Ignored (latent space is unbounded).
            n_samples: Number of random samples.

        Returns:
            Suggested embeddings [n_candidates, D].
        """
        z_suggested = self.suggest_latent(n_candidates, n_samples)
        x_suggested = self.forward(z_suggested)
        return x_suggested

    def get_latent_training_data(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get training data in latent space (Z, Y)."""
        return self._train_Z, self._train_Y
