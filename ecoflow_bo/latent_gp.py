"""
Latent Space GP: Gaussian Process surrogate for Bayesian Optimization in 8D latent space.

Key features:
- Coarse-to-fine optimization: Start with 2D, progressively unlock to 8D
- ARD kernel: Automatic Relevance Determination for Matryoshka structure
- BoTorch integration for acquisition optimization
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import warnings

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

from .config import GPConfig


class LatentSpaceGP:
    """
    GP surrogate f(z) → y for Bayesian Optimization in latent space.

    Supports coarse-to-fine optimization:
    1. Stage 0: GP on dims [0,1] - 2D is easy with 20 points
    2. Stage 1: GP on dims [0:4] - 4D with accumulated data
    3. Stage 2: GP on dims [0:8] - Full 8D for fine-tuning

    Uses Matryoshka encoder's property that first dims carry most info.
    """

    def __init__(self, config: Optional[GPConfig] = None):
        if config is None:
            config = GPConfig()

        self.config = config
        self.active_dims_schedule = config.active_dims_schedule
        self.points_per_stage = config.points_per_stage

        self.current_stage = 0
        self.active_dims = self.active_dims_schedule[0]

        self.gp = None
        self.train_z = None
        self.train_y = None
        self.device = None
        self.dtype = None

    @property
    def n_active_dims(self) -> int:
        """Number of currently active dimensions."""
        return len(self.active_dims)

    @property
    def n_points(self) -> int:
        """Number of training points."""
        if self.train_z is None:
            return 0
        return self.train_z.shape[0]

    def _project_to_active(self, z: torch.Tensor) -> torch.Tensor:
        """Project z to active dimensions only."""
        return z[:, self.active_dims]

    def _should_advance_stage(self) -> bool:
        """Check if we should advance to next stage."""
        if self.current_stage >= len(self.active_dims_schedule) - 1:
            return False

        points_needed = sum(self.points_per_stage[: self.current_stage + 1])
        return self.n_points >= points_needed

    def advance_stage(self) -> bool:
        """
        Advance to next stage if conditions are met.

        Returns:
            True if advanced, False otherwise
        """
        if not self._should_advance_stage():
            return False

        self.current_stage += 1
        self.active_dims = self.active_dims_schedule[self.current_stage]
        print(f"[GP] Advanced to stage {self.current_stage}, active dims: {self.active_dims}")

        # Refit GP with new dimensions
        if self.train_z is not None:
            self._fit_gp()

        return True

    def _create_gp(
        self, train_z: torch.Tensor, train_y: torch.Tensor
    ) -> SingleTaskGP:
        """Create GP model with proper priors."""
        n_dims = train_z.shape[1]

        # Matern 5/2 kernel with ARD
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=n_dims,
                lengthscale_prior=GammaPrior(3.0, 6.0),  # Prior mean ≈ 0.5
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),  # Prior mean ≈ 13
        )

        gp = SingleTaskGP(
            train_z,
            train_y.unsqueeze(-1) if train_y.dim() == 1 else train_y,
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
        )

        return gp

    def _fit_gp(self):
        """Fit GP to current training data."""
        if self.train_z is None or self.train_z.shape[0] < 2:
            return

        # Project to active dimensions
        train_z_active = self._project_to_active(self.train_z)
        train_y = self.train_y

        # Create and fit GP
        self.gp = self._create_gp(train_z_active, train_y)
        self.gp = self.gp.to(self.device)

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

        import logging
        logger = logging.getLogger(__name__)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                logger.error(
                    f"GP fitting failed: {e}. "
                    f"Train data: z shape={train_z_active.shape}, "
                    f"y range=[{train_y.min():.3f}, {train_y.max():.3f}]"
                )
                raise
            # Log warnings at appropriate severity level
            for w in caught_warnings:
                msg = str(w.message)
                if "cholesky" in msg.lower() or "singular" in msg.lower():
                    logger.error(f"GP fitting critical warning: {msg}")
                else:
                    logger.warning(f"GP fitting warning: {msg}")

    def fit(self, z: torch.Tensor, y: torch.Tensor):
        """
        Fit GP to (z, y) pairs.

        Args:
            z: Latent codes [N, latent_dim]
            y: Objective values [N]
        """
        self.device = z.device
        self.dtype = z.dtype
        self.train_z = z.clone()
        self.train_y = y.clone()

        self._fit_gp()

    def update(self, z_new: torch.Tensor, y_new: torch.Tensor):
        """
        Add new observations and refit.

        Args:
            z_new: New latent codes [M, latent_dim]
            y_new: New objective values [M]
        """
        if self.train_z is None:
            self.fit(z_new, y_new)
            return

        # Concatenate new data
        self.train_z = torch.cat([self.train_z, z_new], dim=0)
        self.train_y = torch.cat([self.train_y, y_new], dim=0)

        # Check if we should advance stage
        self.advance_stage()

        # Refit GP
        self._fit_gp()

    def predict(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and variance at z.

        Args:
            z: Test points [M, latent_dim]

        Returns:
            mean: Posterior mean [M]
            var: Posterior variance [M]
        """
        if self.gp is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        z_active = self._project_to_active(z)
        self.gp.eval()

        with torch.no_grad():
            posterior = self.gp.posterior(z_active)
            mean = posterior.mean.squeeze(-1)
            var = posterior.variance.squeeze(-1)

        return mean, var

    def get_best(self) -> Tuple[torch.Tensor, float]:
        """
        Get best observed (z, y) pair.

        Returns:
            best_z: Best latent code [latent_dim]
            best_y: Best objective value
        """
        if self.train_z is None:
            raise RuntimeError("No observations yet.")

        best_idx = self.train_y.argmax()
        return self.train_z[best_idx], self.train_y[best_idx].item()

    def get_lengthscales(self) -> torch.Tensor:
        """Get learned lengthscales (reflects dimension importance)."""
        if self.gp is None:
            return None

        return self.gp.covar_module.base_kernel.lengthscale.detach().squeeze()


class CoarseToFineGP(LatentSpaceGP):
    """
    GP with explicit coarse-to-fine optimization strategy.

    Provides helper methods for the optimizer to know which
    dimensions to perturb during candidate generation.
    """

    def __init__(self, config: Optional[GPConfig] = None):
        super().__init__(config)

    def get_perturbation_dims(self) -> List[int]:
        """
        Get dimensions that should be perturbed during optimization.

        At each stage, we perturb only the newly active dimensions
        to focus search on unexplored aspects.
        """
        if self.current_stage == 0:
            # First stage: perturb all active dims
            return self.active_dims
        else:
            # Later stages: focus on newly added dims
            prev_dims = set(self.active_dims_schedule[self.current_stage - 1])
            curr_dims = set(self.active_dims)
            new_dims = list(curr_dims - prev_dims)
            # Also include some perturbation of previous dims
            return self.active_dims  # Full perturbation for now

    def get_search_bounds(
        self, z_center: torch.Tensor, stage_scale: float = 2.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get search bounds for candidate generation.

        Bounds are tighter in early stages (focus on promising region)
        and expand in later stages (fine-tuning).

        Args:
            z_center: Center point for search [latent_dim]
            stage_scale: Scale factor for bounds

        Returns:
            lower, upper: Bounds for each dimension [latent_dim]
        """
        latent_dim = z_center.shape[-1]

        # Base bounds from prior N(0, I): roughly [-3, 3]
        lower = torch.full((latent_dim,), -3.0, device=z_center.device)
        upper = torch.full((latent_dim,), 3.0, device=z_center.device)

        # Tighter bounds for inactive dimensions (keep at 0)
        inactive = set(range(latent_dim)) - set(self.active_dims)
        for dim in inactive:
            lower[dim] = 0.0
            upper[dim] = 0.0

        # Tighter bounds for active dims in early stages
        scale = 1.0 + (self.current_stage / len(self.active_dims_schedule)) * stage_scale
        for dim in self.active_dims:
            # Center around z_center with scaled width
            half_width = scale * 1.5  # ±1.5σ * scale
            lower[dim] = max(-3.0, z_center[dim].item() - half_width)
            upper[dim] = min(3.0, z_center[dim].item() + half_width)

        return lower, upper

    def get_stage_info(self) -> dict:
        """Get current stage information for logging."""
        return {
            "stage": self.current_stage,
            "max_stages": len(self.active_dims_schedule),
            "active_dims": self.active_dims,
            "n_active_dims": len(self.active_dims),
            "n_points": self.n_points,
            "points_for_next_stage": (
                sum(self.points_per_stage[: self.current_stage + 1])
                if self.current_stage < len(self.active_dims_schedule) - 1
                else None
            ),
        }
