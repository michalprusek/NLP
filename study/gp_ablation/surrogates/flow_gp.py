"""Flow-based GP surrogates.

Implements:
- FlowGuidedGP: Standard GP with flow model projection to manifold
- BAxUSFlowGP: BAxUS projection + flow refinement
- BayesianFlowBO: GP with flow velocity as informative prior
- VelocityGuidedGP: Acquisition modified by flow velocity alignment

These methods leverage the trained flow model to improve GP optimization
by keeping candidates on the learned data manifold.
"""

import logging
import math
import warnings
from typing import Optional, Tuple, Callable

import torch
import torch.nn.functional as F
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate
from study.gp_ablation.surrogates.standard_gp import create_kernel

logger = logging.getLogger(__name__)


class FlowModel:
    """Wrapper for trained flow model with forward/backward integration."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        n_steps: int = 50,
    ):
        self.device = device
        self.n_steps = n_steps
        self._model = None
        self._stats = None
        self.checkpoint_path = checkpoint_path

        self._load_model()

    def _load_model(self):
        """Load trained flow model from checkpoint."""
        logger.info(f"Loading flow model from {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        model_config = checkpoint.get("config", {})
        arch = model_config.get("arch", "mlp")
        scale = model_config.get("scale", "small")

        from study.flow_matching.models import create_model

        self._model = create_model(arch, scale=scale)

        if "ema_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["ema_state_dict"])
        else:
            self._model.load_state_dict(checkpoint["model_state_dict"])

        self._model = self._model.to(self.device)
        self._model.eval()

        self._stats = checkpoint.get("stats", None)
        if self._stats is not None:
            self._stats = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in self._stats.items()
            }

    def velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Get velocity at point x and time t."""
        with torch.no_grad():
            t_tensor = torch.full((x.shape[0],), t, device=self.device)
            return self._model(x, t_tensor)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Integrate from noise z (t=0) to data x (t=1)."""
        from torchdiffeq import odeint

        def velocity_fn(t, x):
            return self.velocity(x, t.item())

        t_span = torch.linspace(0, 1, self.n_steps, device=self.device)

        with torch.no_grad():
            trajectory = odeint(
                velocity_fn, z, t_span,
                method="rk4",
                options={"step_size": 1.0 / self.n_steps},
            )
            return trajectory[-1]

    def project_to_manifold(self, x: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """Project x closer to data manifold using flow.

        Uses a few integration steps to move x toward the data distribution.
        """
        # Normalize if stats available
        if self._stats is not None:
            mean = self._stats.get("mean", torch.zeros_like(x[0]))
            std = self._stats.get("std", torch.ones_like(x[0]))
            x_norm = (x - mean) / (std + 1e-8)
        else:
            x_norm = x

        # Integrate for a short time (e.g., t=0.5 to t=1)
        # This moves x toward higher-density regions
        from torchdiffeq import odeint

        def velocity_fn(t, x):
            return self.velocity(x, 0.5 + 0.5 * t.item())

        t_span = torch.linspace(0, 1, n_steps, device=self.device)

        with torch.no_grad():
            trajectory = odeint(
                velocity_fn, x_norm, t_span,
                method="rk4",
            )
            x_projected = trajectory[-1]

        # Unnormalize
        if self._stats is not None:
            x_projected = x_projected * (std + 1e-8) + mean

        return x_projected


class FlowGuidedGP(BaseGPSurrogate):
    """GP with flow-guided candidate projection.

    Standard GP in embedding space, but candidates suggested by
    acquisition optimization are projected onto the learned manifold
    using the flow model.
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.flow_checkpoint = config.flow_checkpoint
        self.flow_steps = config.flow_steps
        self.initial_lengthscale = math.sqrt(self.D) / 10

        self._flow: Optional[FlowModel] = None

    def _get_flow(self) -> FlowModel:
        """Lazily load flow model."""
        if self._flow is None:
            if self.flow_checkpoint is None:
                raise ValueError("FlowGuidedGP requires flow_checkpoint")
            self._flow = FlowModel(
                self.flow_checkpoint, self.device, self.flow_steps
            )
        return self._flow

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create standard GP model."""
        covar_module = create_kernel(
            self.config.kernel,
            self.D,
            self.device,
            use_msr_prior=True,
        ).to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=covar_module,
                input_transform=Normalize(d=self.D),
                outcome_transform=Standardize(m=1),
            )

        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.D,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        self.model = self._create_model(self._train_X, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and std."""
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X = self._prepare_input(X)
            posterior = self.model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def suggest(
        self,
        n_candidates: int = 1,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_samples: int = 512,
        project_to_manifold: bool = True,
    ) -> torch.Tensor:
        """Suggest candidates with flow projection.

        Args:
            n_candidates: Number of candidates.
            bounds: Optional bounds.
            n_samples: Number of random samples.
            project_to_manifold: If True, project candidates using flow.

        Returns:
            Suggested embeddings [n_candidates, D].
        """
        from botorch.acquisition import LogExpectedImprovement

        self._ensure_fitted("suggestion")

        # Sample candidates
        if self._train_X is not None:
            mean = self._train_X.mean(dim=0)
            std = self._train_X.std(dim=0)
            candidates = mean + std * torch.randn(n_samples, self.D, device=self.device)
        else:
            candidates = torch.randn(n_samples, self.D, device=self.device)

        # Evaluate acquisition
        best_f = self._train_Y.max().item()
        ei = LogExpectedImprovement(model=self.model, best_f=best_f)

        with torch.no_grad():
            ei_values = ei(candidates.unsqueeze(-2))

        # Select top candidates
        top_indices = ei_values.argsort(descending=True)[:n_candidates]
        selected = candidates[top_indices]

        # Project to manifold using flow
        if project_to_manifold:
            flow = self._get_flow()
            selected = flow.project_to_manifold(selected)

        return selected


class BAxUSFlowGP(BaseGPSurrogate):
    """BAxUS projection + flow refinement.

    1. Project 1024D → 128D using BAxUS random matrix
    2. GP operates in 128D subspace
    3. Candidates lifted back to 1024D via S^T
    4. Flow model refines candidates to valid embeddings
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.target_dim = config.target_dim
        self.flow_checkpoint = config.flow_checkpoint
        self.flow_steps = config.flow_steps
        self.initial_lengthscale = math.sqrt(self.target_dim) / 10

        # Create BAxUS projection matrix
        torch.manual_seed(config.seed)
        self.S = self._create_embedding_matrix()
        self._train_X_embedded: Optional[torch.Tensor] = None
        self._flow: Optional[FlowModel] = None

    def _create_embedding_matrix(self) -> torch.Tensor:
        """Create sparse random embedding matrix."""
        S = torch.zeros(self.target_dim, self.D, device=self.device)

        for i in range(self.D):
            j = i % self.target_dim
            sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            S[j, i] = sign

        return S / math.sqrt(self.D / self.target_dim)

    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        """Project to subspace."""
        return X @ self.S.T

    def _lift(self, X_embedded: torch.Tensor) -> torch.Tensor:
        """Lift back to original space."""
        return X_embedded @ self.S

    def _get_flow(self) -> FlowModel:
        """Lazily load flow model."""
        if self._flow is None:
            if self.flow_checkpoint is None:
                raise ValueError("BAxUSFlowGP requires flow_checkpoint")
            self._flow = FlowModel(
                self.flow_checkpoint, self.device, self.flow_steps
            )
        return self._flow

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create GP in projected space."""
        covar_module = create_kernel(
            self.config.kernel,
            self.target_dim,
            self.device,
            use_msr_prior=True,
        ).to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=covar_module,
                input_transform=Normalize(d=self.target_dim),
                outcome_transform=Standardize(m=1),
            )

        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.target_dim,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        self._train_X_embedded = self._embed(self._train_X)

        self.model = self._create_model(self._train_X_embedded, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and std."""
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X = self._prepare_input(X)
            X_embedded = self._embed(X)
            posterior = self.model.posterior(X_embedded)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def suggest(
        self,
        n_candidates: int = 1,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_samples: int = 512,
    ) -> torch.Tensor:
        """Suggest candidates with BAxUS + flow."""
        from botorch.acquisition import LogExpectedImprovement

        self._ensure_fitted("suggestion")

        # Sample in embedded space
        candidates_embedded = torch.randn(
            n_samples, self.target_dim, device=self.device
        )

        # Scale by training data statistics
        if self._train_X_embedded is not None:
            mean = self._train_X_embedded.mean(dim=0)
            std = self._train_X_embedded.std(dim=0)
            candidates_embedded = candidates_embedded * std + mean

        # Evaluate acquisition
        best_f = self._train_Y.max().item()
        ei = LogExpectedImprovement(model=self.model, best_f=best_f)

        with torch.no_grad():
            ei_values = ei(candidates_embedded.unsqueeze(-2))

        # Select top candidates
        top_indices = ei_values.argsort(descending=True)[:n_candidates]
        selected_embedded = candidates_embedded[top_indices]

        # Lift to original space
        selected = self._lift(selected_embedded)

        # Refine with flow
        flow = self._get_flow()
        selected = flow.project_to_manifold(selected)

        return selected


class BayesianFlowBO(BaseGPSurrogate):
    """GP with flow velocity as informative prior.

    Uses flow velocity field to define a prior mean:
    - High velocity = far from data manifold = lower prior value
    - Low velocity = near data manifold = higher prior value

    This regularizes the GP to prefer on-manifold regions.
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.flow_checkpoint = config.flow_checkpoint
        self.flow_steps = config.flow_steps
        self.prior_weight = config.prior_weight
        self.initial_lengthscale = math.sqrt(self.D) / 10

        self._flow: Optional[FlowModel] = None

    def _get_flow(self) -> FlowModel:
        """Lazily load flow model."""
        if self._flow is None:
            if self.flow_checkpoint is None:
                raise ValueError("BayesianFlowBO requires flow_checkpoint")
            self._flow = FlowModel(
                self.flow_checkpoint, self.device, self.flow_steps
            )
        return self._flow

    def _compute_velocity_prior(self, X: torch.Tensor) -> torch.Tensor:
        """Compute prior value based on flow velocity magnitude.

        Low velocity = near manifold = high prior value.
        """
        flow = self._get_flow()

        # Get velocity at t=0.1 (near noise end, captures distance from manifold)
        v = flow.velocity(X, t=0.1)
        v_norm = v.norm(dim=-1)

        # Prior: negative velocity norm (lower velocity = better)
        # Normalize to roughly unit scale
        prior = -v_norm / (v_norm.mean() + 1e-6)

        return prior

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create GP model."""
        covar_module = create_kernel(
            self.config.kernel,
            self.D,
            self.device,
            use_msr_prior=True,
        ).to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=covar_module,
                input_transform=Normalize(d=self.D),
                outcome_transform=Standardize(m=1),
            )

        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.D,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        self.model = self._create_model(self._train_X, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and std, incorporating velocity prior."""
        self._ensure_fitted("prediction")
        self.model.eval()

        X = self._prepare_input(X)

        with torch.no_grad():
            posterior = self.model.posterior(X)
            gp_mean = posterior.mean.squeeze(-1)
            gp_std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

            # Add velocity prior
            velocity_prior = self._compute_velocity_prior(X)
            mean = gp_mean + self.prior_weight * velocity_prior

        return mean, gp_std


class VelocityGuidedGP(BaseGPSurrogate):
    """GP with velocity-guided acquisition function.

    Modifies acquisition by adding bonus for candidates where
    EI gradient aligns with flow velocity (toward manifold).
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.flow_checkpoint = config.flow_checkpoint
        self.flow_steps = config.flow_steps
        self.alignment_weight = config.alignment_weight
        self.initial_lengthscale = math.sqrt(self.D) / 10

        self._flow: Optional[FlowModel] = None

    def _get_flow(self) -> FlowModel:
        """Lazily load flow model."""
        if self._flow is None:
            if self.flow_checkpoint is None:
                raise ValueError("VelocityGuidedGP requires flow_checkpoint")
            self._flow = FlowModel(
                self.flow_checkpoint, self.device, self.flow_steps
            )
        return self._flow

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create GP model."""
        covar_module = create_kernel(
            self.config.kernel,
            self.D,
            self.device,
            use_msr_prior=True,
        ).to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=covar_module,
                input_transform=Normalize(d=self.D),
                outcome_transform=Standardize(m=1),
            )

        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.D,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        self.model = self._create_model(self._train_X, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and std."""
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X = self._prepare_input(X)
            posterior = self.model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def velocity_guided_acquisition(
        self, X: torch.Tensor, best_f: float
    ) -> torch.Tensor:
        """Compute velocity-guided acquisition values.

        Adds bonus for candidates where UCB gradient aligns with flow velocity.
        """
        from botorch.acquisition import LogExpectedImprovement

        self._ensure_fitted("acquisition")

        X = X.to(self.device).requires_grad_(True)

        # Get EI values
        ei = LogExpectedImprovement(model=self.model, best_f=best_f)
        ei_values = ei(X.unsqueeze(-2))

        # Compute gradient of EI
        ei_grad = torch.autograd.grad(
            ei_values.sum(), X, create_graph=False
        )[0]

        # Get flow velocity
        flow = self._get_flow()
        velocity = flow.velocity(X.detach(), t=0.5)

        # Compute alignment (cosine similarity)
        ei_grad_norm = F.normalize(ei_grad, dim=-1)
        velocity_norm = F.normalize(velocity, dim=-1)
        alignment = (ei_grad_norm * velocity_norm).sum(dim=-1)

        # Modified acquisition: EI * (1 + alignment_weight * alignment)
        modified = ei_values.squeeze() * (1 + self.alignment_weight * alignment)

        return modified.detach()
