"""Configuration for GP ablation study.

Provides GPConfig dataclass with all GP method, kernel, acquisition,
and method-specific hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPConfig:
    """Configuration for GP surrogate ablation experiments.

    Core settings:
        method: GP method name (standard_msr, turbo, saasbo, baxus, etc.)
        kernel: Kernel type (matern52, matern32, rbf, arccosine, geodesic_matern52)
        acquisition: Acquisition function (log_ei, ucb, thompson)
        seed: Random seed for reproducibility

    Projection methods (BAxUS):
        target_dim: Target dimensionality for subspace projection

    SAASBO settings:
        nuts_warmup: NUTS MCMC warmup steps
        nuts_samples: NUTS MCMC samples

    TuRBO settings:
        length_init: Initial trust region length
        length_min: Minimum trust region length
        length_max: Maximum trust region length
        success_tolerance: Successes before expansion
        failure_tolerance: Failures before shrinking

    Heteroscedastic settings:
        n_eval: Samples per accuracy evaluation (for binomial noise)

    Riemannian BO settings:
        normalize_inputs: Project inputs to unit sphere S^{d-1}

    GEBO settings:
        use_full_gradient: Use full gradient (expensive) vs directional
        n_directions: Number of random directions for directional GEBO

    LaMCTS settings:
        max_depth: Maximum tree depth
        n_samples_per_leaf: Samples per leaf node

    Hybrid method settings:
        n_grad_steps: Gradient refinement steps (turbo_grad)
        grad_lr: Gradient refinement learning rate
        flow_checkpoint: Path to trained flow model checkpoint
        flow_steps: ODE integration steps for flow sampling

    Novel method settings:
        invert_method: Flow inversion method (ode, fixed_point)
        prior_weight: Weight for flow velocity prior (bayesian_flow_bo)
        alignment_weight: Weight for velocity alignment (velocity_acq)
        fidelity_start: Starting fidelity for curriculum BO
        fidelity_schedule: Fidelity schedule type (linear, exponential)
    """

    # Core settings
    method: str = "standard_msr"
    kernel: str = "matern52"
    acquisition: str = "log_ei"
    seed: int = 42

    # Dimensionality
    input_dim: int = 1024

    # Projection methods (BAxUS)
    target_dim: int = 128

    # SAASBO settings
    nuts_warmup: int = 256
    nuts_samples: int = 128

    # TuRBO settings
    length_init: float = 0.8
    length_min: float = 0.01
    length_max: float = 1.6
    success_tolerance: int = 3
    failure_tolerance: int = 5

    # Heteroscedastic settings
    n_eval: int = 150

    # Riemannian BO settings
    normalize_inputs: bool = True

    # GEBO settings
    use_full_gradient: bool = False
    n_directions: int = 1

    # LaMCTS settings
    max_depth: int = 20
    n_samples_per_leaf: int = 5

    # Hybrid method settings
    n_grad_steps: int = 5
    grad_lr: float = 0.01
    flow_checkpoint: Optional[str] = None
    flow_steps: int = 50

    # Novel method settings (latent_bo, bayesian_flow_bo, etc.)
    invert_method: str = "ode"
    prior_weight: float = 0.1
    alignment_weight: float = 0.3
    fidelity_start: float = 0.1
    fidelity_schedule: str = "linear"

    # Evaluation settings
    n_initial: int = 10
    n_iterations: int = 50
    batch_size: int = 1

    # Paths
    data_path: str = "datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt"
    results_dir: str = "study/results/gp_ablation"

    # Wandb settings
    wandb_project: str = "gp-ablation-study"
    wandb_group: str = "gp-ablation"

    @property
    def run_name(self) -> str:
        """Generate unique run name from config fields."""
        parts = [self.method]

        # Add kernel if not default
        if self.kernel != "matern52":
            parts.append(self.kernel)

        # Add acquisition if not default
        if self.acquisition != "log_ei":
            parts.append(self.acquisition)

        # Add method-specific parameters
        if self.method == "baxus":
            parts.append(f"d{self.target_dim}")
        elif self.method == "turbo":
            parts.append(f"l{self.length_init}")
        elif self.method == "saasbo":
            parts.append(f"s{self.nuts_samples}")
        elif self.method == "turbo_grad":
            parts.append(f"g{self.n_grad_steps}")
        elif self.method == "gebo":
            parts.append(f"dir{self.n_directions}")
        elif self.method == "latent_bo":
            parts.append(self.invert_method)

        # Add seed if not default
        if self.seed != 42:
            parts.append(f"s{self.seed}")

        return "-".join(parts)

    def to_dict(self) -> dict:
        """Convert config to dict for logging."""
        return {
            "method": self.method,
            "kernel": self.kernel,
            "acquisition": self.acquisition,
            "seed": self.seed,
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
            "nuts_warmup": self.nuts_warmup,
            "nuts_samples": self.nuts_samples,
            "length_init": self.length_init,
            "length_min": self.length_min,
            "length_max": self.length_max,
            "success_tolerance": self.success_tolerance,
            "failure_tolerance": self.failure_tolerance,
            "n_eval": self.n_eval,
            "normalize_inputs": self.normalize_inputs,
            "use_full_gradient": self.use_full_gradient,
            "n_directions": self.n_directions,
            "max_depth": self.max_depth,
            "n_samples_per_leaf": self.n_samples_per_leaf,
            "n_grad_steps": self.n_grad_steps,
            "grad_lr": self.grad_lr,
            "flow_checkpoint": self.flow_checkpoint,
            "flow_steps": self.flow_steps,
            "invert_method": self.invert_method,
            "prior_weight": self.prior_weight,
            "alignment_weight": self.alignment_weight,
            "fidelity_start": self.fidelity_start,
            "fidelity_schedule": self.fidelity_schedule,
            "n_initial": self.n_initial,
            "n_iterations": self.n_iterations,
            "batch_size": self.batch_size,
            "data_path": self.data_path,
            "results_dir": self.results_dir,
            "run_name": self.run_name,
        }
