"""Staged Latent Space Bayesian Optimization (Staged LSBO).

Implements ECI-BO style staged optimization for Matryoshka Funnel Flow latents.
Instead of optimizing all dimensions at once (which GPs struggle with),
we decompose into stages of 16 dimensions each.

Example with 64D latent (4 stages):
    Stage 1: GP optimizes z[0:16], z[16:64]=0  → 50 trials (coarse search)
    Stage 2: Fix z[0:16], GP optimizes z[16:32], z[32:64]=0  → 40 trials
    Stage 3: Fix z[0:32], GP optimizes z[32:48], z[48:64]=0  → 30 trials
    Stage 4: Fix z[0:48], GP optimizes z[48:64]  → 20 trials
    Total: 140 evaluations

For 128D latent (primary architecture), use 8 stages with ~225 total evaluations.

References:
    - Expected Coordinate Improvement (2024): https://arxiv.org/abs/2404.11917
    - TuRBO: Trust Region BO: https://arxiv.org/abs/1910.01739
    - Matryoshka Representation Learning (NeurIPS 2022)
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Conditional imports for Bayesian optimization
try:
    from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    logger.warning(
        "BoTorch not available. Staged LSBO will not work. "
        "Install with: pip install botorch gpytorch"
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StagedLSBOConfig:
    """Configuration for Staged LSBO.

    Attributes:
        matryoshka_dims: Matryoshka dimension levels (defines stage boundaries)
        trials_per_stage: Number of BO trials for each stage
        latent_dim: Full latent dimension
        latent_bounds: Bounds for latent space search (default: [-3, 3])
        random_init_per_stage: Random initial points per stage
        use_turbo: Use TuRBO-style trust regions within stages
        turbo_batch_size: Batch size for TuRBO
        dtype: Tensor dtype for BO
        device: Device for BO computations
    """

    matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64)
    trials_per_stage: Tuple[int, ...] = (50, 40, 30, 20)
    latent_dim: int = 64
    latent_bounds: Tuple[float, float] = (-3.0, 3.0)
    random_init_per_stage: int = 5
    use_turbo: bool = False
    turbo_batch_size: int = 1
    dtype: torch.dtype = field(default=torch.float64)
    device: str = "cuda"

    def __post_init__(self):
        assert len(self.matryoshka_dims) == len(self.trials_per_stage), (
            f"matryoshka_dims ({len(self.matryoshka_dims)}) and trials_per_stage "
            f"({len(self.trials_per_stage)}) must have same length"
        )
        assert self.matryoshka_dims[-1] == self.latent_dim, (
            f"Last matryoshka_dim ({self.matryoshka_dims[-1]}) must equal latent_dim ({self.latent_dim})"
        )

    @property
    def total_trials(self) -> int:
        """Total number of objective evaluations."""
        return sum(self.trials_per_stage)

    @property
    def stage_boundaries(self) -> List[Tuple[int, int]]:
        """Get (start, end) index pairs for each stage."""
        boundaries = []
        prev = 0
        for dim in self.matryoshka_dims:
            boundaries.append((prev, dim))
            prev = dim
        return boundaries


# =============================================================================
# Stage Result
# =============================================================================


@dataclass
class StageResult:
    """Result from a single optimization stage."""

    stage: int
    dim_range: Tuple[int, int]  # (start, end) of dimensions optimized
    best_z: Tensor  # Best latent found in this stage (full latent_dim)
    best_value: float  # Best objective value
    all_z: Tensor  # All evaluated latents (n_trials, latent_dim)
    all_values: Tensor  # All objective values (n_trials,)
    n_trials: int


@dataclass
class StagedLSBOResult:
    """Complete result from staged optimization."""

    best_z: Tensor  # Final best latent
    best_value: float  # Final best objective value
    stage_results: List[StageResult]  # Per-stage results
    total_evaluations: int  # Total objective evaluations

    @property
    def all_values(self) -> Tensor:
        """Concatenate all objective values across stages."""
        return torch.cat([sr.all_values for sr in self.stage_results])

    def get_convergence_curve(self) -> Tensor:
        """Get cumulative best value over all evaluations."""
        all_vals = self.all_values
        cummin = torch.zeros_like(all_vals)
        best = float('inf')
        for i, v in enumerate(all_vals):
            best = min(best, v.item())
            cummin[i] = best
        return cummin


# =============================================================================
# Staged LSBO Optimizer
# =============================================================================


class StagedLSBO:
    """Staged Latent Space Bayesian Optimization.

    Decomposes high-dimensional latent optimization into sequential low-dimensional
    stages, leveraging Matryoshka importance ordering.

    Usage:
        ```python
        config = StagedLSBOConfig()
        optimizer = StagedLSBO(config)

        def objective(z: Tensor) -> Tensor:
            # z: (batch, 64) latent vectors
            # returns: (batch,) objective values (lower is better)
            embeddings = flow.decode(z)
            scores = evaluate_embeddings(embeddings)
            return -scores  # Negate if higher is better

        result = optimizer.optimize(objective)
        best_z = result.best_z
        ```
    """

    def __init__(self, config: StagedLSBOConfig):
        """Initialize Staged LSBO optimizer.

        Args:
            config: Optimization configuration
        """
        if not BOTORCH_AVAILABLE:
            raise ImportError(
                "BoTorch is required for StagedLSBO. "
                "Install with: pip install botorch"
            )

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        logger.info(
            f"StagedLSBO initialized: {len(config.matryoshka_dims)} stages, "
            f"dims={config.matryoshka_dims}, trials={config.trials_per_stage}, "
            f"total={config.total_trials} evaluations"
        )

    def optimize(
        self,
        objective: Callable[[Tensor], Tensor],
        initial_z: Optional[Tensor] = None,
        verbose: bool = True,
    ) -> StagedLSBOResult:
        """Run staged Bayesian optimization.

        Args:
            objective: Function (batch, latent_dim) -> (batch,) returning values to MINIMIZE
            initial_z: Optional initial latent (uses zeros if not provided)
            verbose: Print progress

        Returns:
            StagedLSBOResult with best latent and optimization history
        """
        config = self.config
        boundaries = config.stage_boundaries

        # Initialize full latent
        if initial_z is None:
            current_z = torch.zeros(
                config.latent_dim, device=self.device, dtype=self.dtype
            )
        else:
            current_z = initial_z.to(device=self.device, dtype=self.dtype).clone()

        stage_results = []
        global_best_z = current_z.clone()
        global_best_value = float('inf')

        for stage_idx, ((start, end), n_trials) in enumerate(
            zip(boundaries, config.trials_per_stage)
        ):
            if verbose:
                logger.info(
                    f"\n{'='*60}\n"
                    f"Stage {stage_idx + 1}/{len(boundaries)}: "
                    f"Optimizing z[{start}:{end}] ({end - start}D), {n_trials} trials\n"
                    f"{'='*60}"
                )

            # Run stage optimization
            stage_result = self._optimize_stage(
                objective=objective,
                current_z=current_z,
                dim_start=start,
                dim_end=end,
                n_trials=n_trials,
                stage_idx=stage_idx,
                verbose=verbose,
            )

            stage_results.append(stage_result)

            # Update current_z with best from this stage
            current_z = stage_result.best_z.clone()

            # Update global best
            if stage_result.best_value < global_best_value:
                global_best_value = stage_result.best_value
                global_best_z = stage_result.best_z.clone()

            if verbose:
                logger.info(
                    f"Stage {stage_idx + 1} complete: best_value={stage_result.best_value:.6f}"
                )

        total_evals = sum(sr.n_trials for sr in stage_results)

        if verbose:
            logger.info(
                f"\n{'='*60}\n"
                f"Staged LSBO complete: {total_evals} total evaluations\n"
                f"Final best value: {global_best_value:.6f}\n"
                f"{'='*60}"
            )

        return StagedLSBOResult(
            best_z=global_best_z,
            best_value=global_best_value,
            stage_results=stage_results,
            total_evaluations=total_evals,
        )

    def _optimize_stage(
        self,
        objective: Callable[[Tensor], Tensor],
        current_z: Tensor,
        dim_start: int,
        dim_end: int,
        n_trials: int,
        stage_idx: int,
        verbose: bool,
    ) -> StageResult:
        """Optimize a single stage (subset of dimensions).

        Args:
            objective: Objective function
            current_z: Current full latent vector
            dim_start: Start index of dimensions to optimize
            dim_end: End index of dimensions to optimize
            n_trials: Number of trials for this stage
            stage_idx: Stage index (for active_dim inference)
            verbose: Print progress

        Returns:
            StageResult for this stage
        """
        config = self.config
        stage_dim = dim_end - dim_start
        bounds = torch.tensor(
            [[config.latent_bounds[0]] * stage_dim,
             [config.latent_bounds[1]] * stage_dim],
            device=self.device,
            dtype=self.dtype,
        )

        # Storage for this stage
        all_stage_z = []
        all_stage_values = []

        # Random initialization
        n_init = min(config.random_init_per_stage, n_trials)
        init_z_stage = torch.rand(
            n_init, stage_dim, device=self.device, dtype=self.dtype
        )
        init_z_stage = bounds[0] + (bounds[1] - bounds[0]) * init_z_stage

        # Evaluate initial points
        for z_stage in init_z_stage:
            z_full = self._construct_full_z(current_z, z_stage, dim_start, dim_end)
            value = objective(z_full.unsqueeze(0)).squeeze()
            all_stage_z.append(z_full)
            all_stage_values.append(value)

        train_z_stage = init_z_stage
        train_y = torch.tensor(all_stage_values, device=self.device, dtype=self.dtype)

        # BO loop
        n_bo_trials = n_trials - n_init
        for trial in range(n_bo_trials):
            # Fit GP on stage dimensions only
            gp = SingleTaskGP(
                train_z_stage,
                train_y.unsqueeze(-1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Get next point via Expected Improvement
            ei = ExpectedImprovement(gp, best_f=train_y.min())
            candidate, acq_value = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )

            # Evaluate
            z_stage_new = candidate.squeeze(0)
            z_full = self._construct_full_z(current_z, z_stage_new, dim_start, dim_end)
            value = objective(z_full.unsqueeze(0)).squeeze()

            # Update storage
            all_stage_z.append(z_full)
            all_stage_values.append(value)

            # Update training data
            train_z_stage = torch.cat([train_z_stage, z_stage_new.unsqueeze(0)])
            train_y = torch.cat([train_y, value.unsqueeze(0)])

            if verbose and (trial + 1) % 10 == 0:
                logger.info(
                    f"  Trial {n_init + trial + 1}/{n_trials}: "
                    f"value={value.item():.6f}, best={train_y.min().item():.6f}"
                )

        # Find best
        all_values_tensor = torch.tensor(all_stage_values, device=self.device)
        best_idx = all_values_tensor.argmin().item()
        best_z = all_stage_z[best_idx]
        best_value = all_values_tensor[best_idx].item()

        return StageResult(
            stage=stage_idx,
            dim_range=(dim_start, dim_end),
            best_z=best_z,
            best_value=best_value,
            all_z=torch.stack(all_stage_z),
            all_values=all_values_tensor,
            n_trials=n_trials,
        )

    def _construct_full_z(
        self,
        current_z: Tensor,
        z_stage: Tensor,
        dim_start: int,
        dim_end: int,
    ) -> Tensor:
        """Construct full latent by inserting stage values.

        Args:
            current_z: Current full latent (fixed dimensions)
            z_stage: Values for current stage dimensions
            dim_start: Start index
            dim_end: End index

        Returns:
            Full latent tensor (latent_dim,)
        """
        z_full = current_z.clone()
        z_full[dim_start:dim_end] = z_stage
        # Zero out dimensions beyond current stage (Matryoshka property)
        z_full[dim_end:] = 0.0
        return z_full


# =============================================================================
# Comparison: Full LSBO (for benchmarking)
# =============================================================================


class FullLSBO:
    """Standard (non-staged) LSBO for comparison.

    Optimizes all 64 dimensions simultaneously using a single GP.
    Expected to perform worse than StagedLSBO due to GP dimensionality issues.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        latent_bounds: Tuple[float, float] = (-3.0, 3.0),
        dtype: torch.dtype = torch.float64,
        device: str = "cuda",
    ):
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch required")

        self.latent_dim = latent_dim
        self.latent_bounds = latent_bounds
        self.dtype = dtype
        self.device = torch.device(device)

    def optimize(
        self,
        objective: Callable[[Tensor], Tensor],
        n_trials: int = 140,
        n_init: int = 20,
        verbose: bool = True,
    ) -> Tuple[Tensor, float, Tensor]:
        """Run standard BO on full latent space.

        Args:
            objective: Function (batch, latent_dim) -> (batch,)
            n_trials: Total trials
            n_init: Random initialization points
            verbose: Print progress

        Returns:
            best_z: Best latent found
            best_value: Best objective value
            all_values: All objective values (for convergence comparison)
        """
        bounds = torch.tensor(
            [[self.latent_bounds[0]] * self.latent_dim,
             [self.latent_bounds[1]] * self.latent_dim],
            device=self.device,
            dtype=self.dtype,
        )

        # Random initialization
        train_z = torch.rand(
            n_init, self.latent_dim, device=self.device, dtype=self.dtype
        )
        train_z = bounds[0] + (bounds[1] - bounds[0]) * train_z

        train_y = []
        for z in train_z:
            value = objective(z.unsqueeze(0)).squeeze()
            train_y.append(value)
        train_y = torch.tensor(train_y, device=self.device, dtype=self.dtype)

        if verbose:
            logger.info(f"FullLSBO: {n_init} init points, best={train_y.min().item():.6f}")

        # BO loop
        for trial in range(n_trials - n_init):
            gp = SingleTaskGP(train_z, train_y.unsqueeze(-1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            ei = ExpectedImprovement(gp, best_f=train_y.min())
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=50,
            )

            z_new = candidate.squeeze(0)
            value = objective(z_new.unsqueeze(0)).squeeze()

            train_z = torch.cat([train_z, z_new.unsqueeze(0)])
            train_y = torch.cat([train_y, value.unsqueeze(0)])

            if verbose and (trial + 1) % 20 == 0:
                logger.info(
                    f"  Trial {n_init + trial + 1}/{n_trials}: "
                    f"value={value.item():.6f}, best={train_y.min().item():.6f}"
                )

        best_idx = train_y.argmin().item()
        return train_z[best_idx], train_y[best_idx].item(), train_y


# =============================================================================
# Objective wrapper for Matryoshka Flow
# =============================================================================


def create_flow_objective(
    flow,
    target_embedding: Tensor,
    text_evaluator: Optional[Callable] = None,
    stage_idx: Optional[int] = None,
    matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64),
) -> Callable[[Tensor], Tensor]:
    """Create objective function for flow-based LSBO.

    The objective measures how well decoded embeddings match a target,
    optionally using text-level evaluation.

    Args:
        flow: MatryoshkaGTRFunnelFlow model
        target_embedding: Target GTR embedding to approach (768D)
        text_evaluator: Optional function embedding -> score (higher = better)
        stage_idx: Current stage index (for active_dim)
        matryoshka_dims: Matryoshka dimensions

    Returns:
        Objective function z -> value (to minimize)
    """
    device = target_embedding.device

    # Determine active_dim for current stage
    if stage_idx is not None and stage_idx < len(matryoshka_dims):
        active_dim = matryoshka_dims[stage_idx]
    else:
        active_dim = None

    def objective(z: Tensor) -> Tensor:
        """Objective: negative similarity to target (minimize = maximize similarity)."""
        z = z.to(device)

        # Decode with Matryoshka awareness
        with torch.no_grad():
            embedding = flow.decode(z, active_dim=active_dim, deterministic=True)
            embedding = F.normalize(embedding, p=2, dim=-1)

        # Cosine similarity to target
        target_norm = F.normalize(target_embedding, p=2, dim=-1)
        cos_sim = F.cosine_similarity(embedding, target_norm.expand(z.size(0), -1), dim=-1)

        # If text evaluator provided, combine scores
        if text_evaluator is not None:
            text_score = text_evaluator(embedding)
            # Combine: prioritize text score, use cos_sim as tiebreaker
            score = text_score + 0.1 * cos_sim
            return -score  # Minimize negative score
        else:
            return -cos_sim  # Minimize negative similarity

    return objective


# =============================================================================
# Benchmark utility
# =============================================================================


def benchmark_staged_vs_full(
    objective: Callable[[Tensor], Tensor],
    staged_config: StagedLSBOConfig,
    n_trials_full: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, any]:
    """Benchmark StagedLSBO against FullLSBO.

    Args:
        objective: Objective function
        staged_config: Config for staged optimizer
        n_trials_full: Trials for full optimizer (default: same as staged total)
        seed: Random seed

    Returns:
        Dictionary with comparison metrics
    """
    torch.manual_seed(seed)

    if n_trials_full is None:
        n_trials_full = staged_config.total_trials

    logger.info("Running StagedLSBO...")
    staged_optimizer = StagedLSBO(staged_config)
    staged_result = staged_optimizer.optimize(objective, verbose=True)

    torch.manual_seed(seed)  # Reset for fair comparison

    logger.info("\nRunning FullLSBO...")
    full_optimizer = FullLSBO(
        latent_dim=staged_config.latent_dim,
        latent_bounds=staged_config.latent_bounds,
        dtype=staged_config.dtype,
        device=staged_config.device,
    )
    full_z, full_value, full_history = full_optimizer.optimize(
        objective, n_trials=n_trials_full, verbose=True
    )

    # Compare results
    results = {
        'staged_best_value': staged_result.best_value,
        'full_best_value': full_value,
        'staged_total_evals': staged_result.total_evaluations,
        'full_total_evals': n_trials_full,
        'staged_convergence': staged_result.get_convergence_curve(),
        'full_convergence': torch.zeros(n_trials_full),
        'improvement': (full_value - staged_result.best_value) / abs(full_value) * 100,
    }

    # Compute full convergence curve
    best = float('inf')
    for i, v in enumerate(full_history):
        best = min(best, v.item())
        results['full_convergence'][i] = best

    logger.info(
        f"\nBenchmark Results:\n"
        f"  StagedLSBO: best={staged_result.best_value:.6f} ({staged_result.total_evaluations} evals)\n"
        f"  FullLSBO:   best={full_value:.6f} ({n_trials_full} evals)\n"
        f"  Improvement: {results['improvement']:.2f}%"
    )

    return results
