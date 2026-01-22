"""
Flow-Guided Acquisition for FlowPO-HD.

Combines GP-based acquisition with manifold velocity penalty for robust optimization.

IMPORTANT: Based on experimental findings (see FINDINGS.md), we use velocity
magnitude as a PENALTY, not as a direction force. The original approach of
using v(x, t) as a direction to push towards the manifold doesn't work because
flow matching learns transport, not manifold membership.

Acquisition options:
1. UCB (Upper Confidence Bound) - original, gradient-based
2. qLogEI (Log Expected Improvement) - NEW, via SAAS GP (Spearman 0.87 in benchmark)

Update rule (UCB mode):
    x_{k+1} = x_k + η·∇[α_GP(x_k) - λ·||v_θ(x_k, t)||²]

SAAS mode (recommended):
    Uses qLogEI acquisition from fully Bayesian SAAS GP, then filters by velocity penalty.

where:
- α_GP: GP acquisition function (UCB or qLogEI)
- ||v_θ(x, t)||²: Velocity magnitude penalty (high velocity = off-manifold)
- λ: Penalty weight (encourages staying near training distribution)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from flowpo_hd.manifold_keeper import ManifoldKeeperMLP

logger = logging.getLogger(__name__)


def _unpack_gp_prediction(prediction) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack GP prediction to (mean, std) regardless of GP type.

    Handles both:
    - SAAS GP which returns SaasPrediction object with .mean and .std attributes
    - Regular GP which returns tuple (mean, std)
    """
    if hasattr(prediction, 'mean') and hasattr(prediction, 'std'):
        # SAAS GP returns SaasPrediction object
        return prediction.mean, prediction.std
    else:
        # Regular GP returns tuple
        return prediction


class FlowGuidedAcquisition:
    """
    Flow-guided acquisition optimizer for 1024D SONAR space.

    Combines:
    1. GP acquisition gradient (∇UCB or ∇LogEI) for optimization direction
    2. Velocity magnitude penalty ||v_θ(x, t)||² to stay near training distribution
    3. Trust region bounds from TuRBO-1024

    Key insight: We DON'T use velocity as a force direction (that doesn't work).
    Instead, we penalize high velocity magnitude, which indicates being far from
    the training distribution.
    """

    def __init__(
        self,
        manifold_keeper: Optional[ManifoldKeeperMLP],
        device: torch.device,
        manifold_time: float = 0.9,
        num_steps: int = 50,
        step_size: float = 0.01,
        num_restarts: int = 32,
        use_velocity_penalty: bool = True,
        seed_from_training: bool = True,
        training_seed_ratio: float = 0.8,
    ):
        """
        Initialize Flow-Guided Acquisition.

        Args:
            manifold_keeper: Trained ManifoldKeeper (can be None if not using penalty)
            device: Torch device
            manifold_time: Time for velocity computation (default 0.9)
            num_steps: Gradient steps per optimization
            step_size: Step size for gradient updates
            num_restarts: Number of random restarts
            use_velocity_penalty: Whether to use velocity magnitude penalty
            seed_from_training: Whether to seed starting points from training data
            training_seed_ratio: Ratio of starting points seeded from training (0-1)
        """
        self.manifold_keeper = manifold_keeper
        self.device = device
        self.manifold_time = manifold_time
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_restarts = num_restarts
        self.use_velocity_penalty = use_velocity_penalty and manifold_keeper is not None
        self.seed_from_training = seed_from_training
        self.training_seed_ratio = training_seed_ratio

        # Ensure manifold keeper is in eval mode
        if self.manifold_keeper is not None:
            self.manifold_keeper.eval()

    def _compute_velocity_penalty(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity magnitude penalty.

        High velocity magnitude indicates the point is far from training distribution.
        We use ||v||² as a soft penalty to encourage staying near manifold.

        Args:
            x: (B, D) current positions

        Returns:
            (B,) velocity magnitude squared for each point
        """
        if self.manifold_keeper is None:
            return torch.zeros(x.shape[0], device=x.device)

        with torch.no_grad():
            t_tensor = torch.full((x.shape[0],), self.manifold_time, device=x.device)
            # Note: model forward is (t, x), not (x, t)!
            velocity = self.manifold_keeper(t_tensor, x)
            return (velocity ** 2).sum(dim=-1)  # ||v||²

    def _compute_combined_gradient(
        self,
        x: torch.Tensor,
        gp,
        ucb_beta: float = 2.0,
        lambda_penalty: float = 0.001,
    ) -> torch.Tensor:
        """
        Compute gradient of combined acquisition function.

        Combined objective: maximize [UCB(x) - λ·||v(x,t)||²]
        = maximize [-μ(x) + β·σ(x) - λ·||v(x,t)||²]

        Args:
            x: (B, D) current positions
            gp: GP model with predict(x) -> (mean, std)
            ucb_beta: Exploration parameter
            lambda_penalty: Weight for velocity penalty

        Returns:
            (B, D) combined gradients
        """
        x = x.detach().requires_grad_(True)

        # GP prediction (handle both SAAS and regular GP)
        mean, std = _unpack_gp_prediction(gp.predict(x))

        # UCB acquisition (we minimize error rate, so we want low mean)
        ucb = -mean + ucb_beta * std

        # Velocity penalty (if enabled)
        if self.use_velocity_penalty and self.manifold_keeper is not None:
            t_tensor = torch.full((x.shape[0],), self.manifold_time, device=x.device)
            velocity = self.manifold_keeper(t_tensor, x)
            velocity_penalty = (velocity ** 2).sum(dim=-1)  # ||v||²
            # Combined: UCB - penalty
            combined = ucb - lambda_penalty * velocity_penalty
        else:
            combined = ucb

        # Compute gradient (we want to maximize, so gradient ascent)
        combined.sum().backward()

        if x.grad is None:
            logger.warning("Combined gradient is None, returning zeros")
            return torch.zeros_like(x)

        grad = x.grad.detach()

        # Check for numerical issues
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            logger.warning("NaN/Inf in gradient, clipping")
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)

        return grad

    def _project_to_bounds(
        self,
        x: torch.Tensor,
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Project points to be within bounds."""
        return torch.clamp(x, bounds[0], bounds[1])

    def _sample_initial_points(
        self,
        bounds: torch.Tensor,
        n_samples: int,
        X_train: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample initial points for optimization.

        Strategy (based on FINDINGS.md recommendations):
        1. Majority from perturbations of training data (if available)
        2. Rest from uniform random in bounds

        Args:
            bounds: (2, D) bounds
            n_samples: Number of samples
            X_train: Optional training data for seeding

        Returns:
            (n_samples, D) initial points
        """
        dim = bounds.shape[1]
        samples = []

        # Compute split
        if self.seed_from_training and X_train is not None and X_train.shape[0] > 0:
            n_seeded = int(n_samples * self.training_seed_ratio)
            n_random = n_samples - n_seeded
        else:
            n_seeded = 0
            n_random = n_samples

        # Seeded samples from training data (RECOMMENDED)
        if n_seeded > 0 and X_train is not None:
            n_available = X_train.shape[0]
            indices = torch.randint(0, n_available, (n_seeded,), device=self.device)
            seeds = X_train[indices].clone()

            # Add small noise (10% of embedding norm)
            seed_norms = seeds.norm(dim=-1, keepdim=True).mean()
            noise = torch.randn_like(seeds) * 0.1 * seed_norms
            seeded = seeds + noise

            seeded = self._project_to_bounds(seeded, bounds)
            samples.append(seeded)

        # Random samples in bounds
        if n_random > 0:
            uniform = torch.rand(n_random, dim, device=self.device)
            random_samples = bounds[0] + uniform * (bounds[1] - bounds[0])
            samples.append(random_samples)

        return torch.cat(samples, dim=0) if samples else torch.empty(0, dim, device=self.device)

    def optimize(
        self,
        gp,
        bounds: torch.Tensor,
        lambda_penalty: float = 0.001,
        ucb_beta: float = 2.0,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """
        Optimize acquisition function with velocity penalty.

        Main algorithm:
        1. Sample initial points (mostly from training data perturbations)
        2. For each step:
           a. Compute combined gradient: ∇[UCB - λ·||v||²]
           b. Update: x = x + η·grad
           c. Project to bounds
        3. Select best candidate by acquisition value

        NOTE: We do NOT project to manifold at the end (see FINDINGS.md).

        Args:
            gp: GP model with predict(x) -> (mean, std)
            bounds: Trust region bounds (2, D)
            lambda_penalty: Velocity penalty weight
            ucb_beta: UCB exploration parameter
            X_train: Optional training inputs for seeding
            y_train: Optional training targets (unused)
            verbose: Print debug info

        Returns:
            (best_x, best_acq) - Best candidate and its acquisition value
        """
        # Sample initial points
        x = self._sample_initial_points(bounds, self.num_restarts, X_train)

        if verbose:
            logger.info(f"FlowGuidedAcq: {self.num_restarts} restarts, {self.num_steps} steps")
            logger.info(f"  lambda_penalty={lambda_penalty:.6f}, ucb_beta={ucb_beta:.3f}")
            logger.info(f"  use_velocity_penalty={self.use_velocity_penalty}")
            if X_train is not None:
                n_seeded = int(self.num_restarts * self.training_seed_ratio)
                logger.info(f"  seeded_from_training={n_seeded}/{self.num_restarts}")

        # Optimization loop
        for step in range(self.num_steps):
            # Compute combined gradient
            grad = self._compute_combined_gradient(x, gp, ucb_beta, lambda_penalty)

            # Gradient ascent (maximize acquisition)
            x = x + self.step_size * grad

            # Project to bounds
            x = self._project_to_bounds(x, bounds)

            if verbose and step % 10 == 0:
                with torch.no_grad():
                    mean, std = _unpack_gp_prediction(gp.predict(x))
                    acq = -mean + ucb_beta * std
                    if self.use_velocity_penalty:
                        v_penalty = self._compute_velocity_penalty(x)
                        logger.info(f"  Step {step}: acq={acq.mean():.4f}, v_penalty={v_penalty.mean():.4f}")
                    else:
                        logger.info(f"  Step {step}: acq={acq.mean():.4f}")

        # Evaluate final acquisition values (without penalty for fair comparison)
        with torch.no_grad():
            mean, std = _unpack_gp_prediction(gp.predict(x))
            acquisition_values = -mean + ucb_beta * std

        # Select best
        best_idx = acquisition_values.argmax()
        best_x = x[best_idx]
        best_acq = acquisition_values[best_idx].item()

        if verbose:
            logger.info(f"  Best acquisition: {best_acq:.4f}")
            if self.use_velocity_penalty:
                v_penalty = self._compute_velocity_penalty(best_x.unsqueeze(0))
                logger.info(f"  Best velocity penalty: {v_penalty.item():.4f}")

        return best_x, best_acq

    def optimize_batch(
        self,
        gp,
        bounds: torch.Tensor,
        batch_size: int,
        lambda_penalty: float = 0.001,
        ucb_beta: float = 2.0,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
        diversity_threshold: float = 0.05,
    ) -> List[torch.Tensor]:
        """
        Optimize to get a batch of diverse candidates.

        Uses greedy selection with diversity threshold.

        Args:
            gp: GP model
            bounds: Trust region bounds
            batch_size: Number of candidates to return
            lambda_penalty: Velocity penalty weight
            ucb_beta: UCB exploration parameter
            X_train: Training inputs for seeding
            y_train: Training targets
            diversity_threshold: Minimum cosine distance between candidates

        Returns:
            List of batch_size candidate tensors
        """
        candidates = []
        candidate_tensors = []

        for i in range(batch_size):
            x_opt, acq = self.optimize(
                gp=gp,
                bounds=bounds,
                lambda_penalty=lambda_penalty,
                ucb_beta=ucb_beta,
                X_train=X_train,
                y_train=y_train,
            )

            # Check diversity (cosine distance)
            if candidate_tensors:
                all_prev = torch.stack(candidate_tensors, dim=0)
                # Cosine similarity
                cos_sim = F.cosine_similarity(
                    all_prev, x_opt.unsqueeze(0).expand_as(all_prev), dim=-1
                )
                if cos_sim.max() > (1 - diversity_threshold):
                    # Too similar - add noise to diversify
                    noise = torch.randn_like(x_opt) * 0.1 * x_opt.norm()
                    x_opt = self._project_to_bounds(x_opt + noise, bounds)

            candidates.append(x_opt)
            candidate_tensors.append(x_opt)

            # Kriging believer: add hallucinated observation
            with torch.no_grad():
                mean, _ = _unpack_gp_prediction(gp.predict(x_opt.unsqueeze(0)))
                if X_train is not None:
                    X_train = torch.cat([X_train, x_opt.unsqueeze(0)], dim=0)
                    if y_train is not None:
                        y_train = torch.cat([y_train, mean], dim=0)

        return candidates


def create_flow_guided_acquisition(
    config,
    manifold_keeper: Optional[ManifoldKeeperMLP],
) -> FlowGuidedAcquisition:
    """Factory function to create FlowGuidedAcquisition from config.

    Args:
        config: FlowPOHDConfig
        manifold_keeper: Trained ManifoldKeeper (can be None)

    Returns:
        Configured FlowGuidedAcquisition
    """
    device = torch.device(config.device)

    return FlowGuidedAcquisition(
        manifold_keeper=manifold_keeper,
        device=device,
        manifold_time=config.fga_manifold_time,
        num_steps=config.fga_num_steps,
        step_size=config.fga_step_size,
        num_restarts=config.fga_num_restarts,
        use_velocity_penalty=config.fga_use_velocity_penalty,
        seed_from_training=config.fga_seed_from_training,
        training_seed_ratio=config.fga_training_seed_ratio,
    )


if __name__ == "__main__":
    print("Testing FlowGuidedAcquisition (velocity penalty version)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create manifold keeper (untrained, for testing)
    from flowpo_hd.manifold_keeper import ManifoldKeeperMLP

    manifold_keeper = ManifoldKeeperMLP(
        dim=1024,
        hidden_dim=2048,
        num_blocks=3,
    ).to(device)
    manifold_keeper.eval()

    # Create mock GP for testing
    class MockGP:
        def __init__(self, device):
            self.device = device
            self.center = torch.randn(1024, device=device) * 0.1

        def predict(self, x):
            # Simple quadratic bowl centered at self.center
            diff = x - self.center
            mean = (diff ** 2).sum(dim=-1) * 0.001  # Error rate
            std = torch.ones(x.shape[0], device=x.device) * 0.1
            return mean, std

    gp = MockGP(device)

    # Create fake training data
    X_train = torch.randn(50, 1024, device=device) * 0.2

    # Create acquisition optimizer
    fga = FlowGuidedAcquisition(
        manifold_keeper=manifold_keeper,
        device=device,
        num_steps=20,
        num_restarts=16,
        use_velocity_penalty=True,
        seed_from_training=True,
        training_seed_ratio=0.8,
    )

    # Test optimization
    print("\n--- Single Optimization ---")
    bounds = torch.stack([
        torch.full((1024,), -1.0, device=device),
        torch.full((1024,), 1.0, device=device),
    ])

    best_x, best_acq = fga.optimize(
        gp=gp,
        bounds=bounds,
        lambda_penalty=0.001,
        ucb_beta=2.0,
        X_train=X_train,
        verbose=True,
    )

    print(f"Best x shape: {best_x.shape}")
    print(f"Best x norm: {best_x.norm():.4f}")
    print(f"Best acquisition: {best_acq:.4f}")

    # Verify it's close to the GP center
    dist_to_center = (best_x - gp.center).norm()
    print(f"Distance to GP center: {dist_to_center:.4f}")

    # Test without velocity penalty
    print("\n--- Without Velocity Penalty ---")
    fga_no_penalty = FlowGuidedAcquisition(
        manifold_keeper=None,
        device=device,
        num_steps=20,
        num_restarts=16,
        use_velocity_penalty=False,
    )

    best_x2, best_acq2 = fga_no_penalty.optimize(
        gp=gp,
        bounds=bounds,
        ucb_beta=2.0,
        X_train=X_train,
        verbose=True,
    )

    dist_to_center2 = (best_x2 - gp.center).norm()
    print(f"Distance to GP center (no penalty): {dist_to_center2:.4f}")

    # Test batch optimization
    print("\n--- Batch Optimization ---")
    candidates = fga.optimize_batch(
        gp=gp,
        bounds=bounds,
        batch_size=3,
        lambda_penalty=0.001,
        ucb_beta=2.0,
        X_train=X_train,
    )

    print(f"Got {len(candidates)} candidates")
    for i, c in enumerate(candidates):
        dist = (c - gp.center).norm()
        print(f"  Candidate {i}: norm={c.norm():.4f}, dist_to_center={dist:.4f}")

    print("\n[OK] FlowGuidedAcquisition tests passed!")


# =============================================================================
# SAAS-based Acquisition (NEW - benchmark winner)
# =============================================================================


class SaasFlowGuidedAcquisition:
    """
    SAAS GP + qLogEI acquisition with optional velocity penalty.

    This is the benchmark-validated approach (Spearman 0.87):
    1. SAAS GP identifies relevant dimensions via Bayesian ARD
    2. qLogEI provides acquisition values
    3. Velocity penalty filters candidates to stay near manifold

    Unlike FlowGuidedAcquisition (gradient-based UCB), this uses
    BoTorch's optimize_acqf for acquisition optimization.
    """

    def __init__(
        self,
        manifold_keeper: Optional[ManifoldKeeperMLP],
        device: torch.device,
        manifold_time: float = 0.9,
        lambda_penalty: float = 0.001,
        use_velocity_penalty: bool = True,
        saas_warmup_steps: int = 128,
        saas_num_samples: int = 64,
        num_restarts: int = 32,
        raw_samples: int = 512,
    ):
        """
        Initialize SAAS-based acquisition.

        Args:
            manifold_keeper: Trained ManifoldKeeper (can be None)
            device: Torch device
            manifold_time: Time for velocity computation
            lambda_penalty: Velocity penalty weight
            use_velocity_penalty: Whether to use velocity penalty
            saas_warmup_steps: MCMC warmup steps
            saas_num_samples: MCMC posterior samples
            num_restarts: Restarts for acquisition optimization
            raw_samples: Raw samples for acquisition optimization
        """
        self.manifold_keeper = manifold_keeper
        self.device = device
        self.manifold_time = manifold_time
        self.lambda_penalty = lambda_penalty
        self.use_velocity_penalty = use_velocity_penalty and manifold_keeper is not None

        # SAAS config
        self.saas_warmup_steps = saas_warmup_steps
        self.saas_num_samples = saas_num_samples
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

        # GP will be created on first fit
        self._gp = None

        if self.manifold_keeper is not None:
            self.manifold_keeper.eval()

    def _ensure_gp(self):
        """Lazily create SAAS GP."""
        if self._gp is None:
            from flowpo_hd.saas_gp import SaasConfig, SaasGPWithAcquisition

            config = SaasConfig(
                warmup_steps=self.saas_warmup_steps,
                num_samples=self.saas_num_samples,
                num_restarts=16,  # Reduce from 32 to save memory
                raw_samples=256,  # Reduce from 512 to save memory
                use_noisy_ei=False,  # Use qLogEI (qLogNEI too memory-intensive)
            )
            # Fit SAAS on CPU to avoid GPU OOM with vLLM
            self._gp = SaasGPWithAcquisition(config=config, device=str(self.device), fit_on_cpu=True)

    def _compute_velocity_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute velocity magnitude penalty."""
        if self.manifold_keeper is None:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        with torch.no_grad():
            # Store original dtype (SAAS GP uses float64, ManifoldKeeper uses float32)
            orig_dtype = x.dtype

            # Move tensors to ManifoldKeeper's device (CUDA) and convert to float32
            x_device = x.to(device=self.device, dtype=torch.float32)
            t = torch.full((x.shape[0],), self.manifold_time, device=self.device, dtype=torch.float32)
            velocity = self.manifold_keeper(t, x_device)

            # Return on same device and dtype as input for consistency with SAAS GP
            penalty = (velocity ** 2).sum(dim=-1)
            return penalty.to(device=x.device, dtype=orig_dtype)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        variances: Optional[torch.Tensor] = None,
    ) -> bool:
        """Fit SAAS GP on training data.

        Args:
            X: Training embeddings (N, 1024)
            y: Training error rates (N,)
            variances: Optional observation variances

        Returns:
            True if fitting succeeded
        """
        self._ensure_gp()
        return self._gp.fit(X, y, variances)

    def optimize(
        self,
        bounds: torch.Tensor,
        batch_size: int = 1,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """
        Optimize acquisition to get best candidate(s).

        Uses qLogEI from SAAS GP, optionally filtered by velocity penalty.

        Args:
            bounds: Trust region bounds (2, D)
            batch_size: Number of candidates
            X_train: Not used (SAAS uses internal training data)
            y_train: Not used
            verbose: Print debug info

        Returns:
            (best_candidate, acquisition_value)
        """
        if self._gp is None or self._gp.gp_model is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        if verbose:
            logger.info(f"SaasFlowGuidedAcq: qLogEI optimization")
            logger.info(f"  use_velocity_penalty={self.use_velocity_penalty}")

        # Get candidates from qLogEI
        # Request extra candidates if filtering by velocity
        n_request = batch_size * 4 if self.use_velocity_penalty else batch_size

        candidates, acq_value = self._gp.get_best_candidate(
            bounds=bounds,
            batch_size=n_request,
        )

        if verbose:
            logger.info(f"  Got {candidates.shape[0]} candidates, acq={acq_value:.4f}")

        # Filter by velocity penalty if enabled
        if self.use_velocity_penalty and candidates.shape[0] > batch_size:
            v_penalty = self._compute_velocity_penalty(candidates)

            # Get predictions
            pred = self._gp.predict(candidates)

            # Combined score: predicted error + velocity penalty
            combined = pred.mean + self.lambda_penalty * v_penalty

            # Select best (lowest combined)
            best_idx = combined.argsort()[:batch_size]
            candidates = candidates[best_idx]

            if verbose:
                logger.info(f"  After velocity filter: {candidates.shape[0]} candidates")
                logger.info(f"  Velocity penalty range: [{v_penalty.min():.4f}, {v_penalty.max():.4f}]")

        # Return single candidate if batch_size=1
        if batch_size == 1 and candidates.dim() > 1:
            candidates = candidates.squeeze(0)

        return candidates, acq_value

    def add_observation(self, x: torch.Tensor, y: float, refit: bool = True):
        """Add observation and optionally refit."""
        if self._gp is not None:
            self._gp.add_observation(x, y, refit=refit)

    @property
    def relevant_dims(self) -> List[int]:
        """Get relevant dimensions identified by SAAS."""
        if self._gp is not None:
            return self._gp.relevant_dims
        return []

    @property
    def gp(self):
        """Access underlying SAAS GP."""
        return self._gp


def create_saas_flow_guided_acquisition(
    config,
    manifold_keeper: Optional[ManifoldKeeperMLP],
) -> SaasFlowGuidedAcquisition:
    """Factory function to create SAAS-based acquisition from config.

    Args:
        config: FlowPOHDConfig
        manifold_keeper: Trained ManifoldKeeper (can be None)

    Returns:
        Configured SaasFlowGuidedAcquisition
    """
    device = torch.device(config.device)

    return SaasFlowGuidedAcquisition(
        manifold_keeper=manifold_keeper,
        device=device,
        manifold_time=config.fga_manifold_time,
        lambda_penalty=config.fga_lambda_penalty,
        use_velocity_penalty=config.fga_use_velocity_penalty,
        saas_warmup_steps=getattr(config, 'saas_warmup_steps', 128),
        saas_num_samples=getattr(config, 'saas_num_samples', 64),
        num_restarts=config.fga_num_restarts,
        raw_samples=getattr(config, 'saas_raw_samples', 512),
    )
