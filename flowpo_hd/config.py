"""Configuration for FlowPO-HD: High-Dimensional Prompt Optimization.

Single Source of Truth (SSOT) for all parameters.
CLI arguments in run_flowpo_hd.py override these defaults.
"""

from dataclasses import dataclass

import torch

# Valid options for configuration fields
VALID_GP_TYPES = ("vanilla", "isotropic", "saas", "adaptive")
VALID_TIMESTEP_SAMPLING = ("uniform", "u_shaped")


def get_device(device: str = "auto") -> str:
    """Determine device to use for computation.

    Args:
        device: Device specification:
            - "auto": Use CUDA if available, else CPU
            - "cuda": Force CUDA (will fail if unavailable)
            - "cpu": Force CPU

    Returns:
        Device string ("cuda" or "cpu")
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class FlowPOHDConfig:
    """Configuration for FlowPO-HD pipeline.

    Key design decisions:
    - sonar_dim=1024: Full SONAR fidelity, no compression loss
    - normalize=False: SONAR decoder requires unnormalized embeddings (~0.18 norm)
    - manifold_time=0.9: Near-clean samples for meaningful manifold direction
    - lambda adaptive: 0.5→2.0 (start GP exploration, end manifold-guided)
    - turbo_tau_fail=128: ceil(1024/8) for high-dimensional space

    Organized into logical sections:
    - SONAR Embedding
    - ManifoldKeeper Architecture
    - ManifoldKeeper Training
    - TuRBO (Trust Region)
    - Flow-Guided Acquisition
    - GP Configuration
    - LLM Evaluation
    - Device/Paths
    """

    # === SONAR Embedding ===
    sonar_dim: int = 1024  # SONAR native dimension (NO compression)
    sonar_normalize: bool = False  # Keep unnormalized for decoder (~0.18 norm)
    sonar_source_lang: str = "eng_Latn"
    sonar_target_lang: str = "eng_Latn"

    # === ManifoldKeeper Architecture ===
    # MLP velocity field with AdaLN conditioning (NO bottleneck)
    mk_hidden_dim: int = 2048  # Hidden dimension (1024 → 2048 → 1024)
    mk_num_blocks: int = 3  # Number of residual blocks
    mk_time_dim: int = 256  # Timestep embedding dimension
    mk_dropout: float = 0.1  # Dropout for regularization
    # ~15M parameters: 3 blocks × (1024×2048 + 2048×1024) × 2 ≈ 25M, but shared time_emb reduces it

    # === ManifoldKeeper Training ===
    mk_epochs: int = 50000  # Training epochs
    mk_lr: float = 1e-4  # Learning rate
    mk_batch_size: int = 256  # Batch size (larger for stability)
    mk_grad_clip: float = 1.0  # Gradient clipping
    mk_warmup_steps: int = 1000  # LR warmup steps
    mk_patience: int = 2000  # Early stopping patience
    mk_use_ot: bool = True  # Use optimal transport pairing
    mk_timestep_sampling: str = "u_shaped"  # "uniform" or "u_shaped"
    mk_u_shaped_a: float = 4.0  # Concentration parameter for U-shaped

    # === TuRBO (Trust Region) for 1024D ===
    # Adapted from TuRBO paper (Eriksson et al., NeurIPS 2019)
    turbo_enabled: bool = True  # Enable trust region management
    turbo_L_init: float = 0.4  # Initial side length (reduced for high-D)
    turbo_L_max: float = 1.6  # Maximum side length
    turbo_L_min: float = 0.0078  # Minimum side length (2^-7)
    turbo_tau_succ: int = 3  # Consecutive successes to expand
    turbo_tau_fail: int = 128  # Consecutive failures to shrink (ceil(1024/8))

    # === Flow-Guided Acquisition ===
    # Update: x_{k+1} = x_k + η·∇[α_GP(x_k) - λ·||v_θ(x_k, t)||²]
    # NOTE: Based on FINDINGS.md, we use velocity magnitude as PENALTY, not direction
    fga_manifold_time: float = 0.9  # Time for velocity computation
    fga_lambda_penalty: float = 0.001  # Velocity penalty weight (small, just regularization)
    fga_num_steps: int = 50  # Optimization steps per candidate
    fga_step_size: float = 0.01  # Step size for gradient ascent
    fga_num_restarts: int = 32  # Number of random restarts
    fga_use_velocity_penalty: bool = True  # Use ||v||² as penalty
    fga_seed_from_training: bool = True  # Seed starting points from training data
    fga_training_seed_ratio: float = 0.8  # Ratio of seeded vs random starts

    # === GP Configuration ===
    # Options: "isotropic", "saas", "adaptive"
    # "saas" is recommended with warm-start (benchmark winner: Spearman 0.87)
    gp_type: str = "saas"  # Use SAAS by default (benchmark winner)
    gp_switch_threshold: int = 30  # Switch to SAAS when n >= this (for adaptive)
    gp_ucb_beta_start: float = 4.0  # UCB beta at start (high exploration)
    gp_ucb_beta_end: float = 2.0  # UCB beta at end (more exploitation)
    gp_trust_region_scale: float = 2.0  # Trust region scale for GP
    gp_epochs: int = 100  # GP hyperparameter fitting iterations

    # === SAAS GP Configuration (benchmark-validated) ===
    # MCMC settings for fully Bayesian inference
    saas_warmup_steps: int = 128  # NUTS warmup (burn-in)
    saas_num_samples: int = 64  # Posterior samples
    saas_thinning: int = 2  # Keep every N-th sample
    # Acquisition optimization
    saas_raw_samples: int = 512  # Raw samples for acqf optimization
    # qLogEI is used by default (better than UCB for BO)

    # === Warm-Start Configuration ===
    # Use pre-evaluated HbBoPs data to initialize GP
    use_warm_start: bool = True  # Enable warm-start
    warm_start_min_fidelity: int = 600  # medium_600 strategy (benchmark winner)
    warm_start_cache_path: str = "flowpo_hd/data/warm_start_embeddings.pt"

    # === LLM Evaluation ===
    eval_model: str = "Qwen/Qwen2.5-7B-Instruct"
    eval_backend: str = "vllm"
    eval_minibatch_size: int = 100  # Examples per evaluation
    eval_max_tokens: int = 512  # Max tokens for LLM response

    # === Data ===
    ape_instructions_path: str = "lipo/data/ape_instructions.json"
    sonar_embeddings_path: str = "flowpo_hd/data/sonar_unnorm.pt"
    manifold_keeper_path: str = "flowpo_hd/checkpoints/manifold_keeper.pt"
    hbbops_results_path: str = "lipo/data/hbbops_results_20260102.json"

    # === Device/Paths ===
    device: str = "cuda"
    seed: int = 42
    results_dir: str = "flowpo_hd/results"
    checkpoints_dir: str = "flowpo_hd/checkpoints"

    def __post_init__(self):
        """Validate configuration invariants."""
        # SONAR dimension is fixed
        if self.sonar_dim != 1024:
            raise ValueError(f"sonar_dim must be 1024 (SONAR native), got {self.sonar_dim}")

        # ManifoldKeeper architecture constraints
        if self.mk_hidden_dim < self.sonar_dim:
            raise ValueError(
                f"mk_hidden_dim ({self.mk_hidden_dim}) should be >= sonar_dim ({self.sonar_dim})"
            )

        # TuRBO constraints
        if self.turbo_L_init > self.turbo_L_max:
            raise ValueError(
                f"turbo_L_init ({self.turbo_L_init}) must be <= turbo_L_max ({self.turbo_L_max})"
            )
        if self.turbo_L_min >= self.turbo_L_init:
            raise ValueError(
                f"turbo_L_min ({self.turbo_L_min}) must be < turbo_L_init ({self.turbo_L_init})"
            )

        # Flow-guided acquisition constraints
        if not 0.0 < self.fga_manifold_time <= 1.0:
            raise ValueError(
                f"fga_manifold_time must be in (0, 1], got {self.fga_manifold_time}"
            )
        if self.fga_lambda_penalty < 0:
            raise ValueError(f"fga_lambda_penalty must be non-negative, got {self.fga_lambda_penalty}")
        if not 0.0 <= self.fga_training_seed_ratio <= 1.0:
            raise ValueError(
                f"fga_training_seed_ratio must be in [0, 1], got {self.fga_training_seed_ratio}"
            )

        # GP constraints
        if self.gp_type not in VALID_GP_TYPES:
            raise ValueError(f"gp_type must be one of {VALID_GP_TYPES}, got {self.gp_type}")

        # Timestep sampling
        if self.mk_timestep_sampling not in VALID_TIMESTEP_SAMPLING:
            raise ValueError(
                f"mk_timestep_sampling must be one of {VALID_TIMESTEP_SAMPLING}, got {self.mk_timestep_sampling}"
            )

    def get_ucb_beta(self, iteration: int, total_iterations: int) -> float:
        """Compute UCB beta for given iteration.

        Linear decay from beta_start to beta_end.

        Args:
            iteration: Current iteration (0-indexed)
            total_iterations: Total number of iterations

        Returns:
            UCB beta for this iteration
        """
        if total_iterations <= 1:
            return self.gp_ucb_beta_start

        progress = iteration / (total_iterations - 1)
        return self.gp_ucb_beta_start + (self.gp_ucb_beta_end - self.gp_ucb_beta_start) * progress
