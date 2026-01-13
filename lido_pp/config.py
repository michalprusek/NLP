"""
Configuration for FlowPO (Flow Matching for Prompt Optimization).

This module defines the complete configuration for the FlowPO architecture:
- SONAR backbone (reconstruction-optimized embeddings)
- Text Flow Autoencoder (TFA) with Lipschitz regularization
- GP-Guided Flow Generation
- Flow Curvature Uncertainty (FCU) Gating
- Cross-Attention Decoder conditioning

NeurIPS 2026 submission: FlowPO - Unified Flow Matching for Prompt Optimization
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import torch


def get_device(device_str: str = "auto") -> str:
    """Resolve device string to actual device."""
    if device_str == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_str


@dataclass
class FlowPOConfig:
    """
    FlowPO configuration.

    Organized into sections:
    - Device: GPU allocation
    - SONAR Encoder: Reconstruction-optimized embeddings (1024D)
    - Text Flow Autoencoder (TFA): Flow-based compression (1024D → 128D)
    - Flow-DiT: Rectified Flow network
    - GP-Guided Generation: Acquisition function gradient injection
    - FCU Gating: Flow Curvature Uncertainty for adaptive evaluation
    - Cross-Attention Decoder: Memory slot conditioning
    - Regularization: Lipschitz, reconstruction losses
    - Data: Dataset paths
    - Results: Output directories
    """

    # === Device Configuration ===
    device: str = "cuda:0"
    eval_device: str = "cuda:1"  # Separate GPU for LLM evaluation

    # === SONAR Encoder (replaces GritLM) ===
    encoder_type: str = "sonar"  # "sonar" (recommended) or "gritlm" (legacy)
    sonar_source_lang: str = "eng_Latn"
    sonar_normalize: bool = True

    # === Embedding Dimensions ===
    embedding_dim: int = 1024  # SONAR native dimension (was 768 for GTR, 4096 for GritLM)

    # === Text Flow Autoencoder (TFA) ===
    tfa_latent_dim: int = 128  # Increased from 32 for better reconstruction (8:1 compression)
    tfa_flow_dim: int = 256  # Intermediate flow space dimension
    tfa_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    tfa_ode_steps: int = 20  # ODE integration steps
    tfa_time_embed_dim: int = 64

    # === Flow-DiT Architecture ===
    flow_latent_dim: int = 128  # Must match tfa_latent_dim
    flow_hidden_dim: int = 768  # Increased for capacity (was 512)
    flow_num_layers: int = 6  # Number of transformer blocks
    flow_num_heads: int = 8  # Attention heads
    flow_mlp_ratio: float = 4.0  # MLP expansion ratio
    flow_time_embed_dim: int = 256  # Timestep embedding dimension
    flow_context_dim: int = 1024  # Context (SONAR embedding), must match embedding_dim
    flow_dropout: float = 0.1
    flow_cross_attention: bool = True  # Use cross-attention for conditioning

    # === Flow Training ===
    flow_lr: float = 1e-4
    flow_epochs: int = 10000
    flow_batch_size: int = 64
    flow_warmup_epochs: int = 500
    flow_patience: int = 500
    flow_grad_clip: float = 1.0

    # === GP-Guided Flow Generation (Novel Contribution #2) ===
    guidance_enabled: bool = True
    guidance_scale: float = 1.0  # Strength of GP gradient injection
    guidance_schedule: Literal["linear", "cosine", "warmup", "sqrt"] = "linear"
    guidance_ucb_beta: float = 2.0  # UCB exploration parameter

    # === FCU Gating (Novel Contribution #3) ===
    fcu_enabled: bool = True
    fcu_percentile: float = 90.0  # Top 10% get LLM evaluation
    fcu_min_threshold: float = 0.1  # Minimum FCU for evaluation
    fcu_steps: int = 20  # Steps for curvature estimation
    min_evals_before_gating: int = 50  # Build up GP first

    # === Cross-Attention Decoder (replaces prefix tokens) ===
    decoder_type: str = "cross_attention"  # "cross_attention" or "prefix" (legacy)
    num_memory_slots: int = 16  # K,V pairs for cross-attention (was 4 prefix tokens)
    decoder_hidden_dim: int = 4096  # Match decoder model hidden dim
    decoder_num_heads: int = 32
    decoder_dropout: float = 0.1
    decoder_use_gate: bool = True  # GLU-style gating

    # === Regularization ===
    lambda_recon: float = 0.1  # Reconstruction loss weight
    lambda_lip: float = 0.01  # Lipschitz regularization (BO-friendly smoothness)
    lambda_gw: float = 0.0  # Gromov-Wasserstein (optional)
    lipschitz_bound: float = 10.0  # Maximum Lipschitz constant

    # === OAT-FM (Optimal Acceleration Transport) ===
    oat_enabled: bool = True
    oat_weight: float = 0.1  # λ in L_CFM + λ·L_OAT
    oat_steps: int = 10  # Steps for acceleration estimation
    sigma_min: float = 0.001  # Minimum noise for numerical stability

    # === Reflow (Trajectory Straightening) ===
    use_reflow: bool = True
    reflow_start_epoch: int = 5000  # Start reflow after initial training
    reflow_epochs: int = 5000
    reflow_ode_steps: int = 20
    reflow_lr_factor: float = 0.1  # Lower LR for reflow phase

    # === Inference ===
    inference_steps: int = 20  # ODE steps for generation
    inference_method: str = "euler"  # "euler", "midpoint", "rk4"
    diversity_scale: float = 0.05  # Noise injection for diversity
    temperature: float = 1.0  # Initial noise scaling

    # === Value Head (for FCU gating) ===
    value_head_hidden: int = 256  # Increased from 128
    value_head_lr: float = 1e-3

    # === Data Paths ===
    train_path: str = "datasets/gsm8k/train.json"
    test_path: str = "datasets/gsm8k/test.json"
    ape_instructions_path: str = "lipo/data/ape_instructions.json"
    hyperband_results_path: Optional[str] = None  # Load pre-evaluated results

    # === Hyperband (Multi-Fidelity) ===
    bmin: int = 10  # Minimum fidelity (samples)
    eta: float = 2.0  # Successive halving rate
    min_fidelity_pct: float = 0.75  # GP trains on top 25% by fidelity

    # === GP Configuration ===
    gp_epochs: int = 10000
    gp_lr: float = 0.0025
    gp_patience: int = 100
    gp_retrain_epochs: int = 1000  # Per-iteration retraining

    # === UCB Acquisition ===
    ucb_beta: float = 8.0  # Initial exploration
    ucb_beta_final: float = 2.0  # Final exploitation
    ucb_beta_adaptive: bool = True
    num_restarts: int = 64  # L-BFGS-B multi-start
    raw_samples: int = 4096  # Initialization samples

    # === Results ===
    results_dir: str = "lido_pp/results"
    checkpoint_dir: str = "lido_pp/checkpoints"
    log_interval: int = 100  # Logging frequency (epochs)

    # === Reproducibility ===
    seed: int = 42

    # === DEPRECATED (kept for backward compatibility) ===
    # These are no longer used but kept to avoid breaking existing code
    gritlm_model: str = "GritLM/GritLM-7B"  # DEPRECATED: use SONAR
    gritlm_quantize: bool = False
    gritlm_dtype: str = "float16"
    gritlm_trust_remote_code: bool = True
    latent_attention_enabled: bool = False  # DEPRECATED
    latent_attention_queries: int = 512
    latent_attention_heads: int = 8
    latent_attention_dropout: float = 0.1
    vae_latent_dim: int = 128  # Alias for tfa_latent_dim
    vae_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    vae_beta: float = 0.005
    vae_mse_weight: float = 0.2
    vae_epochs: int = 50000
    vae_batch_size: int = 64
    vae_lr: float = 0.0006
    vae_annealing_epochs: int = 2500
    vae_patience: int = 1000
    vae_curriculum: bool = True
    vae_curriculum_start_pct: float = 0.3
    vae_curriculum_epochs: int = 5000
    projector_num_tokens: int = 4  # DEPRECATED: use num_memory_slots
    projector_lr: float = 1e-4
    projector_epochs: int = 50
    projector_batch_size: int = 8
    projector_max_length: int = 128
    projector_dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        # Ensure latent dimensions match
        if self.flow_latent_dim != self.tfa_latent_dim:
            raise ValueError(
                f"flow_latent_dim ({self.flow_latent_dim}) must match "
                f"tfa_latent_dim ({self.tfa_latent_dim})"
            )

        # Ensure context dimension matches embedding output
        if self.flow_context_dim != self.embedding_dim:
            raise ValueError(
                f"flow_context_dim ({self.flow_context_dim}) must match "
                f"embedding_dim ({self.embedding_dim})"
            )

        # Sync deprecated aliases
        self.vae_latent_dim = self.tfa_latent_dim

        # Device validation
        if self.device == "auto":
            self.device = get_device("auto")
        if self.eval_device == "auto":
            self.eval_device = get_device("auto")


# Alias for backward compatibility
LIDOPPConfig = FlowPOConfig


# Convenience function for creating config from CLI args
def config_from_args(args) -> LIDOPPConfig:
    """Create config from argparse namespace."""
    config = LIDOPPConfig()
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    return config
