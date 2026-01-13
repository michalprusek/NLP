"""
Configuration for LID-O++ (Latent Instruction Diffusion Optimization++).

This module defines the complete configuration for the LID-O++ architecture,
including GritLM backbone, Rectified Flow, and Active Learning with FCU.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


def get_device(device_str: str = "auto") -> str:
    """Resolve device string to actual device."""
    if device_str == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_str


@dataclass
class LIDOPPConfig:
    """
    LID-O++ configuration.

    Organized into sections:
    - Device: GPU allocation
    - GritLM Backbone: Unified encoder settings
    - Latent Attention: NV-Embed style pooling
    - VAE: Latent space compression (reused from LIPO)
    - Flow-DiT: Rectified Flow network
    - OAT-FM: Optimal Acceleration Transport regularization
    - Reflow: Trajectory straightening for 1-step inference
    - Inference: Sampling parameters
    - Active Learning: FCU and cost-aware acquisition
    - ZSInvert: Zero-shot embedding inversion
    - Data: Dataset paths
    - Results: Output directories
    """

    # === Device Configuration ===
    device: str = "cuda:0"
    eval_device: str = "cuda:1"  # Separate GPU for LLM evaluation

    # === GritLM Backbone ===
    gritlm_model: str = "GritLM/GritLM-7B"
    gritlm_quantize: bool = False  # Not needed with 2x L40S (96GB total)
    gritlm_dtype: str = "float16"  # or "bfloat16"
    gritlm_trust_remote_code: bool = True

    # === Embedding Output ===
    embedding_dim: int = 768  # GTR-compatible output dimension

    # === Latent Attention (NV-Embed style) ===
    latent_attention_enabled: bool = True
    latent_attention_queries: int = 512  # Number of learnable query vectors
    latent_attention_heads: int = 8
    latent_attention_dropout: float = 0.1

    # === VAE Configuration (reuse from LIPO) ===
    vae_latent_dim: int = 32  # Compression: 768D → 32D
    vae_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    vae_beta: float = 0.005  # KL regularization weight
    vae_mse_weight: float = 0.2  # 20% MSE + 80% cosine in reconstruction
    vae_epochs: int = 50000
    vae_batch_size: int = 64
    vae_lr: float = 0.0006
    vae_annealing_epochs: int = 2500  # KL warmup period
    vae_patience: int = 1000  # Early stopping
    vae_curriculum: bool = True  # Curriculum learning
    vae_curriculum_start_pct: float = 0.3  # Start with 30% easiest
    vae_curriculum_epochs: int = 5000  # Ramp to 100%

    # === Flow-DiT Architecture ===
    flow_latent_dim: int = 32  # Operates in VAE latent space
    flow_hidden_dim: int = 512  # Transformer hidden dimension
    flow_num_layers: int = 6  # Number of transformer blocks
    flow_num_heads: int = 8  # Attention heads
    flow_mlp_ratio: float = 4.0  # MLP expansion ratio
    flow_time_embed_dim: int = 256  # Timestep embedding dimension
    flow_context_dim: int = 768  # Context (GritLM embedding)
    flow_context_tokens: int = 4  # Number of context tokens
    flow_dropout: float = 0.1
    flow_cross_attention: bool = True  # Use cross-attention for conditioning

    # === Flow Training ===
    flow_lr: float = 1e-4
    flow_epochs: int = 10000
    flow_batch_size: int = 64
    flow_warmup_epochs: int = 500
    flow_patience: int = 500
    flow_grad_clip: float = 1.0

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
    inference_steps: int = 1  # 1-step after reflow!
    inference_method: str = "euler"  # "euler", "midpoint", "rk4"
    guidance_scale: float = 1.0  # GP guidance strength
    diversity_scale: float = 0.05  # Noise injection for diversity
    temperature: float = 1.0  # Initial noise scaling

    # === Active Learning (FCU) ===
    fcu_enabled: bool = True
    lambda_cost: float = 0.1  # Curvature penalty in acquisition
    curvature_percentile: float = 90.0  # Top 10% get evaluated
    curvature_steps: int = 20  # Steps for curvature estimation
    value_head_hidden: int = 128  # Value head network hidden dim
    value_head_lr: float = 1e-3
    min_evals_before_gating: int = 50  # Build up GP first

    # === Latent Injection (Decoder) ===
    projector_num_tokens: int = 4  # Number of prefix tokens
    projector_lr: float = 1e-4
    projector_epochs: int = 50
    projector_batch_size: int = 8
    projector_max_length: int = 128
    projector_dropout: float = 0.1

    # === ZSInvert (Zero-Shot Inversion) - DEPRECATED, use Latent Injection ===
    zsinvert_enabled: bool = False  # Replaced by Latent Injection
    zsinvert_beam_width: int = 8
    zsinvert_max_length: int = 128
    correction_model: str = "Qwen/Qwen2.5-3B-Instruct"
    cosine_sim_threshold: float = 0.90  # Min reconstruction quality
    max_rejection_attempts: int = 10

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

    def __post_init__(self):
        """Validate configuration."""
        # Ensure latent dimensions match
        if self.flow_latent_dim != self.vae_latent_dim:
            raise ValueError(
                f"flow_latent_dim ({self.flow_latent_dim}) must match "
                f"vae_latent_dim ({self.vae_latent_dim})"
            )

        # Ensure context dimension matches embedding output
        if self.flow_context_dim != self.embedding_dim:
            raise ValueError(
                f"flow_context_dim ({self.flow_context_dim}) must match "
                f"embedding_dim ({self.embedding_dim})"
            )

        # Device validation
        if self.device == "auto":
            self.device = get_device("auto")
        if self.eval_device == "auto":
            self.eval_device = get_device("auto")


# Convenience function for creating config from CLI args
def config_from_args(args) -> LIDOPPConfig:
    """Create config from argparse namespace."""
    config = LIDOPPConfig()
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    return config
