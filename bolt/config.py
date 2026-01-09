"""Configuration for BOLT pipeline.

BOLT: Bayesian Optimization over Latent Templates
Joint instruction + exemplar optimization using VAE latent space and GP.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


def get_device(device: str = "auto") -> str:
    """Determine device to use for computation."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class BOLTConfig:
    """Configuration for BOLT (Bayesian Optimization over Latent Templates).

    Architecture features:
    - CrossAttentionScorer for instruction↔exemplar matching
    - Set Transformer for permutation-invariant exemplar encoding
    - ListMLE ranking loss for direct rank optimization
    - Deep Kernel Learning GP for latent space modeling
    - Fixed K=8 exemplars, joint 32D latent space
    """

    # === Latent Dimensions ===
    embedding_dim: int = 768  # GTR embedding dimension (fixed)
    instruction_latent_dim: int = 16  # Instruction VAE latent
    exemplar_latent_dim: int = 16  # Exemplar VAE latent
    # Total: 32D joint latent for GP

    # === Set Transformer (Exemplar Encoding) ===
    set_transformer_hidden: int = 128  # ISAB hidden dimension
    set_transformer_heads: int = 4  # Attention heads
    num_inducing_points: int = 4  # ISAB inducing points

    # === Exemplar Selection (Fixed K=8) ===
    num_exemplars: int = 8  # Always select exactly 8 exemplars
    scorer_hidden_dim: int = 128  # ExemplarScorer MLP hidden dimension
    exemplar_pool_path: str = "datasets/hbbops/examples_25.txt"

    # === Cross-Attention Scorer ===
    cross_attn_heads: int = 4  # Number of attention heads
    cross_attn_dropout: float = 0.1  # Dropout rate in attention

    # === Ranking Loss ===
    ranking_loss_type: str = "listmle"  # "listmle" or "bce"

    # === Training Data (Q/A pool from train.json) ===
    train_data_path: str = "hbbops_improved_2/data/train.json"
    qa_pool_size: int = 6154  # All Q/A pairs from training set (default: all)
    use_train_json: bool = True  # Use train.json instead of examples_25.txt

    # === VAE Training ===
    vae_beta: float = 0.005  # KL weight
    vae_mse_weight: float = 0.2  # 20% MSE + 80% cosine
    selection_weight: float = 1.0  # Exemplar selection loss weight (BCE on scores)
    vae_epochs: int = 50000
    vae_annealing_epochs: int = 2500  # 5% for KL warmup
    vae_patience: int = 1000
    vae_lr: float = 0.0006
    vae_batch_size: int = 64
    vae_grad_clip: float = 1.0
    vae_eta_min: float = 1e-4

    # === Hyperband (for initial evaluation) ===
    bmin: int = 10  # Minimum fidelity
    eta: float = 2.0  # Successive halving rate
    random_interleaving_prob: float = 0.1  # 10% random proposals
    # Note: GP trains on ALL fidelity levels with Beta smoothing + heteroscedastic noise

    # === GP Training ===
    gp_epochs: int = 10000
    gp_lr: float = 0.0025
    gp_patience: int = 100

    # === Deep Kernel Learning ===
    use_deep_kernel: bool = True  # Use DKL feature extractor before kernel
    dkl_feature_dim: int = 16  # Output dim of feature extractor (per component)
    dkl_hidden_dim: int = 32  # Hidden layer size in feature extractor

    # === Inference ===
    iterations: int = 50  # BO iterations
    num_restarts: int = 64  # L-BFGS-B restarts
    raw_samples: int = 4096  # Raw samples for initialization
    acquisition_type: str = "ucb"  # "ucb" or "logei"
    ucb_beta: float = 8.0  # Initial exploration
    ucb_beta_adaptive: bool = True
    ucb_beta_final: float = 2.0  # Final after decay
    cosine_sim_threshold: float = 0.90  # Instruction acceptance threshold
    max_rejection_attempts: int = 10
    latent_noise_scale: float = 0.05
    latent_margin: float = 0.1  # Margin for latent bounds during optimization

    # === Vec2Text ===
    vec2text_beam: int = 8
    vec2text_model: str = "32_tokens"
    vec2text_max_length: int = 128

    # === Distance Penalty ===
    distance_penalty_enabled: bool = True
    distance_weight: float = 2.0
    distance_threshold: float = 0.3

    # === GP Retrain ===
    gp_retrain_epochs: int = 1000
    gp_retrain_lr: float = 0.001
    gp_retrain_patience: int = 50

    # === APE Generation ===
    ape_num_instructions: int = 2000
    ape_model: str = "Qwen/Qwen2.5-7B-Instruct"
    ape_backend: str = "vllm"
    ape_cache_path: str = "bolt/data/ape_instructions.json"

    # === Training Data Generation ===
    num_training_samples: int = 2000  # Samples for VAE training
    eval_model: str = "Qwen/Qwen2.5-7B-Instruct"
    eval_backend: str = "vllm"

    # === Device/Paths ===
    device: str = "cuda"
    validation_path: str = "hbbops_improved_2/data/validation.json"
    seed: int = 42

    # === Skip Modes ===
    skip_ape: bool = False  # Skip APE generation, use cached
    skip_hbbops: bool = False  # Skip HbBoPs, load from file
    hyperband_evals_path: str = ""  # Path to load HbBoPs results

    @property
    def total_latent_dim(self) -> int:
        """Total joint latent dimension."""
        return self.instruction_latent_dim + self.exemplar_latent_dim

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.vae_mse_weight <= 1.0:
            raise ValueError(f"vae_mse_weight must be in [0, 1], got {self.vae_mse_weight}")
        if self.instruction_latent_dim <= 0:
            raise ValueError(f"instruction_latent_dim must be positive")
        if self.exemplar_latent_dim <= 0:
            raise ValueError(f"exemplar_latent_dim must be positive")
        if self.num_exemplars < 1:
            raise ValueError(f"num_exemplars must be >= 1")
        if self.ucb_beta_adaptive and self.ucb_beta_final > self.ucb_beta:
            raise ValueError(f"ucb_beta_final must be <= ucb_beta when adaptive")
