"""Unified configuration for LIPO pipeline.

Single Source of Truth (SSOT) for all parameters.
CLI arguments in run.py override these defaults.
"""

import torch
from dataclasses import dataclass


def get_device(device: str = "auto") -> str:
    """Determine device to use for computation.

    Centralized device detection to avoid duplication across modules.

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
class Config:
    """Unified configuration for LIPO pipeline.

    Organized into logical sections:
    - APE Generation
    - VAE Training
    - Latent Dimensions
    - Hyperband
    - GP Training
    - Inference
    - Inversion Optimization
    - GP Retrain (during inference)
    - Device/Paths
    """

    # === APE Generation ===
    ape_num_instructions: int = 2000
    ape_model: str = "Qwen/Qwen2.5-7B-Instruct"
    ape_backend: str = "vllm"
    ape_cache_path: str = "lipo/data/ape_instructions.json"
    ape_batch_size: int = 10
    ape_max_tokens: int = 100
    ape_max_length: int = 500

    # === VAE Training ===
    vae_beta: float = 0.005  # KL regularization weight (lowered from 0.01 for better reconstruction)
    vae_gamma: float = 0.0  # Cycle consistency disabled (compensated by higher beta)
    vae_mse_weight: float = 0.2  # MSE weight in reconstruction loss (0.2 = 20% MSE + 80% cosine)
    vae_epochs: int = 50000  # Increased for better reconstruction with 32D latent
    vae_annealing_epochs: int = 2500  # 5% of epochs for KL warmup
    vae_patience: int = 1000  # More patience for longer training
    vae_lr: float = 0.0006
    vae_batch_size: int = 64
    vae_grad_clip: float = 1.0
    vae_eta_min: float = 1e-4
    vae_curriculum: bool = True  # Curriculum learning: start with shorter instructions
    vae_curriculum_start_pct: float = 0.3  # Start with 30% of instructions (shortest)
    vae_curriculum_epochs: int = 5000  # Epochs over which to increase to 100%

    # === Latent Dimensions ===
    embedding_dim: int = 768  # GTR embedding dimension
    latent_dim: int = 32  # VAE latent dimension (768/32 = 24x compression, balance between fidelity and smoothness)
    # Note: No adapter - GP works directly on 32D VAE latent with ARD kernel

    # === Round-Trip Validation ===
    roundtrip_validation_threshold: float = 0.90  # Min cosine sim for VAE quality
    roundtrip_validation_samples: int = 20  # Number of samples to test

    # === Hyperband ===
    bmin: int = 10  # Minimum fidelity (samples)
    eta: float = 2.0  # Downsampling rate
    random_interleaving_prob: float = 0.1
    min_fidelity_pct: float = 0.75  # Train GP only on fidelity >= 75% of full (top 25%)

    # === GP Training ===
    gp_epochs: int = 10000
    gp_lr: float = 0.0025  # Scaled down for 32D latent (0.01 * 0.25 per gp.py:446)
    gp_patience: int = 100

    # === Inference ===
    num_restarts: int = 64  # L-BFGS-B restarts for BoTorch optimization
    raw_samples: int = 4096  # Raw samples for initialization (higher for better coverage in high-D)
    acquisition_type: str = "ucb"  # Acquisition function: "ucb" or "logei"
    ucb_beta: float = 8.0  # UCB exploration parameter (higher = more exploration)
    ucb_beta_adaptive: bool = True  # Enable adaptive UCB beta (decay over iterations)
    ucb_beta_final: float = 2.0  # Final UCB beta at last iteration (when adaptive)
    cosine_sim_threshold: float = 0.90  # Min cosine similarity for candidate acceptance
    max_rejection_attempts: int = 10  # Max attempts (increased for better candidates)
    latent_noise_scale: float = 0.05  # Noise scale for diversity injection in latent space

    # === Vec2Text ===
    vec2text_beam: int = 8  # Beam width for Vec2Text generation
    vec2text_model: str = "32_tokens"  # "32_tokens" (recommended, no unicode issues) or "512_tokens" (longer but produces garbage)
    vec2text_max_length: int = 128  # Maximum output tokens for Vec2Text
    vec2text_finetuned_path: str = ""  # Path to fine-tuned Vec2Text corrector (empty = use pre-trained)
    vec2text_finetuned_inverter_path: str = ""  # Path to fine-tuned InversionModel (empty = use pre-trained)
    # Adaptive correction: iteratively refine until embedding matches target
    vec2text_adaptive_correction: bool = False  # Enable adaptive iterative correction
    vec2text_adaptive_max_steps: int = 100  # Maximum correction steps
    vec2text_adaptive_threshold: float = 0.98  # Stop when cosine >= this (high = strict)

    # === TuRBO (Trust Region) ===
    # Parameters from Eriksson et al. "Scalable Global Optimization via Local BO" (NeurIPS 2019)
    turbo_enabled: bool = False  # Disabled for global exploration (use distance penalty instead)
    turbo_L_init: float = 0.8  # Initial side length (paper default)
    turbo_L_max: float = 1.6  # Maximum side length (paper default)
    turbo_L_min: float = 0.0078  # Minimum side length (2^-7, triggers restart)
    turbo_tau_succ: int = 3  # Consecutive successes to double L (paper default)
    turbo_tau_fail: int = 32  # Consecutive failures to halve L (paper: ⌈d/q⌉ = ⌈32/1⌉)

    # === PAS (Potential-Aware Anchor Selection) ===
    # From InvBO paper - Thompson Sampling based anchor selection
    pas_enabled: bool = True  # Enable potential-aware anchor selection
    pas_n_candidates: int = 100  # Candidates per anchor for Thompson Sampling

    # === Distance Penalty (when TuRBO disabled) ===
    # Penalizes candidates far from training data to prevent GP optimization in empty space
    distance_penalty_enabled: bool = True  # Enable when turbo_enabled=False
    distance_weight: float = 2.0  # Penalty strength (higher = stronger constraint)
    distance_threshold: float = 0.3  # Min distance before penalty kicks in (normalized space)

    # === Latent Space ===
    latent_margin: float = 0.2  # Margin for latent bounds expansion

    # === GP Retrain (during inference) ===
    gp_retrain_epochs: int = 1000
    gp_retrain_lr: float = 0.001
    gp_retrain_patience: int = 50  # Increased from 10 to allow better convergence

    # === Device/Paths ===
    device: str = "cuda"
    validation_path: str = "hbbops_improved_2/data/validation.json"
    seed: int = 42

    def for_hyperband_gp(self) -> dict:
        """Get GP training params for Hyperband (reduced for speed).

        During Hyperband, GP is retrained frequently so we use shorter training.

        Returns:
            Dict with epochs, lr, patience for GP training during Hyperband.
        """
        return {
            "epochs": 3000,
            "lr": self.gp_lr,
            "patience": 50,
        }
