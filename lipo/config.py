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
    vae_beta: float = 0.003  # KL regularization weight (low for text VAE)
    vae_gamma: float = 5.0  # Cycle consistency weight: ensures z ≈ encode(decode(z))
    vae_epochs: int = 15000
    vae_annealing_epochs: int = 500
    vae_patience: int = 500
    vae_lr: float = 0.0006
    vae_batch_size: int = 64
    vae_grad_clip: float = 1.0
    vae_eta_min: float = 1e-4

    # === Latent Dimensions ===
    embedding_dim: int = 768  # GTR embedding dimension
    latent_dim: int = 64  # VAE latent dimension (768/64 = 12x compression)
    gp_latent_dim: int = 10  # Adapter output dimension for GP

    # === Hyperband ===
    bmin: int = 10  # Minimum fidelity (samples)
    eta: float = 2.0  # Downsampling rate
    random_interleaving_prob: float = 0.1
    min_fidelity_pct: float = 0.75  # Train GP only on fidelity >= 75% of full (top 25%)

    # === GP Training ===
    gp_epochs: int = 10000
    gp_lr: float = 0.01
    gp_patience: int = 100

    # === Inference ===
    num_restarts: int = 64  # L-BFGS-B restarts for BoTorch optimization
    raw_samples: int = 1024  # Raw samples for initialization seeding
    use_inversion: bool = True  # Use InvBO inversion loop
    max_inversion_iters: int = 3  # Maximum inversion iterations per step
    gap_threshold: float = 0.1  # Gap threshold for re-inversion (cosine distance)
    cosine_sim_threshold: float = 0.90  # Min cosine similarity between decoder(z) and GTR(text)
    max_rejection_attempts: int = 5  # Max attempts to find aligned candidate before falling back

    # === Vec2Text ===
    vec2text_beam: int = 8  # Beam width for Vec2Text generation
    vec2text_model: str = "512_tokens"  # "32_tokens" or "512_tokens"
    vec2text_max_length: int = 128  # Maximum output tokens for Vec2Text

    # === TuRBO (Trust Region) ===
    turbo_enabled: bool = True  # Enable trust region optimization
    turbo_L_init: float = 0.8  # Initial side length (fraction of unit cube)
    turbo_L_max: float = 1.6  # Maximum side length
    turbo_L_min: float = 0.0078  # Minimum side length (0.5^7, triggers restart)
    turbo_tau_succ: int = 3  # Consecutive successes to double L
    turbo_tau_fail: int = 40  # Consecutive failures to halve L

    # === PAS (Potential-Aware Anchor Selection) ===
    pas_enabled: bool = True  # Enable potential-aware anchor selection
    pas_n_candidates: int = 100  # Candidates per anchor for Thompson Sampling

    # === Inversion Optimization ===
    inversion_n_steps: int = 100  # Adam optimization steps
    inversion_lr: float = 0.1  # Adam learning rate
    inversion_convergence_threshold: float = 0.01  # Early stop threshold
    latent_margin: float = 0.2  # Margin for latent bounds expansion

    # === ZSInvert Refinement ===
    zsinvert_enabled: bool = True  # Enable ZSInvert refinement after Vec2Text
    zsinvert_iterations: int = 15  # Max refinement iterations
    zsinvert_lr: float = 0.1  # Learning rate for gradient refinement
    zsinvert_steps_per_iter: int = 50  # Optimization steps per iteration
    zsinvert_improvement_threshold: float = 0.01  # Min improvement to continue
    zsinvert_patience: int = 5  # Patience for early stopping in ZSInvert

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
