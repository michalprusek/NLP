"""Unified configuration for LIPO pipeline.

Single Source of Truth (SSOT) for all parameters.
CLI arguments in run.py override these defaults.
"""

from dataclasses import dataclass


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
    ape_num_instructions: int = 1000
    ape_model: str = "Qwen/Qwen2.5-7B-Instruct"
    ape_backend: str = "vllm"
    ape_cache_path: str = "lipo/data/ape_instructions.json"
    ape_batch_size: int = 10
    ape_max_tokens: int = 100
    ape_max_length: int = 500

    # === VAE Training ===
    vae_beta: float = 0.003  # KL regularization weight (scaled for latent_dim=64)
    vae_epochs: int = 10000
    vae_annealing_epochs: int = 500
    vae_patience: int = 500
    vae_lr: float = 0.0003
    vae_batch_size: int = 64
    vae_grad_clip: float = 1.0
    vae_eta_min: float = 1e-4

    # === Latent Dimensions ===
    embedding_dim: int = 768  # GTR embedding dimension
    latent_dim: int = 64  # VAE latent dimension
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

    # === Vec2Text ===
    vec2text_beam: int = 8  # Beam width for Vec2Text generation
    vec2text_model: str = "512_tokens"  # "32_tokens" or "512_tokens"

    # === Inversion Optimization ===
    inversion_n_steps: int = 100  # Adam optimization steps
    inversion_lr: float = 0.1  # Adam learning rate
    inversion_convergence_threshold: float = 0.01  # Early stop threshold
    latent_margin: float = 0.2  # Margin for latent bounds expansion

    # === GP Retrain (during inference) ===
    gp_retrain_epochs: int = 500
    gp_retrain_lr: float = 0.001
    gp_retrain_patience: int = 10

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
