"""COWBOYS Vec2Text: pCN MCMC optimization with trust regions.

Implements Continuous Optimization With BOx constraints and Yielding Samples.

Key differences from robust_vec2text:
- pCN MCMC sampling instead of gradient ascent
- TuRBO-style trust region constraints
- Weighted VAE retraining for latent space focusing

Usage:
    # Run from command line:
    uv run python -m generation.cowboys_vec2text.run --iterations 10

    # Or programmatically:
    from generation.cowboys_vec2text import (
        CowboysOptimizer,
        CowboysInference,
        MCMCConfig,
        TRConfig,
    )

    optimizer = CowboysOptimizer(instructions, exemplars)
    optimizer.train_vae()
    optimizer.train_exemplar_gp_on_decoded(grid_path)

    inference = CowboysInference(
        vae=optimizer.get_vae(),
        exemplar_selector=optimizer.get_exemplar_selector(),
        exemplar_emb=optimizer.exemplar_embeddings[best_id],
        gtr=optimizer.gtr,
    )

    result = inference.full_pipeline(
        initial_latent=optimizer.get_best_latent(),
        best_y=best_error,
        mcmc_config=MCMCConfig(),
        trust_region=optimizer.initialize_trust_region(),
    )

References:
- COWBOYS: "Return of the Latent Space COWBOYS" (ICML 2026)
- TuRBO: "Scalable Global Optimization via Local BO" (NeurIPS 2019)
- HbBoPs: "Hyperband-based BO for Prompt Selection"
"""

# Reused from robust_vec2text
from .encoder import GTRPromptEncoder
from .gp import LatentGP, GPTrainer
from .exemplar_selector import ExemplarSelector, DeepKernelGP, FeatureExtractor

# Extended with weighted training
from .vae import InstructionVAE, WeightedVAELoss, VAELoss
from .training import WeightedVAETrainer, VAETrainer, RetrainConfig

# New COWBOYS components
from .mcmc import pCNSampler, MCMCConfig, MCMCResult
from .trust_region import TrustRegionManager, TRConfig, TRState

# Orchestration
from .optimizer import CowboysOptimizer, GridPrompt
from .inference import CowboysInference, InversionResult, CowboysResult

__all__ = [
    # Encoder
    "GTRPromptEncoder",
    # VAE
    "InstructionVAE",
    "WeightedVAELoss",
    "VAELoss",
    # Training
    "WeightedVAETrainer",
    "VAETrainer",
    "RetrainConfig",
    # GP
    "LatentGP",
    "GPTrainer",
    "ExemplarSelector",
    "DeepKernelGP",
    "FeatureExtractor",
    # MCMC
    "pCNSampler",
    "MCMCConfig",
    "MCMCResult",
    # Trust Region
    "TrustRegionManager",
    "TRConfig",
    "TRState",
    # Optimizer
    "CowboysOptimizer",
    "GridPrompt",
    # Inference
    "CowboysInference",
    "InversionResult",
    "CowboysResult",
]
