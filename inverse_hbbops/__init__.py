"""Inverse HbBoPs: Hyperband-based BO with VAE latent space for instruction optimization.

Pipeline:
1. APE generation: Create diverse instructions
2. VAE training: Learn smooth latent space (768D → 10D → 768D)
3. Hyperband: Successive halving with GP-guided selection
4. InvBO inference: LogEI optimization + 512-token Vec2Text inversion

Self-contained module - no dependencies on other project modules.

Usage:
    uv run python -m inverse_hbbops.run --iterations 10 --ape-instructions 1000

Debug with 10 instructions:
    uv run python -m inverse_hbbops.run --iterations 1 --ape-instructions 10 --debug
"""

from inverse_hbbops.config import Config
from inverse_hbbops.instruction import InstructionOnlyPrompt
from inverse_hbbops.encoder import GTRInstructionEncoder, InstructionVAE, VAEWithAdapter
from inverse_hbbops.gp import GPWithEI, InstructionDeepKernelGP
from inverse_hbbops.hyperband import InverseHbBoPs
from inverse_hbbops.training import InverseHbBoPsTrainer, APEGenerator
from inverse_hbbops.inference import InverseHbBoPsInference, Vec2TextInverter, InversionResult
from inverse_hbbops.evaluate import GSM8KEvaluator, create_evaluator_function

__all__ = [
    # Config (SSOT)
    "Config",
    # Instruction
    "InstructionOnlyPrompt",
    # Encoder & VAE
    "GTRInstructionEncoder",
    "InstructionVAE",
    "VAEWithAdapter",
    # GP
    "GPWithEI",
    "InstructionDeepKernelGP",
    # Hyperband
    "InverseHbBoPs",
    # Training
    "InverseHbBoPsTrainer",
    "APEGenerator",
    # Inference
    "InverseHbBoPsInference",
    "Vec2TextInverter",
    "InversionResult",
    # Evaluation
    "GSM8KEvaluator",
    "create_evaluator_function",
]
