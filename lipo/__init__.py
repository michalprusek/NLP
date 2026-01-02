"""LIPO: Hyperband-based BO with VAE latent space for instruction optimization.

Pipeline:
1. APE generation: Create diverse instructions
2. VAE training: Learn smooth latent space (768D → 64D → 768D)
3. Hyperband: Successive halving with GP-guided selection
4. InvBO inference: LogEI optimization + 512-token Vec2Text inversion

Self-contained module - no dependencies on other project modules.

Usage:
    uv run python -m lipo.run --iterations 10 --ape-instructions 1000

Debug with 10 instructions:
    uv run python -m lipo.run --iterations 1 --ape-instructions 10 --debug
"""

from lipo.config import Config
from lipo.instruction import InstructionOnlyPrompt
from lipo.encoder import GTRInstructionEncoder, InstructionVAE, VAEWithAdapter
from lipo.gp import GPWithEI, InstructionDeepKernelGP
from lipo.hyperband import LIPOHyperband
from lipo.training import LIPOHyperbandTrainer, APEGenerator
from lipo.inference import LIPOHyperbandInference, Vec2TextInverter, InversionResult
from lipo.evaluate import GSM8KEvaluator, create_evaluator_function

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
    "LIPOHyperband",
    # Training
    "LIPOHyperbandTrainer",
    "APEGenerator",
    # Inference
    "LIPOHyperbandInference",
    "Vec2TextInverter",
    "InversionResult",
    # Evaluation
    "GSM8KEvaluator",
    "create_evaluator_function",
]
