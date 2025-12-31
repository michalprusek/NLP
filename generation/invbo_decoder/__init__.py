"""InvBO-style VAE Decoder Inversion for Instruction Optimization.

VAE-based Bayesian optimization for instruction generation. Uses BoTorch qLogEI
for gradient-based optimization in 10D latent space with Vec2Text inversion.

Architecture:
    Instruction Text -> GTR (768D) -> VAE Encoder (10D) -> GP -> LogEI optimization
                                          ^
                                          | KL regularization
                                          v
    10D optimum -> VAE Decoder (768D) -> Vec2Text -> Novel Instruction Text
"""

from generation.invbo_decoder.encoder import (
    GTRInstructionEncoder,
    InstructionVAE,
)
from generation.invbo_decoder.gp import InstructionDeepKernelGP
from generation.invbo_decoder.training import InvBOTrainer
from generation.invbo_decoder.inference import InvBOInference

__all__ = [
    "GTRInstructionEncoder",
    "InstructionVAE",
    "InstructionDeepKernelGP",
    "InvBOTrainer",
    "InvBOInference",
]
