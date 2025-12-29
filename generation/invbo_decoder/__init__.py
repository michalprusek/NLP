"""InvBO-style Decoder Inversion for Instruction Optimization.

Implements a decoder from GP latent space (10D) to Vec2Text embedding space (768D)
with cyclic loss training. Addresses the "misalignment problem" from InvBO (NeurIPS 2024).

Architecture:
    Instruction Text -> GTR (768D) -> Encoder (10D) -> GP -> EI optimization
                                          ^
                                          | cyclic loss
                                          v
    10D optimum -> Decoder (768D) -> Vec2Text -> Novel Instruction Text
"""

from generation.invbo_decoder.encoder import (
    InstructionFeatureExtractor,
    GTRInstructionEncoder,
)
from generation.invbo_decoder.gp import InstructionDeepKernelGP
from generation.invbo_decoder.decoder import LatentDecoder, DecoderCyclicLoss
from generation.invbo_decoder.training import InvBOTrainer
from generation.invbo_decoder.inference import InvBOInference

__all__ = [
    "InstructionFeatureExtractor",
    "GTRInstructionEncoder",
    "InstructionDeepKernelGP",
    "LatentDecoder",
    "DecoderCyclicLoss",
    "InvBOTrainer",
    "InvBOInference",
]
