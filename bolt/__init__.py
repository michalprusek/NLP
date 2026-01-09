"""BOLT: Bayesian Optimization over Latent Templates.

Joint optimization of instructions and few-shot exemplars using:
- Structure-Aware VAE with Set Transformer encoding
- CrossAttentionScorer for instruction↔exemplar matching
- ListMLE ranking loss for direct rank optimization
- Deep Kernel Learning GP for latent space modeling
"""

from bolt.config import BOLTConfig
from bolt.encoder import CrossAttentionScorer, StructureAwareVAE
from bolt.gp import DeepKernelGP, GPWithEI, JointPromptGP
from bolt.hyperband import BOLTHyperband
from bolt.inference import BOLTInference

__all__ = [
    "BOLTConfig",
    "CrossAttentionScorer",
    "StructureAwareVAE",
    "DeepKernelGP",
    "GPWithEI",
    "JointPromptGP",
    "BOLTHyperband",
    "BOLTInference",
]
