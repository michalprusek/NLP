"""Latent GP - Reused from robust_vec2text.

Operates on VAE latent space (32D) or decoded embeddings.
The COWBOYS fix (training on decoded embeddings) is applied at the optimizer level.
"""

from robust_vec2text.gp import LatentGP, GPTrainer

__all__ = ["LatentGP", "GPTrainer"]
