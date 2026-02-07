"""Molecular encoder/decoder for GuacaMol optimization.

This module provides a codec for converting between SMILES strings and
continuous embeddings suitable for flow-based optimization.

Uses SELFIES VAE (from LOLBO) with:
- 256D latent space (2 bottleneck tokens × 128 d_model)
- Transformer encoder/decoder
- SELFIES molecular representation (guarantees valid molecules)
"""

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import selfies as sf
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Add lolbo_ref to path for imports
LOLBO_REF_PATH = (Path(__file__).parent.parent.parent / "lolbo_ref").resolve()
_lolbo_path_str = str(LOLBO_REF_PATH)
if _lolbo_path_str not in sys.path:
    sys.path.insert(0, _lolbo_path_str)

# Default path to SELFIES VAE weights
DEFAULT_SELFIES_VAE_WEIGHTS = (
    LOLBO_REF_PATH
    / "lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt"
)


class MolecularCodec(ABC):
    """Abstract base class for molecular encoders/decoders."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding space."""
        pass

    @abstractmethod
    def encode(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode SMILES strings to embeddings."""
        pass

    @abstractmethod
    def decode(self, embeddings: torch.Tensor, temperature: float = 1.0) -> list[str]:
        """Decode embeddings to SMILES strings."""
        pass


class SELFIESVAECodec(MolecularCodec):
    """SELFIES VAE codec for 256D molecular embeddings.

    Uses the SELFIES VAE from LOLBO which provides:
    - 256D latent space (2 bottleneck tokens × 128 d_model)
    - Transformer-based encoder/decoder
    - SELFIES encoding that guarantees valid molecules

    Example:
        >>> codec = SELFIESVAECodec.from_pretrained()
        >>> embeddings = codec.encode(["CCO", "CCC", "c1ccccc1"])  # [3, 256]
        >>> smiles = codec.decode(embeddings)
    """

    EMBEDDING_DIM = 256  # 2 bottleneck × 128 d_model

    def __init__(
        self,
        model: nn.Module,
        dataset,  # SELFIESDataset for vocab
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.device = device

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Optional[str] = None,
        device: str = "cuda",
    ) -> "SELFIESVAECodec":
        """Load codec from pretrained SELFIES VAE weights.

        Args:
            weights_path: Path to state_dict file (default: LOLBO pretrained)
            device: Device for model
        """
        from lolbo.utils.mol_utils.selfies_vae.data import SELFIESDataset
        from lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import (
            InfoTransformerVAE,
        )

        weights_path = Path(weights_path) if weights_path else DEFAULT_SELFIES_VAE_WEIGHTS

        # Create dataset with default vocab
        dataset = SELFIESDataset()

        # Create model with default architecture
        model = InfoTransformerVAE(
            dataset=dataset,
            bottleneck_size=2,
            d_model=128,
        )

        # Load weights
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded SELFIES VAE weights from {weights_path}")
        else:
            raise FileNotFoundError(
                f"Weights not found at {weights_path}\n"
                f"See CLAUDE.md for instructions on obtaining SELFIES VAE weights."
            )

        return cls(model, dataset, device)

    def _smiles_to_selfies(self, smiles: str) -> Optional[str]:
        """Convert SMILES to SELFIES. Returns None if conversion fails."""
        try:
            return sf.encoder(smiles)
        except sf.EncoderError as e:
            logger.debug(f"SELFIES encoding failed for '{smiles[:50]}...': {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error encoding '{smiles[:50]}...': {type(e).__name__}: {e}")
            return None

    def _selfies_to_smiles(self, selfies_str: str) -> Optional[str]:
        """Convert SELFIES to SMILES. Returns None if conversion fails."""
        try:
            return sf.decoder(selfies_str)
        except sf.DecoderError as e:
            logger.debug(f"SELFIES decoding failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error decoding SELFIES: {type(e).__name__}: {e}")
            return None

    def _tokenize_selfies(self, selfies_str: str) -> torch.Tensor:
        """Tokenize SELFIES string to tensor.

        Filters out tokens not in the vocabulary (e.g., stereochemistry).
        """
        tokens = list(sf.split_selfies(selfies_str))
        # Filter to only known tokens
        known_tokens = [t for t in tokens if t in self.dataset.vocab2idx]
        if not known_tokens:
            # If no known tokens, return a minimal valid sequence
            known_tokens = ['[C]']
        return self.dataset.encode(known_tokens)

    def encode(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode SMILES strings to 256D embeddings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Embeddings tensor [N, 256]
        """
        if not smiles_list:
            return torch.empty(0, self.EMBEDDING_DIM, device=self.device)

        embeddings = []
        invalid_count = 0
        with torch.no_grad():
            for smiles in smiles_list:
                # Convert SMILES to SELFIES
                selfies_str = self._smiles_to_selfies(smiles)
                if selfies_str is None:
                    # Invalid molecule - return zeros
                    invalid_count += 1
                    embeddings.append(torch.zeros(self.EMBEDDING_DIM, device=self.device))
                    continue

                # Tokenize
                tokens = self._tokenize_selfies(selfies_str).unsqueeze(0).to(self.device)

                # Encode (get mu from VAE)
                mu, sigma = self.model.encode(tokens)

                # Flatten: [1, bottleneck_size, d_model] -> [1, bottleneck_size * d_model]
                z = mu.reshape(1, -1)
                embeddings.append(z[0])

        # Log warning if significant number of invalid molecules
        if invalid_count > 0:
            logger.warning(
                f"Encoded {invalid_count}/{len(smiles_list)} invalid molecules as zeros"
            )

        return torch.stack(embeddings)

    def encode_batch(
        self, smiles_list: list[str], batch_size: int = 64
    ) -> torch.Tensor:
        """Encode SMILES strings in batches for efficiency.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for encoding

        Returns:
            Embeddings tensor [N, 256]
        """
        all_embeddings = []

        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def decode(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
    ) -> list[str]:
        """Decode embeddings to SMILES strings.

        Args:
            embeddings: Embeddings tensor [N, 256]
            temperature: Sampling temperature (not used in current implementation)

        Returns:
            List of SMILES strings
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        B = embeddings.shape[0]
        smiles_list = []

        with torch.no_grad():
            for i in range(B):
                # Reshape to [1, bottleneck_size, d_model]
                z = embeddings[i : i + 1].reshape(1, 2, 128).to(self.device)

                # Sample from VAE decoder
                tokens = self.model.sample(z=z)

                # Convert tokens to SELFIES string
                selfies_str = self.dataset.decode(tokens[0].cpu().tolist())

                # Convert SELFIES to SMILES
                smiles = self._selfies_to_smiles(selfies_str)
                if smiles is None:
                    smiles = ""

                smiles_list.append(smiles)

        return smiles_list

    def test_roundtrip(
        self, smiles_list: list[str], temperature: float = 0.0
    ) -> tuple[torch.Tensor, list[str], list[bool]]:
        """Test encode-decode roundtrip quality.

        Returns:
            Tuple of (embeddings, decoded_smiles, match_flags)
        """
        embeddings = self.encode(smiles_list)
        decoded = self.decode(embeddings, temperature=temperature)
        matches = [orig == dec for orig, dec in zip(smiles_list, decoded)]
        return embeddings, decoded, matches


# Alias for backwards compatibility
MiniCDDDCodec = SELFIESVAECodec  # miniCDDD was replaced with SELFIES VAE


def create_molecular_codec(
    codec_type: str = "selfies_vae",
    model_path: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> MolecularCodec:
    """Factory function to create a molecular codec.

    Args:
        codec_type: Type of codec:
            - "selfies_vae" (default): LOLBO SELFIES VAE (256D)
            - "minicddd": Alias for selfies_vae
        model_path: Path to model weights (optional)
        device: Device for computation
        **kwargs: Additional arguments for specific codec

    Returns:
        MolecularCodec instance
    """
    if codec_type in ("selfies_vae", "minicddd"):
        return SELFIESVAECodec.from_pretrained(model_path, device=device, **kwargs)
    elif codec_type == "smi_ted":
        from shared.guacamol.smi_ted_codec import SmiTedCodec
        return SmiTedCodec.from_pretrained(device=device)
    else:
        raise ValueError(
            f"Unknown codec type: {codec_type}. Use 'selfies_vae' or 'smi_ted'."
        )
