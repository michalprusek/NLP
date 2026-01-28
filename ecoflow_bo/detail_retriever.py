"""
Detail Retriever: Get z_detail from training set based on z_core similarity.

The key insight: During BO, we only optimize z_core (16D). But the decoder
needs z_full (48D) = [z_core, z_detail] for high-fidelity reconstruction.

Strategies for z_detail:
1. "zero": All zeros (simplest, may hurt quality)
2. "mean": Average z_detail from training set
3. "nearest": Copy z_detail from nearest neighbor by z_core (recommended!)
4. "k_nearest": Average z_detail from k nearest neighbors

The "nearest" strategy leverages the training data effectively:
"If my z_core is similar to training sample X, my z_detail should also be similar."
"""

import torch
import torch.nn.functional as F
from typing import Optional, Literal, Union
import numpy as np

# FAISS is optional - only needed for large datasets (>50k)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class DetailRetriever:
    """
    Retrieve z_detail from training set based on z_core similarity.

    Uses FAISS for fast nearest neighbor search in the training z_core space.

    Usage:
        # Setup (once, with training data)
        retriever = DetailRetriever(
            z_cores_train,    # [N, 16]
            z_details_train,  # [N, 32]
            mode="nearest"
        )

        # During BO (every iteration)
        z_detail = retriever.get_detail(z_core_candidate)  # [B, 32]
    """

    def __init__(
        self,
        z_cores: torch.Tensor,
        z_details: torch.Tensor,
        mode: Literal["zero", "mean", "nearest", "k_nearest"] = "nearest",
        k: int = 1,
        device: str = "cuda",
    ):
        """
        Initialize detail retriever with training set latents.

        Args:
            z_cores: Training z_core values [N, core_dim]
            z_details: Training z_detail values [N, detail_dim]
            mode: Retrieval strategy
            k: Number of neighbors for k_nearest mode
            device: Device for computations

        Raises:
            ImportError: If FAISS is not installed
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for DetailRetriever (large dataset support). "
                "Install with: pip install faiss-cpu or pip install faiss-gpu. "
                "For smaller datasets (<50K), use SimpleDetailRetriever instead."
            )
        self.mode = mode
        self.k = k
        self.device = device
        self.core_dim = z_cores.shape[1]
        self.detail_dim = z_details.shape[1]

        # Store training data
        self.z_cores = z_cores.to(device)
        self.z_details = z_details.to(device)

        # Precompute statistics
        self.z_detail_mean = z_details.mean(dim=0).to(device)

        # Build FAISS index for fast nearest neighbor search
        if mode in ["nearest", "k_nearest"]:
            self._build_index(z_cores)

    def _build_index(self, z_cores: torch.Tensor):
        """Build FAISS index for fast NN search."""
        # Normalize for cosine similarity
        z_cores_np = z_cores.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(z_cores_np)

        # Build index
        self.index = faiss.IndexFlatIP(self.core_dim)  # Inner product (cosine after L2 norm)
        self.index.add(z_cores_np)

        print(f"[DetailRetriever] Built FAISS index with {len(z_cores_np)} vectors")

    def get_detail(self, z_core: torch.Tensor) -> torch.Tensor:
        """
        Get z_detail for given z_core candidates.

        Args:
            z_core: Query z_core values [B, core_dim]

        Returns:
            z_detail: Retrieved/computed z_detail [B, detail_dim]
        """
        B = z_core.shape[0]

        if self.mode == "zero":
            return torch.zeros(B, self.detail_dim, device=self.device, dtype=z_core.dtype)

        elif self.mode == "mean":
            return self.z_detail_mean.unsqueeze(0).expand(B, -1).clone()

        elif self.mode == "nearest":
            indices = self._find_nearest(z_core, k=1)
            return self.z_details[indices.squeeze(-1)]

        elif self.mode == "k_nearest":
            indices = self._find_nearest(z_core, k=self.k)  # [B, k]
            # Average z_detail from k neighbors
            z_details_k = self.z_details[indices]  # [B, k, detail_dim]
            return z_details_k.mean(dim=1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _find_nearest(self, z_core: torch.Tensor, k: int) -> torch.Tensor:
        """
        Find k nearest neighbors in training set.

        Args:
            z_core: Query vectors [B, core_dim]
            k: Number of neighbors

        Returns:
            indices: Indices of nearest neighbors [B, k]
        """
        # Normalize query
        z_core_np = z_core.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(z_core_np)

        # Search
        _, indices = self.index.search(z_core_np, k)

        return torch.from_numpy(indices).to(self.device)

    def get_full_latent(self, z_core: torch.Tensor) -> torch.Tensor:
        """
        Get full latent z_full = [z_core, z_detail].

        Args:
            z_core: Core latent [B, core_dim]

        Returns:
            z_full: Full latent [B, core_dim + detail_dim]
        """
        z_detail = self.get_detail(z_core)
        return torch.cat([z_core, z_detail], dim=-1)


class SimpleDetailRetriever:
    """
    Simple detail retriever without FAISS dependency.

    Uses brute-force search, suitable for smaller datasets (<100K).
    """

    def __init__(
        self,
        z_cores: torch.Tensor,
        z_details: torch.Tensor,
        mode: Literal["zero", "mean", "nearest", "k_nearest"] = "nearest",
        k: int = 1,
        device: str = "cuda",
    ):
        self.mode = mode
        self.k = k
        self.device = device
        self.core_dim = z_cores.shape[1]
        self.detail_dim = z_details.shape[1]

        # Store training data (normalized for cosine similarity)
        self.z_cores = F.normalize(z_cores.to(device), dim=-1)
        self.z_details = z_details.to(device)

        # Precompute mean
        self.z_detail_mean = z_details.mean(dim=0).to(device)

    def get_detail(self, z_core: torch.Tensor) -> torch.Tensor:
        """Get z_detail for given z_core candidates."""
        B = z_core.shape[0]

        if self.mode == "zero":
            return torch.zeros(B, self.detail_dim, device=self.device, dtype=z_core.dtype)

        elif self.mode == "mean":
            return self.z_detail_mean.unsqueeze(0).expand(B, -1).clone()

        elif self.mode in ["nearest", "k_nearest"]:
            # Normalize query
            z_core_norm = F.normalize(z_core.to(self.device), dim=-1)

            # Compute cosine similarity: [B, N]
            sim = torch.mm(z_core_norm, self.z_cores.t())

            if self.mode == "nearest":
                # Get single nearest neighbor
                indices = sim.argmax(dim=-1)  # [B]
                return self.z_details[indices]
            else:
                # Get k nearest neighbors and average
                _, indices = sim.topk(self.k, dim=-1)  # [B, k]
                z_details_k = self.z_details[indices]  # [B, k, detail_dim]
                return z_details_k.mean(dim=1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_full_latent(self, z_core: torch.Tensor) -> torch.Tensor:
        """Get full latent z_full = [z_core, z_detail]."""
        z_detail = self.get_detail(z_core)
        return torch.cat([z_core, z_detail], dim=-1)


def create_detail_retriever(
    z_cores: torch.Tensor,
    z_details: torch.Tensor,
    mode: str = "nearest",
    k: int = 1,
    device: str = "cuda",
    use_faiss: bool = True,
) -> Union["DetailRetriever", "SimpleDetailRetriever"]:
    """
    Factory function to create detail retriever.

    Uses FAISS for large datasets (>50K), simple brute-force for small ones.
    Falls back to SimpleDetailRetriever if FAISS is not installed.
    """
    N = z_cores.shape[0]

    # Use FAISS for large datasets if available
    if use_faiss and FAISS_AVAILABLE and N > 50000:
        try:
            return DetailRetriever(z_cores, z_details, mode, k, device)
        except Exception as e:
            print(f"[Warning] FAISS failed ({e}), falling back to simple retriever")

    return SimpleDetailRetriever(z_cores, z_details, mode, k, device)
