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
            ValueError: If inputs have invalid shapes or mismatched dimensions
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for DetailRetriever (large dataset support). "
                "Install with: pip install faiss-cpu or pip install faiss-gpu. "
                "For smaller datasets (<50K), use SimpleDetailRetriever instead."
            )

        # Validate inputs
        if z_cores.dim() != 2:
            raise ValueError(f"z_cores must be 2D [N, core_dim], got shape {z_cores.shape}")
        if z_details.dim() != 2:
            raise ValueError(f"z_details must be 2D [N, detail_dim], got shape {z_details.shape}")
        if z_cores.shape[0] != z_details.shape[0]:
            raise ValueError(
                f"z_cores and z_details must have same batch size: "
                f"z_cores has {z_cores.shape[0]}, z_details has {z_details.shape[0]}"
            )
        if z_cores.shape[0] == 0:
            raise ValueError("Cannot create DetailRetriever with empty training set")
        if mode == "k_nearest" and k > z_cores.shape[0]:
            raise ValueError(
                f"k={k} exceeds training set size {z_cores.shape[0]} for k_nearest mode"
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
        # Ensure dtype matches z_core to avoid mixed precision issues
        if z_detail.dtype != z_core.dtype:
            z_detail = z_detail.to(dtype=z_core.dtype)
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
        """
        Initialize simple detail retriever with training set latents.

        Args:
            z_cores: Training z_core values [N, core_dim]
            z_details: Training z_detail values [N, detail_dim]
            mode: Retrieval strategy
            k: Number of neighbors for k_nearest mode
            device: Device for computations

        Raises:
            ValueError: If inputs have invalid shapes or mismatched dimensions
        """
        # Validate inputs
        if z_cores.dim() != 2:
            raise ValueError(f"z_cores must be 2D [N, core_dim], got shape {z_cores.shape}")
        if z_details.dim() != 2:
            raise ValueError(f"z_details must be 2D [N, detail_dim], got shape {z_details.shape}")
        if z_cores.shape[0] != z_details.shape[0]:
            raise ValueError(
                f"z_cores and z_details must have same batch size: "
                f"z_cores has {z_cores.shape[0]}, z_details has {z_details.shape[0]}"
            )
        if z_cores.shape[0] == 0:
            raise ValueError("Cannot create SimpleDetailRetriever with empty training set")
        if mode == "k_nearest" and k > z_cores.shape[0]:
            raise ValueError(
                f"k={k} exceeds training set size {z_cores.shape[0]} for k_nearest mode"
            )

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
        query_dtype = z_core.dtype

        if self.mode == "zero":
            return torch.zeros(B, self.detail_dim, device=self.device, dtype=query_dtype)

        elif self.mode == "mean":
            result = self.z_detail_mean.unsqueeze(0).expand(B, -1).clone()
            return result.to(dtype=query_dtype) if result.dtype != query_dtype else result

        elif self.mode in ["nearest", "k_nearest"]:
            # Normalize query - use float32 for similarity computation then cast back
            z_core_f32 = z_core.to(device=self.device, dtype=torch.float32)
            z_core_norm = F.normalize(z_core_f32, dim=-1)

            # Compute cosine similarity in float32: [B, N]
            z_cores_f32 = self.z_cores.to(dtype=torch.float32) if self.z_cores.dtype != torch.float32 else self.z_cores
            sim = torch.mm(z_core_norm, z_cores_f32.t())

            if self.mode == "nearest":
                # Get single nearest neighbor
                indices = sim.argmax(dim=-1)  # [B]
                result = self.z_details[indices]
            else:
                # Get k nearest neighbors and average
                _, indices = sim.topk(self.k, dim=-1)  # [B, k]
                z_details_k = self.z_details[indices]  # [B, k, detail_dim]
                result = z_details_k.mean(dim=1)

            return result.to(dtype=query_dtype) if result.dtype != query_dtype else result

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_full_latent(self, z_core: torch.Tensor) -> torch.Tensor:
        """Get full latent z_full = [z_core, z_detail]."""
        z_detail = self.get_detail(z_core)
        # Ensure dtype matches z_core to avoid mixed precision issues
        if z_detail.dtype != z_core.dtype:
            z_detail = z_detail.to(dtype=z_core.dtype)
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
    import logging
    logger = logging.getLogger(__name__)

    N = z_cores.shape[0]

    # Use FAISS for large datasets if available
    if use_faiss and FAISS_AVAILABLE and N > 50000:
        try:
            return DetailRetriever(z_cores, z_details, mode, k, device)
        except (RuntimeError, MemoryError) as e:
            # Only catch FAISS-specific runtime errors, not programming bugs
            logger.warning(
                f"FAISS index construction failed: {e}. "
                f"Falling back to SimpleDetailRetriever (brute-force). "
                f"This may be slower for {N} samples."
            )
        # Let other exceptions (ValueError, TypeError, etc.) propagate

    return SimpleDetailRetriever(z_cores, z_details, mode, k, device)
