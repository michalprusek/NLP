"""Reflow pair generator for rectified flow training.

Generates synthetic (x0, x1) pairs by running ODE integration on a
trained teacher flow model. These pairs form the training data for
the next rectification iteration.

The reflow procedure:
1. Sample x0 ~ N(0, I)
2. Run ODE: x_t = x0 + integral(v(x_t, t), t=0..1) using Euler
3. Store pair (x0, x1_generated) for training

After training on these pairs, the student flow learns straighter paths
since x0 and x1 are now deterministically coupled via the teacher ODE.
"""

import logging
from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ReflowPairGenerator:
    """Generate synthetic pairs from a trained flow model via ODE integration.

    Produces (x0, x1) pairs where:
    - x0: Sampled from N(0, I)
    - x1: ODE endpoint starting from x0, using teacher model

    These pairs couple noise to data deterministically, enabling straighter
    flow trajectories in the retrained student model.

    Attributes:
        teacher_model: Frozen velocity network for ODE integration.
        n_steps: Number of Euler integration steps.
        dim: Embedding dimension (default 1024 for SONAR).

    Example:
        >>> teacher = load_model(checkpoint_path)
        >>> generator = ReflowPairGenerator(teacher, n_steps=100)
        >>> x0, x1 = generator.generate_pairs(n_pairs=10000, device='cuda')
    """

    def __init__(
        self,
        teacher_model: torch.nn.Module,
        n_steps: int = 100,
        dim: int = 1024,
    ):
        """Initialize pair generator.

        Args:
            teacher_model: Trained velocity network (will be set to eval mode).
            n_steps: Number of Euler integration steps (default 100).
            dim: Embedding dimension (default 1024 for SONAR).
        """
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # Frozen for inference
        self.n_steps = n_steps
        self.dim = dim

    @torch.no_grad()
    def generate_pairs(
        self,
        n_pairs: int,
        device: str | torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Generate synthetic (x0, x1) pairs via ODE integration.

        Samples x0 from N(0, I) and integrates teacher ODE to get x1.
        Uses Euler method matching the evaluate.py implementation.

        Args:
            n_pairs: Number of pairs to generate.
            device: Device for computation.

        Returns:
            Tuple of (x0, x1) tensors:
            - x0: Starting noise [n_pairs, dim]
            - x1: ODE endpoints [n_pairs, dim]
        """
        device = torch.device(device) if isinstance(device, str) else device

        # Sample noise starting points
        x0 = torch.randn(n_pairs, self.dim, device=device)

        # Euler ODE integration
        dt = 1.0 / self.n_steps
        x = x0.clone()

        for i in tqdm(range(self.n_steps), desc="ODE integration", leave=False):
            t = i / self.n_steps
            # Expand t to batch dimension
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
            # Velocity prediction
            v = self.teacher_model(x, t_batch)
            # Euler step
            x = x + dt * v

        x1 = x

        logger.info(
            f"Generated {n_pairs} reflow pairs: "
            f"x0 mean={x0.mean():.4f}, x1 mean={x1.mean():.4f}"
        )

        return x0, x1

    @torch.no_grad()
    def generate_dataset(
        self,
        n_total: int,
        batch_size: int,
        device: str | torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Generate pairs in batches for memory efficiency.

        Accumulates pairs on CPU to avoid GPU OOM for large datasets.

        Args:
            n_total: Total number of pairs to generate.
            batch_size: Pairs per batch (affects GPU memory).
            device: Device for computation (GPU recommended).

        Returns:
            Tuple of (x0, x1) tensors on CPU:
            - x0: All starting noise [n_total, dim]
            - x1: All ODE endpoints [n_total, dim]
        """
        device = torch.device(device) if isinstance(device, str) else device

        x0_all = []
        x1_all = []

        n_batches = (n_total + batch_size - 1) // batch_size
        remaining = n_total

        logger.info(f"Generating {n_total} pairs in {n_batches} batches of {batch_size}")

        for batch_idx in tqdm(range(n_batches), desc="Generating batches"):
            current_batch = min(batch_size, remaining)

            x0_batch, x1_batch = self.generate_pairs(current_batch, device)

            # Move to CPU for accumulation
            x0_all.append(x0_batch.cpu())
            x1_all.append(x1_batch.cpu())

            remaining -= current_batch

        x0 = torch.cat(x0_all, dim=0)
        x1 = torch.cat(x1_all, dim=0)

        logger.info(
            f"Generated dataset: x0 shape={x0.shape}, x1 shape={x1.shape}, "
            f"x0 std={x0.std():.4f}, x1 std={x1.std():.4f}"
        )

        return x0, x1
