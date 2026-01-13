"""
Value Head for Cheap Reward Prediction.

The Value Head is an auxiliary network that predicts the expected
error rate of a latent vector without expensive LLM evaluation.

It's used in cost-aware active learning:
    - When FCU is low (model confident): use Value Head prediction
    - When FCU is high (model uncertain): evaluate with LLM

The Value Head is trained on:
    1. Initial Hyperband evaluations
    2. Online updates from BO iterations

This enables significant cost savings by avoiding redundant
LLM evaluations in confident regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


class ValueHead(nn.Module):
    """
    Value Head network for error rate prediction.

    Architecture:
        z (latent_dim) → Linear → GELU → LayerNorm
        → Linear → GELU → LayerNorm
        → Linear → Sigmoid → error_rate ∈ [0, 1]

    The network is small and fast, designed to provide
    cheap predictions when the model is confident.

    Args:
        latent_dim: Input dimension (VAE latent)
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict error rate from latent vector.

        Args:
            z: Latent vectors (B, latent_dim)

        Returns:
            error_rate: Predicted error rates (B,) in [0, 1]
        """
        return self.network(z).squeeze(-1)


@dataclass
class ValueHeadPrediction:
    """Result of Value Head prediction."""

    # Predicted error rates
    error_rate: torch.Tensor  # (B,)

    # Confidence (based on distance to training data)
    confidence: Optional[torch.Tensor] = None  # (B,)


class ValueHeadWithUncertainty(nn.Module):
    """
    Value Head with uncertainty estimation via MC Dropout.

    In addition to the mean prediction, this version provides
    uncertainty estimates by running multiple forward passes
    with different dropout masks.

    This can be used to:
    1. Weight predictions by confidence
    2. Identify samples that need more training data
    3. Combine with FCU for robust uncertainty
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,  # Higher dropout for MC
        num_mc_samples: int = 10,
    ):
        super().__init__()

        self.num_mc_samples = num_mc_samples

        # Network with dropout (kept on during inference for MC)
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict with optional MC dropout uncertainty.

        Args:
            z: Latent vectors (B, latent_dim)
            return_uncertainty: Whether to compute MC uncertainty

        Returns:
            mean_pred: Mean prediction (B,)
            uncertainty: Optional std from MC samples (B,)
        """
        if not return_uncertainty:
            return self.network(z).squeeze(-1), None

        # MC Dropout: multiple forward passes with dropout
        self.train()  # Enable dropout
        predictions = []

        for _ in range(self.num_mc_samples):
            pred = self.network(z).squeeze(-1)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (num_mc, B)

        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean_pred, uncertainty


class ValueHeadTrainer:
    """
    Trainer for Value Head with replay buffer.

    Handles:
    1. Storing observations (z, error_rate) in buffer
    2. Training on buffered data
    3. Online updates during BO
    """

    def __init__(
        self,
        value_head: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        buffer_size: int = 10000,
    ):
        self.value_head = value_head
        self.buffer_size = buffer_size

        # Replay buffer
        self.buffer_z: List[torch.Tensor] = []
        self.buffer_y: List[float] = []

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            value_head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def add_observation(
        self,
        z: torch.Tensor,
        error_rate: float,
    ):
        """Add observation to buffer."""
        self.buffer_z.append(z.detach().cpu())
        self.buffer_y.append(error_rate)

        # Trim buffer if too large
        if len(self.buffer_z) > self.buffer_size:
            self.buffer_z = self.buffer_z[-self.buffer_size:]
            self.buffer_y = self.buffer_y[-self.buffer_size:]

    def add_batch(
        self,
        z_batch: torch.Tensor,
        error_rates: torch.Tensor,
    ):
        """Add batch of observations."""
        for i in range(z_batch.shape[0]):
            self.add_observation(z_batch[i], error_rates[i].item())

    def train_step(
        self,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Train on random batch from buffer.

        Args:
            batch_size: Training batch size
            device: Device for training

        Returns:
            metrics: Dict with loss and other metrics
        """
        if len(self.buffer_z) < batch_size:
            return {"loss": 0.0, "buffer_size": len(self.buffer_z)}

        # Sample random batch
        indices = torch.randint(0, len(self.buffer_z), (batch_size,))
        z_batch = torch.stack([self.buffer_z[i] for i in indices]).to(device)
        y_batch = torch.tensor([self.buffer_y[i] for i in indices]).to(device)

        # Forward
        self.value_head.train()
        pred = self.value_head(z_batch)

        # MSE loss
        loss = F.mse_loss(pred, y_batch)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "buffer_size": len(self.buffer_z),
            "pred_mean": pred.mean().item(),
            "target_mean": y_batch.mean().item(),
        }

    def train_epoch(
        self,
        steps_per_epoch: int = 100,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            metrics = self.train_step(batch_size, device)
            total_loss += metrics["loss"]

        return {
            "avg_loss": total_loss / steps_per_epoch,
            "buffer_size": len(self.buffer_z),
        }

    def get_buffer_statistics(self) -> Dict[str, float]:
        """Get statistics of buffer data."""
        if len(self.buffer_y) == 0:
            return {}

        y = torch.tensor(self.buffer_y)
        return {
            "buffer_size": len(self.buffer_y),
            "error_rate_mean": y.mean().item(),
            "error_rate_std": y.std().item(),
            "error_rate_min": y.min().item(),
            "error_rate_max": y.max().item(),
        }


def create_value_head_from_hyperband(
    hyperband_results: Dict,
    vae_encoder: nn.Module,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    device: str = "cuda",
    train_epochs: int = 100,
) -> Tuple[ValueHead, ValueHeadTrainer]:
    """
    Create and train Value Head from Hyperband results.

    Args:
        hyperband_results: Dict with "results" containing
            instruction embeddings and error rates
        vae_encoder: VAE encoder to convert embeddings to latents
        latent_dim: Latent dimension
        hidden_dim: Value head hidden dimension
        device: Training device
        train_epochs: Number of training epochs

    Returns:
        value_head: Trained Value Head
        trainer: Trainer with populated buffer
    """
    # Create Value Head
    value_head = ValueHead(latent_dim, hidden_dim).to(device)
    trainer = ValueHeadTrainer(value_head, lr=1e-3)

    # Extract data from Hyperband results
    for result in hyperband_results.get("results", {}).values():
        if "embedding" in result and "error_rate" in result:
            embedding = torch.tensor(result["embedding"], device=device)

            # Encode to latent
            with torch.no_grad():
                z = vae_encoder.encode_mu(embedding.unsqueeze(0)).squeeze(0)

            trainer.add_observation(z, result["error_rate"])

    print(f"Loaded {len(trainer.buffer_z)} samples from Hyperband")

    # Train
    for epoch in range(train_epochs):
        metrics = trainer.train_epoch(steps_per_epoch=50, batch_size=64, device=device)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}: loss={metrics['avg_loss']:.6f}")

    return value_head, trainer


if __name__ == "__main__":
    print("Testing Value Head...")

    # Create Value Head
    value_head = ValueHead(latent_dim=32, hidden_dim=128)

    # Test forward pass
    z = torch.randn(16, 32)
    pred = value_head(z)
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Predictions: {pred}")

    # Test trainer
    print("\nTesting ValueHeadTrainer...")
    trainer = ValueHeadTrainer(value_head, lr=1e-3)

    # Add synthetic data
    for i in range(200):
        z_sample = torch.randn(32)
        # Synthetic target: error rate correlated with z norm
        error_rate = 0.3 + 0.2 * torch.tanh(z_sample.norm() / 5).item()
        trainer.add_observation(z_sample, error_rate)

    print(f"Buffer size: {len(trainer.buffer_z)}")

    # Train
    print("\nTraining...")
    for epoch in range(5):
        metrics = trainer.train_epoch(steps_per_epoch=20, batch_size=32, device="cpu")
        print(f"Epoch {epoch + 1}: loss={metrics['avg_loss']:.6f}")

    # Test Value Head with uncertainty
    print("\nTesting ValueHeadWithUncertainty...")
    value_head_mc = ValueHeadWithUncertainty(latent_dim=32, hidden_dim=128)

    z = torch.randn(8, 32)
    mean_pred, uncertainty = value_head_mc(z, return_uncertainty=True)
    print(f"Mean predictions: {mean_pred}")
    print(f"Uncertainty: {uncertainty}")

    print("\nAll tests passed!")
