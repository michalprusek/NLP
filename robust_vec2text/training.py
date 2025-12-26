"""VAE Training Pipeline.

Trains InstructionVAE with cosine-priority loss and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple
import numpy as np

from robust_vec2text.vae import InstructionVAE, VAELoss


class VAETrainer:
    """Trainer for InstructionVAE with early stopping.

    Tracks cosine similarity as the primary metric since it's
    critical for Vec2Text compatibility.

    Attributes:
        vae: The VAE model to train
        loss_fn: VAELoss instance
        device: Training device (cuda/cpu)
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        device: str = "cuda",
        lambda_cosine: float = 1.0,
        lambda_mse: float = 0.1,
        lambda_kld: float = 0.001,
    ):
        """Initialize trainer.

        Args:
            input_dim: Input embedding dimension (768 for GTR)
            latent_dim: VAE latent dimension (32)
            device: Device to use
            lambda_cosine: Weight for cosine loss (priority)
            lambda_mse: Weight for MSE loss
            lambda_kld: Weight for KL divergence
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.vae = InstructionVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        self.loss_fn = VAELoss(
            lambda_cosine=lambda_cosine,
            lambda_mse=lambda_mse,
            lambda_kld=lambda_kld,
        )

        self.best_state: Optional[Dict] = None
        self.best_cosine: float = 0.0

    def train(
        self,
        embeddings: torch.Tensor,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 30,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train VAE on instruction embeddings.

        Args:
            embeddings: Tensor of shape (N, 768) - GTR embeddings
            epochs: Maximum epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            val_split: Validation split ratio
            verbose: Print progress

        Returns:
            Training history dict with loss and cosine metrics
        """
        embeddings = embeddings.to(self.device)
        n_samples = len(embeddings)

        if n_samples < 5:
            if verbose:
                print(f"Warning: Only {n_samples} samples, training may be unstable")

        # Train/val split
        indices = torch.randperm(n_samples)
        n_val = max(1, int(n_samples * val_split))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_data = embeddings[train_indices]
        val_data = embeddings[val_indices]

        if verbose:
            print(f"Training VAE: {len(train_data)} train, {len(val_data)} val samples")

        # DataLoader
        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
        )

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_cosine": [],
            "val_cosine": [],
        }

        self.best_cosine = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.vae.train()
            train_losses = []
            train_cosines = []

            for (batch,) in train_loader:
                optimizer.zero_grad()

                x_recon, mu, logvar = self.vae(batch)
                loss, components = self.loss_fn(batch, x_recon, mu, logvar)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()

                train_losses.append(components["total"])
                train_cosines.append(components["cosine_sim_mean"])

            # Validation
            val_loss, val_cosine = self._evaluate(val_data)

            # Record history
            avg_train_loss = np.mean(train_losses)
            avg_train_cosine = np.mean(train_cosines)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["train_cosine"].append(avg_train_cosine)
            history["val_cosine"].append(val_cosine)

            # Learning rate scheduling
            scheduler.step(val_cosine)

            # Early stopping based on validation cosine
            if val_cosine > self.best_cosine:
                self.best_cosine = val_cosine
                self.best_state = {k: v.cpu().clone() for k, v in self.vae.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging
            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1:3d}: "
                    f"train_loss={avg_train_loss:.4f}, "
                    f"val_cosine={val_cosine:.4f} "
                    f"(best={self.best_cosine:.4f})"
                )

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_state is not None:
            self.vae.load_state_dict(self.best_state)
            self.vae.to(self.device)

        if verbose:
            print(f"VAE training complete. Best val cosine: {self.best_cosine:.4f}")

        return history

    def _evaluate(self, data: torch.Tensor) -> Tuple[float, float]:
        """Evaluate on data.

        Args:
            data: Embeddings tensor

        Returns:
            Tuple of (loss, cosine_similarity)
        """
        self.vae.eval()

        with torch.no_grad():
            x_recon, mu, logvar = self.vae(data)
            _, components = self.loss_fn(data, x_recon, mu, logvar)

        return components["total"], components["cosine_sim_mean"]

    def get_vae(self) -> InstructionVAE:
        """Get the trained VAE model."""
        return self.vae

    def save(self, path: str):
        """Save VAE weights."""
        torch.save(self.vae.state_dict(), path)

    def load(self, path: str):
        """Load VAE weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(state_dict)
