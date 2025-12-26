"""Training pipeline for PromptAutoencoder.

Handles data preparation, training loop, and model checkpointing.
Optimized for low-data regime (~200 samples) with early stopping.
"""

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from vec2text_hbbops.autoencoder import PromptAutoencoder, AutoencoderLoss


@dataclass
class TrainingConfig:
    """Configuration for autoencoder training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 1000
    patience: int = 20
    val_split: float = 0.15
    grad_clip: float = 1.0
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class TrainingHistory:
    """Training history tracker."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


class AutoencoderTrainer:
    """Training pipeline for PromptAutoencoder.

    Features:
        - Early stopping with patience
        - Learning rate scheduling
        - Gradient clipping
        - Best model checkpointing
        - Train/validation split
    """

    def __init__(
        self,
        autoencoder: PromptAutoencoder,
        config: Optional[TrainingConfig] = None,
        device: str = "auto",
    ):
        """Initialize trainer.

        Args:
            autoencoder: PromptAutoencoder instance
            config: Training configuration
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.ae = autoencoder
        self.config = config or TrainingConfig()
        self.device = self._get_device(device)

        self.ae.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.ae.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.min_lr,
        )

        self.loss_fn = AutoencoderLoss()
        self.history = TrainingHistory()
        self.best_state: Optional[Dict] = None

    def _get_device(self, device: str) -> torch.device:
        """Determine device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        elif device in ("cuda", "mps", "cpu"):
            return torch.device(device)
        return torch.device("cpu")

    def prepare_data(
        self,
        instruction_embeddings: Dict[int, np.ndarray],
        exemplar_embeddings: Dict[int, np.ndarray],
        use_all_combinations: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data from cached embeddings.

        Creates concatenated (instruction, exemplar) embeddings.

        Args:
            instruction_embeddings: {inst_id: 768D array}
            exemplar_embeddings: {ex_id: 768D array}
            use_all_combinations: If True, use cartesian product (I x E)

        Returns:
            (train_data, val_data) tensors of shape (N, 1536)
        """
        data = []

        if use_all_combinations:
            # Cartesian product of all instruction x exemplar combinations
            for inst_id, inst_emb in instruction_embeddings.items():
                for ex_id, ex_emb in exemplar_embeddings.items():
                    concat = np.concatenate([inst_emb, ex_emb])
                    data.append(concat)
        else:
            # Only use diagonal (matching indices)
            for idx in instruction_embeddings.keys():
                if idx in exemplar_embeddings:
                    concat = np.concatenate(
                        [instruction_embeddings[idx], exemplar_embeddings[idx]]
                    )
                    data.append(concat)

        data = torch.tensor(np.array(data), dtype=torch.float32)

        # Shuffle
        perm = torch.randperm(len(data))
        data = data[perm]

        # Split
        val_size = int(len(data) * self.config.val_split)
        val_size = max(1, val_size)  # At least 1 validation sample

        val_data = data[:val_size]
        train_data = data[val_size:]

        return train_data, val_data

    def train(
        self,
        instruction_embeddings: Dict[int, np.ndarray],
        exemplar_embeddings: Dict[int, np.ndarray],
        use_all_combinations: bool = True,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Train autoencoder on prompt embeddings.

        Args:
            instruction_embeddings: {inst_id: 768D numpy array}
            exemplar_embeddings: {ex_id: 768D numpy array}
            use_all_combinations: Whether to use cartesian product
            verbose: Print progress

        Returns:
            TrainingHistory with losses
        """
        train_data, val_data = self.prepare_data(
            instruction_embeddings,
            exemplar_embeddings,
            use_all_combinations,
        )

        if verbose:
            print(f"Training autoencoder:")
            print(f"  Train samples: {len(train_data)}")
            print(f"  Val samples: {len(val_data)}")
            print(f"  Max epochs: {self.config.max_epochs}")
            print(f"  Patience: {self.config.patience}")
            print(f"  LR: {self.config.learning_rate}")

        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_data_dev = val_data.to(self.device)
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            # Training phase
            self.ae.train()
            train_losses = []

            for (batch,) in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                x_recon, z = self.ae(batch)
                loss, _ = self.loss_fn(batch, x_recon, z)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.ae.parameters(), self.config.grad_clip
                )

                self.optimizer.step()
                train_losses.append(loss.item())

            # Validation phase
            self.ae.eval()
            with torch.no_grad():
                x_recon, z = self.ae(val_data_dev)
                val_loss, val_components = self.loss_fn(val_data_dev, x_recon, z)

            train_loss = np.mean(train_losses)
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss.item())

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping check
            if val_loss.item() < self.history.best_val_loss:
                self.history.best_val_loss = val_loss.item()
                self.history.best_epoch = epoch
                patience_counter = 0
                # Save best weights
                self.best_state = copy.deepcopy(self.ae.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            # Progress logging
            if verbose and (epoch + 1) % 50 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}: "
                    f"train={train_loss:.6f}, "
                    f"val={val_loss:.6f}, "
                    f"lr={current_lr:.2e}"
                )

        # Restore best weights
        if self.best_state is not None:
            self.ae.load_state_dict(self.best_state)

        if verbose:
            print(f"Training complete:")
            print(f"  Best epoch: {self.history.best_epoch + 1}")
            print(f"  Best val loss: {self.history.best_val_loss:.6f}")

        return self.history

    def evaluate_reconstruction(
        self,
        instruction_embeddings: Dict[int, np.ndarray],
        exemplar_embeddings: Dict[int, np.ndarray],
        n_samples: int = 10,
    ) -> Dict[str, float]:
        """Evaluate reconstruction quality.

        Args:
            instruction_embeddings: Instruction embeddings
            exemplar_embeddings: Exemplar embeddings
            n_samples: Number of samples to evaluate

        Returns:
            Dictionary with reconstruction metrics
        """
        self.ae.eval()

        # Prepare some test samples
        test_data = []
        for i, (inst_id, inst_emb) in enumerate(instruction_embeddings.items()):
            if i >= n_samples:
                break
            for j, (ex_id, ex_emb) in enumerate(exemplar_embeddings.items()):
                if j >= 1:  # Just one exemplar per instruction for testing
                    break
                concat = np.concatenate([inst_emb, ex_emb])
                test_data.append(concat)

        test_tensor = torch.tensor(
            np.array(test_data), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            x_recon, z = self.ae(test_tensor)

            # MSE
            mse = torch.nn.functional.mse_loss(x_recon, test_tensor).item()

            # Cosine similarity (per sample, then average)
            cos_sims = []
            for i in range(len(test_tensor)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    test_tensor[i : i + 1], x_recon[i : i + 1]
                )
                cos_sims.append(cos_sim.item())

            # Split cosine similarity
            inst_orig, ex_orig = test_tensor[:, :768], test_tensor[:, 768:]
            inst_recon, ex_recon = x_recon[:, :768], x_recon[:, 768:]

            inst_cos = torch.nn.functional.cosine_similarity(
                inst_orig, inst_recon
            ).mean()
            ex_cos = torch.nn.functional.cosine_similarity(ex_orig, ex_recon).mean()

        return {
            "mse": mse,
            "cosine_similarity": np.mean(cos_sims),
            "instruction_cosine": inst_cos.item(),
            "exemplar_cosine": ex_cos.item(),
            "latent_sparsity": z.abs().mean().item(),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save(
            {
                "model_state_dict": self.ae.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.ae.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
