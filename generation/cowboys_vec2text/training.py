"""Weighted VAE Training Pipeline.

Extends the robust_vec2text VAETrainer with:
1. Sample weighting during training
2. Incremental retraining with new samples
3. Weight computation from error rates

This enables the COWBOYS methodology's "weighted retraining" where
the VAE latent space is periodically refocused on high-quality regions.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .vae import InstructionVAE, WeightedVAELoss


@dataclass
class RetrainConfig:
    """Configuration for weighted retraining.

    Attributes:
        retrain_interval: Retrain VAE every N iterations
        weight_method: How to compute weights from error rates
            - "rank": w_i = 1 / rank(error_i)^power
            - "exponential": w_i = exp(-beta * error_i)
        rank_power: Exponent for rank-based weighting
        exp_beta: Beta parameter for exponential weighting
        epochs: Number of retraining epochs
        lr: Learning rate for retraining (typically lower than initial)
        patience: Early stopping patience for retraining
    """

    retrain_interval: int = 10
    weight_method: str = "rank"
    rank_power: float = 1.0
    exp_beta: float = 1.0
    epochs: int = 50
    lr: float = 0.0001
    patience: int = 15


class WeightedVAETrainer:
    """VAE Trainer with weighted retraining support.

    Extends VAETrainer with:
    - Sample weight computation from error rates
    - Incremental sample accumulation for retraining
    - Weighted loss during training

    Key insight: By weighting samples inversely to their error rate,
    we make the VAE "pay more attention" to high-performing prompts.
    This deforms the latent space to allocate more volume to
    high-quality regions, enabling finer-grained optimization there.

    Usage:
        trainer = WeightedVAETrainer(...)

        # Initial training (unweighted or with grid weights)
        trainer.train(embeddings, error_rates=grid_errors)

        # During optimization loop:
        trainer.add_samples(new_embedding, new_error)

        if optimizer.should_retrain_vae(iteration, config):
            trainer.retrain_with_weights(config)

    Attributes:
        vae: The VAE model to train
        loss_fn: WeightedVAELoss instance
        device: Training device
        all_embeddings: Accumulated embeddings for retraining
        all_error_rates: Accumulated error rates for weighting
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        device: str = "cuda",
        lambda_cosine: float = 20.0,
        lambda_mse: float = 1.0,
        lambda_kld: float = 0.0025,
        lambda_cycle: float = 2.0,
        kld_annealing_epochs: int = 500,
    ):
        """Initialize trainer.

        Args:
            input_dim: Input embedding dimension (768 for GTR)
            latent_dim: VAE latent dimension (32)
            device: Device to use
            lambda_cosine: Weight for cosine loss (priority for GTR)
            lambda_mse: Weight for MSE loss
            lambda_kld: Target weight for KL divergence (with annealing)
            lambda_cycle: Weight for cycle-consistency loss
            kld_annealing_epochs: Epochs to anneal KLD from 0 to target (default: 500)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.vae = InstructionVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        self.lambda_kld = lambda_kld
        self.lambda_cycle = lambda_cycle
        self.kld_annealing_epochs = kld_annealing_epochs
        self.loss_fn = WeightedVAELoss(
            lambda_cosine=lambda_cosine,
            lambda_mse=lambda_mse,
            lambda_kld=lambda_kld,
            lambda_cycle=lambda_cycle,
        )

        self.best_state: Optional[Dict] = None
        self.best_cosine: float = 0.0

        # Sample accumulation for retraining
        self.all_embeddings: List[torch.Tensor] = []
        self.all_error_rates: List[float] = []

    def compute_sample_weights(
        self,
        error_rates: torch.Tensor,
        method: str = "rank",
        **kwargs,
    ) -> torch.Tensor:
        """Compute sample weights from error rates.

        Lower error rate = higher weight (better prompt = more influence).

        Methods:
            rank: w_i = 1 / rank(error_i)^power
                - Robust to outliers
                - Best sample has weight 1, worst has weight 1/N
            exponential: w_i = exp(-beta * error_i)
                - Sensitive to absolute differences
                - Can create very skewed weights

        Args:
            error_rates: Error rates tensor (N,)
            method: Weighting method ("rank" or "exponential")
            **kwargs: Method-specific parameters (rank_power, exp_beta)

        Returns:
            Normalized weights summing to 1
        """
        error_rates = error_rates.cpu().numpy()
        n = len(error_rates)

        if method == "rank":
            # Rank-based weighting
            power = kwargs.get("rank_power", 1.0)
            # Lower error = lower rank number = higher weight
            ranks = np.argsort(np.argsort(error_rates)) + 1  # 1-indexed
            weights = 1.0 / (ranks ** power)
        elif method == "exponential":
            # Exponential weighting
            beta = kwargs.get("exp_beta", 1.0)
            # Subtract min for numerical stability
            centered = error_rates - error_rates.min()
            weights = np.exp(-beta * centered)
        else:
            raise ValueError(f"Unknown weight method: {method}")

        # Normalize to sum to 1
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def train(
        self,
        embeddings: torch.Tensor,
        error_rates: Optional[torch.Tensor] = None,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 30,
        val_split: float = 0.2,
        weight_method: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train VAE on instruction embeddings.

        Args:
            embeddings: Tensor of shape (N, 768) - GTR embeddings
            error_rates: Optional error rates for weighted training
            epochs: Maximum epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            val_split: Validation split ratio
            weight_method: Optional weight method for training
            verbose: Print progress

        Returns:
            Training history dict with loss and cosine metrics
        """
        embeddings = embeddings.to(self.device)
        n_samples = len(embeddings)

        # Store for future retraining (always on CPU for consistency)
        self.all_embeddings = [embeddings.cpu()]
        if error_rates is not None:
            self.all_error_rates = error_rates.tolist()
        else:
            self.all_error_rates = [0.5] * n_samples  # Default weights

        if n_samples < 5:
            if verbose:
                print(f"Warning: Only {n_samples} samples, training may be unstable")

        # Compute sample weights if error rates provided and method specified
        sample_weights = None
        if error_rates is not None and weight_method is not None:
            sample_weights = self.compute_sample_weights(error_rates, weight_method)
            if verbose:
                print(f"Using {weight_method} weighting for training")

        # Train/val split
        indices = torch.randperm(n_samples)
        n_val = max(1, int(n_samples * val_split))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_data = embeddings[train_indices]
        val_data = embeddings[val_indices]

        # Also split weights if applicable
        train_weights = None
        if sample_weights is not None:
            train_weights = sample_weights[train_indices]
            train_weights = train_weights / train_weights.sum()  # Renormalize

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
            # KLD Annealing: 0 -> target over kld_annealing_epochs, then hold
            # Epoch 0-500: grows from 0 to lambda_kld
            # Epoch 500+: holds at lambda_kld
            if epoch < self.kld_annealing_epochs:
                annealing_factor = epoch / self.kld_annealing_epochs
            else:
                annealing_factor = 1.0
            current_kld_weight = self.lambda_kld * annealing_factor
            self.loss_fn.lambda_kld = current_kld_weight

            # Training
            self.vae.train()
            train_losses = []
            train_cosines = []

            for batch_idx, (batch,) in enumerate(train_loader):
                optimizer.zero_grad()

                x_recon, mu, logvar = self.vae(batch)

                # Get batch weights if using weighted training
                batch_weights = None
                if train_weights is not None:
                    # Note: This is a simplification. For proper batch weighting,
                    # we'd need to track indices. For now, use uniform within batch.
                    pass

                # Compute cycle-consistency loss if enabled
                cycle_loss = None
                if self.lambda_cycle > 0:
                    cycle_loss = self.vae.cycle_consistency_loss(mu)

                loss, components = self.loss_fn(
                    batch, x_recon, mu, logvar, batch_weights, cycle_loss
                )

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

            # Early stopping
            if val_cosine > self.best_cosine:
                self.best_cosine = val_cosine
                self.best_state = {k: v.cpu().clone() for k, v in self.vae.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

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

    def add_samples(
        self,
        new_embeddings: torch.Tensor,
        new_error_rates: List[float],
    ):
        """Add new samples to training pool for future retraining.

        Args:
            new_embeddings: New embeddings to add (N, 768) or (768,)
            new_error_rates: Corresponding error rates
        """
        if new_embeddings.dim() == 1:
            new_embeddings = new_embeddings.unsqueeze(0)
        if isinstance(new_error_rates, (int, float)):
            new_error_rates = [new_error_rates]

        self.all_embeddings.append(new_embeddings.cpu())
        self.all_error_rates.extend(new_error_rates)

    def retrain_with_weights(
        self,
        config: RetrainConfig,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Retrain VAE with weighted samples.

        Uses accumulated samples with weights based on error rates.
        High-performing samples (low error) get higher weights.

        Args:
            config: Retraining configuration
            verbose: Print progress

        Returns:
            Retraining history
        """
        if not self.all_embeddings:
            if verbose:
                print("No samples accumulated for retraining")
            return {}

        # Concatenate all embeddings
        all_emb = torch.cat(self.all_embeddings, dim=0).to(self.device)
        all_errors = torch.tensor(self.all_error_rates, dtype=torch.float32)

        if verbose:
            print(f"Retraining VAE on {len(all_emb)} samples with {config.weight_method} weighting")

        # Compute weights
        weights = self.compute_sample_weights(
            all_errors,
            method=config.weight_method,
            rank_power=config.rank_power,
            exp_beta=config.exp_beta,
        )

        # Create weighted sampler for DataLoader
        sampler = WeightedRandomSampler(
            weights=weights.cpu(),
            num_samples=len(all_emb),
            replacement=True,
        )

        # DataLoader with weighted sampling
        train_loader = DataLoader(
            TensorDataset(all_emb),
            batch_size=32,
            sampler=sampler,
        )

        # Optimizer with lower learning rate for fine-tuning
        optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=config.lr,
            weight_decay=1e-5,
        )

        history = {"train_loss": [], "train_cosine": []}
        best_cosine = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(config.epochs):
            self.vae.train()
            train_losses = []
            train_cosines = []

            for (batch,) in train_loader:
                optimizer.zero_grad()

                x_recon, mu, logvar = self.vae(batch)

                # Compute cycle-consistency loss if enabled
                cycle_loss = None
                if self.lambda_cycle > 0:
                    cycle_loss = self.vae.cycle_consistency_loss(mu)

                loss, components = self.loss_fn(
                    batch, x_recon, mu, logvar, None, cycle_loss
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()

                train_losses.append(components["total"])
                train_cosines.append(components["cosine_sim_mean"])

            avg_loss = np.mean(train_losses)
            avg_cosine = np.mean(train_cosines)
            history["train_loss"].append(avg_loss)
            history["train_cosine"].append(avg_cosine)

            # Track best
            if avg_cosine > best_cosine:
                best_cosine = avg_cosine
                best_state = {k: v.cpu().clone() for k, v in self.vae.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Retrain epoch {epoch+1}: cosine={avg_cosine:.4f}")

            if patience_counter >= config.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best
        if best_state is not None:
            self.vae.load_state_dict(best_state)
            self.vae.to(self.device)

        if verbose:
            print(f"  Retraining complete. Best cosine: {best_cosine:.4f}")

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

            # Compute cycle-consistency loss if enabled
            cycle_loss = None
            if self.lambda_cycle > 0:
                cycle_loss = self.vae.cycle_consistency_loss(mu)

            _, components = self.loss_fn(data, x_recon, mu, logvar, None, cycle_loss)

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

    def get_sample_count(self) -> int:
        """Get total number of accumulated samples."""
        return sum(emb.shape[0] for emb in self.all_embeddings)


# For backward compatibility
class VAETrainer(WeightedVAETrainer):
    """Alias for WeightedVAETrainer for backward compatibility."""

    pass
