"""VAE-based training for InvBO decoder.

Training pipeline:
1. Train VAE on diverse instructions with KL annealing
2. Train GP using VAE.encode_mu for latent representation
3. Use VAE.decode as decoder (no separate training needed)

The VAE provides a smooth, continuous latent space for BO optimization.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from generation.invbo_decoder.encoder import (
    GTRInstructionEncoder,
    InstructionVAE,
)
from generation.invbo_decoder.gp import GPWithEI


class VAEWithAdapter(nn.Module):
    """Wrapper combining frozen VAE encoder with trainable adapter.

    Used in VAE mode to allow GP's feature extractor to learn
    while keeping VAE weights fixed. The adapter is a small MLP
    that reduces VAE latents (64D) to GP latent dimension (10D).
    """

    def __init__(self, vae: nn.Module, vae_latent_dim: int, gp_latent_dim: int = 10):
        super().__init__()
        # Freeze VAE (don't register as submodule to avoid saving it twice)
        object.__setattr__(self, '_vae', vae)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        # Trainable adapter: reduces VAE latents (64D) to GP latent dimension (10D)
        self.adapter = nn.Sequential(
            nn.Linear(vae_latent_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, gp_latent_dim),
        )

    def encode_vae(self, x: torch.Tensor) -> torch.Tensor:
        """Encode embedding to 64D VAE latent (deterministic mu).

        Use this for getting latent for optimization bounds.
        """
        return self._vae.encode_mu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode embedding to 10D GP latent (through adapter).

        Pipeline: x (768D) → VAE encoder → z (64D) → adapter → z_gp (10D)

        This is used for GP training.
        """
        z = self._vae.encode_mu(x)
        return self.adapter(z)

    def adapt(self, z: torch.Tensor) -> torch.Tensor:
        """Apply adapter to VAE latent.

        Args:
            z: VAE latent tensor of shape (..., 64)

        Returns:
            GP latent tensor of shape (..., 10)
        """
        return self.adapter(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode 64D VAE latent to 768D embedding.

        Args:
            z: VAE latent tensor of shape (..., 64)

        Returns:
            Embedding tensor of shape (..., 768)
        """
        return self._vae.decode(z)


@dataclass
class TrainingConfig:
    """Configuration for InvBO training (VAE mode only)."""

    # Data paths
    instructions_path: str = "datasets/inversion/instructions_100.txt"
    grid_path: str = "datasets/inversion/grid_100_qend.jsonl"
    diverse_instructions_path: str = "datasets/inversion/diverse_instructions_1000.json"

    # Architecture
    latent_dim: int = 10  # GP latent dimension (after adapter reduction)
    vae_latent_dim: int = 64  # VAE latent dimension (before adapter reduction)
    embedding_dim: int = 768

    # GP training
    gp_epochs: int = 5000
    gp_lr: float = 0.01
    gp_patience: int = 50
    gp_initial_top_k: int = 0  # 0 = train on all prompts, >0 = train on top-k only

    # VAE training
    vae_beta: float = 0.1  # KL regularization weight
    vae_epochs: int = 10000
    vae_lr: float = 0.0003
    vae_annealing_epochs: int = 500  # Epochs for KL annealing (0 → beta)
    vae_patience: int = 500
    vae_batch_size: int = 64

    # Device
    device: str = "cuda"


class InvBOTrainer:
    """VAE-based trainer for InvBO decoder inversion.

    Training pipeline:
    1. Train VAE on diverse instructions with KL annealing
    2. Train GP using VAE.encode_mu for latent representation
    3. Use VAE.decode as decoder (no separate training needed)
    """

    def __init__(self, config: TrainingConfig):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Initialize GTR encoder (pre-trained, frozen)
        print("Loading GTR encoder...")
        self.gtr = GTRInstructionEncoder(device=str(self.device))

        # Components (initialized during training)
        self.gp: Optional[GPWithEI] = None
        self.decoder: Optional[nn.Module] = None  # VAEDecoderWrapper
        self.vae: Optional[InstructionVAE] = None

        # Data storage (100 instructions with error rates for GP)
        self.instructions: List[str] = []
        self.instruction_embeddings: Dict[int, torch.Tensor] = {}
        self.error_rates: Dict[int, float] = {}

        # Diverse instructions (1000 for VAE training)
        self.diverse_instructions: List[str] = []
        self.diverse_embeddings: Optional[torch.Tensor] = None  # (1000, 768)

    def load_data(self, verbose: bool = True) -> None:
        """Load instructions and grid data.

        Loads instructions directly from grid file (which contains instruction_text),
        ensuring instructions and error rates are always in sync.

        Args:
            verbose: Print progress
        """
        if verbose:
            print(f"Loading instructions and error rates from {self.config.grid_path}...")

        # Load instructions and error rates directly from grid
        grid_entries = []
        with open(self.config.grid_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                grid_entries.append(entry)

        # Sort by instruction_id for consistent ordering
        grid_entries.sort(key=lambda x: x["instruction_id"])

        # Re-index from 0 to ensure contiguous IDs
        for new_idx, entry in enumerate(grid_entries):
            inst_text = entry["instruction_text"]
            self.instructions.append(inst_text)
            self.error_rates[new_idx] = entry["error_rate"]

        if verbose:
            print(f"  Loaded {len(self.instructions)} instructions with error rates")
            rates = list(self.error_rates.values())
            print(f"  Error rate range: [{min(rates):.4f}, {max(rates):.4f}]")

        # Encode instructions with GTR
        if verbose:
            print("Encoding instructions with GTR...")
        for idx, inst in enumerate(self.instructions):
            self.instruction_embeddings[idx] = self.gtr.encode_tensor(inst)

    def _validate_gp(self, verbose: bool = True) -> dict:
        """Validate GP predictions on training data.

        Returns:
            Dictionary with MAE, RMSE, and max error metrics
        """
        if verbose:
            print("\nValidating GP predictions on training data...")

        # Collect all embeddings and true error rates
        X = torch.stack([
            self.instruction_embeddings[i] for i in sorted(self.error_rates.keys())
        ])
        y = torch.tensor([
            self.error_rates[i] for i in sorted(self.error_rates.keys())
        ], dtype=torch.float32)

        # Use the new validate_predictions method
        metrics = self.gp.validate_predictions(X, y)

        if verbose:
            print(f"  Mean Absolute Error: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Max Error: {metrics['max_error']:.4f}")

        return metrics

    def train_vae(self, verbose: bool = True) -> bool:
        """Train VAE on diverse instructions with KL annealing.

        The VAE provides:
        - Smooth latent space via KL regularization to N(0,1)
        - encode_mu() for GP latent (deterministic)
        - decode() for Vec2Text inversion

        KL annealing starts with beta=0 (autoencoder) and slowly
        increases to target beta to prevent posterior collapse.

        Args:
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Training VAE with KL Annealing")
            print("=" * 60)

        # Load diverse instructions for VAE training (if path provided)
        embeddings_list = []

        if self.config.diverse_instructions_path:
            if verbose:
                print(f"Loading diverse instructions from {self.config.diverse_instructions_path}...")

            with open(self.config.diverse_instructions_path, "r") as f:
                data = json.load(f)
                self.diverse_instructions = data["instructions"]

            if verbose:
                print(f"  Using {len(self.diverse_instructions)} instructions for VAE")

            # Encode with GTR
            if verbose:
                print("  Encoding with GTR...")
            for inst in self.diverse_instructions:
                emb = self.gtr.encode_tensor(inst)
                embeddings_list.append(emb)
        else:
            self.diverse_instructions = []
            if verbose:
                print("Using only grid_100 instructions for VAE training")

        # Add the grid instructions
        grid_embeddings = []
        for inst_id in sorted(self.instruction_embeddings.keys()):
            grid_embeddings.append(self.instruction_embeddings[inst_id])

        if grid_embeddings:
            embeddings_list.extend(grid_embeddings)

        self.diverse_embeddings = torch.stack(embeddings_list).to(self.device)

        if verbose:
            print(f"  Total training embeddings: {self.diverse_embeddings.shape}")

        # Initialize VAE with vae_latent_dim (64D by default)
        self.vae = InstructionVAE(
            input_dim=self.config.embedding_dim,
            latent_dim=self.config.vae_latent_dim,
            beta=self.config.vae_beta,
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.config.vae_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.vae_epochs * 2, eta_min=1e-4  # Slower decay for stable reconstruction
        )

        X = self.diverse_embeddings
        n_samples = X.shape[0]
        best_recon = float("inf")  # Track reconstruction loss, not total (avoids early stop during KL annealing)
        patience_counter = 0

        if verbose:
            print(f"Training VAE (beta annealing: 0 → {self.config.vae_beta} over {self.config.vae_annealing_epochs} epochs)...")

        self.vae.train()

        for epoch in range(self.config.vae_epochs):
            # KL annealing: linearly increase beta from 0 to target
            if epoch < self.config.vae_annealing_epochs:
                current_beta = self.config.vae_beta * (epoch / self.config.vae_annealing_epochs)
            else:
                current_beta = self.config.vae_beta

            # Shuffle data
            perm = torch.randperm(n_samples, device=self.device)
            X_shuffled = X[perm]

            epoch_losses = []
            epoch_recon = []
            epoch_kl = []
            epoch_cosine = []

            # Mini-batch training
            for i in range(0, n_samples, self.config.vae_batch_size):
                batch_x = X_shuffled[i:i + self.config.vae_batch_size]

                optimizer.zero_grad()

                # Forward pass
                x_recon, mu, log_var, z = self.vae(batch_x)

                # Compute loss with current beta
                loss, loss_dict = self.vae.loss(batch_x, x_recon, mu, log_var, beta=current_beta)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(loss_dict["total"])
                epoch_recon.append(loss_dict["recon"])
                epoch_kl.append(loss_dict["kl"])
                epoch_cosine.append(loss_dict["cosine_mean"])

            scheduler.step()

            # Epoch stats
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_recon = sum(epoch_recon) / len(epoch_recon)
            avg_kl = sum(epoch_kl) / len(epoch_kl)
            avg_cosine = sum(epoch_cosine) / len(epoch_cosine)

            if avg_recon < best_recon:
                best_recon = avg_recon
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.vae_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, "
                    f"kl={avg_kl:.4f}, cosine={avg_cosine:.4f}, beta={current_beta:.4f}"
                )

        if verbose:
            print(f"  VAE training complete (epochs={epoch + 1})")
            print(f"  Final cosine similarity: {avg_cosine:.4f}")
            self._validate_vae(verbose=True)

        return True

    def _validate_vae(self, verbose: bool = True) -> None:
        """Validate VAE reconstruction quality on grid instructions."""
        if verbose:
            print("\nValidating VAE on grid_100 instructions...")

        self.vae.eval()
        cosine_sims = []
        kl_values = []

        with torch.no_grad():
            for inst_id in sorted(self.error_rates.keys()):
                emb_original = self.instruction_embeddings[inst_id].unsqueeze(0)

                # Encode and decode
                x_recon, mu, log_var, z = self.vae(emb_original)

                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    x_recon, emb_original
                ).item()
                cosine_sims.append(cos_sim)

                # KL divergence
                kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum().item()
                kl_values.append(kl)

        if verbose:
            print(f"  Samples: {len(cosine_sims)}")
            print(f"  Cosine similarity: {sum(cosine_sims)/len(cosine_sims):.4f} "
                  f"[{min(cosine_sims):.4f}, {max(cosine_sims):.4f}]")
            print(f"  KL divergence: {sum(kl_values)/len(kl_values):.4f}")

    def evaluate_vae_quality(self, verbose: bool = True) -> Dict[str, float]:
        """Compute comprehensive VAE quality metrics.

        Similar to COWBOYS VAE quality evaluation. Computes:
        - Reconstruction quality (cosine similarity, MSE)
        - Latent space statistics (norm, variance per dimension)
        - KL divergence statistics
        - Posterior collapse detection

        NOTE: Evaluates on grid_100 instructions only (not diverse) because:
        1. These are the task-relevant instructions with error rates
        2. Diverse instructions are only for VAE regularization, not optimization
        3. Avoids confusing metrics from out-of-domain instructions

        Args:
            verbose: Print metrics

        Returns:
            Dictionary with quality metrics
        """
        if self.vae is None:
            raise RuntimeError("VAE not trained. Call train_vae() first.")

        self.vae.eval()

        # Evaluate on grid_100 only (the task-relevant instructions)
        embeddings = torch.stack([
            self.instruction_embeddings[i] for i in sorted(self.error_rates.keys())
        ]).to(self.device)

        # Note: NOT using diverse_embeddings because:
        # - Diverse instructions have no error rates (not used in optimization)
        # - They're only used to regularize VAE latent space
        # - Including them gives misleadingly low reconstruction quality
        if verbose and self.diverse_embeddings is not None:
            print(f"  Note: Evaluating on grid_100 ({embeddings.shape[0]} samples), "
                  f"not diverse ({self.diverse_embeddings.shape[0]} samples)")

        with torch.no_grad():
            x_recon, mu, log_var, z = self.vae(embeddings)

        # Reconstruction metrics
        cosine_sims = torch.nn.functional.cosine_similarity(embeddings, x_recon, dim=-1)
        mse_values = ((embeddings - x_recon) ** 2).mean(dim=-1)
        l2_relative = torch.norm(embeddings - x_recon, dim=-1) / torch.norm(embeddings, dim=-1)

        # Latent space statistics
        latent_norms = z.norm(dim=-1)
        latent_var_per_dim = z.var(dim=0)

        # KL divergence
        kld_per_sample = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1)

        # Posterior collapse detection (dimensions with very low variance)
        active_dims = (latent_var_per_dim > 0.01).sum().item()
        total_dims = z.shape[-1]

        metrics = {
            "cosine_mean": cosine_sims.mean().item(),
            "cosine_std": cosine_sims.std().item(),
            "cosine_min": cosine_sims.min().item(),
            "cosine_max": cosine_sims.max().item(),
            "mse_mean": mse_values.mean().item(),
            "mse_std": mse_values.std().item(),
            "l2_relative_error": l2_relative.mean().item(),
            "latent_norm_mean": latent_norms.mean().item(),
            "latent_norm_std": latent_norms.std().item(),
            "latent_var_mean": latent_var_per_dim.mean().item(),
            "latent_var_min": latent_var_per_dim.min().item(),
            "latent_var_max": latent_var_per_dim.max().item(),
            "active_dims": int(active_dims),
            "total_dims": int(total_dims),
            "kld_mean": kld_per_sample.mean().item(),
            "kld_std": kld_per_sample.std().item(),
            "posterior_collapsed": active_dims < total_dims * 0.5,
        }

        if verbose:
            print("\n--- VAE Quality Metrics ---")
            print("Reconstruction (Cosine Similarity):")
            print(f"  Mean: {metrics['cosine_mean']:.4f} | Std: {metrics['cosine_std']:.4f} | "
                  f"Min: {metrics['cosine_min']:.4f} | Max: {metrics['cosine_max']:.4f}")
            print("Reconstruction (MSE):")
            print(f"  Mean: {metrics['mse_mean']:.6f} | Std: {metrics['mse_std']:.6f}")
            print(f"  Relative L2 Error: {metrics['l2_relative_error']:.4f}")
            print("Latent Space:")
            print(f"  Norm: mean={metrics['latent_norm_mean']:.4f}, std={metrics['latent_norm_std']:.4f}")
            print(f"  Variance per dim: mean={metrics['latent_var_mean']:.4f}, "
                  f"min={metrics['latent_var_min']:.4f}, max={metrics['latent_var_max']:.4f}")
            print(f"  Active dimensions (var>0.01): {metrics['active_dims']}/{metrics['total_dims']}")
            print("KL Divergence:")
            print(f"  Mean: {metrics['kld_mean']:.4f} | Std: {metrics['kld_std']:.4f}")
            if metrics['posterior_collapsed']:
                print("  WARNING: Posterior may be collapsed!")
            print("----------------------------")

        return metrics

    def train_gp_with_vae(self, verbose: bool = True) -> bool:
        """Train GP using VAE encoder for latent representation.

        Uses VAE.encode_mu() (deterministic) for GP training.
        Trains on all prompts by default (gp_initial_top_k=0).

        Args:
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if self.vae is None:
            raise RuntimeError("VAE not trained. Call train_vae() first.")

        if verbose:
            print("\n" + "=" * 60)
            print("Training GP with VAE Encoder")
            print("=" * 60)

        # Use all prompts or top-k for initial training
        sorted_ids = sorted(self.error_rates.keys(), key=lambda x: self.error_rates[x])
        top_k = self.config.gp_initial_top_k
        if top_k > 0 and top_k < len(sorted_ids):
            train_ids = sorted_ids[:top_k]
            if verbose:
                print(f"  Using top-{top_k} prompts (lowest error rate)")
                print(f"  Error rate range: [{self.error_rates[train_ids[0]]:.4f}, {self.error_rates[train_ids[-1]]:.4f}]")
        else:
            # Use all prompts for better GP generalization
            train_ids = sorted_ids
            if verbose:
                rates = [self.error_rates[i] for i in train_ids]
                print(f"  Using all {len(train_ids)} prompts")
                print(f"  Error rate range: [{min(rates):.4f}, {max(rates):.4f}]")

        # Prepare training data
        embeddings = []
        targets = []
        for inst_id in train_ids:
            embeddings.append(self.instruction_embeddings[inst_id])
            targets.append(self.error_rates[inst_id])

        X = torch.stack(embeddings).to(self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)

        if verbose:
            print(f"  Training data: X={X.shape}, y={y.shape}")

        # Get VAE latent (mu) for GP training
        self.vae.eval()
        with torch.no_grad():
            z_train = self.vae.encode_mu(X)

        if verbose:
            print(f"  VAE latents: {z_train.shape}")
            print(f"  Latent range: [{z_train.min():.3f}, {z_train.max():.3f}]")

        # Initialize GP with VAE encoder wrapper
        self.gp = GPWithEI(device=str(self.device), latent_dim=self.config.latent_dim)

        # Use VAEWithAdapter: frozen VAE + trainable adapter for GP feature extraction
        # Adapter reduces vae_latent_dim (64D) to latent_dim (10D) for GP
        self.gp.vae_with_adapter = VAEWithAdapter(
            self.vae, self.config.vae_latent_dim, self.config.latent_dim
        ).to(self.device)
        self.gp.set_training_data(X, y)

        success = self.gp.train(
            epochs=self.config.gp_epochs,
            lr=self.config.gp_lr,
            patience=self.config.gp_patience,
            verbose=verbose,
        )

        if success and verbose:
            self._validate_gp(verbose=True)

        return success

    def train(self, verbose: bool = True) -> Tuple[GPWithEI, nn.Module]:
        """Run full training pipeline.

        1. Train VAE on diverse instructions with KL annealing
        2. Train GP using VAE.encode_mu for latent representation
        3. Use VAE.decode as decoder (no separate training needed)

        Args:
            verbose: Print progress

        Returns:
            (gp, decoder) tuple of trained models
        """
        self.load_data(verbose=verbose)

        if not self.train_vae(verbose=verbose):
            raise RuntimeError("VAE training failed")

        if not self.train_gp_with_vae(verbose=verbose):
            raise RuntimeError("GP with VAE training failed")

        # Create decoder wrapper from VAE (without registering VAE as submodule)
        class VAEDecoderWrapper(nn.Module):
            """Wraps VAE.decode() as a decoder-compatible module."""

            def __init__(self, vae):
                super().__init__()
                object.__setattr__(self, '_vae', vae)

            def forward(self, z):
                return self._vae.decode(z)

        self.decoder = VAEDecoderWrapper(self.vae).to(self.device)

        return self.gp, self.decoder

    def save(self, path: str) -> None:
        """Save trained models.

        Args:
            path: Directory to save models
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save VAE and GP
        torch.save(self.vae.state_dict(), save_dir / "vae.pt")
        torch.save(
            {
                "gp_model": self.gp.gp_model.state_dict(),
                "likelihood": self.gp.likelihood.state_dict(),
                "X_min": self.gp.X_min,
                "X_max": self.gp.X_max,
                "y_mean": self.gp.y_mean,
                "y_std": self.gp.y_std,
                "y_best": self.gp.y_best,
            },
            save_dir / "gp.pt",
        )

        # Save config
        torch.save(self.config, save_dir / "config.pt")

        print(f"Models saved to {save_dir}")

    def load(self, path: str) -> None:
        """Load trained models.

        Args:
            path: Directory with saved models
        """
        save_dir = Path(path)

        # Load config (weights_only=False needed for dataclass)
        self.config = torch.load(save_dir / "config.pt", weights_only=False)

        # Reload data (needed for inference)
        self.load_data(verbose=False)

        # Load GP
        gp_state = torch.load(save_dir / "gp.pt", map_location=self.device)

        # Reconstruct GP model (needs training data)
        embeddings = [self.instruction_embeddings[i] for i in sorted(self.error_rates.keys())]
        targets = [self.error_rates[i] for i in sorted(self.error_rates.keys())]
        X = torch.stack(embeddings).to(self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)

        self.gp = GPWithEI(device=str(self.device), latent_dim=self.config.latent_dim)

        # Load VAE and use as encoder (with vae_latent_dim)
        self.vae = InstructionVAE(
            input_dim=self.config.embedding_dim,
            latent_dim=self.config.vae_latent_dim,
            beta=self.config.vae_beta,
        ).to(self.device)
        self.vae.load_state_dict(torch.load(save_dir / "vae.pt", map_location=self.device))

        # Use VAEWithAdapter (same as training) to ensure state_dict compatibility
        # Adapter reduces vae_latent_dim (64D) to latent_dim (10D) for GP
        self.gp.vae_with_adapter = VAEWithAdapter(
            self.vae, self.config.vae_latent_dim, self.config.latent_dim
        ).to(self.device)

        # Create VAE decoder wrapper (without registering VAE as submodule)
        class VAEDecoderWrapper(nn.Module):
            def __init__(self, vae):
                super().__init__()
                object.__setattr__(self, '_vae', vae)

            def forward(self, z):
                return self._vae.decode(z)

        self.decoder = VAEDecoderWrapper(self.vae).to(self.device)

        # Set training data and GP params (set_training_data encodes to 64D VAE latents)
        self.gp.set_training_data(X, y)
        self.gp.X_min = gp_state["X_min"]
        self.gp.X_max = gp_state["X_max"]
        self.gp.y_mean = gp_state["y_mean"]
        self.gp.y_std = gp_state["y_std"]
        self.gp.y_best = gp_state["y_best"]

        # Reinitialize GP model with stored weights
        from gpytorch.constraints import Interval
        from gpytorch.likelihoods import GaussianLikelihood
        from generation.invbo_decoder.gp import InstructionDeepKernelGP

        # X_train is already 64D VAE latents (set by set_training_data)
        denom = self.gp.X_max - self.gp.X_min
        denom[denom == 0] = 1.0
        X_norm = (self.gp.X_train - self.gp.X_min) / denom
        y_norm = (self.gp.y_train - self.gp.y_mean) / self.gp.y_std

        # Use same noise constraint as in training
        self.gp.likelihood = GaussianLikelihood(noise_constraint=Interval(0.001, 0.1)).to(self.device)
        self.gp.gp_model = InstructionDeepKernelGP(
            X_norm, y_norm, self.gp.likelihood, self.gp.vae_with_adapter.adapter
        ).to(self.device)
        self.gp.gp_model.load_state_dict(gp_state["gp_model"])
        self.gp.likelihood.load_state_dict(gp_state["likelihood"])

        print(f"Models loaded from {save_dir}")
