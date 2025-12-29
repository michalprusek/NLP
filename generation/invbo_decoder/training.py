"""Two-phase training for InvBO decoder.

Phase 1: Train GP + Encoder jointly on instruction embeddings
Phase 2: Train Decoder with frozen encoder using cyclic loss

This ensures the 10D latent space is semantically meaningful
before training the decoder to invert it.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from generation.invbo_decoder.encoder import (
    GTRInstructionEncoder,
    InstructionFeatureExtractor,
    InstructionVAE,
)
from generation.invbo_decoder.gp import GPWithEI
from generation.invbo_decoder.decoder import LatentDecoder, DecoderCyclicLoss


@dataclass
class TrainingConfig:
    """Configuration for InvBO training."""

    # Data paths
    instructions_path: str = "datasets/inversion/instructions_100.txt"
    grid_path: str = "datasets/inversion/grid_100_qend.jsonl"
    diverse_instructions_path: str = "datasets/inversion/diverse_instructions_1000.json"

    # Architecture
    latent_dim: int = 10
    embedding_dim: int = 768

    # Phase 1: GP training
    gp_epochs: int = 3000
    gp_lr: float = 0.01
    gp_patience: int = 10

    # Phase 2: Decoder training (on diverse instructions)
    decoder_epochs: int = 500
    decoder_lr: float = 0.001
    decoder_patience: int = 30
    decoder_batch_size: int = 64

    # Loss weights (simplified: only cyclic + cosine)
    lambda_cycle: float = 1.0
    lambda_cosine: float = 5.0
    cycle_tolerance: float = 0.0  # Strict cyclic loss

    # VAE mode (alternative to separate encoder+decoder)
    use_vae: bool = False
    vae_beta: float = 0.1  # KL regularization weight
    vae_epochs: int = 1000
    vae_lr: float = 0.001
    vae_annealing_epochs: int = 500  # Epochs for KL annealing (0 → beta)
    vae_patience: int = 100  # Patience for VAE early stopping (higher than decoder)

    # Device
    device: str = "cuda"


class InvBOTrainer:
    """Two-phase trainer for InvBO decoder inversion.

    Phase 1: Train GP + InstructionFeatureExtractor jointly
        - Loads instruction embeddings and error rates from grid
        - Trains deep kernel GP to predict error rates

    Phase 2: Train LatentDecoder with frozen encoder
        - Freezes encoder from Phase 1
        - Trains decoder to minimize cyclic loss
        - ||z - encoder(decoder(z))||^2 with soft tolerance
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
        self.decoder: Optional[LatentDecoder] = None
        self.vae: Optional[InstructionVAE] = None  # VAE mode

        # Data storage (100 instructions with error rates for GP)
        self.instructions: List[str] = []
        self.instruction_embeddings: Dict[int, torch.Tensor] = {}
        self.error_rates: Dict[int, float] = {}

        # Diverse instructions (1000 for decoder training)
        self.diverse_instructions: List[str] = []
        self.diverse_embeddings: Optional[torch.Tensor] = None  # (1000, 768)
        self.diverse_latents: Optional[torch.Tensor] = None  # (1000, 10)

    def load_data(self, verbose: bool = True) -> None:
        """Load instructions and grid data.

        Args:
            verbose: Print progress
        """
        if verbose:
            print(f"Loading instructions from {self.config.instructions_path}...")

        # Load instructions (skip comments and empty lines)
        with open(self.config.instructions_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Parse numbered instruction: "1. Solve:" -> "Solve:"
                    if ". " in line[:5]:
                        line = line.split(". ", 1)[1]
                    self.instructions.append(line)

        if verbose:
            print(f"  Loaded {len(self.instructions)} instructions")

        # Encode instructions
        if verbose:
            print("Encoding instructions with GTR...")
        for idx, inst in enumerate(self.instructions):
            self.instruction_embeddings[idx] = self.gtr.encode_tensor(inst)

        # Load grid data (error rates)
        if verbose:
            print(f"Loading grid from {self.config.grid_path}...")

        with open(self.config.grid_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                inst_id = entry["instruction_id"]
                self.error_rates[inst_id] = entry["error_rate"]

        if verbose:
            print(f"  Loaded {len(self.error_rates)} error rates")
            rates = list(self.error_rates.values())
            print(f"  Error rate range: [{min(rates):.4f}, {max(rates):.4f}]")

    def train_phase1(self, verbose: bool = True) -> bool:
        """Phase 1: Train GP + Encoder jointly.

        Args:
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Phase 1: Training GP + Encoder")
            print("=" * 60)

        # Prepare training data
        embeddings = []
        targets = []

        for inst_id in sorted(self.error_rates.keys()):
            embeddings.append(self.instruction_embeddings[inst_id])
            targets.append(self.error_rates[inst_id])

        X = torch.stack(embeddings).to(self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)

        if verbose:
            print(f"  Training data: X={X.shape}, y={y.shape}")

        # Initialize and train GP
        self.gp = GPWithEI(device=str(self.device), latent_dim=self.config.latent_dim)
        self.gp.set_training_data(X, y)

        success = self.gp.train(
            epochs=self.config.gp_epochs,
            lr=self.config.gp_lr,
            patience=self.config.gp_patience,
            verbose=verbose,
        )

        if success and verbose:
            # Validate GP predictions
            self._validate_gp(verbose=True)

        return success

    def _validate_gp(self, verbose: bool = True) -> None:
        """Validate GP predictions on training data."""
        if verbose:
            print("\nValidating GP predictions...")

        errors = []
        for inst_id in sorted(self.error_rates.keys()):
            emb = self.instruction_embeddings[inst_id]
            pred_mean, pred_std = self.gp.predict(emb)
            true_error = self.error_rates[inst_id]
            errors.append(abs(pred_mean - true_error))

        mae = sum(errors) / len(errors)
        if verbose:
            print(f"  Mean Absolute Error: {mae:.4f}")
            print(f"  Max Error: {max(errors):.4f}")

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

        # Load diverse instructions for VAE training
        if verbose:
            print(f"Loading diverse instructions from {self.config.diverse_instructions_path}...")

        with open(self.config.diverse_instructions_path, "r") as f:
            data = json.load(f)
            self.diverse_instructions = data["instructions"]

        if verbose:
            print(f"  Loaded {len(self.diverse_instructions)} instructions")

        # Encode with GTR
        if verbose:
            print("  Encoding with GTR...")
        embeddings_list = []
        for inst in self.diverse_instructions:
            emb = self.gtr.encode_tensor(inst)
            embeddings_list.append(emb)
        self.diverse_embeddings = torch.stack(embeddings_list).to(self.device)

        # Also add the 100 grid instructions
        grid_embeddings = []
        for inst_id in sorted(self.instruction_embeddings.keys()):
            grid_embeddings.append(self.instruction_embeddings[inst_id])
        if grid_embeddings:
            grid_emb_tensor = torch.stack(grid_embeddings).to(self.device)
            self.diverse_embeddings = torch.cat([self.diverse_embeddings, grid_emb_tensor], dim=0)

        if verbose:
            print(f"  Total training embeddings: {self.diverse_embeddings.shape}")

        # Initialize VAE
        self.vae = InstructionVAE(
            input_dim=self.config.embedding_dim,
            latent_dim=self.config.latent_dim,
            beta=self.config.vae_beta,
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.config.vae_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.vae_epochs, eta_min=1e-5
        )

        X = self.diverse_embeddings
        n_samples = X.shape[0]
        best_loss = float("inf")
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
            for i in range(0, n_samples, self.config.decoder_batch_size):
                batch_x = X_shuffled[i:i + self.config.decoder_batch_size]

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

            if avg_loss < best_loss:
                best_loss = avg_loss
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

    def train_gp_with_vae(self, verbose: bool = True) -> bool:
        """Train GP using VAE encoder for latent representation.

        Uses VAE.encode_mu() (deterministic) for GP training.

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

        # Prepare training data (100 grid instructions)
        embeddings = []
        targets = []
        for inst_id in sorted(self.error_rates.keys()):
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

        # Create a wrapper feature extractor that uses VAE.encode_mu
        class VAEEncoderWrapper(nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, x):
                return self.vae.encode_mu(x)

        # Set training data - use raw embeddings, GP will use feature extractor
        self.gp.feature_extractor = VAEEncoderWrapper(self.vae).to(self.device)
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

    def load_diverse_instructions(self, verbose: bool = True) -> None:
        """Load and encode 1000 diverse instructions for decoder training.

        Must be called AFTER train_phase1() since we need the trained encoder.

        Args:
            verbose: Print progress
        """
        if self.gp is None or self.gp.feature_extractor is None:
            raise RuntimeError("Phase 1 not completed. Train GP+Encoder first.")

        if verbose:
            print(f"\nLoading diverse instructions from {self.config.diverse_instructions_path}...")

        # Load JSON
        with open(self.config.diverse_instructions_path, "r") as f:
            data = json.load(f)
            self.diverse_instructions = data["instructions"]

        if verbose:
            print(f"  Loaded {len(self.diverse_instructions)} diverse instructions")

        # Encode with GTR -> 768D
        if verbose:
            print("  Encoding with GTR...")
        embeddings_list = []
        for inst in self.diverse_instructions:
            emb = self.gtr.encode_tensor(inst)
            embeddings_list.append(emb)

        self.diverse_embeddings = torch.stack(embeddings_list).to(self.device)

        # Encode with trained GP encoder -> 10D
        # Apply same normalization as GP training
        if verbose:
            print("  Encoding with Deep Kernel encoder -> 10D latent...")

        encoder = self.gp.feature_extractor
        encoder.eval()

        denom = self.gp.X_max - self.gp.X_min
        denom[denom == 0] = 1.0
        embeddings_norm = (self.diverse_embeddings - self.gp.X_min) / denom

        with torch.no_grad():
            self.diverse_latents = encoder(embeddings_norm)

        if verbose:
            print(f"  Diverse embeddings: {self.diverse_embeddings.shape}")
            print(f"  Diverse latents: {self.diverse_latents.shape}")
            print(f"  Latent range: [{self.diverse_latents.min():.3f}, {self.diverse_latents.max():.3f}]")

    def train_phase2(self, verbose: bool = True) -> bool:
        """Phase 2: Train Decoder on 1000 diverse instructions.

        Uses (latent, embedding) pairs from diverse_instructions_1000.json.
        The decoder learns to map 10D latent -> 768D embedding.

        Loss = lambda_cycle * ||z - encoder(decoder(z))||^2 + lambda_cosine * (1 - cos_sim)

        Args:
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if self.gp is None or self.gp.feature_extractor is None:
            raise RuntimeError("Phase 1 not completed. Run train_phase1() first.")

        # Load diverse instructions if not already loaded
        if self.diverse_embeddings is None:
            self.load_diverse_instructions(verbose=verbose)

        if verbose:
            print("\n" + "=" * 60)
            print("Phase 2: Training Decoder on Diverse Instructions")
            print("=" * 60)

        # Freeze encoder
        encoder = self.gp.feature_extractor
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        if verbose:
            print("  Encoder frozen")

        # Initialize decoder
        self.decoder = LatentDecoder(
            latent_dim=self.config.latent_dim,
            output_dim=self.config.embedding_dim,
            normalize=True,
        ).to(self.device)

        # Training data
        Z = self.diverse_latents  # (1000, 10)
        X_emb = self.diverse_embeddings  # (1000, 768)

        if verbose:
            print(f"  Training data: latents={Z.shape}, embeddings={X_emb.shape}")

        # Optimizer
        optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.config.decoder_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.decoder_epochs, eta_min=1e-5
        )

        best_loss = float("inf")
        best_cosine = 0.0
        patience_counter = 0
        n_samples = Z.shape[0]

        # Normalization params for re-encoding
        denom = self.gp.X_max - self.gp.X_min
        denom[denom == 0] = 1.0

        if verbose:
            print("Training decoder...")

        self.decoder.train()

        for epoch in range(self.config.decoder_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples, device=self.device)
            Z_shuffled = Z[perm]
            X_emb_shuffled = X_emb[perm]

            epoch_cycle_losses = []
            epoch_cosine_losses = []
            epoch_cosines = []

            # Mini-batch training
            for i in range(0, n_samples, self.config.decoder_batch_size):
                batch_z = Z_shuffled[i:i + self.config.decoder_batch_size]
                batch_emb = X_emb_shuffled[i:i + self.config.decoder_batch_size]

                optimizer.zero_grad()

                # Forward: z -> decoder -> embedding_decoded (768D, L2-normalized)
                embedding_decoded = self.decoder(batch_z)

                # Re-encode: embedding_decoded -> normalize -> encoder -> z_recon
                emb_norm = (embedding_decoded - self.gp.X_min) / denom
                with torch.no_grad():
                    z_recon = encoder(emb_norm)

                # Cyclic loss: ||z - z_recon||^2
                cycle_loss = torch.nn.functional.mse_loss(batch_z, z_recon)

                # Cosine loss: 1 - cosine_similarity
                cosine_sim = torch.nn.functional.cosine_similarity(
                    embedding_decoded, batch_emb, dim=-1
                )
                cosine_loss = (1 - cosine_sim).mean()

                # Total loss
                loss = (
                    self.config.lambda_cycle * cycle_loss +
                    self.config.lambda_cosine * cosine_loss
                )

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_cycle_losses.append(cycle_loss.item())
                epoch_cosine_losses.append(cosine_loss.item())
                epoch_cosines.append(cosine_sim.mean().item())

            scheduler.step()

            # Epoch stats
            avg_cycle = sum(epoch_cycle_losses) / len(epoch_cycle_losses)
            avg_cosine_loss = sum(epoch_cosine_losses) / len(epoch_cosine_losses)
            avg_cosine = sum(epoch_cosines) / len(epoch_cosines)
            avg_loss = self.config.lambda_cycle * avg_cycle + self.config.lambda_cosine * avg_cosine_loss

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_cosine = avg_cosine
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.decoder_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, "
                    f"cycle={avg_cycle:.4f}, cosine={avg_cosine:.4f}"
                )

        if verbose:
            print(f"  Decoder training complete (epochs={epoch + 1})")
            print(f"  Best loss: {best_loss:.4f}, Best cosine: {best_cosine:.4f}")
            self._validate_decoder(verbose=True)

        return True

    def _validate_decoder(self, verbose: bool = True) -> None:
        """Validate decoder on 100 grid instructions (with error rates).

        Measures:
        - Cosine similarity between decoded and original embeddings
        - Cyclic distance ||z - encoder(decoder(z))||
        """
        if verbose:
            print("\nValidating decoder on grid_100 instructions...")

        encoder = self.gp.feature_extractor
        encoder.eval()
        self.decoder.eval()

        denom = self.gp.X_max - self.gp.X_min
        denom[denom == 0] = 1.0

        cosine_sims = []
        cycle_dists = []

        with torch.no_grad():
            for inst_id in sorted(self.error_rates.keys()):
                emb_original = self.instruction_embeddings[inst_id]

                # Get latent (same normalization as training)
                emb_norm = (emb_original - self.gp.X_min) / denom
                latent = encoder(emb_norm.unsqueeze(0)).squeeze(0)

                # Decode
                emb_decoded = self.decoder(latent)

                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    emb_decoded.unsqueeze(0), emb_original.unsqueeze(0)
                ).item()
                cosine_sims.append(cos_sim)

                # Cyclic distance: z -> decode -> normalize -> encode -> z_recon
                emb_decoded_norm = (emb_decoded - self.gp.X_min) / denom
                z_recon = encoder(emb_decoded_norm.unsqueeze(0)).squeeze(0)
                cycle_dist = torch.norm(latent - z_recon).item()
                cycle_dists.append(cycle_dist)

        if verbose:
            print(f"  Samples: {len(cosine_sims)} (grid instructions)")
            print(f"  Cosine similarity: {sum(cosine_sims)/len(cosine_sims):.4f} "
                  f"[{min(cosine_sims):.4f}, {max(cosine_sims):.4f}]")
            print(f"  Cycle distance: {sum(cycle_dists)/len(cycle_dists):.4f} "
                  f"[{min(cycle_dists):.4f}, {max(cycle_dists):.4f}]")

    def train(self, verbose: bool = True) -> Tuple[GPWithEI, LatentDecoder]:
        """Run full training pipeline.

        In standard mode (use_vae=False):
            Phase 1: Train GP + InstructionFeatureExtractor jointly
            Phase 2: Train LatentDecoder with frozen encoder

        In VAE mode (use_vae=True):
            Phase 1: Train VAE on diverse instructions (KL annealed)
            Phase 2: Train GP using VAE.encode_mu for latent
            Decoder = VAE.decode (no separate training needed)

        Args:
            verbose: Print progress

        Returns:
            (gp, decoder) tuple of trained models
        """
        self.load_data(verbose=verbose)

        if self.config.use_vae:
            # VAE mode
            if verbose:
                print("\n[VAE MODE ENABLED]")

            if not self.train_vae(verbose=verbose):
                raise RuntimeError("VAE training failed")

            if not self.train_gp_with_vae(verbose=verbose):
                raise RuntimeError("GP with VAE training failed")

            # Create decoder wrapper from VAE
            class VAEDecoderWrapper(nn.Module):
                """Wraps VAE.decode() as a LatentDecoder-compatible module."""

                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae

                def forward(self, z):
                    return self.vae.decode(z)

            self.decoder = VAEDecoderWrapper(self.vae).to(self.device)

        else:
            # Standard mode
            if not self.train_phase1(verbose=verbose):
                raise RuntimeError("Phase 1 (GP) training failed")

            if not self.train_phase2(verbose=verbose):
                raise RuntimeError("Phase 2 (Decoder) training failed")

        return self.gp, self.decoder

    def save(self, path: str) -> None:
        """Save trained models.

        Args:
            path: Directory to save models
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_vae:
            # VAE mode: save VAE and GP
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
        else:
            # Standard mode: save GP + encoder + decoder
            torch.save(
                {
                    "feature_extractor": self.gp.feature_extractor.state_dict(),
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
            torch.save(self.decoder.state_dict(), save_dir / "decoder.pt")

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

        if self.config.use_vae:
            # VAE mode: load VAE and use as encoder
            self.vae = InstructionVAE(
                input_dim=self.config.embedding_dim,
                latent_dim=self.config.latent_dim,
                beta=self.config.vae_beta,
            ).to(self.device)
            self.vae.load_state_dict(torch.load(save_dir / "vae.pt", map_location=self.device))

            # Create VAE encoder wrapper
            class VAEEncoderWrapper(nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae

                def forward(self, x):
                    return self.vae.encode_mu(x)

            self.gp.feature_extractor = VAEEncoderWrapper(self.vae).to(self.device)

            # Create VAE decoder wrapper
            class VAEDecoderWrapper(nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae

                def forward(self, z):
                    return self.vae.decode(z)

            self.decoder = VAEDecoderWrapper(self.vae).to(self.device)
        else:
            # Standard mode: load feature extractor and decoder separately
            self.gp.feature_extractor = InstructionFeatureExtractor(
                input_dim=768, latent_dim=self.config.latent_dim
            ).to(self.device)
            self.gp.feature_extractor.load_state_dict(gp_state["feature_extractor"])

            self.decoder = LatentDecoder(
                latent_dim=self.config.latent_dim,
                output_dim=self.config.embedding_dim,
                normalize=True,
            ).to(self.device)
            self.decoder.load_state_dict(torch.load(save_dir / "decoder.pt", map_location=self.device))

        # Set training data and GP params
        self.gp.set_training_data(X, y)
        self.gp.X_min = gp_state["X_min"]
        self.gp.X_max = gp_state["X_max"]
        self.gp.y_mean = gp_state["y_mean"]
        self.gp.y_std = gp_state["y_std"]
        self.gp.y_best = gp_state["y_best"]

        # Reinitialize GP model with stored weights
        from gpytorch.likelihoods import GaussianLikelihood
        from generation.invbo_decoder.gp import InstructionDeepKernelGP

        denom = self.gp.X_max - self.gp.X_min
        denom[denom == 0] = 1.0
        X_norm = (X - self.gp.X_min) / denom
        y_norm = (y - self.gp.y_mean) / self.gp.y_std

        self.gp.likelihood = GaussianLikelihood().to(self.device)
        self.gp.gp_model = InstructionDeepKernelGP(
            X_norm, y_norm, self.gp.likelihood, self.gp.feature_extractor
        ).to(self.device)
        self.gp.gp_model.load_state_dict(gp_state["gp_model"])
        self.gp.likelihood.load_state_dict(gp_state["likelihood"])

        print(f"Models loaded from {save_dir}")
