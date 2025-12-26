"""Robust HbBoPs Optimizer.

Main optimization pipeline using VAE latent space and GP.
Instruction-only optimization with fixed exemplar from grid.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from robust_vec2text.encoder import GTRPromptEncoder
from robust_vec2text.vae import InstructionVAE
from robust_vec2text.gp import GPTrainer
from robust_vec2text.training import VAETrainer


@dataclass
class GridPrompt:
    """A prompt from the pre-evaluated grid."""

    instruction_id: int
    exemplar_id: int
    instruction: str
    exemplar: str
    error_rate: float


class RobustHbBoPs:
    """Robust HbBoPs optimizer with VAE latent space.

    Pipeline:
        1. Load instructions and exemplars
        2. Pre-compute GTR embeddings
        3. Train VAE on instruction embeddings
        4. Load top-k prompts from grid
        5. Train GP on latent representations
        6. Gradient-based optimization in latent space
        7. Invert optimized latent via Vec2Text

    Attributes:
        gtr: GTR encoder for embeddings
        vae_trainer: VAE trainer instance
        gp_trainer: GP trainer instance
    """

    def __init__(
        self,
        instructions: List[str],
        exemplars: List[str],
        device: str = "cuda",
        latent_dim: int = 32,
    ):
        """Initialize optimizer.

        Args:
            instructions: List of instruction texts
            exemplars: List of exemplar texts
            device: Device to use
            latent_dim: VAE latent dimension
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        self.instructions = instructions
        self.exemplars = exemplars

        # Initialize encoder
        print("Initializing GTR encoder...")
        self.gtr = GTRPromptEncoder(device=str(self.device))

        # Pre-compute embeddings (using encode_tensor for torch tensors)
        print("Pre-computing instruction embeddings...")
        self.instruction_embeddings: Dict[int, torch.Tensor] = {}
        for i, inst in enumerate(instructions):
            emb = self.gtr.encode_tensor(inst)
            self.instruction_embeddings[i] = emb.to(self.device)

        print("Pre-computing exemplar embeddings...")
        self.exemplar_embeddings: Dict[int, torch.Tensor] = {}
        for i, ex in enumerate(exemplars):
            emb = self.gtr.encode_tensor(ex)
            self.exemplar_embeddings[i] = emb.to(self.device)

        print(f"  Cached {len(self.instruction_embeddings)} instruction embeddings")
        print(f"  Cached {len(self.exemplar_embeddings)} exemplar embeddings")

        # Initialize trainers (will be trained later)
        self.vae_trainer: Optional[VAETrainer] = None
        self.gp_trainer: Optional[GPTrainer] = None

        # Best results from grid
        self.best_grid_prompt: Optional[GridPrompt] = None
        self.grid_prompts: List[GridPrompt] = []

    def train_vae(
        self,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 30,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train VAE on instruction embeddings.

        Args:
            epochs: Maximum epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            Training history
        """
        # Stack all instruction embeddings
        embeddings = torch.stack(list(self.instruction_embeddings.values()))

        self.vae_trainer = VAETrainer(
            input_dim=768,
            latent_dim=self.latent_dim,
            device=str(self.device),
        )

        history = self.vae_trainer.train(
            embeddings=embeddings,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            verbose=verbose,
        )

        return history

    def load_grid(
        self,
        grid_path: str,
        top_k: int = 25,
    ) -> List[GridPrompt]:
        """Load top-k prompts from pre-evaluated grid.

        Grid format: {instruction_id, exemplar_id, error_rate, ...}
        Instruction/exemplar texts are looked up from self.instructions/exemplars.

        Args:
            grid_path: Path to grid JSONL file
            top_k: Number of top prompts to load

        Returns:
            List of top-k GridPrompt objects
        """
        grid_prompts = []

        with open(grid_path, "r") as f:
            for line in f:
                data = json.loads(line)
                inst_id = data["instruction_id"]
                ex_id = data["exemplar_id"]

                # Look up texts from stored lists
                instruction = self.instructions[inst_id] if inst_id < len(self.instructions) else ""
                exemplar = self.exemplars[ex_id] if ex_id < len(self.exemplars) else ""

                grid_prompts.append(
                    GridPrompt(
                        instruction_id=inst_id,
                        exemplar_id=ex_id,
                        instruction=instruction,
                        exemplar=exemplar,
                        error_rate=data["error_rate"],
                    )
                )

        # Sort by error rate (ascending - lower is better)
        grid_prompts.sort(key=lambda p: p.error_rate)

        # Take top-k
        self.grid_prompts = grid_prompts[:top_k]
        self.best_grid_prompt = self.grid_prompts[0]

        print(f"Loaded {len(self.grid_prompts)} prompts from grid")
        print(f"  Best error rate: {self.best_grid_prompt.error_rate:.4f}")
        print(f"  Worst in top-k: {self.grid_prompts[-1].error_rate:.4f}")

        return self.grid_prompts

    def train_gp(
        self,
        max_epochs: int = 500,
        lr: float = 0.01,
        patience: int = 20,
        verbose: bool = True,
    ) -> bool:
        """Train GP on latent representations from grid.

        Requires VAE to be trained first.

        Args:
            max_epochs: Maximum training epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if self.vae_trainer is None:
            raise RuntimeError("VAE must be trained first. Call train_vae().")

        if not self.grid_prompts:
            raise RuntimeError("Grid must be loaded first. Call load_grid().")

        vae = self.vae_trainer.get_vae()
        vae.eval()

        # Get latent representations for grid instructions
        X_list = []
        y_list = []

        for prompt in self.grid_prompts:
            # Get instruction embedding
            if prompt.instruction_id in self.instruction_embeddings:
                emb = self.instruction_embeddings[prompt.instruction_id]
            else:
                # Instruction not in cache - encode it
                emb = self.gtr.encode_tensor(prompt.instruction).to(self.device)

            # Get latent representation (mu only, no sampling)
            with torch.no_grad():
                latent = vae.get_latent(emb.unsqueeze(0)).squeeze(0)

            X_list.append(latent)
            y_list.append(prompt.error_rate)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, device=self.device)

        if verbose:
            print(f"Training GP on {len(X)} latent points")
            print(f"  Error rate range: {y.min():.4f} - {y.max():.4f}")

        # Train GP
        self.gp_trainer = GPTrainer(
            latent_dim=self.latent_dim,
            device=str(self.device),
        )

        success = self.gp_trainer.train(
            X=X,
            y=y,
            max_epochs=max_epochs,
            lr=lr,
            patience=patience,
            verbose=verbose,
        )

        return success

    def get_best_latent(self) -> torch.Tensor:
        """Get latent representation of best instruction from grid.

        Returns:
            Latent tensor of shape (32,)
        """
        if self.best_grid_prompt is None:
            raise RuntimeError("Grid must be loaded first.")

        if self.vae_trainer is None:
            raise RuntimeError("VAE must be trained first.")

        vae = self.vae_trainer.get_vae()
        vae.eval()

        emb = self.instruction_embeddings[self.best_grid_prompt.instruction_id]

        with torch.no_grad():
            latent = vae.get_latent(emb.unsqueeze(0)).squeeze(0)

        return latent

    def get_vae(self) -> InstructionVAE:
        """Get trained VAE model."""
        if self.vae_trainer is None:
            raise RuntimeError("VAE not trained.")
        return self.vae_trainer.get_vae()

    def get_gp_trainer(self) -> GPTrainer:
        """Get trained GP trainer."""
        if self.gp_trainer is None:
            raise RuntimeError("GP not trained.")
        return self.gp_trainer

    def save_checkpoint(self, path: str):
        """Save VAE checkpoint."""
        if self.vae_trainer is not None:
            self.vae_trainer.save(path)

    def load_checkpoint(self, path: str):
        """Load VAE checkpoint."""
        if self.vae_trainer is None:
            self.vae_trainer = VAETrainer(
                input_dim=768,
                latent_dim=self.latent_dim,
                device=str(self.device),
            )
        self.vae_trainer.load(path)
