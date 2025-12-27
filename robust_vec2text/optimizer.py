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
from robust_vec2text.training import VAETrainer
from robust_vec2text.exemplar_selector import ExemplarSelector


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
        3. Load top-k prompts from grid + train HbBoPs GP
        4. Train VAE on instruction embeddings
        5. Gradient-based EI optimization (VAE decode → GP predict)
        6. Invert optimized embedding via Vec2Text

    Attributes:
        gtr: GTR encoder for embeddings
        vae_trainer: VAE trainer instance
        exemplar_selector: HbBoPs-style GP for error prediction
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

        # Exemplar selector (HbBoPs-style GP on instruction+exemplar pairs)
        self.exemplar_selector: Optional[ExemplarSelector] = None

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
        train_exemplar_gp: bool = False,
    ) -> List[GridPrompt]:
        """Load top-k prompts from pre-evaluated grid.

        Grid format: {instruction_id, exemplar_id, error_rate, ...}
        Instruction/exemplar texts are looked up from self.instructions/exemplars.

        Args:
            grid_path: Path to grid JSONL file
            top_k: Number of top prompts to load
            train_exemplar_gp: If True, train HbBoPs-style GP for exemplar selection

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

        # Train exemplar selector GP if requested
        if train_exemplar_gp:
            self._train_exemplar_gp(grid_path, top_k)

        return self.grid_prompts

    def _train_exemplar_gp(self, grid_path: str, top_k: int) -> None:
        """Train HbBoPs-style GP for exemplar selection.

        This GP operates on (instruction, exemplar) GTR embeddings
        and can be used to predict error rates for any combination.

        NOTE: This trains on ORIGINAL embeddings. For EI optimization,
        use train_exemplar_gp_on_decoded() which trains on VAE-decoded
        embeddings to match the distribution during gradient optimization.
        """
        print("\nTraining exemplar selection GP...")

        self.exemplar_selector = ExemplarSelector(
            instructions=self.instructions,
            exemplars=self.exemplars,
            gtr=self.gtr,
            device=str(self.device),
        )

        self.exemplar_selector.train_from_grid(
            grid_path=grid_path,
            top_k=top_k,
            epochs=3000,
            patience=10,
            verbose=True,
        )

    def train_exemplar_gp_on_decoded(
        self,
        grid_path: str,
        top_k: int = 25,
        epochs: int = 3000,
        patience: int = 10,
        verbose: bool = True,
    ) -> None:
        """Train GP on VAE-decoded instruction embeddings.

        This is the key fix for EI=0 problem. By training GP on decoded
        embeddings, we ensure it operates in the same distribution as
        the gradient-based EI optimization.

        Reference: COWBOYS paper (Return of the Latent Space COWBOYS)
        https://arxiv.org/abs/2507.03910

        Pipeline:
            1. For each instruction in grid:
               - latent = VAE.encode(GTR.encode(instruction))
               - decoded_emb = VAE.decode(latent)
            2. Train GP on (decoded_emb, exemplar_emb) pairs

        Args:
            grid_path: Path to grid JSONL file
            top_k: Number of top prompts to use for training
            epochs: GP training epochs
            patience: Early stopping patience
            verbose: Print progress
        """
        if self.vae_trainer is None:
            raise RuntimeError("VAE must be trained before training GP on decoded embeddings.")

        print("\nComputing VAE-decoded instruction embeddings...")
        vae = self.get_vae()
        vae.eval()

        # Compute decoded embeddings for each instruction
        decoded_instruction_embeddings: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for inst_id, orig_emb in self.instruction_embeddings.items():
                latent = vae.get_latent(orig_emb.unsqueeze(0))
                decoded_emb = vae.decode(latent).squeeze(0)
                decoded_instruction_embeddings[inst_id] = decoded_emb

                if verbose and inst_id < 3:
                    cosine = torch.nn.functional.cosine_similarity(
                        orig_emb.unsqueeze(0), decoded_emb.unsqueeze(0)
                    ).item()
                    print(f"  Instruction {inst_id}: orig→decoded cosine = {cosine:.4f}")

        print(f"  Computed {len(decoded_instruction_embeddings)} decoded embeddings")

        # Initialize exemplar selector
        print("\nTraining GP on decoded embeddings...")
        self.exemplar_selector = ExemplarSelector(
            instructions=self.instructions,
            exemplars=self.exemplars,
            gtr=self.gtr,
            device=str(self.device),
        )

        # Train with decoded embeddings
        self.exemplar_selector.train_with_decoded_embeddings(
            decoded_instruction_embeddings=decoded_instruction_embeddings,
            exemplar_embeddings=self.exemplar_embeddings,
            grid_path=grid_path,
            top_k=top_k,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
        )

    def get_exemplar_selector(self) -> ExemplarSelector:
        """Get trained exemplar selector."""
        if self.exemplar_selector is None:
            raise RuntimeError("Exemplar GP not trained. Call load_grid with train_exemplar_gp=True.")
        return self.exemplar_selector


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
