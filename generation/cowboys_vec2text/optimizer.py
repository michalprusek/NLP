"""COWBOYS Optimizer (Instruction-Only Version).

Main optimization pipeline using pCN MCMC with trust regions
and weighted VAE retraining.

This version works with instructions only (no exemplars).

Extends RobustHbBoPs with:
- Weighted VAE retraining
- Trust region management
- Sample accumulation for retraining
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .encoder import GTRPromptEncoder
from .vae import InstructionVAE
from .training import WeightedVAETrainer, RetrainConfig
from .trust_region import TrustRegionManager, TRConfig

# Import instruction-only GP classes
from robust_vec2text.exemplar_selector import InstructionSelector


@dataclass
class GridPrompt:
    """A prompt from the pre-evaluated grid (instruction-only)."""

    instruction_id: int
    instruction: str
    error_rate: float


class CowboysOptimizer:
    """COWBOYS optimizer: pCN MCMC with trust regions and weighted retraining.

    This is the instruction-only version (no exemplars).

    Pipeline:
        1. Load instructions, pre-compute GTR embeddings
        2. Load top-k prompts from grid + train instruction GP
        3. Train VAE on instruction embeddings
        4. Run iterative optimization:
           a. pCN MCMC sampling within trust region
           b. Vec2Text inversion + perplexity filtering
           c. Evaluate best candidate
           d. Update GP with new observation
           e. Update trust region
           f. Periodically retrain VAE with weighted samples

    Attributes:
        gtr: GTR encoder for embeddings
        vae_trainer: WeightedVAETrainer instance
        instruction_selector: InstructionSelector GP for error prediction
        trust_region: Optional TrustRegionManager
    """

    def __init__(
        self,
        instructions: List[str],
        device: str = "cuda",
        latent_dim: int = 32,
    ):
        """Initialize optimizer.

        Args:
            instructions: List of instruction texts
            device: Device to use
            latent_dim: VAE latent dimension
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        self.instructions = instructions

        # Initialize encoder
        print("Initializing GTR encoder...")
        self.gtr = GTRPromptEncoder(device=str(self.device))

        # Pre-compute embeddings
        print("Pre-computing instruction embeddings...")
        self.instruction_embeddings: Dict[int, torch.Tensor] = {}
        for i, inst in enumerate(instructions):
            emb = self.gtr.encode_tensor(inst)
            self.instruction_embeddings[i] = emb.to(self.device)

        print(f"  Cached {len(self.instruction_embeddings)} instruction embeddings")

        # Initialize trainers
        self.vae_trainer: Optional[WeightedVAETrainer] = None
        self.instruction_selector: Optional[InstructionSelector] = None

        # Best results from grid
        self.best_grid_prompt: Optional[GridPrompt] = None
        self.grid_prompts: List[GridPrompt] = []

        # Trust region (initialized when needed)
        self.trust_region: Optional[TrustRegionManager] = None

        # Track accumulated observations for retraining
        self.new_observations: List[Tuple[str, torch.Tensor, float]] = []

    def train_vae(
        self,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 30,
        lambda_cycle: float = 0.0,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train VAE on instruction embeddings.

        Args:
            epochs: Maximum epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            lambda_cycle: Weight for cycle-consistency loss (default: 0.0)
            verbose: Print progress

        Returns:
            Training history
        """
        # Stack all instruction embeddings
        embeddings = torch.stack(list(self.instruction_embeddings.values()))

        # Get error rates for weighted training if grid is loaded
        error_rates = None
        if self.grid_prompts:
            # Create error rate tensor aligned with embeddings
            error_rate_dict = {}
            for gp in self.grid_prompts:
                if gp.instruction_id not in error_rate_dict:
                    error_rate_dict[gp.instruction_id] = gp.error_rate
            # Default to 0.5 for instructions not in grid
            error_rates = torch.tensor([
                error_rate_dict.get(i, 0.5)
                for i in range(len(embeddings))
            ])

        self.vae_trainer = WeightedVAETrainer(
            input_dim=768,
            latent_dim=self.latent_dim,
            device=str(self.device),
            lambda_cycle=lambda_cycle,
        )

        if verbose:
            print(f"VAE loss weights: cosine=20, mse=1, kld=0.0025 (annealed 0-500 epochs), cycle={lambda_cycle}")

        history = self.vae_trainer.train(
            embeddings=embeddings,
            error_rates=error_rates,
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
        train_instruction_gp: bool = False,
    ) -> List[GridPrompt]:
        """Load top-k prompts from pre-evaluated grid.

        Args:
            grid_path: Path to grid JSONL file (instruction-only format)
            top_k: Number of top prompts to load
            train_instruction_gp: If True, train instruction GP

        Returns:
            List of top-k GridPrompt objects
        """
        grid_prompts = []

        with open(grid_path, "r") as f:
            for line in f:
                data = json.loads(line)
                inst_id = data["instruction_id"]

                # Get instruction text (from data if available, else from list)
                if "instruction_text" in data:
                    instruction = data["instruction_text"]
                elif inst_id < len(self.instructions):
                    instruction = self.instructions[inst_id]
                else:
                    instruction = ""

                grid_prompts.append(
                    GridPrompt(
                        instruction_id=inst_id,
                        instruction=instruction,
                        error_rate=data["error_rate"],
                    )
                )

        # Sort by error rate
        grid_prompts.sort(key=lambda p: p.error_rate)

        self.grid_prompts = grid_prompts[:top_k]
        self.best_grid_prompt = self.grid_prompts[0]

        print(f"Loaded {len(self.grid_prompts)} prompts from grid")
        print(f"  Best error rate: {self.best_grid_prompt.error_rate:.4f}")
        print(f"  Worst in top-k: {self.grid_prompts[-1].error_rate:.4f}")

        if train_instruction_gp:
            self._train_instruction_gp(grid_path, top_k)

        return self.grid_prompts

    def _train_instruction_gp(self, grid_path: str, top_k: int) -> None:
        """Train instruction-only GP."""
        print("\nTraining instruction selection GP...")

        self.instruction_selector = InstructionSelector(
            instructions=self.instructions,
            gtr=self.gtr,
            device=str(self.device),
        )

        self.instruction_selector.train_from_grid(
            grid_path=grid_path,
            top_k=top_k,
            epochs=3000,
            patience=10,
            verbose=True,
        )

    def train_instruction_gp_on_decoded(
        self,
        grid_path: str,
        top_k: int = 25,
        epochs: int = 3000,
        patience: int = 10,
        verbose: bool = True,
        use_invbo_alignment: bool = True,
    ) -> None:
        """Train GP on VAE-decoded instruction embeddings.

        This is the COWBOYS fix: training GP on decoded embeddings
        ensures it operates in the same distribution as the MCMC
        sampling.

        With use_invbo_alignment=True (default), we use InvBO-style decoder
        inversion to create aligned embeddings where decode(z_inv) ≈ GTR(text).
        This reduces the prediction gap from the very first iteration.

        Args:
            grid_path: Path to grid JSONL file
            top_k: Number of top prompts to use
            epochs: GP training epochs
            patience: Early stopping patience
            verbose: Print progress
            use_invbo_alignment: If True, use decoder inversion for aligned embeddings
        """
        if self.vae_trainer is None:
            raise RuntimeError("VAE must be trained before training GP on decoded embeddings.")

        print("\nComputing VAE-decoded instruction embeddings...")
        if use_invbo_alignment:
            print("  Using InvBO-style decoder inversion for alignment")
        vae = self.get_vae()
        vae.eval()

        decoded_instruction_embeddings: Dict[int, torch.Tensor] = {}

        for inst_id, orig_emb in self.instruction_embeddings.items():
            if use_invbo_alignment:
                # InvBO: invert decoder to find z where decode(z) ≈ orig_emb
                z_inv = vae.invert_decoder(orig_emb)
                with torch.no_grad():
                    decoded_emb = vae.decode(z_inv.unsqueeze(0)).squeeze(0)
            else:
                # Original: encode then decode (has reconstruction error)
                with torch.no_grad():
                    latent = vae.get_latent(orig_emb.unsqueeze(0))
                    decoded_emb = vae.decode(latent).squeeze(0)

            decoded_instruction_embeddings[inst_id] = decoded_emb

            if verbose and inst_id < 3:
                cosine = torch.nn.functional.cosine_similarity(
                    orig_emb.unsqueeze(0), decoded_emb.unsqueeze(0)
                ).item()
                print(f"  Instruction {inst_id}: orig→decoded cosine = {cosine:.4f}")

        print(f"  Computed {len(decoded_instruction_embeddings)} decoded embeddings")

        print("\nTraining instruction-only GP on decoded embeddings...")
        self.instruction_selector = InstructionSelector(
            instructions=self.instructions,
            gtr=self.gtr,
            device=str(self.device),
        )

        self.instruction_selector.train_with_decoded_embeddings(
            decoded_instruction_embeddings=decoded_instruction_embeddings,
            grid_path=grid_path,
            top_k=top_k,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
        )

    def initialize_trust_region(
        self,
        anchor: Optional[torch.Tensor] = None,
        config: Optional[TRConfig] = None,
    ) -> TrustRegionManager:
        """Initialize trust region around anchor point.

        Args:
            anchor: Anchor point (default: best grid latent)
            config: Trust region configuration

        Returns:
            Initialized TrustRegionManager
        """
        if anchor is None:
            anchor = self.get_best_latent()

        config = config or TRConfig()

        self.trust_region = TrustRegionManager(
            anchor=anchor,
            config=config,
            device=str(self.device),
        )

        return self.trust_region

    def get_trust_region(self) -> Optional[TrustRegionManager]:
        """Get current trust region manager."""
        return self.trust_region

    def should_retrain_vae(self, iteration: int, config: RetrainConfig) -> bool:
        """Check if VAE should be retrained this iteration.

        Args:
            iteration: Current iteration number (0-indexed)
            config: Retraining configuration

        Returns:
            True if VAE should be retrained
        """
        return iteration > 0 and iteration % config.retrain_interval == 0

    def retrain_vae(
        self,
        config: RetrainConfig,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Retrain VAE with weighted samples.

        Uses accumulated observations weighted by error rate.

        Args:
            config: Retraining configuration
            verbose: Print progress

        Returns:
            Retraining history
        """
        if self.vae_trainer is None:
            raise RuntimeError("VAE not trained.")

        return self.vae_trainer.retrain_with_weights(config, verbose)

    def add_observation(
        self,
        instruction_text: str,
        instruction_emb: torch.Tensor,
        error_rate: float,
    ):
        """Add new observation for future retraining.

        Args:
            instruction_text: Generated instruction text
            instruction_emb: GTR embedding of instruction
            error_rate: Evaluated error rate
        """
        self.new_observations.append((instruction_text, instruction_emb.cpu(), error_rate))

        # Also add to VAE trainer for retraining
        if self.vae_trainer is not None:
            self.vae_trainer.add_samples(instruction_emb.cpu(), [error_rate])

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

    def get_instruction_selector(self) -> InstructionSelector:
        """Get trained instruction selector."""
        if self.instruction_selector is None:
            raise RuntimeError("Instruction GP not trained.")
        return self.instruction_selector

    def get_decoded_embedding(self, instruction_text: str) -> torch.Tensor:
        """Encode instruction and decode through VAE.

        Returns embedding in VAE-decoded distribution for GP compatibility.

        Args:
            instruction_text: Instruction text to encode

        Returns:
            VAE-decoded embedding tensor of shape (768,)
        """
        if self.vae_trainer is None:
            raise RuntimeError("VAE not trained.")

        vae = self.vae_trainer.get_vae()
        vae.eval()

        inst_emb = self.gtr.encode_tensor(instruction_text).to(self.device)

        with torch.no_grad():
            latent = vae.get_latent(inst_emb.unsqueeze(0))
            decoded = vae.decode(latent).squeeze(0)

        return decoded

    def save_checkpoint(self, path: str):
        """Save VAE checkpoint."""
        if self.vae_trainer is not None:
            self.vae_trainer.save(path)

    def load_checkpoint(self, path: str):
        """Load VAE checkpoint."""
        if self.vae_trainer is None:
            self.vae_trainer = WeightedVAETrainer(
                input_dim=768,
                latent_dim=self.latent_dim,
                device=str(self.device),
            )
        self.vae_trainer.load(path)
