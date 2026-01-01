"""Training pipeline for Inverse HbBoPs.

Combines:
1. APE instruction generation (or loading from cache)
2. VAE training with KL annealing
3. Hyperband with successive halving
4. GP training ready for InvBO inference

Self-contained - no imports from other modules outside inverse_hbbops/.
"""

import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from tqdm import tqdm

from inverse_hbbops.encoder import GTRInstructionEncoder, InstructionVAE, VAEWithAdapter
from inverse_hbbops.gp import GPWithEI
from inverse_hbbops.hyperband import InverseHbBoPs, HyperbandConfig
from inverse_hbbops.instruction import InstructionOnlyPrompt


@dataclass
class TrainingConfig:
    """Configuration for Inverse HbBoPs training."""

    # APE generation
    ape_num_instructions: int = 1000
    ape_model: str = "Qwen/Qwen2.5-7B-Instruct"
    ape_backend: str = "vllm"
    ape_cache_path: str = "inverse_hbbops/data/ape_instructions.json"

    # Validation data path
    validation_path: str = "hbbops_improved_2/data/validation.json"

    # VAE training
    vae_beta: float = 0.02  # KL regularization weight (matches run.py CLI default)
    vae_epochs: int = 10000
    vae_annealing_epochs: int = 500
    vae_patience: int = 500
    vae_lr: float = 0.0003
    vae_batch_size: int = 64
    vae_hidden_dim: int = 256  # Hidden layer size
    latent_dim: int = 10
    embedding_dim: int = 768

    # Hyperband
    bmin: int = 10
    eta: float = 2.0
    random_interleaving_prob: float = 0.1
    top_fidelity_pct: float = 0.75

    # GP training (synced with invbo_decoder)
    gp_epochs: int = 10000
    gp_lr: float = 0.01
    gp_patience: int = 100

    # Device
    device: str = "cuda"

    # Seed
    seed: int = 42


class APEGenerator:
    """Simple APE instruction generator (self-contained).

    Generates diverse instructions using style-constrained prompts.
    """

    STYLE_CATEGORIES = {
        "minimalist": {
            "description": "Ultra-short, terse (1-5 words only)",
            "examples": ["Solve:", "Answer:", "Calculate.", "Find:"],
            "temperature": 1.0,
        },
        "direct_command": {
            "description": "Short imperative commands (6-15 words)",
            "examples": [
                "Find the answer to this problem.",
                "Calculate the result step by step.",
            ],
            "temperature": 0.9,
        },
        "chain_of_thought": {
            "description": "Step-by-step reasoning triggers",
            "examples": [
                "Let's think step by step.",
                "Break this down step by step.",
            ],
            "temperature": 0.8,
        },
        "pedagogical": {
            "description": "Patient teacher/tutor persona",
            "examples": [
                "You are a patient tutor. Walk through this problem step by step.",
                "As a helpful teacher, guide the solution process.",
            ],
            "temperature": 0.9,
        },
        "analytical": {
            "description": "Focus on logical structure and analysis",
            "examples": [
                "Analyze the logical structure of this problem.",
                "Use deductive reasoning to solve this.",
            ],
            "temperature": 0.8,
        },
    }

    def __init__(self, model: str, backend: str):
        self.model = model
        self.backend = backend
        self._client = None
        self.total_calls: int = 0  # Track LLM calls for APE generation

    def _get_client(self):
        if self._client is None:
            from src.llm_client import create_llm_client
            print(f"Initializing APE LLM client: {self.model}")
            self._client = create_llm_client(self.model, self.backend)
        return self._client

    def _build_prompt(self, style_config: dict, examples: List[dict]) -> str:
        """Build generation prompt with task examples."""
        qa_text = "\n\n".join([
            f"Input: {ex['question']}\nOutput: {ex['answer']}"
            for ex in examples[:3]
        ])

        style_examples = "\n".join([f"  - {ex}" for ex in style_config.get("examples", [])[:2]])

        return f"""Below are examples of a task. Study them to understand what the task involves.

TASK EXAMPLES:
{qa_text}

---

Generate an instruction that would help someone complete similar tasks.

STYLE: {style_config["description"]}
EXAMPLES:
{style_examples}

IMPORTANT:
- Do NOT mention specific details from the examples above
- Create a GENERAL instruction applicable to this type of task
- Output ONLY the instruction, nothing else

Your instruction:"""

    def generate(
        self,
        validation_data: List[dict],
        num_instructions: int = 1000,
        verbose: bool = True,
    ) -> List[str]:
        """Generate diverse instructions."""
        client = self._get_client()
        instructions = set()
        num_per_style = max(10, num_instructions // len(self.STYLE_CATEGORIES))

        if verbose:
            print(f"\nGenerating {num_instructions} diverse instructions...")

        for style_name, style_config in self.STYLE_CATEGORIES.items():
            if verbose:
                print(f"  [{style_name}] Generating {num_per_style} instructions...")

            temperature = style_config.get("temperature", 1.0)
            batch_size = 10
            attempts = 0
            max_attempts = (num_per_style * 3) // batch_size + 2

            while len([i for i in instructions if i.startswith(f"[{style_name}]")]) < num_per_style and attempts < max_attempts:
                attempts += 1

                prompts = []
                for _ in range(batch_size):
                    examples = random.sample(validation_data, min(3, len(validation_data)))
                    prompts.append(self._build_prompt(style_config, examples))

                responses = client.generate_batch(prompts, max_tokens=100, temperature=temperature)
                self.total_calls += len(prompts)  # Track LLM calls

                for response in responses:
                    instruction = self._parse_instruction(response)
                    if instruction:
                        instructions.add(instruction)

        result = list(instructions)[:num_instructions]
        if verbose:
            print(f"  Generated {len(result)} unique instructions")

        return result

    def _parse_instruction(self, response: str) -> Optional[str]:
        """Parse and clean instruction from LLM response."""
        if not response:
            return None

        instruction = response.strip()

        # Remove common prefixes
        for prefix in ["Instruction:", "Here is an instruction:", "Answer:", "Output:"]:
            if instruction.lower().startswith(prefix.lower()):
                instruction = instruction[len(prefix):].strip()

        # Remove quotes if wrapped
        if len(instruction) > 2 and instruction[0] in '"\'':
            if instruction[-1] == instruction[0]:
                instruction = instruction[1:-1]

        # Take only first line
        instruction = instruction.split("\n")[0].strip()

        if len(instruction) < 1 or len(instruction) > 500:
            return None

        return instruction

    def generate_or_load(
        self,
        cache_path: str,
        validation_data: List[dict],
        num_instructions: int = 1000,
        force_regenerate: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        """Generate or load from cache."""
        cache_file = Path(cache_path)

        if cache_file.exists() and not force_regenerate:
            if verbose:
                print(f"Loading cached instructions from {cache_path}...")
            with open(cache_path, "r") as f:
                data = json.load(f)
            instructions = data if isinstance(data, list) else data.get("instructions", data)
            if verbose:
                print(f"  Loaded {len(instructions)} instructions")
            return instructions[:num_instructions]

        # Generate new
        instructions = self.generate(validation_data, num_instructions, verbose)

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"instructions": instructions}, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"  Saved to {cache_path}")

        return instructions


class InverseHbBoPsTrainer:
    """Complete training pipeline for Inverse HbBoPs.

    Pipeline:
    1. Generate/load instructions via APE
    2. Train VAE on instruction embeddings
    3. Run Hyperband with VAEWithAdapter as GP feature extractor
    4. Output: trained VAE + GP ready for InvBO inference
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Initialize GTR encoder
        print("Loading GTR encoder...")
        self.gtr = GTRInstructionEncoder(device=str(self.device))

        # Components (initialized during training)
        self.vae: Optional[InstructionVAE] = None
        self.vae_with_adapter: Optional[VAEWithAdapter] = None
        self.gp: Optional[GPWithEI] = None
        self.hyperband: Optional[InverseHbBoPs] = None

        # Data
        self.instructions: List[str] = []
        self.validation_data: List[dict] = []
        self.instruction_embeddings: Dict[int, torch.Tensor] = {}

        # Grid data (for load_from_grid mode)
        self.grid_data: Optional[List[dict]] = None
        self.grid_error_rates: List[float] = []

        # LLM call counters
        self.total_llm_calls: int = 0  # Hyperband LLM calls
        self.ape_llm_calls: int = 0    # APE generation LLM calls

        # Training stats (for detailed logging)
        self.vae_stats: Dict[str, Any] = {}
        self.gp_stats: Dict[str, Any] = {}

    def load_validation_data(self, verbose: bool = True) -> List[dict]:
        """Load validation data for evaluation."""
        if verbose:
            print(f"Loading validation data from {self.config.validation_path}...")

        with open(self.config.validation_path, "r") as f:
            self.validation_data = json.load(f)

        if verbose:
            print(f"  Loaded {len(self.validation_data)} validation samples")

        return self.validation_data

    def generate_instructions(
        self,
        num_instructions: Optional[int] = None,
        force_regenerate: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        """Generate or load instructions via APE."""
        if not self.validation_data:
            self.load_validation_data(verbose=verbose)

        num = num_instructions or self.config.ape_num_instructions

        generator = APEGenerator(
            model=self.config.ape_model,
            backend=self.config.ape_backend,
        )

        self.instructions = generator.generate_or_load(
            cache_path=self.config.ape_cache_path,
            validation_data=self.validation_data,
            num_instructions=num,
            force_regenerate=force_regenerate,
            verbose=verbose,
        )

        # Track APE LLM calls (only non-zero if we actually generated)
        self.ape_llm_calls = generator.total_calls
        if verbose and self.ape_llm_calls > 0:
            print(f"  APE generation used {self.ape_llm_calls} LLM calls")

        # Pre-compute embeddings
        if verbose:
            print("Encoding instructions with GTR...")
        for idx, inst in enumerate(self.instructions):
            self.instruction_embeddings[idx] = self.gtr.encode_tensor(inst)

        if verbose:
            print(f"  Encoded {len(self.instruction_embeddings)} instructions")

        return self.instructions

    def train_vae(self, verbose: bool = True) -> InstructionVAE:
        """Train VAE on instruction embeddings with KL annealing."""
        if not self.instruction_embeddings:
            raise RuntimeError("No instructions. Call generate_instructions() first.")

        if verbose:
            print("\n" + "=" * 60)
            print("Training VAE with KL Annealing")
            print("=" * 60)

        # Prepare embeddings
        embeddings = torch.stack(list(self.instruction_embeddings.values())).to(self.device)

        if verbose:
            print(f"  Training on {embeddings.shape[0]} embeddings")

        # Initialize VAE
        self.vae = InstructionVAE(
            input_dim=self.config.embedding_dim,
            latent_dim=self.config.latent_dim,
            beta=self.config.vae_beta,
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.config.vae_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.vae_epochs * 2, eta_min=1e-4
        )

        best_recon = float("inf")
        patience_counter = 0

        if verbose:
            print(f"  KL annealing: β = 0 → {self.config.vae_beta} over {self.config.vae_annealing_epochs} epochs")

        self.vae.train()

        for epoch in range(self.config.vae_epochs):
            # KL annealing
            if epoch < self.config.vae_annealing_epochs:
                current_beta = self.config.vae_beta * (epoch / self.config.vae_annealing_epochs)
            else:
                current_beta = self.config.vae_beta

            # Shuffle data
            perm = torch.randperm(embeddings.shape[0], device=self.device)
            X_shuffled = embeddings[perm]

            epoch_losses = []
            epoch_cosine = []

            for i in range(0, len(embeddings), self.config.vae_batch_size):
                batch = X_shuffled[i:i + self.config.vae_batch_size]

                optimizer.zero_grad()
                x_recon, mu, log_var, z = self.vae(batch)
                loss, loss_dict = self.vae.loss(batch, x_recon, mu, log_var, beta=current_beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(loss_dict["recon"])
                epoch_cosine.append(loss_dict["cosine_mean"])

            scheduler.step()

            avg_recon = sum(epoch_losses) / len(epoch_losses)
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

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}: recon={avg_recon:.4f}, cosine={avg_cosine:.4f}, β={current_beta:.4f}")

        # Store VAE training stats
        self.vae_stats = {
            "epochs_trained": epoch + 1,
            "final_recon_loss": float(avg_recon),
            "final_cosine_similarity": float(avg_cosine),
            "final_beta": float(current_beta),
            "early_stopped": patience_counter >= self.config.vae_patience,
            "num_embeddings": len(embeddings),
            "latent_dim": self.config.latent_dim,
        }

        if verbose:
            print(f"  VAE training complete (epochs={epoch + 1}, final cosine={avg_cosine:.4f})")

        # Create VAEWithAdapter
        self.vae_with_adapter = VAEWithAdapter(self.vae, self.config.latent_dim).to(self.device)

        return self.vae

    def run_hyperband(
        self,
        llm_evaluator: Callable[[InstructionOnlyPrompt, List[dict]], float],
        verbose: bool = True,
    ) -> Tuple[InstructionOnlyPrompt, float]:
        """Run Hyperband with VAEWithAdapter as GP feature extractor."""
        if self.vae_with_adapter is None:
            raise RuntimeError("VAE not trained. Call train_vae() first.")

        if not self.validation_data:
            self.load_validation_data(verbose=verbose)

        if verbose:
            print("\n" + "=" * 60)
            print("Running Hyperband")
            print("=" * 60)

        hb_config = HyperbandConfig(
            bmin=self.config.bmin,
            eta=self.config.eta,
            random_interleaving_prob=self.config.random_interleaving_prob,
            gp_epochs=self.config.gp_epochs,
            gp_lr=self.config.gp_lr,
            gp_patience=self.config.gp_patience,
            top_fidelity_pct=self.config.top_fidelity_pct,
            seed=self.config.seed,
        )

        self.hyperband = InverseHbBoPs(
            instructions=self.instructions,
            validation_data=self.validation_data,
            llm_evaluator=llm_evaluator,
            vae_with_adapter=self.vae_with_adapter,
            encoder=self.gtr,
            config=hb_config,
            device=str(self.device),
        )

        best_prompt, best_error = self.hyperband.run_hyperband(verbose=verbose)

        # Track LLM calls
        self.total_llm_calls += self.hyperband.total_llm_calls

        return best_prompt, best_error

    def get_gp_for_inference(self) -> GPWithEI:
        """Get trained GP ready for InvBO inference."""
        if self.hyperband is None:
            raise RuntimeError("Hyperband not run. Call run_hyperband() first.")

        self.gp = self.hyperband.get_gp_with_ei()

        # Store GP stats if available
        if hasattr(self.gp, 'training_stats') and self.gp.training_stats:
            self.gp_stats = self.gp.training_stats.copy()
            if self.gp.y_best is not None:
                self.gp_stats["best_observed_error"] = float(self.gp.y_best)

        return self.gp

    def train(
        self,
        llm_evaluator: Callable[[InstructionOnlyPrompt, List[dict]], float],
        num_instructions: Optional[int] = None,
        force_regenerate_ape: bool = False,
        verbose: bool = True,
    ) -> Tuple[GPWithEI, InstructionVAE]:
        """Run complete training pipeline.

        1. Generate/load instructions
        2. Train VAE
        3. Run Hyperband
        4. Return GP + VAE for inference

        Returns:
            (gp, vae) tuple ready for InvBO inference
        """
        # Load validation data
        self.load_validation_data(verbose=verbose)

        # Generate or load instructions
        self.generate_instructions(
            num_instructions=num_instructions,
            force_regenerate=force_regenerate_ape,
            verbose=verbose,
        )

        # Train VAE
        self.train_vae(verbose=verbose)

        # Run Hyperband
        best_prompt, best_error = self.run_hyperband(llm_evaluator, verbose=verbose)

        if verbose:
            print(f"\nBest prompt found:")
            print(f"  Error: {best_error:.4f}")
            print(f"  Instruction:\n{best_prompt.instruction}")

        # Get GP for inference
        gp = self.get_gp_for_inference()

        if verbose:
            print(f"\nTotal LLM calls: {self.total_llm_calls}")
            print("Training complete! GP + VAE ready for InvBO inference.")

        return gp, self.vae

    def save(self, path: str) -> None:
        """Save trained models."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save VAE
        torch.save(self.vae.state_dict(), save_dir / "vae.pt")

        # Save config
        torch.save(self.config, save_dir / "config.pt")

        # Save instructions
        with open(save_dir / "instructions.json", "w") as f:
            json.dump({"instructions": self.instructions}, f, ensure_ascii=False, indent=2)

        print(f"Models saved to {save_dir}")

    def load(self, path: str) -> None:
        """Load trained models."""
        save_dir = Path(path)

        # Load config
        self.config = torch.load(save_dir / "config.pt", weights_only=False)

        # Load instructions
        with open(save_dir / "instructions.json", "r") as f:
            data = json.load(f)
        self.instructions = data["instructions"]

        # Re-encode instructions
        for idx, inst in enumerate(self.instructions):
            self.instruction_embeddings[idx] = self.gtr.encode_tensor(inst)

        # Load VAE
        self.vae = InstructionVAE(
            input_dim=self.config.embedding_dim,
            latent_dim=self.config.latent_dim,
            beta=self.config.vae_beta,
        ).to(self.device)
        self.vae.load_state_dict(torch.load(save_dir / "vae.pt", map_location=self.device))

        # Create VAEWithAdapter
        self.vae_with_adapter = VAEWithAdapter(self.vae, self.config.latent_dim).to(self.device)

        print(f"Models loaded from {save_dir}")

    def load_from_grid(
        self,
        grid_path: str,
        top_k: Optional[int] = None,
        instructions_path: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Tuple[str, float]]:
        """Load pre-evaluated instructions from grid.

        Skips APE generation and Hyperband - loads directly from pre-evaluated grid.

        Args:
            grid_path: Path to grid JSONL file
            top_k: Number of top instructions to load (None = all)
            instructions_path: Path to instructions file (for grids with instruction_id)
            verbose: Print progress

        Returns:
            List of (instruction, error_rate) tuples sorted by error
        """
        if verbose:
            print(f"\nLoading from grid: {grid_path}")

        # Load grid data
        grid_data = []
        with open(grid_path, "r") as f:
            for line in f:
                grid_data.append(json.loads(line))

        if verbose:
            print(f"  Loaded {len(grid_data)} entries from grid")

        # Check if grid has instruction_text or instruction_id
        has_text = "instruction_text" in grid_data[0] if grid_data else False

        # If grid only has IDs, load instruction texts from file
        if not has_text:
            if instructions_path is None:
                # Auto-detect instructions file in same directory
                grid_dir = Path(grid_path).parent
                instructions_path = str(grid_dir / "instructions_25.txt")
                if verbose:
                    print(f"  Auto-detected instructions path: {instructions_path}")

            # Load instruction texts from file
            instruction_texts = self._load_instructions_file(instructions_path)
            if verbose:
                print(f"  Loaded {len(instruction_texts)} instruction texts")

            # Add instruction_text to grid_data
            for d in grid_data:
                inst_id = d["instruction_id"]
                if inst_id < len(instruction_texts):
                    d["instruction_text"] = instruction_texts[inst_id]
                else:
                    d["instruction_text"] = f"[Unknown instruction {inst_id}]"

        # Aggregate by instruction (average error rates across exemplars)
        # Grid has instruction+exemplar combinations, but GP uses only instruction embeddings
        instruction_errors = {}  # instruction_text -> list of error_rates
        for d in grid_data:
            inst = d["instruction_text"]
            if inst not in instruction_errors:
                instruction_errors[inst] = []
            instruction_errors[inst].append(d["error_rate"])

        # Compute mean error for each unique instruction
        aggregated = []
        for inst, errors in instruction_errors.items():
            avg_error = sum(errors) / len(errors)
            aggregated.append({"instruction_text": inst, "error_rate": avg_error})

        if verbose:
            print(f"  Aggregated {len(grid_data)} entries to {len(aggregated)} unique instructions")

        # Sort by error_rate (ascending - best first)
        aggregated.sort(key=lambda x: x["error_rate"])

        # Take top-k if specified
        if top_k is not None:
            aggregated = aggregated[:top_k]
            if verbose:
                print(f"  Selected top {top_k} instructions")

        # Store instructions and their error rates
        self.instructions = [d["instruction_text"] for d in aggregated]
        self.grid_error_rates = [d["error_rate"] for d in aggregated]

        # Pre-compute embeddings
        if verbose:
            print("Encoding instructions with GTR...")
        for idx, inst in enumerate(self.instructions):
            self.instruction_embeddings[idx] = self.gtr.encode_tensor(inst)

        if verbose:
            print(f"  Encoded {len(self.instruction_embeddings)} instructions")
            print(f"  Best error in grid: {self.grid_error_rates[0]:.4f}")
            print(f"  Worst error in selection: {self.grid_error_rates[-1]:.4f}")

        # Store grid data for GP training
        self.grid_data = grid_data

        return list(zip(self.instructions, self.grid_error_rates))

    def _load_instructions_file(self, path: str) -> List[str]:
        """Load instructions from numbered text file.

        Format:
        # Comment lines start with #
        1. First instruction text
        2. Second instruction text
        ...

        Args:
            path: Path to instructions text file

        Returns:
            List of instruction texts (indexed by instruction_id)
        """
        instructions = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse numbered lines: "1. text" -> "text"
                if line[0].isdigit():
                    # Find the first ". " and take everything after
                    dot_idx = line.find(". ")
                    if dot_idx > 0:
                        instructions.append(line[dot_idx + 2:])
        return instructions

    def load_diverse_instructions(
        self,
        diverse_path: str,
        verbose: bool = True,
    ) -> List[str]:
        """Load diverse instructions for VAE training.

        This allows training VAE on a larger, more diverse set of instructions
        while using only top-k from grid for GP training.

        Args:
            diverse_path: Path to diverse instructions JSON file
            verbose: Print progress

        Returns:
            List of diverse instructions
        """
        if verbose:
            print(f"\nLoading diverse instructions: {diverse_path}")

        with open(diverse_path, "r") as f:
            data = json.load(f)

        # Handle both list and {"instructions": [...]} format
        if isinstance(data, list):
            diverse_instructions = data
        else:
            diverse_instructions = data.get("instructions", data)

        if verbose:
            print(f"  Loaded {len(diverse_instructions)} diverse instructions")

        # Store for VAE training (separate from grid instructions)
        self.diverse_instructions = diverse_instructions

        # Pre-compute embeddings for diverse instructions
        if verbose:
            print("Encoding diverse instructions with GTR...")

        # Use a separate embedding dict to avoid overwriting grid embeddings
        self.diverse_embeddings = {}
        for idx, inst in enumerate(diverse_instructions):
            self.diverse_embeddings[idx] = self.gtr.encode_tensor(inst)

        if verbose:
            print(f"  Encoded {len(self.diverse_embeddings)} diverse instructions")

        return diverse_instructions

    def train_vae_on_diverse(self, verbose: bool = True) -> InstructionVAE:
        """Train VAE on diverse instruction embeddings.

        Uses diverse_embeddings instead of instruction_embeddings.
        Must call load_diverse_instructions() first.
        """
        if not hasattr(self, 'diverse_embeddings') or not self.diverse_embeddings:
            raise RuntimeError("No diverse instructions. Call load_diverse_instructions() first.")

        if verbose:
            print("\n" + "=" * 60)
            print("Training VAE on Diverse Instructions")
            print("=" * 60)

        # Use diverse embeddings for training
        embeddings = torch.stack(list(self.diverse_embeddings.values())).to(self.device)

        if verbose:
            print(f"  Training on {embeddings.shape[0]} diverse embeddings")

        # Initialize VAE
        self.vae = InstructionVAE(
            input_dim=self.config.embedding_dim,
            latent_dim=self.config.latent_dim,
            beta=self.config.vae_beta,
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.config.vae_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.vae_epochs * 2, eta_min=1e-4
        )

        best_recon = float("inf")
        patience_counter = 0

        if verbose:
            print(f"  KL annealing: β = 0 → {self.config.vae_beta} over {self.config.vae_annealing_epochs} epochs")

        self.vae.train()

        for epoch in range(self.config.vae_epochs):
            # KL annealing
            if epoch < self.config.vae_annealing_epochs:
                current_beta = self.config.vae_beta * (epoch / self.config.vae_annealing_epochs)
            else:
                current_beta = self.config.vae_beta

            # Shuffle data
            perm = torch.randperm(embeddings.shape[0], device=self.device)
            X_shuffled = embeddings[perm]

            epoch_losses = []
            epoch_cosine = []

            for i in range(0, embeddings.shape[0], self.config.vae_batch_size):
                batch = X_shuffled[i:i + self.config.vae_batch_size]

                optimizer.zero_grad()
                recon, mu, log_var, z = self.vae(batch)

                # Losses
                recon_loss = 1 - F.cosine_similarity(recon, batch, dim=1).mean()
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch.shape[0]
                loss = recon_loss + current_beta * kl_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_cosine.append(1 - recon_loss.item())

            scheduler.step()

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_cosine = sum(epoch_cosine) / len(epoch_cosine)

            # Track reconstruction (not total loss) for early stopping
            if avg_cosine > (1 - best_recon + 0.001):
                best_recon = 1 - avg_cosine
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 500 == 0:
                print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, cosine={avg_cosine:.4f}, β={current_beta:.4f}")

            if patience_counter >= self.config.vae_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        self.vae.eval()

        # Create VAEWithAdapter
        self.vae_with_adapter = VAEWithAdapter(self.vae, self.config.latent_dim).to(self.device)

        if verbose:
            print(f"  Final cosine similarity: {avg_cosine:.4f}")
            print(f"  VAEWithAdapter created")

        return self.vae

    def train_gp_from_grid(self, verbose: bool = True) -> GPWithEI:
        """Train GP directly from grid data (no Hyperband).

        Must call load_from_grid() and train_vae() first.

        Returns:
            Trained GPWithEI ready for inference
        """
        if not hasattr(self, 'grid_data') or self.grid_data is None:
            raise RuntimeError("No grid data. Call load_from_grid() first.")

        if self.vae_with_adapter is None:
            raise RuntimeError("VAE not trained. Call train_vae() first.")

        if verbose:
            print("\n" + "=" * 60)
            print("Training GP from Grid Data")
            print("=" * 60)

        # Prepare embeddings and error rates
        embeddings = torch.stack([
            self.instruction_embeddings[idx]
            for idx in range(len(self.instructions))
        ]).to(self.device)

        error_rates = torch.tensor(
            self.grid_error_rates,
            dtype=torch.float32,
            device=self.device
        )

        if verbose:
            print(f"  Training on {len(embeddings)} grid samples")
            print(f"  Error range: [{error_rates.min():.4f}, {error_rates.max():.4f}]")

        # Create and train GP
        self.gp = GPWithEI(
            device=str(self.device),
            latent_dim=self.config.latent_dim,
        )

        # Set feature extractor (VAEWithAdapter)
        self.gp.feature_extractor = self.vae_with_adapter

        # Set training data
        self.gp.set_training_data(embeddings, error_rates)

        # Train GP
        self.gp.train(
            epochs=self.config.gp_epochs,
            lr=self.config.gp_lr,
            patience=self.config.gp_patience,
            verbose=verbose,
        )

        # Store GP training stats
        self.gp_stats = self.gp.training_stats.copy()
        self.gp_stats["best_observed_error"] = float(self.gp.y_best)
        self.gp_stats["error_range"] = {
            "min": float(error_rates.min()),
            "max": float(error_rates.max()),
        }

        if verbose:
            print(f"  GP trained, best error: {self.gp.y_best:.4f}")

        return self.gp

    def get_best_from_grid(self) -> Tuple[InstructionOnlyPrompt, float]:
        """Get best prompt from loaded grid data.

        Returns:
            (best_prompt, best_error) tuple
        """
        if not hasattr(self, 'grid_data') or not self.grid_data:
            raise RuntimeError("No grid data. Call load_from_grid() first.")

        best = self.grid_data[0]  # Already sorted by error
        best_prompt = InstructionOnlyPrompt(
            instruction=best["instruction_text"],
            instruction_id=0,
        )
        return best_prompt, best["error_rate"]
