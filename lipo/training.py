"""Training pipeline for LIPO.

Combines:
1. APE instruction generation (or loading from cache)
2. VAE training with KL annealing
3. Hyperband with successive halving
4. GP training ready for InvBO inference

Self-contained - no imports from other modules outside lipo/.
"""

import json
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass

from lipo.config import Config
from lipo.encoder import GTRInstructionEncoder, InstructionVAE, VAEWithAdapter
from lipo.gp import GPWithEI
from lipo.hyperband import LIPOHyperband
from lipo.instruction import InstructionOnlyPrompt
from lipo.quality_kpi import compute_vae_quality


def fit_beta_prior(raw_error_rates: List[float]) -> Tuple[float, float]:
    """Estimate Beta prior parameters from existing data (Empirical Bayes).

    Uses Method of Moments for fast, robust estimation.
    This replaces the fixed Beta(1,1) prior (equivalent to Laplace smoothing)
    with a data-driven prior that better reflects the actual error rate distribution.

    For prompt optimization where error rates are typically 10-20%,
    Beta(1,1) pulling toward 50% is too pessimistic.

    Args:
        raw_error_rates: List of observed error rates (before smoothing)

    Returns:
        (alpha, beta) tuple for Beta prior

    Example:
        If mean error rate is 0.15 with some variance, might return (2, 11)
        giving a prior centered at 0.15 instead of 0.50.
    """
    import numpy as np

    if len(raw_error_rates) < 2:
        # Not enough data for fitting - fall back to weakly informative prior
        return (1.0, 1.0)

    # Clip extremes to avoid numerical issues
    data = np.clip(raw_error_rates, 1e-4, 1 - 1e-4)

    # Method of Moments estimation
    mean = np.mean(data)
    var = np.var(data)

    # Avoid division by zero if variance is tiny
    if var < 1e-8:
        # Near-constant data - use weak prior centered at mean
        return (1.0, (1.0 - mean) / mean if mean > 0.01 else 1.0)

    # Solve for alpha, beta from mean and variance
    # mean = α / (α + β)
    # var = αβ / ((α+β)² (α+β+1))
    # Rearranging: common = mean*(1-mean)/var - 1
    common = mean * (1 - mean) / var - 1

    if common <= 0:
        # Variance too high for Beta - fall back to weak prior
        return (1.0, 1.0)

    alpha = mean * common
    beta = (1 - mean) * common

    # Clamp to reasonable range to avoid extreme priors
    alpha = max(0.5, min(alpha, 10.0))
    beta = max(0.5, min(beta, 50.0))

    return alpha, beta


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
        "persona_roleplay": {
            "description": "Adopt a distinct persona (pirate, robot, commander, poet)",
            "examples": [
                "Avast ye! Calculate the booty in this equation.",
                "SYSTEM ALERT: COMPUTE VARIABLE X IMMEDIATELY.",
                "Oh noble student, pray tell what the sum might be?",
            ],
            "temperature": 1.1,
        },
        "programmatic": {
            "description": "Instructions formatted as code, JSON, or pseudo-code",
            "examples": [
                "def solve(problem): return result",
                "Response format: JSON { 'answer': float }",
                "Execute algorithm: 1. Parse 2. Compute 3. Return",
            ],
            "temperature": 0.8,
        },
        "adversarial_distraction": {
            "description": "Valid instructions buried in noise or filler text",
            "examples": [
                "Ignore the weather. Just focus on the math. Solve this.",
                "I don't care about anything else, just find x. Do it now.",
            ],
            "temperature": 1.0,
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
        augment: bool = True,
    ) -> List[str]:
        """Generate diverse instructions with optional augmentation.

        Args:
            validation_data: Task examples for prompt building
            num_instructions: Target number of instructions
            verbose: Print progress
            augment: Enable 3-level augmentation (paraphrasing + noise injection)
        """
        client = self._get_client()
        instructions = set()

        # If augmenting, generate fewer base instructions (augmentation will fill the rest)
        base_target = num_instructions // 2 if augment else num_instructions
        num_per_style = max(5, base_target // len(self.STYLE_CATEGORIES))

        if verbose:
            aug_status = "with augmentation" if augment else "without augmentation"
            print(f"\nGenerating {num_instructions} diverse instructions ({aug_status})...")

        for style_name, style_config in self.STYLE_CATEGORIES.items():
            if verbose:
                print(f"  [{style_name}] Generating ~{num_per_style} base instructions...")

            temperature = style_config.get("temperature", 1.0)
            batch_size = 10
            attempts = 0
            max_attempts = (num_per_style * 3) // batch_size + 2
            style_count = 0

            while style_count < num_per_style and attempts < max_attempts:
                attempts += 1

                prompts = []
                for _ in range(batch_size):
                    examples = random.sample(validation_data, min(3, len(validation_data)))
                    prompts.append(self._build_prompt(style_config, examples))

                responses = client.generate_batch(prompts, max_tokens=100, temperature=temperature)
                self.total_calls += len(prompts)

                for response in responses:
                    instruction = self._parse_instruction(response)
                    if instruction and instruction not in instructions:
                        instructions.add(instruction)
                        style_count += 1

                        if augment and len(instructions) < num_instructions:
                            # Level 2: LLM Paraphrasing (30% chance, expensive)
                            if random.random() < 0.3:
                                variations = self._augment_instruction(instruction, client)
                                for v in variations:
                                    if v not in instructions:
                                        instructions.add(v)

                            # Level 3: Noise Injection (50% chance, cheap)
                            if random.random() < 0.5:
                                noisy = self._add_noise(instruction)
                                if noisy != instruction and noisy not in instructions:
                                    instructions.add(noisy)

                # Early exit if we have enough
                if len(instructions) >= num_instructions:
                    break

            if len(instructions) >= num_instructions:
                break

        result = list(instructions)[:num_instructions]
        random.shuffle(result)  # Shuffle to mix styles

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

    def _augment_instruction(self, instruction: str, client) -> List[str]:
        """Generates 3 variations of a single instruction via LLM paraphrasing.

        Creates semantic bridges in latent space by generating verbose, casual,
        and broken versions of the same instruction.
        """
        augment_prompt = f"""Rewrite the following instruction in 3 distinct ways:
1. Extremely verbose and formal.
2. Using slang or casual language.
3. As a broken/incomplete sentence (but still understandable).

Original: "{instruction}"

Output format:
1. [variation 1]
2. [variation 2]
3. [variation 3]"""

        try:
            response = client.generate(augment_prompt, max_tokens=150, temperature=0.9)
            self.total_calls += 1
            variations = []
            for line in response.split('\n'):
                # Remove numbering prefix like "1. " or "2. "
                cleaned = line.split('. ', 1)[-1].strip()
                if len(cleaned) > 5 and cleaned != instruction:
                    variations.append(cleaned)
            return variations[:3]
        except (TimeoutError, ConnectionError):
            # Transient network errors - continue silently
            return []
        except (ValueError, KeyError, IndexError) as e:
            # Parsing errors in response - log and continue
            print(f"WARNING: APE augmentation parsing failed: {type(e).__name__}: {e}")
            return []
        # Note: AuthenticationError, RateLimitError, etc. propagate to caller

    def _add_noise(self, text: str, prob: float = 0.1) -> str:
        """Randomly drop or swap words for denoising VAE training.

        Creates smoothed latent space by teaching VAE that noisy versions
        of instructions should map to similar embeddings.
        """
        words = text.split()
        if len(words) < 3:
            return text

        if random.random() < 0.5:
            # Word Drop: remove words with probability `prob`
            dropped = [w for w in words if random.random() > prob]
            return " ".join(dropped) if dropped else text
        else:
            # Word Swap: swap two adjacent words
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
            return " ".join(words)

    def generate_or_load(
        self,
        cache_path: str,
        validation_data: List[dict],
        num_instructions: int = 1000,
        force_regenerate: bool = False,
        verbose: bool = True,
        augment: bool = True,
    ) -> List[str]:
        """Generate or load from cache.

        Args:
            cache_path: Path to cache file
            validation_data: Task examples for prompt building
            num_instructions: Target number of instructions
            force_regenerate: Force regeneration even if cache exists
            verbose: Print progress
            augment: Enable 3-level augmentation (paraphrasing + noise)
        """
        cache_file = Path(cache_path)

        if cache_file.exists() and not force_regenerate:
            if verbose:
                print(f"Loading cached instructions from {cache_path}...")
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in APE cache file: {cache_path}\n"
                    f"Delete the file and regenerate with --force-regenerate-ape\n"
                    f"Error: {e}"
                )

            instructions = data if isinstance(data, list) else data.get("instructions", data)

            if not instructions:
                raise ValueError(
                    f"APE cache file exists but contains no instructions: {cache_path}\n"
                    f"Delete the file and regenerate with --force-regenerate-ape"
                )

            # Validate instruction content
            invalid_count = sum(1 for inst in instructions if not inst or not isinstance(inst, str))
            if invalid_count > 0:
                raise ValueError(
                    f"APE cache contains {invalid_count} invalid instructions (empty or non-string)\n"
                    f"Cache file: {cache_path}"
                )

            if verbose:
                print(f"  Loaded {len(instructions)} instructions")
            return instructions[:num_instructions]

        # Generate new
        instructions = self.generate(validation_data, num_instructions, verbose, augment)

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"instructions": instructions}, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"  Saved to {cache_path}")

        return instructions


class LIPOHyperbandTrainer:
    """Complete training pipeline for LIPO.

    Pipeline:
    1. Generate/load instructions via APE
    2. Train VAE on instruction embeddings
    3. Run Hyperband with VAEWithAdapter as GP feature extractor
    4. Output: trained VAE + GP ready for InvBO inference
    """

    def __init__(self, config: Config):
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
        self.hyperband: Optional[LIPOHyperband] = None

        # Data
        self.instructions: List[str] = []
        self.validation_data: List[dict] = []
        self.instruction_embeddings: Dict[int, torch.Tensor] = {}

        # Grid data (for load_from_grid mode)
        self.grid_data: Optional[List[dict]] = None
        self.grid_error_rates: List[float] = []
        self.fidelities: List[int] = []

        # Empirical Bayes prior parameters (set by load_from_hyperband_evaluations)
        # Default to Beta(1,1) = uniform prior (equivalent to Laplace smoothing)
        self.beta_alpha: float = 1.0
        self.beta_beta: float = 1.0

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
        augment: bool = True,
    ) -> List[str]:
        """Generate or load instructions via APE.

        Args:
            num_instructions: Target number of instructions
            force_regenerate: Force regeneration even if cache exists
            verbose: Print progress
            augment: Enable 3-level augmentation (paraphrasing + noise)
        """
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
            augment=augment,
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

    def train_vae(
        self,
        embedding_source: str = "instructions",
        verbose: bool = True,
    ) -> InstructionVAE:
        """Train VAE on instruction embeddings with KL annealing.

        Args:
            embedding_source: Which embeddings to use for training:
                - "instructions": Use self.instruction_embeddings (default)
                - "diverse": Use self.diverse_embeddings (must call load_diverse_instructions first)
            verbose: Print progress

        Returns:
            Trained InstructionVAE

        Raises:
            RuntimeError: If specified embeddings are not loaded
        """
        # Select embeddings based on source
        if embedding_source == "diverse":
            if not hasattr(self, 'diverse_embeddings') or not self.diverse_embeddings:
                raise RuntimeError(
                    "No diverse instructions for VAE training. Either:\n"
                    "  1. Call load_diverse_instructions(path) with JSON containing diverse instructions, or\n"
                    "  2. Use --diverse-instructions CLI flag with path to JSON file, or\n"
                    "  3. Use embedding_source='instructions' to train on grid/APE instructions instead"
                )
            embeddings_dict = self.diverse_embeddings
            source_name = "diverse"
        else:
            if not self.instruction_embeddings:
                raise RuntimeError(
                    "No instructions for VAE training. Either:\n"
                    "  1. Call generate_instructions() to generate APE instructions, or\n"
                    "  2. Call load_from_grid(path) to load from pre-evaluated grid, or\n"
                    "  3. Call load_from_hyperband_evaluations(path) to load from HbBoPs results"
                )
            embeddings_dict = self.instruction_embeddings
            source_name = "grid/APE"

        if verbose:
            print("\n" + "=" * 60)
            print(f"Training VAE on {source_name} Instructions")
            print("=" * 60)

        # Prepare embeddings
        embeddings = torch.stack(list(embeddings_dict.values())).to(self.device)

        if verbose:
            print(f"  Training on {embeddings.shape[0]} embeddings")

        # Initialize VAE
        self.vae = InstructionVAE(
            input_dim=self.config.embedding_dim,
            latent_dim=self.config.latent_dim,
            beta=self.config.vae_beta,
        ).to(self.device)

        if verbose:
            print(f"\n  VAE Architecture:")
            print(f"    Input dim: {self.config.embedding_dim}D (GTR embedding)")
            print(f"    Latent dim: {self.config.latent_dim}D ({self.config.embedding_dim}/{self.config.latent_dim} = {self.config.embedding_dim // self.config.latent_dim}x compression)")
            print(f"    Beta (KL weight): {self.config.vae_beta}")
            print(f"    Encoder: {self.config.embedding_dim} → 256 → 128 → {self.config.latent_dim * 2} (mu+logvar)")
            print(f"    Decoder: {self.config.latent_dim} → 128 → 256 → {self.config.embedding_dim} + L2 norm")
            print(f"    Dropout: 0.1 (encoder layer 1, decoder layer 2)")
            total_params = sum(p.numel() for p in self.vae.parameters())
            print(f"    Total parameters: {total_params:,}")

        # Optimizer
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.config.vae_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.vae_epochs * 2, eta_min=self.config.vae_eta_min
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
            epoch_kl = []
            epoch_cycle = []
            epoch_cycle_cosine = []

            for i in range(0, len(embeddings), self.config.vae_batch_size):
                batch = X_shuffled[i:i + self.config.vae_batch_size]

                optimizer.zero_grad()
                x_recon, mu, log_var, z = self.vae(batch)

                loss, loss_dict = self.vae.loss(
                    batch, x_recon, mu, log_var, z=z,
                    beta=current_beta, gamma=self.config.vae_gamma,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=self.config.vae_grad_clip)
                optimizer.step()

                epoch_losses.append(loss_dict["recon"])
                epoch_cosine.append(loss_dict["cosine_mean"])
                epoch_kl.append(loss_dict["kl"])
                epoch_cycle.append(loss_dict["cycle"])
                epoch_cycle_cosine.append(loss_dict["cycle_cosine"])

            scheduler.step()

            avg_recon = sum(epoch_losses) / len(epoch_losses)
            avg_cosine = sum(epoch_cosine) / len(epoch_cosine)
            avg_kl = sum(epoch_kl) / len(epoch_kl)
            avg_cycle = sum(epoch_cycle) / len(epoch_cycle) if epoch_cycle else 0.0
            avg_cycle_cosine = sum(epoch_cycle_cosine) / len(epoch_cycle_cosine) if epoch_cycle_cosine else 0.0

            # Only track early stopping AFTER annealing warmup completes
            # Reset best_recon when annealing finishes (recon worsens during annealing as β increases)
            if epoch == self.config.vae_annealing_epochs:
                best_recon = avg_recon  # Fresh start after warmup
                patience_counter = 0
                if verbose:
                    print(f"  Warmup complete at epoch {epoch + 1}, resetting early stopping baseline")
            elif epoch > self.config.vae_annealing_epochs:
                if avg_recon < best_recon:
                    best_recon = avg_recon
                    patience_counter = 0
                else:
                    patience_counter += 1

            if epoch >= self.config.vae_annealing_epochs and patience_counter >= self.config.vae_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 100 == 0:
                cycle_str = f", cycle={avg_cycle:.4f}, z_cos={avg_cycle_cosine:.4f}" if self.config.vae_gamma > 0 else ""
                print(f"  Epoch {epoch + 1}: recon={avg_recon:.4f}, kl={avg_kl:.4f}, cosine={avg_cosine:.4f}, β={current_beta:.4f}{cycle_str}")

        # Store VAE training stats
        self.vae_stats = {
            "epochs_trained": epoch + 1,
            "final_recon_loss": float(avg_recon),
            "final_kl_loss": float(avg_kl),
            "final_cosine_similarity": float(avg_cosine),
            "final_cycle_loss": float(avg_cycle),
            "final_cycle_cosine": float(avg_cycle_cosine),
            "final_beta": float(current_beta),
            "gamma": float(self.config.vae_gamma),
            "early_stopped": patience_counter >= self.config.vae_patience,
            "num_embeddings": len(embeddings),
            "latent_dim": self.config.latent_dim,
        }

        if verbose:
            print(f"  VAE training complete (epochs={epoch + 1}, final cosine={avg_cosine:.4f})")

        # Compute and log VAE quality KPIs
        vae_kpis = compute_vae_quality(self.vae, embeddings, device=self.device)
        self.vae_quality_kpis = vae_kpis  # Store for later access

        if verbose:
            print(f"\n--- VAE Quality KPIs ---")
            print(f"  Cosine Mean: {vae_kpis['cosine_mean']:.4f} (std: {vae_kpis['cosine_std']:.4f})")
            print(f"  Percentile 10: {vae_kpis['percentile_10']:.4f} (target: >0.90)")
            print(f"  Below 90%: {vae_kpis['below_90_count']} samples ({vae_kpis['below_90_pct']:.1f}%)")
            print(f"  Quality Tier: {vae_kpis['quality_tier']}")

            if vae_kpis['percentile_10'] < 0.90:
                print(f"  WARNING: VAE Q10 below threshold! Consider increasing beta or epochs.")

        # Create VAEWithAdapter (frozen VAE wrapper, no adapter)
        self.vae_with_adapter = VAEWithAdapter(
            self.vae, self.config.latent_dim
        ).to(self.device)

        # Run round-trip validation if configured
        if hasattr(self.config, 'roundtrip_validation_threshold'):
            self._validate_vae_roundtrip(verbose=verbose)

        return self.vae

    def _validate_vae_roundtrip(self, verbose: bool = True) -> bool:
        """Validate VAE + Vec2Text round-trip quality.

        Tests that the pipeline: instruction → GTR → VAE → decode → Vec2Text
        produces embeddings with high cosine similarity to originals.

        If similarity is below threshold, warns that optimization may not work.

        Returns:
            True if quality is acceptable (mean_sim >= threshold)
        """
        from lipo.inference import validate_roundtrip_quality, Vec2TextInverter

        threshold = getattr(self.config, 'roundtrip_validation_threshold', 0.90)
        n_samples = getattr(self.config, 'roundtrip_validation_samples', 20)

        if verbose:
            print("\n--- Round-Trip Validation ---")

        # Get test instructions from grid data (task-relevant)
        if hasattr(self, 'grid_data') and self.grid_data:
            test_instructions = [
                d["instruction_text"]
                for d in self.grid_data[:n_samples]
            ]
        elif self.instructions:
            test_instructions = self.instructions[:n_samples]
        else:
            if verbose:
                print("  Skipping: no instructions loaded for validation")
            return True

        try:
            inverter = Vec2TextInverter(
                device=str(self.device),
                model_type=getattr(self.config, 'vec2text_model', "gtr-base"),
            )
        except Exception as e:
            if verbose:
                print(f"  Skipping: Vec2Text init failed: {e}")
            return True

        results = validate_roundtrip_quality(
            vae=self.vae,
            gtr=self.gtr,
            inverter=inverter,
            instructions=test_instructions,
            n_samples=len(test_instructions),
            verbose=verbose,
        )

        if results["mean_sim"] < threshold:
            print(f"WARNING: VAE round-trip quality below threshold!")
            print(f"  Mean similarity: {results['mean_sim']:.4f} < {threshold}")
            print(f"  Poor samples: {results['poor_count']}/{results['n_samples']}")
            print(f"  Optimization may not improve results.")
            return False

        if verbose:
            print(f"  Round-trip validation PASSED (mean_sim={results['mean_sim']:.4f} >= {threshold})")

        return True

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

        self.hyperband = LIPOHyperband(
            instructions=self.instructions,
            validation_data=self.validation_data,
            llm_evaluator=llm_evaluator,
            vae_with_adapter=self.vae_with_adapter,
            encoder=self.gtr,
            config=self.config,
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
            if self.gp.best_error_rate is not None:
                self.gp_stats["best_observed_error"] = float(self.gp.best_error_rate)

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

        # Create VAEWithAdapter (frozen VAE wrapper, no adapter)
        self.vae_with_adapter = VAEWithAdapter(
            self.vae, self.config.latent_dim
        ).to(self.device)

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
            missing_ids = []
            for d in grid_data:
                inst_id = d["instruction_id"]
                if inst_id < len(instruction_texts):
                    d["instruction_text"] = instruction_texts[inst_id]
                else:
                    missing_ids.append(inst_id)

            # Fail loudly if any instruction IDs are missing
            if missing_ids:
                max_id = max(d["instruction_id"] for d in grid_data)
                raise ValueError(
                    f"Grid references {len(missing_ids)} instruction IDs not found in instructions file.\n"
                    f"  Missing IDs: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}\n"
                    f"  Instructions file has {len(instruction_texts)} entries (IDs 0-{len(instruction_texts)-1})\n"
                    f"  Grid references IDs up to {max_id}\n"
                    f"  Check that the correct instructions file is being used: {instructions_path}"
                )

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

    def load_from_hyperband_evaluations(
        self,
        evals_path: str,
        verbose: bool = True,
    ) -> Tuple[List[str], List[float], List[int]]:
        """Load instructions and evaluations from saved hyperband_evaluations JSON.

        Uses ALL evaluated instructions - FixedNoiseGaussianLikelihood with Bernoulli
        variance weights observations by fidelity (low fidelity = high uncertainty).

        Args:
            evals_path: Path to JSON file with evaluation results
            verbose: Print progress

        Returns:
            Tuple of (instructions, error_rates, fidelities) for all evaluated instructions
        """
        if verbose:
            print(f"\nLoading hyperband evaluations: {evals_path}")

        with open(evals_path, "r") as f:
            data = json.load(f)

        # Detect format: new hbbops_results format vs old hyperband_evaluations format
        if "metadata" in data and "results" in data:
            # New format from hbbops_results.py
            # Results contain instruction text directly
            results = data["results"]
            max_fidelity = data["metadata"].get("max_fidelity", 1319)
            all_instructions = None  # Instructions are in results
            if verbose:
                print(f"  Format: hbbops_results (new)")
                print(f"  Evaluated instructions: {len(results)}")
                print(f"  Max fidelity: {max_fidelity}")
        else:
            # Old format: separate instructions list + hyperband_evaluations
            all_instructions = data.get("instructions", [])
            hb_evals = data.get("hyperband_evaluations", {})
            results = hb_evals.get("results", {})
            max_fidelity = hb_evals.get("max_fidelity", 1319)
            if verbose:
                print(f"  Format: hyperband_evaluations (legacy)")
                print(f"  Total instructions: {len(all_instructions)}")
                print(f"  Evaluated instructions: {len(results)}")
                print(f"  Max fidelity: {max_fidelity}")

        # Build lists of ALL evaluated instructions with their error rates and fidelities
        # FixedNoiseGaussianLikelihood will weight by fidelity (Beta posterior variance)

        # First pass: collect raw error rates for Empirical Bayes prior fitting
        raw_error_rates = []
        instruction_data = []  # (instruction, raw_error, fidelity)

        for idx_str, result in results.items():
            idx = int(idx_str)
            fidelity = result.get("fidelity", max_fidelity)

            # Validate fidelity to prevent division by zero in noise computation
            if fidelity < 1:
                fidelity = 1

            # Get instruction text based on format
            if all_instructions is not None:
                # Legacy format: instruction from separate list
                if idx >= len(all_instructions):
                    continue
                instruction = all_instructions[idx]
            else:
                # New format: instruction directly in result
                instruction = result.get("instruction")
                if instruction is None:
                    continue

            raw_error = result["error_rate"]
            raw_error_rates.append(raw_error)
            instruction_data.append((instruction, raw_error, fidelity))

        # Fit Empirical Bayes prior from data
        # This replaces fixed Beta(1,1) with data-driven prior
        alpha, beta = fit_beta_prior(raw_error_rates)
        self.beta_alpha = alpha
        self.beta_beta = beta

        if verbose:
            prior_mean = alpha / (alpha + beta)
            print(f"  Empirical Bayes prior: Alpha={alpha:.2f}, Beta={beta:.2f}")
            print(f"  Prior mean error rate: {prior_mean:.2%}")

        # Second pass: apply smoothing with learned prior
        evaluated_instructions = []
        error_rates = []
        fidelities = []

        for instruction, raw_error, fidelity in instruction_data:
            evaluated_instructions.append(instruction)

            # Empirical Bayes posterior mean: (errors + α) / (n + α + β)
            # This pulls toward the data-driven prior mean instead of 50%
            num_errors = raw_error * fidelity
            posterior_mean = (num_errors + alpha) / (fidelity + alpha + beta)
            error_rates.append(posterior_mean)
            fidelities.append(fidelity)

        if verbose:
            print(f"  Using all {len(evaluated_instructions)} evaluated instructions (Empirical Bayes)")
            errors = sorted(error_rates)
            print(f"  Smoothed error range: [{errors[0]:.4f}, {errors[-1]:.4f}]")
            print(f"  Best 5 smoothed errors: {[f'{e:.4f}' for e in errors[:5]]}")
            # Fidelity distribution
            fid_sorted = sorted(fidelities)
            print(f"  Fidelity range: [{fid_sorted[0]}, {fid_sorted[-1]}]")
            full_fid_count = sum(1 for f in fidelities if f == max_fidelity)
            print(f"  Full fidelity ({max_fidelity}): {full_fid_count}/{len(fidelities)}")

            # Show best from full-fidelity vs best overall (for debugging)
            full_fid_errors = [e for e, f in zip(error_rates, fidelities) if f == max_fidelity]
            if full_fid_errors:
                best_full_fid = min(full_fid_errors)
                best_overall = min(error_rates)
                if abs(best_full_fid - best_overall) > 1e-6:
                    print(f"  NOTE: Best overall ({best_overall:.4f}) differs from best full-fidelity ({best_full_fid:.4f})")
                    print(f"        Only full-fidelity best will be used for optimization target.")

        # Store for VAE and GP training
        self.instructions = evaluated_instructions
        self.grid_error_rates = error_rates
        self.fidelities = fidelities

        # Build grid_data (sorted by error_rate) for train_gp_from_grid compatibility
        grid_data = []
        for inst, err, fid in zip(evaluated_instructions, error_rates, fidelities):
            grid_data.append({
                "instruction_text": inst,
                "error_rate": err,
                "fidelity": fid,
            })
        grid_data.sort(key=lambda x: x["error_rate"])
        self.grid_data = grid_data

        # Pre-compute embeddings
        if verbose:
            print("Encoding instructions with GTR...")
        for idx, inst in enumerate(self.instructions):
            self.instruction_embeddings[idx] = self.gtr.encode_tensor(inst)

        if verbose:
            print(f"  Encoded {len(self.instruction_embeddings)} instructions")

        # Always load APE instructions for VAE training (better coverage)
        ape_path = "lipo/data/ape_instructions.json"
        try:
            with open(ape_path, "r") as f:
                ape_data = json.load(f)
            diverse_source = ape_data.get("instructions", [])
            if not diverse_source:
                raise ValueError(f"APE instructions file is empty or has wrong format: {ape_path}")
            if verbose:
                print(f"Loaded {len(diverse_source)} APE instructions for VAE from {ape_path}")
        except FileNotFoundError:
            # Fallback: use evaluated instructions - warn loudly as this is significant degradation
            if all_instructions is not None:
                diverse_source = all_instructions
            else:
                diverse_source = [r.get("instruction") for r in results.values() if r.get("instruction")]

            print(f"WARNING: APE instructions not found at {ape_path}")
            print(f"  Falling back to {len(diverse_source)} evaluated instructions for VAE training.")
            print(f"  This may significantly degrade VAE quality!")
            print(f"  Expected ~2000 diverse instructions, got {len(diverse_source)}.")
            print(f"  To fix: generate APE instructions or provide --ape-cache-path")

        self.diverse_instructions = diverse_source
        self.diverse_embeddings = {}
        if verbose:
            print(f"Encoding {len(diverse_source)} diverse instructions for VAE...")
        for idx, inst in enumerate(diverse_source):
            self.diverse_embeddings[idx] = self.gtr.encode_tensor(inst)
        if verbose:
            print(f"  Encoded {len(self.diverse_embeddings)} diverse instructions")

        return evaluated_instructions, error_rates, fidelities

    def evaluate_vae_quality(self, verbose: bool = True) -> Dict[str, float]:
        """Compute comprehensive VAE quality metrics.

        Computes:
        - Reconstruction quality (cosine similarity, MSE)
        - Latent space statistics (norm, variance per dimension)
        - KL divergence statistics
        - Posterior collapse detection

        NOTE: Evaluates on grid instructions (not diverse) because:
        1. These are the task-relevant instructions with error rates
        2. Diverse instructions are only for VAE regularization

        Args:
            verbose: Print metrics

        Returns:
            Dictionary with quality metrics
        """
        if self.vae is None:
            raise RuntimeError("VAE not trained. Call train_vae() first.")

        self.vae.eval()

        # Evaluate on grid instructions only (the task-relevant ones)
        if hasattr(self, 'grid_data') and self.grid_data is not None:
            # grid_data is a list of dicts with 'instruction_text' key
            embeddings = torch.stack([
                self.gtr.encode_tensor(d["instruction_text"])
                for d in self.grid_data
            ]).to(self.device)
            eval_name = "grid"
        else:
            embeddings = torch.stack([
                self.gtr.encode_tensor(inst) for inst in self.instructions
            ]).to(self.device)
            eval_name = "all"

        if verbose and hasattr(self, 'diverse_embeddings') and self.diverse_embeddings is not None:
            # diverse_embeddings can be tensor or dict with 'embeddings' key
            if isinstance(self.diverse_embeddings, dict):
                diverse_count = len(self.diverse_embeddings.get("instructions", []))
            else:
                diverse_count = self.diverse_embeddings.shape[0]
            print(f"  Note: Evaluating on {eval_name} ({embeddings.shape[0]} samples), "
                  f"not diverse ({diverse_count} samples)")

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

        # Get fidelities (sample counts) for heteroscedastic noise
        if hasattr(self, 'fidelities') and self.fidelities:
            fidelities = torch.tensor(
                self.fidelities,
                dtype=torch.float32,
                device=self.device
            )
        else:
            # Default: assume max fidelity for all
            fidelities = torch.ones(len(embeddings), dtype=torch.float32, device=self.device) * 1319

        if verbose:
            print(f"  Training on {len(embeddings)} grid samples")
            print(f"  Error range: [{error_rates.min():.4f}, {error_rates.max():.4f}]")
            print(f"  Fidelity range: [{fidelities.min():.0f}, {fidelities.max():.0f}]")

        # Create and train GP (works directly on 32D VAE latent, no adapter)
        self.gp = GPWithEI(
            device=str(self.device),
        )

        # Set VAEWithAdapter (frozen VAE wrapper for encoding)
        self.gp.vae_with_adapter = self.vae_with_adapter

        # Set training data with fidelities and Empirical Bayes prior for heteroscedastic noise
        self.gp.set_training_data(
            embeddings, error_rates, fidelities,
            beta_alpha=self.beta_alpha,
            beta_beta=self.beta_beta,
        )

        # Train GP
        train_success = self.gp.train(
            epochs=self.config.gp_epochs,
            lr=self.config.gp_lr,
            patience=self.config.gp_patience,
            verbose=verbose,
        )

        if not train_success:
            raise RuntimeError(
                "GP training failed due to numerical instability. "
                "This typically indicates: duplicate/near-duplicate inputs, "
                "near-constant outputs, or ill-conditioned kernel. "
                "Try reducing the number of training samples or checking for duplicates."
            )

        # Store GP training stats
        self.gp_stats = self.gp.training_stats.copy()
        self.gp_stats["best_observed_error"] = float(self.gp.best_error_rate)
        self.gp_stats["error_range"] = {
            "min": float(error_rates.min()),
            "max": float(error_rates.max()),
        }

        if verbose:
            print(f"  GP trained, best error: {self.gp.best_error_rate:.4f}")

        return self.gp

    def get_best_from_grid(
        self,
        full_fidelity_only: bool = True,
    ) -> Tuple[InstructionOnlyPrompt, float]:
        """Get best prompt from loaded grid data.

        Args:
            full_fidelity_only: If True, only consider prompts evaluated at full fidelity.
                               This is the default since low-fidelity evaluations are unreliable.

        Returns:
            (best_prompt, best_error) tuple
        """
        if not hasattr(self, 'grid_data') or not self.grid_data:
            raise RuntimeError("No grid data. Call load_from_grid() first.")

        # Determine max fidelity from data
        max_fidelity = max(
            d.get("fidelity", 1319) for d in self.grid_data
        )

        if full_fidelity_only:
            # Filter to full-fidelity prompts only
            full_fidelity_data = [
                d for d in self.grid_data
                if d.get("fidelity", 1319) >= max_fidelity
            ]
            if not full_fidelity_data:
                print(f"WARNING: No full-fidelity ({max_fidelity}) prompts found!")
                print(f"  Falling back to best from all {len(self.grid_data)} prompts")
                full_fidelity_data = self.grid_data

            # grid_data is sorted by error, but we need to re-sort filtered data
            best = min(full_fidelity_data, key=lambda d: d["error_rate"])
        else:
            best = self.grid_data[0]  # Already sorted by error

        # Find the index in the instructions list
        try:
            best_idx = self.instructions.index(best["instruction_text"])
        except ValueError:
            best_idx = 0

        best_prompt = InstructionOnlyPrompt(
            instruction=best["instruction_text"],
            instruction_id=best_idx,
        )
        return best_prompt, best["error_rate"]

    def get_full_fidelity_best(self) -> Tuple[Optional[InstructionOnlyPrompt], Optional[float]]:
        """Get best prompt from full-fidelity evaluations only.

        This is the authoritative "best" since low-fidelity evaluations have high variance.

        Returns:
            (best_prompt, best_error) tuple, or (None, None) if no full-fidelity data
        """
        if not hasattr(self, 'grid_data') or not self.grid_data:
            return None, None

        # Determine max fidelity
        fidelities = [d.get("fidelity", 1319) for d in self.grid_data]
        max_fidelity = max(fidelities)

        # Filter to full-fidelity only
        full_fidelity_data = [
            d for d in self.grid_data
            if d.get("fidelity", 1319) >= max_fidelity
        ]

        if not full_fidelity_data:
            return None, None

        # Find best among full-fidelity
        best = min(full_fidelity_data, key=lambda d: d["error_rate"])

        # Find index in instructions list
        try:
            best_idx = self.instructions.index(best["instruction_text"])
        except ValueError:
            best_idx = 0

        best_prompt = InstructionOnlyPrompt(
            instruction=best["instruction_text"],
            instruction_id=best_idx,
        )
        return best_prompt, best["error_rate"]
