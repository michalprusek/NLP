"""HbBoPs with Vec2Text integration.

Extends HbBoPs to use GTR encoder (compatible with Vec2Text) and adds
autoencoder for bidirectional mapping between 10D latent and 768D embedding spaces.
"""

import json
import torch
import torch.nn as nn
import gpytorch
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
from collections import Counter
from dataclasses import dataclass
from scipy.stats import norm

from vec2text_hbbops.encoder import GTRPromptEncoder
from vec2text_hbbops.autoencoder import PromptAutoencoder, InstructionAutoencoder, InstructionAutoencoderLoss
from vec2text_hbbops.training import AutoencoderTrainer, TrainingConfig


@dataclass
class Prompt:
    """A prompt composed of an instruction and few-shot exemplar."""

    instruction: str
    exemplar: str
    instruction_id: int
    exemplar_id: int

    def __str__(self) -> str:
        return f"{self.instruction}\n\n{self.exemplar}"


class FeatureExtractor(nn.Module):
    """Structural-aware feature extractor for deep kernel GP.

    Architecture from HbBoPs paper:
    - Separate encoders: Lin(768, 64) -> ReLU -> Lin(64, 32) -> ReLU
    - Joint encoder: Lin(64, 32) -> ReLU -> Lin(32, 10)
    """

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.instruction_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.exemplar_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(2 * 32, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(
        self, instruction_emb: torch.Tensor, exemplar_emb: torch.Tensor
    ) -> torch.Tensor:
        inst_features = self.instruction_encoder(instruction_emb)
        ex_features = self.exemplar_encoder(exemplar_emb)
        combined = torch.cat([inst_features, ex_features], dim=1)
        return self.joint_encoder(combined)


class DeepKernelGP(gpytorch.models.ExactGP):
    """Gaussian Process with structural-aware deep kernel.

    Uses ARD Matern 5/2 kernel on 10-dim latent features.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        feature_extractor: FeatureExtractor,
        input_dim: int = 768,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=10,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        self.feature_extractor = feature_extractor
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        instruction_emb = x[:, : self.input_dim]
        exemplar_emb = x[:, self.input_dim :]
        latent = self.feature_extractor(instruction_emb, exemplar_emb)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(latent), self.covar_module(latent)
        )


class HbBoPsVec2Text:
    """HbBoPs with Vec2Text integration.

    Key differences from base HbBoPs:
    1. Uses GTRPromptEncoder instead of BERT PromptEncoder
    2. Trains autoencoder on all prompt embeddings before GP
    3. Provides methods to invert optimized latent back to text

    The autoencoder maps:
        1536D (concat inst+ex GTR embeddings) <-> 10D latent

    This enables:
        - Optimization in 10D latent space via GP
        - Inversion back to 768D embeddings via AE decoder
        - Text generation via Vec2Text from decoded embeddings
    """

    def __init__(
        self,
        instructions: List[str],
        exemplars: List[str],
        validation_data: List[Dict],
        llm_evaluator: Callable,
        bmin: int = 10,
        eta: float = 2.0,
        random_interleaving_prob: float = 0.1,
        device: str = "auto",
        seed: int = 42,
        # Autoencoder parameters
        ae_latent_dim: int = 10,
        ae_hidden_dims: Optional[List[int]] = None,
        ae_dropout: float = 0.3,
        ae_noise_std: float = 0.1,
    ):
        """Initialize HbBoPs with Vec2Text support.

        Args:
            instructions: List of instruction strings
            exemplars: List of exemplar strings
            validation_data: Validation instances for evaluation
            llm_evaluator: Function(prompt, instances) -> error_rate
            bmin: Minimum instances per evaluation (Hyperband)
            eta: Halving ratio (Hyperband)
            random_interleaving_prob: Probability of random exploration
            device: Device to use
            seed: Random seed
            ae_latent_dim: Autoencoder latent dimension (should match GP)
            ae_hidden_dims: Autoencoder hidden layer sizes
            ae_dropout: Autoencoder dropout rate
            ae_noise_std: Denoising noise std
        """
        self.instructions = instructions
        self.exemplars = exemplars
        self.llm_evaluator = llm_evaluator

        # Shuffle validation data for unbiased subsets
        random.seed(seed)
        self.validation_data = validation_data.copy()
        random.shuffle(self.validation_data)
        self.nvalid = len(validation_data)

        # Generate all candidate prompts (P = I x E)
        self.prompts = [
            Prompt(inst, ex, i_idx, e_idx)
            for i_idx, inst in enumerate(instructions)
            for e_idx, ex in enumerate(exemplars)
        ]
        print(f"Generated {len(self.prompts)} candidate prompts")
        print(f"  {len(instructions)} instructions x {len(exemplars)} exemplars")

        # Device setup
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")

        # Initialize GTR encoder and pre-compute embeddings
        print("Initializing GTR encoder...")
        self.encoder = GTRPromptEncoder(device=str(self.device))
        print("Pre-computing GTR embeddings for all instructions and exemplars...")
        self._precompute_embeddings()

        # Hyperband parameters
        self.bmin = bmin
        self.eta = eta
        self.random_interleaving_prob = random_interleaving_prob

        r = self.nvalid / self.bmin
        self.smax = int(np.floor(np.log(r) / np.log(eta)))
        self.B = (self.smax + 1) * self.nvalid

        print(f"\nHyperband schedule:")
        print(f"  nvalid={self.nvalid}, bmin={bmin}, eta={eta}")
        print(f"  r={r:.1f}, smax={self.smax}, B={self.B}")

        # Design data storage
        self.design_data: List[Tuple] = []
        self.evaluation_cache: Dict = {}

        # GP model (initialized when enough data)
        self.gp_model: Optional[DeepKernelGP] = None
        self.likelihood = None
        self.feature_extractor: Optional[FeatureExtractor] = None

        # Normalization parameters
        self.X_min: Optional[torch.Tensor] = None
        self.X_max: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        # Best prompt tracking
        self.best_prompt: Optional[Prompt] = None
        self.best_validation_error = float("inf")

        # Autoencoder configuration
        self.ae_config = {
            "latent_dim": ae_latent_dim,
            "hidden_dims": ae_hidden_dims or [512, 128],
            "dropout_rate": ae_dropout,
            "noise_std": ae_noise_std,
        }
        self.autoencoder: Optional[PromptAutoencoder] = None
        self.ae_trained = False

    def _get_device(self, device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _precompute_embeddings(self) -> None:
        """Pre-compute GTR embeddings for all instructions and exemplars."""
        self.instruction_embeddings: Dict[int, np.ndarray] = {}
        for i_idx, inst in enumerate(self.instructions):
            self.instruction_embeddings[i_idx] = self.encoder.encode(inst)

        self.exemplar_embeddings: Dict[int, np.ndarray] = {}
        for e_idx, ex in enumerate(self.exemplars):
            self.exemplar_embeddings[e_idx] = self.encoder.encode(ex)

        print(f"  Cached {len(self.instruction_embeddings)} GTR instruction embeddings")
        print(f"  Cached {len(self.exemplar_embeddings)} GTR exemplar embeddings")

    def embed_prompt(self, prompt: Prompt) -> Tuple[np.ndarray, np.ndarray]:
        """Get embeddings for instruction and exemplar (from cache)."""
        return (
            self.instruction_embeddings[prompt.instruction_id],
            self.exemplar_embeddings[prompt.exemplar_id],
        )

    def train_autoencoder(
        self,
        config: Optional[TrainingConfig] = None,
        verbose: bool = True,
    ) -> None:
        """Train autoencoder on all prompt embeddings.

        Should be called before run_hyperband() to enable latent space operations.

        Args:
            config: Training configuration
            verbose: Print progress
        """
        if self.ae_trained:
            print("Autoencoder already trained, skipping...")
            return

        print("\n" + "=" * 60)
        print("Training Autoencoder for Latent Space Mapping")
        print("=" * 60)

        self.autoencoder = PromptAutoencoder(
            input_dim=1536,
            latent_dim=self.ae_config["latent_dim"],
            hidden_dims=self.ae_config["hidden_dims"],
            dropout_rate=self.ae_config["dropout_rate"],
            noise_std=self.ae_config["noise_std"],
        ).to(self.device)

        trainer = AutoencoderTrainer(
            self.autoencoder,
            config=config,
            device=str(self.device),
        )

        history = trainer.train(
            self.instruction_embeddings,
            self.exemplar_embeddings,
            use_all_combinations=True,
            verbose=verbose,
        )

        # Evaluate reconstruction quality
        metrics = trainer.evaluate_reconstruction(
            self.instruction_embeddings,
            self.exemplar_embeddings,
        )
        if verbose:
            print("\nReconstruction quality:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
            print(f"  Instruction cosine: {metrics['instruction_cosine']:.4f}")
            print(f"  Exemplar cosine: {metrics['exemplar_cosine']:.4f}")

        self.ae_trained = True
        print("Autoencoder training complete\n")

    def get_prompt_latent(self, prompt: Prompt) -> torch.Tensor:
        """Get autoencoder latent representation for a prompt.

        Args:
            prompt: Prompt to encode

        Returns:
            10D latent tensor
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not trained. Call train_autoencoder() first.")

        inst_emb, ex_emb = self.embed_prompt(prompt)
        concat = torch.tensor(
            np.concatenate([inst_emb, ex_emb]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.autoencoder.eval()
        with torch.no_grad():
            latent = self.autoencoder.encode(concat)

        return latent.squeeze(0)

    def decode_latent(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent to instruction and exemplar embeddings.

        Args:
            latent: 10D latent tensor

        Returns:
            Tuple of (instruction_emb, exemplar_emb), each 768D
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not trained. Call train_autoencoder() first.")

        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        self.autoencoder.eval()
        with torch.no_grad():
            recon = self.autoencoder.decode(latent)
            inst_emb, ex_emb = self.autoencoder.split_reconstruction(recon)

        return inst_emb.squeeze(0), ex_emb.squeeze(0)

    def evaluate_prompt(self, prompt: Prompt, fidelity: int) -> float:
        """Evaluate prompt on 'fidelity' validation instances with caching."""
        cache_key = (prompt.instruction_id, prompt.exemplar_id, fidelity)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        # Try to extend from lower fidelity
        for prev_f in sorted(
            [
                f
                for (i, e, f) in self.evaluation_cache.keys()
                if i == prompt.instruction_id
                and e == prompt.exemplar_id
                and f < fidelity
            ],
            reverse=True,
        ):
            prev_key = (prompt.instruction_id, prompt.exemplar_id, prev_f)
            prev_error = self.evaluation_cache[prev_key]
            remaining = self.validation_data[prev_f:fidelity]
            if remaining:
                new_error = self.llm_evaluator(prompt, remaining)
                total_error = (prev_error * prev_f + new_error * len(remaining)) / fidelity
            else:
                total_error = prev_error
            self.evaluation_cache[cache_key] = total_error
            return total_error

        # Full evaluation
        error = self.llm_evaluator(prompt, self.validation_data[:fidelity])
        self.evaluation_cache[cache_key] = error
        return error

    def train_gp(self, fidelities: list, min_observations: int = 4) -> bool:
        """Train GP on design data from multiple fidelity levels."""
        fidelity_set = set(fidelities) if isinstance(fidelities, list) else {fidelities}
        fidelity_data = [
            (ie, ee, ve) for _, ie, ee, ve, f in self.design_data if f in fidelity_set
        ]

        if len(fidelity_data) < min_observations:
            return False

        errors = [ve for _, _, ve in fidelity_data]
        if np.std(errors) < 1e-6:
            return False

        # Prepare training tensors
        X_inst = torch.tensor(
            np.array([ie for ie, _, _ in fidelity_data]), dtype=torch.float32, device=self.device
        )
        X_ex = torch.tensor(
            np.array([ee for _, ee, _ in fidelity_data]), dtype=torch.float32, device=self.device
        )
        X = torch.cat([X_inst, X_ex], dim=1)
        y = torch.tensor(np.array(errors), dtype=torch.float32, device=self.device)

        # Unit cube normalization
        self.X_min = X.min(dim=0)[0]
        self.X_max = X.max(dim=0)[0]
        denominator = self.X_max - self.X_min
        denominator[denominator == 0] = 1.0
        X_norm = (X - self.X_min) / denominator

        # Standardization
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-6
        y_norm = (y - self.y_mean) / self.y_std

        # Create GP model
        self.feature_extractor = FeatureExtractor(input_dim=768).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp_model = DeepKernelGP(
            X_norm, y_norm, self.likelihood, self.feature_extractor
        ).to(self.device)

        # Training
        self.gp_model.train()
        self.likelihood.train()
        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss, patience = float("inf"), 0
        with gpytorch.settings.cholesky_jitter(1e-4):
            for epoch in range(3000):
                try:
                    optimizer.zero_grad()
                    loss = -mll(self.gp_model(X_norm), y_norm)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss, patience = loss.item(), 0
                    else:
                        patience += 1
                    if patience >= 10:
                        break
                except Exception:
                    if epoch == 0:
                        return False
                    break

        return True

    def expected_improvement(self, prompt: Prompt, vmin_b: float) -> float:
        """Compute Expected Improvement acquisition function."""
        if self.gp_model is None:
            return 0.0

        inst_emb, ex_emb = self.embed_prompt(prompt)
        X = torch.tensor(
            np.concatenate([inst_emb, ex_emb]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        denominator = self.X_max - self.X_min
        denominator[denominator == 0] = 1.0
        X_norm = (X - self.X_min) / denominator

        self.gp_model.eval()
        self.likelihood.eval()

        try:
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                gpytorch.settings.cholesky_jitter(1e-4),
            ):
                pred = self.likelihood(self.gp_model(X_norm))
                mean = pred.mean.item() * self.y_std.item() + self.y_mean.item()
                std = pred.stddev.item() * self.y_std.item()
        except Exception:
            return 0.0

        if std <= 0:
            return max(vmin_b - mean, 0)
        z = (vmin_b - mean) / std
        return (vmin_b - mean) * norm.cdf(z) + std * norm.pdf(z)

    def get_prompt_proposal(self, evaluated: List[int], vmin_b: float) -> int:
        """Propose next prompt using BO or random interleaving."""
        unevaluated = [i for i in range(len(self.prompts)) if i not in evaluated]

        if np.random.rand() < self.random_interleaving_prob or self.gp_model is None:
            return np.random.choice(unevaluated)

        return max(
            unevaluated, key=lambda i: self.expected_improvement(self.prompts[i], vmin_b)
        )

    def run_hyperband(self, verbose: bool = True) -> Tuple[Prompt, float]:
        """Run Hyperband with BO proposals (Algorithm 1)."""
        if verbose:
            print("\n" + "=" * 60)
            print("Running HbBoPs-Vec2Text Optimization")
            print("=" * 60)

        for s in range(self.smax, -1, -1):
            n = int(np.ceil((self.B / self.nvalid) * (self.eta**s) / (s + 1)))
            b = int(self.nvalid * (self.eta ** (-s)))

            if verbose:
                print(f"\nBracket s={s}: n={n} prompts, b={b} instances")

            P, V = [], []

            for j in range(n):
                # Get trainable fidelities
                fidelities = Counter(f for _, _, _, _, f in self.design_data)
                trainable = [f for f, c in fidelities.items() if c >= 4]

                if trainable:
                    sorted_fidelities = sorted(trainable, reverse=True)
                    top_75_count = max(1, int(len(sorted_fidelities) * 0.75))
                    top_75_fidelities = sorted_fidelities[:top_75_count]
                    self.train_gp(top_75_fidelities)

                # Compute vmin_b
                vmin_b = float("inf")
                if trainable and self.gp_model:
                    sorted_fidelities = sorted(trainable, reverse=True)
                    top_75_count = max(1, int(len(sorted_fidelities) * 0.75))
                    top_75_fidelities = set(sorted_fidelities[:top_75_count])
                    vmin_b = min(
                        ve
                        for _, _, _, ve, f in self.design_data
                        if f in top_75_fidelities
                    )

                # Propose and evaluate
                p_idx = self.get_prompt_proposal(P, vmin_b)
                prompt = self.prompts[p_idx]
                v = self.evaluate_prompt(prompt, b)

                P.append(p_idx)
                V.append(v)

                # Store design data
                inst_emb, ex_emb = self.embed_prompt(prompt)
                self.design_data.append((p_idx, inst_emb, ex_emb, v, b))

                # Track best
                if v < self.best_validation_error:
                    self.best_validation_error = v
                    self.best_prompt = prompt

            # Successive halving
            for i in range(1, s + 1):
                ni = int(np.floor(n * (self.eta ** (-i))))
                bi = int(b * (self.eta**i))

                if ni == 0:
                    break

                sorted_idx = np.argsort(V)
                P = [P[idx] for idx in sorted_idx[:ni]]

                V = []
                for p_idx in P:
                    prompt = self.prompts[p_idx]
                    v = self.evaluate_prompt(prompt, bi)
                    V.append(v)

                    inst_emb, ex_emb = self.embed_prompt(prompt)
                    self.design_data.append((p_idx, inst_emb, ex_emb, v, bi))

                    if v < self.best_validation_error:
                        self.best_validation_error = v
                        self.best_prompt = prompt

        if verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete")
            print("=" * 60)
            if self.best_prompt:
                print(f"Best validation error: {self.best_validation_error:.4f}")
                print(f"Best instruction ID: {self.best_prompt.instruction_id}")
                print(f"Best exemplar ID: {self.best_prompt.exemplar_id}")

        return self.best_prompt, self.best_validation_error

    def _load_grid_data(
        self,
        grid_path: str = "datasets/hbbops/full_grid_combined.jsonl",
    ) -> List[Dict]:
        """Load and filter grid data.

        Returns:
            List of grid entries with valid instruction/exemplar IDs
        """
        grid_file = Path(grid_path)
        if not grid_file.is_absolute():
            root = Path(__file__).parent.parent
            grid_file = root / grid_path

        if not grid_file.exists():
            raise FileNotFoundError(f"Grid file not found: {grid_file}")

        # Load grid data
        grid_data = []
        with open(grid_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    grid_data.append(json.loads(line))

        # Filter to only include prompts with valid instruction/exemplar IDs
        valid_inst_ids = set(self.instruction_embeddings.keys())
        valid_ex_ids = set(self.exemplar_embeddings.keys())

        grid_data = [
            entry for entry in grid_data
            if entry["instruction_id"] in valid_inst_ids
            and entry["exemplar_id"] in valid_ex_ids
        ]

        return grid_data

    def train_autoencoder_from_grid(
        self,
        grid_path: str = "datasets/hbbops/full_grid_combined.jsonl",
        config: Optional[TrainingConfig] = None,
        verbose: bool = True,
    ) -> None:
        """Train autoencoder on ALL prompts from the grid.

        Uses the entire grid (e.g., 625 prompts) for AE training,
        which provides more data than just the top-k used for GP.

        Args:
            grid_path: Path to grid file
            config: Training configuration
            verbose: Print progress
        """
        if self.ae_trained:
            print("Autoencoder already trained, skipping...")
            return

        grid_data = self._load_grid_data(grid_path)

        if verbose:
            print("\n" + "=" * 60)
            print("Training Autoencoder on Full Grid")
            print("=" * 60)
            print(f"Grid prompts: {len(grid_data)}")

        # Build training data from grid
        # Use only the (instruction_id, exemplar_id) pairs that exist in grid
        grid_inst_embeddings = {}
        grid_ex_embeddings = {}

        for entry in grid_data:
            inst_id = entry["instruction_id"]
            ex_id = entry["exemplar_id"]
            if inst_id not in grid_inst_embeddings:
                grid_inst_embeddings[inst_id] = self.instruction_embeddings[inst_id]
            if ex_id not in grid_ex_embeddings:
                grid_ex_embeddings[ex_id] = self.exemplar_embeddings[ex_id]

        if verbose:
            print(f"Unique instructions in grid: {len(grid_inst_embeddings)}")
            print(f"Unique exemplars in grid: {len(grid_ex_embeddings)}")

        # Initialize autoencoder
        self.autoencoder = PromptAutoencoder(
            input_dim=1536,
            latent_dim=self.ae_config["latent_dim"],
            hidden_dims=self.ae_config["hidden_dims"],
            dropout_rate=self.ae_config["dropout_rate"],
            noise_std=self.ae_config["noise_std"],
        ).to(self.device)

        trainer = AutoencoderTrainer(
            self.autoencoder,
            config=config,
            device=str(self.device),
        )

        # Train on grid combinations (not full cartesian product)
        # Override prepare_data to use specific grid pairs
        train_data = []
        for entry in grid_data:
            inst_emb = self.instruction_embeddings[entry["instruction_id"]]
            ex_emb = self.exemplar_embeddings[entry["exemplar_id"]]
            concat = np.concatenate([inst_emb, ex_emb])
            train_data.append(concat)

        train_data = torch.tensor(np.array(train_data), dtype=torch.float32)

        # Shuffle and split
        perm = torch.randperm(len(train_data))
        train_data = train_data[perm]

        val_split = 0.15
        val_size = max(1, int(len(train_data) * val_split))
        val_data = train_data[:val_size]
        train_data_split = train_data[val_size:]

        if verbose:
            print(f"Train samples: {len(train_data_split)}")
            print(f"Val samples: {len(val_data)}")

        # Manual training loop using trainer's internals
        from torch.utils.data import DataLoader, TensorDataset

        train_loader = DataLoader(
            TensorDataset(train_data_split),
            batch_size=trainer.config.batch_size,
            shuffle=True,
        )

        val_data_dev = val_data.to(trainer.device)
        patience_counter = 0
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(trainer.config.max_epochs):
            # Training
            trainer.ae.train()
            train_losses = []
            for (batch,) in train_loader:
                batch = batch.to(trainer.device)
                trainer.optimizer.zero_grad()
                x_recon, z = trainer.ae(batch)
                loss, _ = trainer.loss_fn(batch, x_recon, z)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.ae.parameters(), trainer.config.grad_clip)
                trainer.optimizer.step()
                train_losses.append(loss.item())

            # Validation
            trainer.ae.eval()
            with torch.no_grad():
                x_recon, z = trainer.ae(val_data_dev)
                val_loss, _ = trainer.loss_fn(val_data_dev, x_recon, z)

            trainer.scheduler.step(val_loss)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_state = {k: v.clone() for k, v in trainer.ae.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= trainer.config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}: train={np.mean(train_losses):.6f}, val={val_loss:.6f}")

        # Restore best weights
        if best_state:
            trainer.ae.load_state_dict(best_state)

        if verbose:
            print(f"Best val loss: {best_val_loss:.6f}")

        self.ae_trained = True
        print("Autoencoder training complete\n")

    def load_from_grid(
        self,
        grid_path: str = "datasets/hbbops/full_grid_combined.jsonl",
        top_k: int = 25,
        verbose: bool = True,
    ) -> Tuple[Optional[Prompt], float]:
        """Load top-k prompts from pre-evaluated grid and train GP.

        NOTE: Call train_autoencoder_from_grid() BEFORE this method
        to train AE on the full grid.

        Args:
            grid_path: Path to full_grid_combined.jsonl
            top_k: Number of top prompts for GP training
            verbose: Print progress

        Returns:
            Tuple of (best_prompt, best_error)
        """
        grid_data = self._load_grid_data(grid_path)

        # Sort by error rate ascending (best first)
        grid_data.sort(key=lambda x: x.get("error_rate", 1.0))
        top_prompts = grid_data[:top_k]

        if verbose:
            print("\n" + "=" * 60)
            print(f"Loading Top-{top_k} Prompts for GP Training")
            print("=" * 60)
            print(f"Total grid prompts: {len(grid_data)}")
            print(f"Using top {len(top_prompts)} for GP")
            if top_prompts:
                print(f"Error rate range: {top_prompts[0]['error_rate']:.4f} - {top_prompts[-1]['error_rate']:.4f}")

        # Populate design_data with top prompts
        fidelity = self.nvalid

        for entry in top_prompts:
            inst_id = entry["instruction_id"]
            ex_id = entry["exemplar_id"]
            error_rate = entry["error_rate"]

            inst_emb = self.instruction_embeddings[inst_id]
            ex_emb = self.exemplar_embeddings[ex_id]

            prompt_idx = inst_id * len(self.exemplars) + ex_id
            self.design_data.append((prompt_idx, inst_emb, ex_emb, error_rate, fidelity))

            if error_rate < self.best_validation_error:
                self.best_validation_error = error_rate
                self.best_prompt = self.prompts[prompt_idx]

        # Train GP on top-k data
        if verbose:
            print(f"\nTraining GP on {len(self.design_data)} prompts...")

        success = self.train_gp([fidelity], min_observations=4)

        if verbose:
            if success:
                print("GP training successful")
            else:
                print("GP training failed (not enough variance?)")

            print(f"\nBest prompt from grid:")
            print(f"  Error rate: {self.best_validation_error:.4f}")
            if self.best_prompt:
                print(f"  Instruction ID: {self.best_prompt.instruction_id}")
                print(f"  Exemplar ID: {self.best_prompt.exemplar_id}")

        return self.best_prompt, self.best_validation_error

    # ========================================================================
    # Instruction-Only Optimization (Vec2Text works better on short text)
    # ========================================================================

    def train_instruction_autoencoder_from_grid(
        self,
        grid_path: str = "datasets/hbbops/full_grid_combined.jsonl",
        config: Optional[TrainingConfig] = None,
        verbose: bool = True,
    ) -> None:
        """Train instruction-only autoencoder (768D → 10D → 768D).

        Uses ALL available instructions (not just grid) for training.
        This provides more training data for better reconstruction.

        Args:
            grid_path: Path to grid file (unused, kept for API compatibility)
            config: Training configuration
            verbose: Print progress
        """
        if hasattr(self, "instruction_ae_trained") and self.instruction_ae_trained:
            print("Instruction autoencoder already trained, skipping...")
            return

        if verbose:
            print("\n" + "=" * 60)
            print("Training Instruction-Only Autoencoder")
            print("=" * 60)

        # Use ALL instruction embeddings for training (not just grid)
        # This provides more data for the autoencoder
        inst_embeddings = [
            self.instruction_embeddings[inst_id]
            for inst_id in sorted(self.instruction_embeddings.keys())
        ]

        if verbose:
            print(f"Total instructions for training: {len(inst_embeddings)}")

        # Initialize instruction autoencoder (768D → 10D → 768D)
        self.instruction_autoencoder = InstructionAutoencoder(
            input_dim=768,
            latent_dim=self.ae_config["latent_dim"],
            hidden_dims=[256, 64],  # Smaller for instruction-only
            dropout_rate=self.ae_config["dropout_rate"],
            noise_std=self.ae_config["noise_std"],
        ).to(self.device)

        # Prepare training data
        train_data = torch.tensor(
            np.array(inst_embeddings), dtype=torch.float32
        )

        # Shuffle and split
        perm = torch.randperm(len(train_data))
        train_data = train_data[perm]

        val_split = 0.15
        val_size = max(1, int(len(train_data) * val_split))
        val_data = train_data[:val_size]
        train_data_split = train_data[val_size:]

        if verbose:
            print(f"Train samples: {len(train_data_split)}")
            print(f"Val samples: {len(val_data)}")

        # Training setup
        from torch.utils.data import DataLoader, TensorDataset

        loss_fn = InstructionAutoencoderLoss(
            lambda_cosine=0.5,
            lambda_sparse=0.001,
        )

        if config is None:
            config = TrainingConfig()

        optimizer = torch.optim.AdamW(
            self.instruction_autoencoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )

        train_loader = DataLoader(
            TensorDataset(train_data_split),
            batch_size=min(config.batch_size, len(train_data_split)),
            shuffle=True,
        )

        val_data_dev = val_data.to(self.device)
        patience_counter = 0
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(config.max_epochs):
            # Training
            self.instruction_autoencoder.train()
            train_losses = []
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                x_recon, z = self.instruction_autoencoder(batch)
                loss, _ = loss_fn(batch, x_recon, z)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.instruction_autoencoder.parameters(), config.grad_clip
                )
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.instruction_autoencoder.eval()
            with torch.no_grad():
                x_recon, z = self.instruction_autoencoder(val_data_dev)
                val_loss, _ = loss_fn(val_data_dev, x_recon, z)

            scheduler.step(val_loss)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_state = {
                    k: v.clone() for k, v in self.instruction_autoencoder.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}: train={np.mean(train_losses):.6f}, val={val_loss:.6f}")

        # Restore best weights
        if best_state:
            self.instruction_autoencoder.load_state_dict(best_state)

        if verbose:
            print(f"Best val loss: {best_val_loss:.6f}")

        self.instruction_ae_trained = True
        print("Instruction autoencoder training complete\n")

    def get_best_exemplar_from_grid(
        self,
        grid_path: str = "datasets/hbbops/full_grid_combined.jsonl",
    ) -> Tuple[int, str]:
        """Get the best exemplar from grid based on lowest error rate.

        Returns:
            Tuple of (exemplar_id, exemplar_text)
        """
        grid_data = self._load_grid_data(grid_path)
        grid_data.sort(key=lambda x: x.get("error_rate", 1.0))

        best_entry = grid_data[0]
        ex_id = best_entry["exemplar_id"]
        return ex_id, self.exemplars[ex_id]

    def encode_instruction(self, instruction_id: int) -> torch.Tensor:
        """Encode instruction to 10D latent using instruction autoencoder.

        Args:
            instruction_id: Index into self.instructions

        Returns:
            10D latent tensor
        """
        if not hasattr(self, "instruction_autoencoder") or self.instruction_autoencoder is None:
            raise RuntimeError(
                "Instruction autoencoder not trained. "
                "Call train_instruction_autoencoder_from_grid() first."
            )

        inst_emb = self.instruction_embeddings[instruction_id]
        inst_tensor = torch.tensor(
            inst_emb, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.instruction_autoencoder.eval()
        with torch.no_grad():
            latent = self.instruction_autoencoder.encode(inst_tensor)

        return latent.squeeze(0)

    def decode_instruction_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode 10D latent to 768D instruction embedding.

        Args:
            latent: 10D latent tensor

        Returns:
            768D instruction embedding tensor
        """
        if not hasattr(self, "instruction_autoencoder") or self.instruction_autoencoder is None:
            raise RuntimeError(
                "Instruction autoencoder not trained. "
                "Call train_instruction_autoencoder_from_grid() first."
            )

        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        self.instruction_autoencoder.eval()
        with torch.no_grad():
            inst_emb = self.instruction_autoencoder.decode(latent)

        return inst_emb.squeeze(0)

    def get_instruction_latent(self, prompt: Prompt) -> torch.Tensor:
        """Get instruction latent for a prompt (ignores exemplar).

        Args:
            prompt: Prompt object

        Returns:
            10D instruction latent tensor
        """
        return self.encode_instruction(prompt.instruction_id)
