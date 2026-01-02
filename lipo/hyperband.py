"""Instruction-only Hyperband for LIPO.

Adapted from hbbops_improved_2 with key changes:
- No exemplars (instruction-only mode)
- Uses GTR encoder instead of BERT
- Uses VAEWithAdapter as GP feature extractor
- Simplified cache key: (inst_id, fidelity)

Algorithm based on Schneider et al. (2025)
"Hyperband-based Bayesian Optimization for Black-box Prompt Selection"
"""

import torch
import gpytorch
from typing import List, Tuple, Dict, Optional, Callable
from collections import Counter
import numpy as np
import random
from scipy.stats import norm

from lipo.config import Config
from lipo.instruction import InstructionOnlyPrompt
from lipo.encoder import GTRInstructionEncoder, VAEWithAdapter
from lipo.gp import GPWithEI, InstructionDeepKernelGP


class LIPOHyperband:
    """Hyperband-based Bayesian Optimization for instruction-only prompts.

    Key differences from standard HbBoPs:
    - No exemplars (P = I, not P = I × E)
    - Uses GTR encoder (Vec2Text compatible)
    - Uses VAEWithAdapter as GP feature extractor
    - Simplified cache key: (inst_id, fidelity)
    """

    def __init__(
        self,
        instructions: List[str],
        validation_data: List[Dict],
        llm_evaluator: Callable[[InstructionOnlyPrompt, List[Dict]], float],
        vae_with_adapter: VAEWithAdapter,
        encoder: GTRInstructionEncoder,
        config: Config,
        device: str = "auto",
    ):
        """Initialize Hyperband.

        Args:
            instructions: List of instruction texts
            validation_data: Q/A pairs for evaluation
            llm_evaluator: Function (prompt, data) -> error_rate
            vae_with_adapter: VAEWithAdapter for GP feature extraction
            encoder: GTR encoder for embeddings
            config: Unified pipeline configuration
            device: Device to use
        """
        self.config = config
        self.device = self._get_device(device)

        # Store instructions as prompts
        self.prompts = [
            InstructionOnlyPrompt(instruction=inst, instruction_id=idx)
            for idx, inst in enumerate(instructions)
        ]
        print(f"Loaded {len(self.prompts)} instruction prompts")

        # Shuffle validation data once for unbiased subsets
        random.seed(self.config.seed)
        self.validation_data = validation_data.copy()
        random.shuffle(self.validation_data)
        self.nvalid = len(validation_data)

        # LLM evaluator
        self.llm_evaluator = llm_evaluator

        # Encoder and VAE
        self.encoder = encoder
        self.vae_with_adapter = vae_with_adapter.to(self.device)

        # Pre-compute embeddings
        print("Pre-computing GTR embeddings for all instructions...")
        self.instruction_embeddings: Dict[int, torch.Tensor] = {}
        for prompt in self.prompts:
            emb = self.encoder.encode_tensor(prompt.instruction)
            self.instruction_embeddings[prompt.instruction_id] = emb
        print(f"  Cached {len(self.instruction_embeddings)} embeddings")

        # Hyperband parameters
        self.bmin = self.config.bmin
        self.eta = self.config.eta
        r = self.nvalid / self.bmin
        self.smax = int(np.floor(np.log(r) / np.log(self.eta)))
        self.B = (self.smax + 1) * self.nvalid

        print(f"\nHyperband schedule:")
        print(f"  nvalid={self.nvalid}, bmin={self.bmin}, η={self.eta}")
        print(f"  smax={self.smax}, B={self.B}")

        # Design data: (prompt_idx, embedding, error_rate, fidelity)
        self.design_data: List[Tuple[int, torch.Tensor, float, int]] = []
        self.evaluation_cache: Dict[Tuple[int, int], float] = {}  # (inst_id, fidelity) -> error

        # GP model (initialized when enough data)
        self.gp_model: Optional[InstructionDeepKernelGP] = None
        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None

        # Normalization parameters
        self.X_min: Optional[torch.Tensor] = None
        self.X_max: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        # Best prompt tracking
        self.best_prompt: Optional[InstructionOnlyPrompt] = None
        self.best_validation_error: float = float('inf')

        # LLM call counter
        self.total_llm_calls: int = 0

        # GP training stats (updated each time train_gp is called)
        self.gp_training_stats: Dict = {}

    def _get_device(self, device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def evaluate_prompt(self, prompt: InstructionOnlyPrompt, fidelity: int) -> float:
        """Evaluate prompt on 'fidelity' validation instances with caching."""
        cache_key = (prompt.instruction_id, fidelity)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        # Try to extend from lower fidelity
        for prev_f in sorted(
            [f for (i, f) in self.evaluation_cache.keys()
             if i == prompt.instruction_id and f < fidelity],
            reverse=True
        ):
            prev_key = (prompt.instruction_id, prev_f)
            prev_error = self.evaluation_cache[prev_key]
            # Extend evaluation
            remaining = self.validation_data[prev_f:fidelity]
            if remaining:
                new_error = self.llm_evaluator(prompt, remaining)
                self.total_llm_calls += len(remaining)
                total_error = (prev_error * prev_f + new_error * len(remaining)) / fidelity
            else:
                total_error = prev_error
            self.evaluation_cache[cache_key] = total_error
            return total_error

        # Full evaluation
        error = self.llm_evaluator(prompt, self.validation_data[:fidelity])
        self.total_llm_calls += fidelity
        self.evaluation_cache[cache_key] = error
        return error

    def train_gp(self, fidelities: List[int], min_observations: int = 4) -> bool:
        """Train GP on design data from specified fidelity levels.

        Uses same training parameters as inference GP (from config.for_hyperband_gp())
        for consistency between HbBoPs exploration and final GP training.

        Args:
            fidelities: List of fidelity levels to train on (e.g., top 75%)
            min_observations: Minimum samples required
        """
        fidelity_set = set(fidelities)
        fidelity_data = [
            (emb, ve)
            for _, emb, ve, f in self.design_data
            if f in fidelity_set
        ]

        if len(fidelity_data) < min_observations:
            return False

        errors = [ve for _, ve in fidelity_data]
        if np.std(errors) < 1e-6:
            return False

        # Prepare training tensors - convert 768D embeddings to 64D VAE latents
        embeddings = torch.stack([emb for emb, _ in fidelity_data]).to(self.device)
        y = torch.tensor(errors, dtype=torch.float32, device=self.device)

        # Convert to 64D VAE latents (GP expects 64D input for adapter)
        self.vae_with_adapter.eval()
        with torch.no_grad():
            X = self.vae_with_adapter.encode_vae(embeddings)

        # Unit cube normalization for 64D VAE latents
        self.X_min = X.min(dim=0)[0]
        self.X_max = X.max(dim=0)[0]
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        X_norm = (X - self.X_min) / denom

        # Standardization for outputs
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-6
        y_norm = (y - self.y_mean) / self.y_std

        # Create GP with VAEWithAdapter
        # Noise constraint per CLAUDE.md: Interval(0.001, 0.1) for GP stability
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(0.001, 0.1)
        ).to(self.device)
        self.gp_model = InstructionDeepKernelGP(
            X_norm, y_norm, self.likelihood, self.vae_with_adapter.adapter
        ).to(self.device)

        # Use same training params as inference GP for consistency
        gp_params = self.config.for_hyperband_gp()

        # Training
        self.gp_model.train()
        self.likelihood.train()
        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=gp_params["lr"])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss, patience_counter = float('inf'), 0
        final_epoch = 0
        with gpytorch.settings.cholesky_jitter(1e-4):
            for epoch in range(gp_params["epochs"]):
                try:
                    optimizer.zero_grad()
                    loss = -mll(self.gp_model(X_norm), y_norm)
                    loss.backward()
                    optimizer.step()

                    final_epoch = epoch + 1
                    if loss.item() < best_loss:
                        best_loss, patience_counter = loss.item(), 0
                    else:
                        patience_counter += 1
                    if patience_counter >= gp_params["patience"]:
                        break
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "cholesky" in error_msg or "singular" in error_msg or "positive definite" in error_msg:
                        print(f"  GP training: numerical error at epoch {epoch + 1}, stopping early")
                    else:
                        print(f"  GP training error at epoch {epoch + 1}: {e}")
                    if epoch == 0:
                        return False
                    break

        # Store training stats for logging
        self.gp_training_stats = {
            "epochs_trained": final_epoch,
            "final_loss": float(best_loss),
            "early_stopped": patience_counter >= gp_params["patience"],
            "num_samples": len(fidelity_data),
        }

        return True

    def expected_improvement(self, prompt: InstructionOnlyPrompt, vmin_b: float) -> float:
        """Compute Expected Improvement for prompt."""
        if self.gp_model is None:
            return 0.0

        emb = self.instruction_embeddings[prompt.instruction_id].unsqueeze(0)

        # Convert 768D embedding to 64D VAE latent (GP expects 64D)
        self.vae_with_adapter.eval()
        with torch.no_grad():
            z_vae = self.vae_with_adapter.encode_vae(emb)

        # Normalize 64D VAE latent
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        X_norm = (z_vae - self.X_min) / denom

        self.gp_model.eval()
        self.likelihood.eval()

        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
                pred = self.likelihood(self.gp_model(X_norm))
                mean = pred.mean.item() * self.y_std.item() + self.y_mean.item()
                std = pred.stddev.item() * self.y_std.item()
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cholesky" in error_msg or "singular" in error_msg or "positive definite" in error_msg:
                print(f"  WARNING: EI prediction failed (numerical issue), using RANDOM selection")
                print(f"    Error: {e}")
            else:
                print(f"  WARNING: Unexpected EI prediction error: {type(e).__name__}: {e}")
            return 0.0

        if std <= 0:
            return max(vmin_b - mean, 0)

        z = (vmin_b - mean) / std
        return (vmin_b - mean) * norm.cdf(z) + std * norm.pdf(z)

    def get_prompt_proposal(self, evaluated: List[int], vmin_b: float) -> int:
        """Propose next prompt using BO or random interleaving."""
        unevaluated = [i for i in range(len(self.prompts)) if i not in evaluated]

        if not unevaluated:
            return evaluated[0] if evaluated else 0

        # Random interleaving (10%)
        if np.random.rand() < self.config.random_interleaving_prob or self.gp_model is None:
            return np.random.choice(unevaluated)

        # BO proposal: maximize Expected Improvement
        return max(unevaluated, key=lambda i: self.expected_improvement(self.prompts[i], vmin_b))

    def run_hyperband(self, verbose: bool = True) -> Tuple[InstructionOnlyPrompt, float]:
        """Run Hyperband with BO proposals (Algorithm 1).

        Returns:
            (best_prompt, best_error) tuple
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Starting LIPO (Instruction-Only)")
            print("=" * 60)

        # Run brackets from smax down to 0
        for s in range(self.smax, -1, -1):
            if verbose:
                print(f"\nBracket s={s}")

            # Initial prompts and budget for this bracket
            n = int(np.ceil((self.B / self.nvalid) * (self.eta ** s) / (s + 1)))
            b = int(self.nvalid * (self.eta ** (-s)))

            # Cap n at number of available prompts
            n = min(n, len(self.prompts))

            if verbose:
                print(f"  Initial: n={n} prompts, b={b} instances")

            P, V = [], []

            # Minimum fidelity threshold = 75% of full fidelity (top 25% quality data)
            min_fidelity = int(self.config.min_fidelity_pct * self.nvalid)

            for j in range(n):
                # Train GP only on high-fidelity data (>= min_fidelity_pct of full)
                high_fidelity_levels = [
                    f for f in set(f for _, _, _, f in self.design_data)
                    if f >= min_fidelity
                ]
                # Need at least 4 observations at high fidelity
                trainable = [
                    f for f in high_fidelity_levels
                    if sum(1 for _, _, _, ff in self.design_data if ff == f) >= 4
                ]

                if trainable:
                    self.train_gp(trainable)

                # Get vmin_b for acquisition (best error from high-fidelity data)
                vmin_b = float('inf')
                if trainable and self.gp_model:
                    high_fidelity_errors = [
                        ve for _, _, ve, f in self.design_data if f >= min_fidelity
                    ]
                    if high_fidelity_errors:
                        vmin_b = min(high_fidelity_errors)

                # Propose and evaluate prompt
                p_idx = self.get_prompt_proposal(P, vmin_b)
                prompt = self.prompts[p_idx]
                v = self.evaluate_prompt(prompt, b)

                if verbose:
                    print(f"    [{j+1}/{n}] Prompt {p_idx}: error={v:.4f} at fidelity={b}")

                P.append(p_idx)
                V.append(v)

                # Add to design data
                emb = self.instruction_embeddings[p_idx]
                self.design_data.append((p_idx, emb, v, b))

            # Successive Halving stages
            for i in range(1, s + 1):
                ni = int(np.floor(n * (self.eta ** (-i))))
                bi = int(b * (self.eta ** i))

                if ni == 0:
                    break

                if verbose:
                    print(f"  Stage i={i}: n={ni} prompts, b={bi} instances")

                # Keep top prompts
                sorted_idx = np.argsort(V)
                P = [P[idx] for idx in sorted_idx[:ni]]

                # Extend evaluations
                V = []
                for p_idx in P:
                    prompt = self.prompts[p_idx]
                    v = self.evaluate_prompt(prompt, bi)
                    V.append(v)

                    if verbose:
                        print(f"    Prompt {p_idx}: error={v:.4f} at fidelity={bi}")

                    # Update design data
                    emb = self.instruction_embeddings[p_idx]
                    self.design_data.append((p_idx, emb, v, bi))

            # Update best prompt (only from full-fidelity evaluations)
            final_b = int(b * (self.eta ** s)) if s > 0 else b
            if final_b >= self.nvalid:
                for p_idx, v in zip(P, V):
                    if v < self.best_validation_error:
                        self.best_validation_error = v
                        self.best_prompt = self.prompts[p_idx]
                        if verbose:
                            print(f"  New best: error={v:.4f}")

        if verbose:
            print("\n" + "=" * 60)
            print(f"Hyperband complete!")
            print(f"  Best error: {self.best_validation_error:.4f}")
            print(f"  Total LLM calls: {self.total_llm_calls}")
            print("=" * 60)

        return self.best_prompt, self.best_validation_error

    def get_gp_with_ei(self) -> GPWithEI:
        """Convert trained GP to GPWithEI for InvBO inference.

        Uses only high-fidelity observations (>= min_fidelity_pct of full fidelity)
        for consistency with train_gp() during HbBoPs.

        Returns:
            GPWithEI instance ready for inference
        """
        if self.gp_model is None:
            raise RuntimeError("GP not trained. Run run_hyperband() first.")

        gp_with_ei = GPWithEI(device=str(self.device), latent_dim=10)
        gp_with_ei.vae_with_adapter = self.vae_with_adapter

        # Filter to high-fidelity observations only (>= min_fidelity_pct of full)
        min_fidelity = int(self.config.min_fidelity_pct * self.nvalid)
        high_fidelity_levels = sorted(
            set(f for _, _, _, f in self.design_data if f >= min_fidelity),
            reverse=True
        )

        print(f"  GP training data: using fidelity >= {min_fidelity} (levels: {high_fidelity_levels})")

        # Use highest fidelity observations for each prompt (from high-fidelity only)
        best_observations: Dict[int, Tuple[torch.Tensor, float, int]] = {}
        for p_idx, emb, error, fidelity in self.design_data:
            if fidelity < min_fidelity:
                continue
            if p_idx not in best_observations or fidelity > best_observations[p_idx][2]:
                best_observations[p_idx] = (emb, error, fidelity)

        embeddings = torch.stack([obs[0] for obs in best_observations.values()])
        errors = torch.tensor([obs[1] for obs in best_observations.values()], dtype=torch.float32)
        fidelities = torch.tensor([obs[2] for obs in best_observations.values()], dtype=torch.float32)

        print(f"  GP training data: {len(best_observations)} unique prompts")
        print(f"  Fidelity range: [{fidelities.min().item():.0f}, {fidelities.max().item():.0f}]")

        # Pass fidelities for heteroscedastic noise (Bernoulli variance weighting)
        gp_with_ei.set_training_data(embeddings, errors, fidelities)

        # Note: train() will recompute X_min, X_max, y_mean, y_std from training data
        # No need to set them manually here

        # Retrain GP with params from config
        hb_gp_params = self.config.for_hyperband_gp()
        gp_with_ei.train(
            epochs=hb_gp_params["epochs"],
            lr=hb_gp_params["lr"],
            patience=hb_gp_params["patience"],
            verbose=False,
        )

        return gp_with_ei
