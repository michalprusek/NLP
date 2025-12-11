"""
HbBoPs Improved 5: Multi-Fidelity GP WITHOUT Logit Transform

This implementation extends HbBoPs Improved 4 but removes the logit output warping.
GP models accuracy directly with standard binomial variance.

Features:
1. Heteroscedastic noise via binomial variance p(1-p)/n
2. Wilson score variance for extreme p values
3. NO LOGIT TRANSFORM - GP models accuracy directly
4. Multi-fidelity GP using TOP 75% of fidelity levels
5. Model persistence for saving/loading trained GP

Key mathematical differences from Improved 4:
- Accuracy is used directly: y = p = 1 - error (no logit transform)
- Standard binomial variance: sigma^2 = p(1-p)/n
- EI for MAXIMIZATION in accuracy space (higher = better)
"""

import torch
import torch.nn as nn
import gpytorch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Optional
from collections import Counter
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import random

from .gp_model import (
    FeatureExtractor,
    MultiFidelityDeepKernelGP,
    GPNormalizationParams,
    prepare_gp_input
)
from .noise_estimation import compute_heteroscedastic_noise
from .model_persistence import GPModelSaver


@dataclass
class Prompt:
    """A prompt composed of an instruction and few-shot exemplar"""
    instruction: str
    exemplar: str
    instruction_id: int
    exemplar_id: int

    def __str__(self) -> str:
        return f"{self.instruction}\n\n{self.exemplar}"


class PromptEncoder:
    """Encode prompts using BERT [CLS] token embeddings"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to [CLS] token embedding (768 dim)"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embedding.squeeze()


class HbBoPs:
    """
    HbBoPs Improved 5: Multi-Fidelity GP WITHOUT Logit Transform.

    Key improvements over Improved 4:
    1. NO LOGIT TRANSFORM - GP models accuracy directly
    2. Standard binomial variance p(1-p)/n (with Wilson score for extremes)
    3. Product kernel (K_deep x K_fidelity) for multi-fidelity modeling
    4. TOP 75% FIDELITY FILTERING - excludes bottom 25% lowest fidelity data
    5. GP model persistence for later inference
    """

    def __init__(
        self,
        instructions: List[str],
        exemplars: List[str],
        validation_data: List[Dict],
        llm_evaluator,
        encoder_name: str = "bert-base-uncased",
        bmin: int = 10,
        eta: float = 2.0,
        random_interleaving_prob: float = 0.1,
        device: str = "auto",
        seed: int = 42,
        # Parameters for noise estimation
        epsilon: float = 0.001,
        use_wilson_score: bool = True
    ):
        self.instructions = instructions
        self.exemplars = exemplars
        self.llm_evaluator = llm_evaluator

        # Shuffle validation data once for unbiased subsets
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

        # Initialize encoder and pre-compute embeddings
        print("Pre-computing embeddings for all instructions and exemplars...")
        self.encoder = PromptEncoder(encoder_name)
        self._precompute_embeddings()

        # Hyperband parameters
        self.bmin = bmin
        self.eta = eta
        self.random_interleaving_prob = random_interleaving_prob

        # Hyperband schedule
        r = self.nvalid / self.bmin
        self.smax = int(np.floor(np.log(r) / np.log(eta)))
        self.B = (self.smax + 1) * self.nvalid

        print(f"\nHyperband schedule:")
        print(f"  nvalid={self.nvalid}, bmin={bmin}, eta={eta}")
        print(f"  r={r:.1f}, smax={self.smax}, B={self.B}")

        # Design data: (prompt_idx, inst_emb, ex_emb, val_error, fidelity)
        self.design_data = []
        self.evaluation_cache = {}

        # GP model components
        self.gp_model = None
        self.likelihood = None
        self.feature_extractor = None
        self.norm_params = None

        # Fidelity tracking for multi-fidelity GP
        self.fidelity_levels = []
        self.fidelity_to_idx = {}

        # Parameters
        self.epsilon = epsilon
        self.use_wilson_score = use_wilson_score

        # Best prompt tracking
        self.best_prompt = None
        self.best_validation_error = float('inf')

    def _get_device(self, device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _precompute_embeddings(self) -> None:
        """Pre-compute BERT embeddings for all instructions and exemplars."""
        self.instruction_embeddings = {}
        for i_idx, inst in enumerate(self.instructions):
            self.instruction_embeddings[i_idx] = self.encoder.encode(inst)

        self.exemplar_embeddings = {}
        for e_idx, ex in enumerate(self.exemplars):
            self.exemplar_embeddings[e_idx] = self.encoder.encode(ex)

        print(f"  Cached {len(self.instruction_embeddings)} instruction embeddings")
        print(f"  Cached {len(self.exemplar_embeddings)} exemplar embeddings")

    def embed_prompt(self, prompt: Prompt) -> Tuple[np.ndarray, np.ndarray]:
        """Get embeddings for instruction and exemplar (from cache)."""
        return (
            self.instruction_embeddings[prompt.instruction_id],
            self.exemplar_embeddings[prompt.exemplar_id]
        )

    def evaluate_prompt(self, prompt: Prompt, fidelity: int) -> float:
        """Evaluate prompt on 'fidelity' validation instances with caching."""
        cache_key = (prompt.instruction_id, prompt.exemplar_id, fidelity)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        # Try to extend from lower fidelity
        for prev_f in sorted([f for (i, e, f) in self.evaluation_cache.keys()
                              if i == prompt.instruction_id and e == prompt.exemplar_id and f < fidelity],
                             reverse=True):
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

    def _update_fidelity_mapping(self) -> None:
        """Update fidelity level to index mapping based on observed data."""
        observed_fidelities = set(f for _, _, _, _, f in self.design_data)
        self.fidelity_levels = sorted(observed_fidelities)
        self.fidelity_to_idx = {f: i for i, f in enumerate(self.fidelity_levels)}

    def train_gp(self, min_observations: int = 4, fidelity_percentile: float = 0.25) -> bool:
        """
        Train Multi-Fidelity GP with heteroscedastic noise.

        NO LOGIT TRANSFORM - GP models accuracy directly.

        This method:
        1. Uses TOP 75% fidelity data (excludes bottom 25% lowest fidelities)
        2. Converts error_rate to accuracy (1 - error)
        3. Uses standard binomial variance: sigma^2 = p(1-p)/n
        4. Trains MultiFidelityDeepKernelGP with product kernel

        Args:
            min_observations: Minimum number of observations to train GP
            fidelity_percentile: Bottom percentile of fidelities to exclude (default: 0.25 = 25%)

        Returns:
            True if training succeeded, False otherwise
        """
        if len(self.design_data) < min_observations:
            return False

        # Update fidelity mapping
        self._update_fidelity_mapping()
        num_fidelities = len(self.fidelity_levels)

        if num_fidelities == 0:
            return False

        # Filter out bottom 25% of fidelity levels
        # Keep only top 75% of fidelity data for training
        if len(self.fidelity_levels) > 1:
            cutoff_idx = max(1, int(len(self.fidelity_levels) * fidelity_percentile))
            min_fidelity_threshold = self.fidelity_levels[cutoff_idx]
            filtered_data = [d for d in self.design_data if d[4] >= min_fidelity_threshold]
        else:
            filtered_data = self.design_data
            min_fidelity_threshold = self.fidelity_levels[0] if self.fidelity_levels else 0

        if len(filtered_data) < min_observations:
            # Fall back to all data if not enough after filtering
            filtered_data = self.design_data
            min_fidelity_threshold = 0

        # Update fidelity mapping for filtered data
        filtered_fidelities = sorted(set(f for _, _, _, _, f in filtered_data))
        filtered_fidelity_to_idx = {f: i for i, f in enumerate(filtered_fidelities)}

        # Extract data from filtered design_data
        # design_data: (prompt_idx, inst_emb, ex_emb, val_error, fidelity)
        inst_embs = torch.tensor(
            [ie for _, ie, _, _, _ in filtered_data],
            dtype=torch.float32, device=self.device
        )
        ex_embs = torch.tensor(
            [ee for _, _, ee, _, _ in filtered_data],
            dtype=torch.float32, device=self.device
        )
        X = torch.cat([inst_embs, ex_embs], dim=1)

        # Convert error rate to accuracy: p = 1 - error
        error_rates = torch.tensor(
            [ve for _, _, _, ve, _ in filtered_data],
            dtype=torch.float32, device=self.device
        )
        accuracies = 1.0 - error_rates

        # Get fidelities (using filtered data and mapping)
        fidelities = torch.tensor(
            [f for _, _, _, _, f in filtered_data],
            dtype=torch.float32, device=self.device
        )
        fidelity_idx = torch.tensor(
            [filtered_fidelity_to_idx[f] for _, _, _, _, f in filtered_data],
            dtype=torch.long, device=self.device
        )
        num_fidelities = len(filtered_fidelities)

        # Compute heteroscedastic noise: standard binomial variance p(1-p)/n
        # Uses Wilson score for extreme p values
        noise_variance = compute_heteroscedastic_noise(
            accuracies, fidelities,
            epsilon=self.epsilon,
            use_wilson=self.use_wilson_score
        )

        # Input normalization to unit cube [0, 1]
        X_min = X.min(dim=0)[0]
        X_max = X.max(dim=0)[0]
        denominator = X_max - X_min
        denominator[denominator == 0] = 1.0
        X_norm = (X - X_min) / denominator

        # Output standardization (zero mean, unit variance) in accuracy space
        y_mean = accuracies.mean()
        y_std = accuracies.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, device=self.device)
        y_norm = (accuracies - y_mean) / y_std

        # Scale noise variance to match standardized scale
        # Var(y_norm) = Var(y) / y_std^2 => noise_norm = noise / y_std^2
        noise_variance_norm = noise_variance / (y_std ** 2)

        # Clamp minimum noise for numerical stability
        noise_variance_norm = torch.clamp(noise_variance_norm, min=1e-6)

        # Prepare GP input: concatenate embeddings with fidelity index
        # X_with_fidelity has shape (N, 2*768 + 1)
        X_with_fidelity = prepare_gp_input(X_norm, fidelity_idx)

        # Store normalization parameters (for X without fidelity column)
        # Use filtered fidelity mapping for GP predictions
        self.norm_params = GPNormalizationParams(
            X_min=X_min,
            X_max=X_max,
            y_mean=float(y_mean),
            y_std=float(y_std),
            fidelity_to_idx=filtered_fidelity_to_idx.copy(),
            max_fidelity_idx=num_fidelities - 1,
            epsilon=self.epsilon
        )

        # Create model components
        self.feature_extractor = FeatureExtractor(input_dim=768).to(self.device)

        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=noise_variance_norm,
            learn_additional_noise=False
        ).to(self.device)

        self.gp_model = MultiFidelityDeepKernelGP(
            train_x=X_with_fidelity,
            train_y=y_norm,
            likelihood=self.likelihood,
            feature_extractor=self.feature_extractor,
            num_fidelities=num_fidelities,
            input_dim=768
        ).to(self.device)

        # Training
        self.gp_model.train()
        self.likelihood.train()
        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss, patience = float('inf'), 0
        with gpytorch.settings.cholesky_jitter(1e-4):
            for epoch in range(3000):
                try:
                    optimizer.zero_grad()
                    output = self.gp_model(X_with_fidelity)
                    loss = -mll(output, y_norm)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss, patience = loss.item(), 0
                    else:
                        patience += 1
                    if patience >= 10:
                        break
                except Exception as e:
                    if epoch == 0:
                        print(f"GP training failed at epoch 0: {e}")
                        return False
                    break

        return True

    def expected_improvement(self, prompt: Prompt, vmin_b: float) -> float:
        """
        Compute Expected Improvement at highest fidelity level.

        NO LOGIT TRANSFORM - predictions are in accuracy space.

        Args:
            prompt: Prompt to evaluate
            vmin_b: Best error rate observed (lower = better)

        Returns:
            Expected improvement value (higher = more promising)
        """
        if self.gp_model is None or self.norm_params is None:
            return 0.0

        inst_emb, ex_emb = self.embed_prompt(prompt)
        X = torch.tensor(
            np.concatenate([inst_emb, ex_emb]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Normalize input
        X_norm = self.norm_params.normalize_x(X)

        # Use highest fidelity for prediction
        max_fidelity_idx = torch.tensor(
            [self.norm_params.max_fidelity_idx],
            dtype=torch.long, device=self.device
        )

        # Prepare GP input with fidelity
        X_with_fidelity = prepare_gp_input(X_norm, max_fidelity_idx)

        self.gp_model.eval()
        self.likelihood.eval()

        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.cholesky_jitter(1e-4):
                pred = self.likelihood(self.gp_model(X_with_fidelity))

                # Predictions are in standardized accuracy space
                mean_norm = pred.mean.item()
                std_norm = pred.stddev.item()

                # Convert best error rate to standardized accuracy
                # best_error -> best_accuracy -> standardized
                best_acc = 1.0 - vmin_b
                best_acc_norm = (best_acc - self.norm_params.y_mean) / self.norm_params.y_std

        except Exception as e:
            return 0.0

        # EI formula for MAXIMIZATION (we want high accuracy)
        # improvement = mean - best
        if std_norm <= 1e-8:
            return max(mean_norm - best_acc_norm, 0)

        from scipy.stats import norm
        z = (mean_norm - best_acc_norm) / std_norm
        ei = (mean_norm - best_acc_norm) * norm.cdf(z) + std_norm * norm.pdf(z)

        return max(ei, 0)

    def get_prompt_proposal(self, evaluated: List[int], vmin_b: float) -> int:
        """Propose next prompt using BO or random interleaving."""
        unevaluated = [i for i in range(len(self.prompts)) if i not in evaluated]

        if not unevaluated:
            return np.random.choice(range(len(self.prompts)))

        # Random interleaving
        if np.random.rand() < self.random_interleaving_prob or self.gp_model is None:
            return np.random.choice(unevaluated)

        # BO proposal: maximize Expected Improvement
        return max(unevaluated, key=lambda i: self.expected_improvement(self.prompts[i], vmin_b))

    def save_gp_model(self, filepath: str, metadata: Dict = None) -> None:
        """Save trained GP model for later inference."""
        if self.gp_model is None:
            raise RuntimeError("No trained GP model to save")

        GPModelSaver.save(
            filepath=Path(filepath),
            gp_model=self.gp_model,
            likelihood=self.likelihood,
            feature_extractor=self.feature_extractor,
            norm_params=self.norm_params,
            design_data=self.design_data,
            metadata={
                'num_prompts': len(self.prompts),
                'num_observations': len(self.design_data),
                'fidelity_levels': self.fidelity_levels,
                'best_error': self.best_validation_error,
                'best_prompt_idx': self.prompts.index(self.best_prompt) if self.best_prompt else None,
                **(metadata or {})
            }
        )

    def load_gp_model(self, filepath: str) -> None:
        """Load pre-trained GP model."""
        loaded = GPModelSaver.load(
            filepath=Path(filepath),
            device=self.device
        )

        self.gp_model = loaded['gp_model']
        self.likelihood = loaded['likelihood']
        self.feature_extractor = loaded['feature_extractor']
        self.norm_params = loaded['norm_params']
        self.fidelity_to_idx = self.norm_params.fidelity_to_idx
        self.fidelity_levels = sorted(self.fidelity_to_idx.keys())

        if loaded.get('design_data'):
            self.design_data = loaded['design_data']

    def run_hyperband(self, verbose: bool = True, output_dir: str = None) -> Tuple[Prompt, float]:
        """
        Run Hyperband with BO proposals.

        If output_dir is provided, saves the GP model at the end.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Starting HbBoPs Improved 5 optimization")
            print("  - Heteroscedastic noise (binomial variance)")
            print("  - NO logit transform (accuracy space)")
            print("  - Multi-fidelity GP (product kernel)")
            print("  - Top 75% fidelity filtering")
            print("=" * 60)

        # Run brackets from smax down to 0
        for s in range(self.smax, -1, -1):
            if verbose:
                print(f"\nBracket s={s}")

            # Initial prompts and budget for this bracket
            n = int(np.ceil((self.B / self.nvalid) * (self.eta ** s) / (s + 1)))
            b = int(self.nvalid * (self.eta ** (-s)))
            n = min(n, len(self.prompts))

            if verbose:
                print(f"  Initial: n={n} prompts, b={b} instances")

            P, V = [], []

            for j in range(n):
                # Train GP if enough data (uses all fidelities)
                if len(self.design_data) >= 4:
                    self.train_gp()

                # Get vmin_b for acquisition
                vmin_b = float('inf')
                if self.design_data:
                    vmin_b = min(ve for _, _, _, ve, _ in self.design_data)

                # Propose and evaluate prompt
                p_idx = self.get_prompt_proposal(P, vmin_b)
                prompt = self.prompts[p_idx]
                v = self.evaluate_prompt(prompt, b)

                if verbose:
                    print(f"    Evaluated prompt {p_idx} with error {v:.4f} at fidelity {b}")

                P.append(p_idx)
                V.append(v)

                # Add to design data
                inst_emb, ex_emb = self.embed_prompt(prompt)
                self.design_data.append((p_idx, inst_emb, ex_emb, v, b))

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

                    # Update design data
                    inst_emb, ex_emb = self.embed_prompt(prompt)
                    self.design_data.append((p_idx, inst_emb, ex_emb, v, bi))

            # Update best prompt (only from full-fidelity evaluations)
            final_b = int(b * (self.eta ** s)) if s > 0 else b
            if final_b >= self.nvalid:
                for p_idx, v in zip(P, V):
                    if v < self.best_validation_error:
                        self.best_validation_error = v
                        self.best_prompt = self.prompts[p_idx]
                        if verbose:
                            print(f"  New best: error={v:.4f}")

        # Final GP training on all data
        if len(self.design_data) >= 4:
            self.train_gp()

        # Save GP model if output_dir provided
        if output_dir:
            gp_path = Path(output_dir) / "gp_model_final.pt"
            self.save_gp_model(str(gp_path))

        if verbose:
            print("\n" + "=" * 60)
            print(f"HbBoPs complete. Best error: {self.best_validation_error:.4f}")
            print(f"  Total observations: {len(self.design_data)}")
            print(f"  Fidelity levels used: {self.fidelity_levels}")
            print("=" * 60)

        return self.best_prompt, self.best_validation_error
