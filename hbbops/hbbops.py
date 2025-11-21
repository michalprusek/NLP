"""
HbBoPs: Hyperband-based Bayesian Optimization for Black-box Prompt Selection

Implementation based on the paper by Schneider et al. (2025)
"""
import torch
import torch.nn as nn
import gpytorch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


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
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            outputs = self.model(**inputs)
            # Extract [CLS] token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embedding.squeeze()


class FeatureExtractor(nn.Module):
    """
    Structural-aware feature extractor for deep kernel GP

    Processes instruction and exemplar embeddings separately, then combines them
    to learn a low-dimensional (10-dim) latent representation
    """

    def __init__(self, input_dim: int = 768):
        super().__init__()

        # Separate encoders for instruction and exemplar
        self.instruction_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.exemplar_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Joint encoder
        self.joint_encoder = nn.Sequential(
            nn.Linear(32 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, instruction_emb: torch.Tensor, exemplar_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            instruction_emb: (batch_size, 768) - BERT embeddings of instructions
            exemplar_emb: (batch_size, 768) - BERT embeddings of exemplars

        Returns:
            (batch_size, 10) - low-dimensional latent representation
        """
        inst_features = self.instruction_encoder(instruction_emb)
        ex_features = self.exemplar_encoder(exemplar_emb)

        # Concatenate and pass through joint encoder
        combined = torch.cat([inst_features, ex_features], dim=1)
        latent = self.joint_encoder(combined)

        return latent


class StructuralAwareDeepKernelGP(gpytorch.models.ExactGP):
    """
    Gaussian Process with structural-aware deep kernel

    Uses separate embeddings of instructions and few-shot exemplars
    to learn a latent representation aligned with prompt performance
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        feature_extractor: FeatureExtractor,
        input_dim: int = 768
    ):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()

        # ARD Matérn 5/2 kernel on latent features
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=10,  # 10-dimensional latent space
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
        )

        self.feature_extractor = feature_extractor
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Args:
            x: (batch_size, 768 * 2) - concatenated [instruction_emb, exemplar_emb]

        Returns:
            MultivariateNormal distribution
        """
        # Split into instruction and exemplar embeddings
        instruction_emb = x[:, :self.input_dim]
        exemplar_emb = x[:, self.input_dim:]

        # Extract features
        latent_features = self.feature_extractor(instruction_emb, exemplar_emb)

        mean_x = self.mean_module(latent_features)
        covar_x = self.covar_module(latent_features)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HbBoPs:
    """
    Hyperband-based Bayesian Optimization for black-box Prompt Selection

    Combines:
    1. Structural-aware deep kernel GP for sample efficiency
    2. Hyperband for query efficiency (multi-fidelity over validation instances)
    3. Bayesian Optimization proposal for prompt selection
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
        device: str = "auto"
    ):
        """
        Args:
            instructions: List of instruction strings
            exemplars: List of exemplar strings
            validation_data: List of validation examples (dict with 'question' and 'answer')
            llm_evaluator: Function that evaluates prompts on data
            encoder_name: Name of HuggingFace model for embeddings
            bmin: Minimum budget (validation instances) for Hyperband
            eta: Halving parameter for Hyperband
            random_interleaving_prob: Probability of random prompt proposal
            device: Device for computation ('auto', 'cuda', 'cpu', 'mps')
        """
        self.instructions = instructions
        self.exemplars = exemplars
        self.validation_data = validation_data
        self.llm_evaluator = llm_evaluator

        # Generate all candidate prompts
        self.prompts = []
        for i_idx, instruction in enumerate(instructions):
            for e_idx, exemplar in enumerate(exemplars):
                self.prompts.append(Prompt(instruction, exemplar, i_idx, e_idx))

        print(f"Generated {len(self.prompts)} candidate prompts")
        print(f"  {len(instructions)} instructions x {len(exemplars)} exemplars")

        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize encoder
        self.encoder = PromptEncoder(encoder_name)

        # Hyperband parameters
        self.bmin = bmin
        self.eta = eta
        self.nvalid = len(validation_data)
        self.random_interleaving_prob = random_interleaving_prob

        # Compute Hyperband schedule
        r = self.nvalid / self.bmin
        self.smax = int(np.floor(np.log(r) / np.log(eta)))
        self.B = (self.smax + 1) * self.nvalid

        print(f"\nHyperband schedule:")
        print(f"  smax = {self.smax}, B = {self.B}, eta = {eta}")
        print(f"  Validation instances: {self.nvalid}, min budget: {bmin}")

        # Design data: (prompt, instruction_emb, exemplar_emb, validation_error, fidelity)
        self.design_data = []

        # Cache for prompt evaluations (prompt_idx, fidelity) -> validation_error
        self.evaluation_cache = {}

        # GP model (initialized when we have enough data)
        self.gp_model = None
        self.likelihood = None
        self.feature_extractor = None

        # Best prompt tracking
        self.best_prompt = None
        self.best_validation_error = float('inf')
        self.best_fidelity = 0

    def embed_prompt(self, prompt: Prompt) -> Tuple[np.ndarray, np.ndarray]:
        """Embed instruction and exemplar separately"""
        instruction_emb = self.encoder.encode(prompt.instruction)
        exemplar_emb = self.encoder.encode(prompt.exemplar)
        return instruction_emb, exemplar_emb

    def evaluate_prompt(
        self,
        prompt: Prompt,
        fidelity: int,
        prompt_idx: Optional[int] = None
    ) -> float:
        """
        Evaluate a prompt on 'fidelity' validation instances

        Uses caching and extends previous evaluations if available
        """
        if prompt_idx is None:
            prompt_idx = self.prompts.index(prompt)

        # Check cache for exact fidelity
        cache_key = (prompt_idx, fidelity)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        # Check if we can extend previous evaluation
        # Find largest cached fidelity <= current fidelity
        cached_fidelities = [f for (p_idx, f), _ in self.evaluation_cache.items()
                             if p_idx == prompt_idx and f < fidelity]

        if cached_fidelities:
            # Extend from largest cached fidelity
            prev_fidelity = max(cached_fidelities)
            prev_error_sum = self.evaluation_cache[(prompt_idx, prev_fidelity)] * prev_fidelity

            # Evaluate on remaining instances
            remaining_instances = self.validation_data[prev_fidelity:fidelity]
            new_error_sum = self.llm_evaluator(prompt, remaining_instances)

            total_error = (prev_error_sum + new_error_sum) / fidelity
        else:
            # Evaluate from scratch
            instances = self.validation_data[:fidelity]
            total_error = self.llm_evaluator(prompt, instances)

        # Cache result
        self.evaluation_cache[cache_key] = total_error
        return total_error

    def train_gp(self, fidelity: int, min_observations: int = 4):
        """
        Train GP on design data at given fidelity level

        Args:
            fidelity: Use design data at this fidelity level
            min_observations: Minimum number of observations required
        """
        # Filter design data for this fidelity
        fidelity_data = [(inst_emb, ex_emb, val_err)
                         for _, inst_emb, ex_emb, val_err, f in self.design_data
                         if f == fidelity]

        if len(fidelity_data) < min_observations:
            return False

        # Prepare training data
        X_inst = torch.tensor([inst_emb for inst_emb, _, _ in fidelity_data],
                              dtype=torch.float32, device=self.device)
        X_ex = torch.tensor([ex_emb for _, ex_emb, _ in fidelity_data],
                            dtype=torch.float32, device=self.device)
        X = torch.cat([X_inst, X_ex], dim=1)  # (n, 1536)
        y = torch.tensor([val_err for _, _, val_err in fidelity_data],
                        dtype=torch.float32, device=self.device)

        # Normalize inputs to unit cube and standardize outputs
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0) + 1e-6
        X_normalized = (X - X_mean) / X_std

        y_mean = y.mean()
        y_std = y.std() + 1e-6
        y_standardized = (y - y_mean) / y_std

        # Initialize models if needed
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(input_dim=768).to(self.device)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.gp_model = StructuralAwareDeepKernelGP(
                X_normalized, y_standardized, self.likelihood,
                self.feature_extractor, input_dim=768
            ).to(self.device)

        # Set to training mode
        self.gp_model.train()
        self.likelihood.train()

        # Optimizer
        optimizer = torch.optim.AdamW([
            {'params': self.gp_model.feature_extractor.parameters()},
            {'params': self.gp_model.covar_module.parameters()},
            {'params': self.gp_model.mean_module.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)

        # Loss function (negative log marginal likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        max_epochs = 3000
        patience = 10

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            output = self.gp_model(X_normalized)
            loss = -mll(output, y_standardized)
            loss.backward()
            optimizer.step()

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  GP training converged at epoch {epoch + 1}")
                break

        # Store normalization parameters
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        self.y_std = y_std

        return True

    def expected_improvement(self, prompt: Prompt, vmin_b: float) -> float:
        """
        Compute Expected Improvement acquisition function

        Args:
            prompt: Candidate prompt
            vmin_b: Best validation error observed at current fidelity

        Returns:
            Expected improvement value
        """
        if self.gp_model is None:
            return 0.0

        # Embed prompt
        inst_emb, ex_emb = self.embed_prompt(prompt)
        X_inst = torch.tensor(inst_emb, dtype=torch.float32, device=self.device).unsqueeze(0)
        X_ex = torch.tensor(ex_emb, dtype=torch.float32, device=self.device).unsqueeze(0)
        X = torch.cat([X_inst, X_ex], dim=1)

        # Normalize
        X_normalized = (X - self.X_mean) / self.X_std

        # Predict
        self.gp_model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self.gp_model(X_normalized))
            mean = prediction.mean.item() * self.y_std.item() + self.y_mean.item()
            std = prediction.stddev.item() * self.y_std.item()

        # Expected Improvement (we minimize, so improvement = vmin_b - mean)
        improvement = vmin_b - mean

        if std > 0:
            z = improvement / std
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        else:
            ei = max(improvement, 0)

        return ei

    def get_prompt_proposal(self, evaluated_prompts: List[int], vmin_b: float) -> int:
        """
        Propose next prompt to evaluate using BO or random interleaving

        Args:
            evaluated_prompts: List of prompt indices already evaluated
            vmin_b: Best validation error at current fidelity

        Returns:
            Index of prompt to evaluate next
        """
        # Random interleaving
        if np.random.rand() < self.random_interleaving_prob or self.gp_model is None:
            unevaluated = [i for i in range(len(self.prompts)) if i not in evaluated_prompts]
            return np.random.choice(unevaluated)

        # BO proposal: maximize Expected Improvement
        unevaluated = [i for i in range(len(self.prompts)) if i not in evaluated_prompts]
        best_ei = -float('inf')
        best_idx = None

        for idx in unevaluated:
            ei = self.expected_improvement(self.prompts[idx], vmin_b)
            if ei > best_ei:
                best_ei = ei
                best_idx = idx

        return best_idx if best_idx is not None else np.random.choice(unevaluated)

    def run_hyperband(self, verbose: bool = True) -> Tuple[Prompt, float]:
        """
        Run Hyperband with BO proposals

        Returns:
            (best_prompt, best_validation_error)
        """
        if verbose:
            print("\n" + "=" * 80)
            print("Starting HbBoPs optimization")
            print("=" * 80)

        # Run brackets
        for s in range(self.smax, -1, -1):
            if verbose:
                print(f"\nBracket s={s}")

            # Initial number of prompts and budget
            n = int(np.ceil((self.B / self.nvalid) * (self.eta ** s) / (s + 1)))
            b = int(self.nvalid * (self.eta ** (-s)))

            if verbose:
                print(f"  Initial: n={n} prompts, b={b} instances")

            # Sample prompts
            P = []  # Prompt indices
            V = []  # Validation errors

            # Use same random validation instances for all prompts in this stage
            np.random.seed(s)  # Reproducible instance selection per bracket
            stage_instances = np.random.choice(self.nvalid, size=b, replace=False)
            stage_val_data = [self.validation_data[i] for i in sorted(stage_instances)]

            # Propose and evaluate prompts
            for j in range(n):
                # Determine highest fidelity with >= 4 observations for GP training
                highest_fidelity = None
                for fid in sorted(set(f for _, _, _, _, f in self.design_data), reverse=True):
                    fid_count = sum(1 for _, _, _, _, f in self.design_data if f == fid)
                    if fid_count >= 4:
                        highest_fidelity = fid
                        break

                # Train GP if we have enough data
                if highest_fidelity is not None:
                    self.train_gp(highest_fidelity, min_observations=4)

                # Get vmin_b for acquisition function
                if highest_fidelity is not None and self.gp_model is not None:
                    vmin_b = min(val_err for _, _, _, val_err, f in self.design_data
                                if f == highest_fidelity)
                else:
                    vmin_b = float('inf')

                # Propose prompt
                p_idx = self.get_prompt_proposal(P, vmin_b)
                p = self.prompts[p_idx]

                # Evaluate
                v = self.evaluate_prompt(p, b, p_idx)

                P.append(p_idx)
                V.append(v)

                # Update design data
                inst_emb, ex_emb = self.embed_prompt(p)
                self.design_data.append((p_idx, inst_emb, ex_emb, v, b))

                if verbose and (j + 1) % 10 == 0:
                    print(f"    Evaluated {j + 1}/{n} prompts")

            # Successive Halving stages
            for i in range(s):
                # Keep top ni prompts
                ni = int(np.floor(n * (self.eta ** (-i - 1))))
                bi = int(b * (self.eta ** (i + 1)))

                if verbose:
                    print(f"  Stage i={i + 1}: n={ni} prompts, b={bi} instances")

                # Select top prompts
                sorted_indices = np.argsort(V)
                P = [P[idx] for idx in sorted_indices[:ni]]

                # Extend evaluations for remaining prompts
                # Use same instances for all prompts (superset of previous)
                np.random.seed(s * 100 + i)
                stage_instances = np.random.choice(self.nvalid, size=bi, replace=False)
                stage_val_data = [self.validation_data[idx] for idx in sorted(stage_instances)]

                V = []
                for p_idx in P:
                    p = self.prompts[p_idx]
                    v = self.evaluate_prompt(p, bi, p_idx)
                    V.append(v)

                    # Update design data
                    inst_emb, ex_emb = self.embed_prompt(p)
                    self.design_data.append((p_idx, inst_emb, ex_emb, v, bi))

            # Update best prompt (only from those evaluated on full validation set)
            if bi == self.nvalid:
                for p_idx, v in zip(P, V):
                    if v < self.best_validation_error:
                        self.best_validation_error = v
                        self.best_prompt = self.prompts[p_idx]
                        self.best_fidelity = bi
                        if verbose:
                            print(f"  New best validation error: {v:.4f}")

        if verbose:
            print("\n" + "=" * 80)
            print("HbBoPs optimization complete")
            print(f"Best validation error: {self.best_validation_error:.4f}")
            print("=" * 80)

        return self.best_prompt, self.best_validation_error
