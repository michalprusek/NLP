"""
HbBoPs: Hyperband-based Bayesian Optimization for Black-box Prompt Selection

Implementation based on Schneider et al. (2025)
Paper: "Hyperband-based Bayesian Optimization for Black-box Prompt Selection"
"""
import torch
import torch.nn as nn
import gpytorch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
import random


@dataclass # dataclass - automatically generate __init__, __repr__, __eq__ methods
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

    def __init__(self, model_name: str = "bert-base-uncased"): # 12 layers, 768D, 110M params, case agnostic
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval() # turn off dropout and batch norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # move weights to gpu/cpu

    def encode(self, text: str) -> np.ndarray:
        """Encode text to [CLS] token embedding (768 dim)"""
        with torch.no_grad(): # context manager - disable gradient computation (memory and speed efficient)
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device) # "pt" = pytorch, truncate long sequences, max length 512 tokens (BERT), pad short sequences
            outputs = self.model(**inputs) # forward pass, dictionary = (input_ids, attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy() # last hiddent state = [batch, seq_len, 768], [:, 0, :] = [CLS] token embedding, move to cpu, convert to numpy array
        return cls_embedding.squeeze() # (768,) - numpy array vector


class FeatureExtractor(nn.Module):
    """
    Structural-aware feature extractor for deep kernel GP (Section 3.2)

    Architecture from paper:
    - Separate encoders: Lin(768, 64) → ReLU → Lin(64, 32) → ReLU
    - Joint encoder: Lin(64, 32) → ReLU → Lin(32, 10)

     Instruction (768-dim)          Exemplar (768-dim)
          ↓                              ↓
     Lin(768→64)                    Lin(768→64)
        ReLU                           ReLU
     Lin(64→32)                     Lin(64→32)
        ReLU                           ReLU
          ↓                              ↓
      [32-dim]                       [32-dim]
          └──────────┬───────────────┘
                     ↓
              Concatenate → [64-dim]
                     ↓
               Lin(64→32)
                  ReLU
               Lin(32→10)
                     ↓
              Output [10-dim]
    """

    def __init__(self, input_dim: int = 768): # 768 hardcoded for BERT
        super().__init__() # call parent class constructor (nn.Module)

        # Separate encoders for instruction and exemplar (separate weights)
        self.instruction_encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.exemplar_encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        # Joint encoder to 10-dim latent space
        self.joint_encoder = nn.Sequential(
            nn.Linear(2 * 32, 32), nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, instruction_emb: torch.Tensor, exemplar_emb: torch.Tensor) -> torch.Tensor: # called on model(x)
        inst_features = self.instruction_encoder(instruction_emb)
        ex_features = self.exemplar_encoder(exemplar_emb)
        combined = torch.cat([inst_features, ex_features], dim=1) # concatenate along dim 1 (columns)
        return self.joint_encoder(combined)


class DeepKernelGP(gpytorch.models.ExactGP): # inherits from ExactGP - exact inference for small datasets
    """
    Gaussian Process with structural-aware deep kernel (Section 3.2)

    Uses ARD Matérn 5/2 kernel on 10-dim latent features

    Input x [batch, 1536]
      ↓
  ┌───────────────────────────────────────┐
  │         Split (768 | 768)             │
  └───────────────────────────────────────┘
      ↓                    ↓
  [instruction]        [exemplar]
      ↓                    ↓
  ┌───────────────────────────────────────┐
  │       FeatureExtractor (768→10)       │
  └───────────────────────────────────────┘
      ↓
  latent [batch, 10]
      ↓
  ┌───────────────────────────────────────┐
  │  Matérn 5/2 Kernel (ARD, 10 dims)     │
  │  + ScaleKernel                        │
  └───────────────────────────────────────┘
      ↓
  MultivariateNormal(mean=0, K(latent))
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.Likelihood,
                 feature_extractor: FeatureExtractor, input_dim: int = 768): # train_x = [batch, 1536], train_y = [batch, 1], likelihood = GaussianLikelihood(), feature_extractor = FeatureExtractor(), input_dim = 768
        super().__init__(train_x, train_y, likelihood) # call parent class constructor (ExactGP) - cache for kernel computation
        self.mean_module = gpytorch.means.ZeroMean() # assume zero mean
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=10, # matérn 5/2 - 2 times differentiable, 10 latent dims from extractor
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0) # Gamma prior for lengthscale for ARD parameters distribution (mean = 0.5 = average distance odf prompts in latent space)
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15) # Gamma prior for outputscale (variance of GP) (mean = 13.3 = average variance of GP prediction (accuracy), std = 89)
        )

        '''
        # data optimized
          self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
          nu=2.5,
          ard_num_dims=10,
          lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 96.8) # mean = 0.031 = average distance of prompts in latent space
      ),
      outputscale_prior=gpytorch.priors.GammaPrior(0.13, 2.0) # mean = 0.065 = average variance of GP prediction (accuracy)
  )
        '''
        self.feature_extractor = feature_extractor
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal: # called on model(x), x = [batch, 1536]
        # Split concatenated embeddings
        instruction_emb = x[:, :self.input_dim] # 1st half of x
        exemplar_emb = x[:, self.input_dim:] # 2nd half of x
        latent = self.feature_extractor(instruction_emb, exemplar_emb)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(latent), self.covar_module(latent)
        )


class HbBoPs:
    """
    Hyperband-based Bayesian Optimization for black-box Prompt Selection

    Algorithm 1 from the paper
    """

    def __init__(
        self,
        instructions: List[str],
        exemplars: List[str],
        validation_data: List[Dict],
        llm_evaluator,
        encoder_name: str = "bert-base-uncased",
        bmin: int = 10, # min number of validation instances per prompt
        eta: float = 2.0, # reduction factor for bracket size
        random_interleaving_prob: float = 0.1,
        device: str = "auto",
        seed: int = 42
    ):
        self.instructions = instructions
        self.exemplars = exemplars
        self.llm_evaluator = llm_evaluator

        # Shuffle validation data once for unbiased subsets (Appendix C)
        # This ensures "first k" instances are random but fixed for paired comparisons
        random.seed(seed)
        self.validation_data = validation_data.copy()
        random.shuffle(self.validation_data) # shuffle in-place - order is fixed for all prompts = fair comaprison
        self.nvalid = len(validation_data)

        # Generate all candidate prompts (P = I × E) - cartesian product of instructions and exemplars
        self.prompts = [
            Prompt(inst, ex, i_idx, e_idx)
            for i_idx, inst in enumerate(instructions)
            for e_idx, ex in enumerate(exemplars)
        ]
        print(f"Generated {len(self.prompts)} candidate prompts")
        print(f"  {len(instructions)} instructions × {len(exemplars)} exemplars")

        # Device setup
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")

        # Initialize encoder and pre-compute all embeddings (for efficiency)
        print("Pre-computing embeddings for all instructions and exemplars...")
        self.encoder = PromptEncoder(encoder_name)
        self._precompute_embeddings() # creates self.instruction_embeddings and self.exemplar_embeddings

        ################################################################################################################################
        # Hyperband parameters (Algorithm 1)
        ################################################################################################################################
        self.bmin = bmin
        self.eta = eta
        self.random_interleaving_prob = random_interleaving_prob

        # Hyperband schedule from paper:
        r = self.nvalid / self.bmin # ratio of total budget to min budget (r = 131.9)

        # log_η(r) = log(r) / log(η)
        self.smax = int(np.floor(np.log(r) / np.log(eta))) # max number of SH brackets (smax = 7)
        self.B = (self.smax + 1) * self.nvalid # SH theoretical bracket budget B = (7 + 1) × 1319 = 10552 evaluations - used for number of prompts computation
        ################################################################################################################################
        # END
        ################################################################################################################################

        print(f"\nHyperband schedule (from paper):")
        print(f"  nvalid={self.nvalid}, bmin={bmin}, η={eta}")
        print(f"  r={r:.1f}, smax={self.smax}, B={self.B}")

        self.design_data = [] # Design data: (prompt_idx, inst_emb, ex_emb, val_error, fidelity)
        self.evaluation_cache = {} # Cache {(inst_id, ex_id, fidelity): error}

        # GP model (initialized when enough data)
        self.gp_model = None
        self.likelihood = None
        self.feature_extractor = None

        # Normalization parameters for unit cube scaling
        #   X (embeddings):  [0.1, 0.5, 0.9] → normalize → [0, 0.5, 1]  (unit cube)
        #   y (accuracy):    [0.3, 0.5, 0.7] → standardize → [-1, 0, 1]  (z-score)
        self.X_min = None
        self.X_max = None
        self.y_mean = None
        self.y_std = None

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
        """
        Pre-compute BERT embeddings for all instructions and exemplars.

        Since P = I × E is fixed (static setting), embeddings don't change.
        This dramatically speeds up GP training and acquisition computation.
        """
        # Embed all unique instructions
        self.instruction_embeddings = {}
        for i_idx, inst in enumerate(self.instructions):
            self.instruction_embeddings[i_idx] = self.encoder.encode(inst)

        # Embed all unique exemplars
        self.exemplar_embeddings = {}
        for e_idx, ex in enumerate(self.exemplars):
            self.exemplar_embeddings[e_idx] = self.encoder.encode(ex)

        print(f"  Cached {len(self.instruction_embeddings)} instruction embeddings")
        print(f"  Cached {len(self.exemplar_embeddings)} exemplar embeddings")

    def embed_prompt(self, prompt: Prompt) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for instruction and exemplar (from cache).

        Returns separate embeddings to leverage structural-aware deep kernel.
        """
        return (
            self.instruction_embeddings[prompt.instruction_id],
            self.exemplar_embeddings[prompt.exemplar_id]
        )

    def evaluate_prompt(self, prompt: Prompt, fidelity: int) -> float:
        """Evaluate prompt on 'fidelity' validation instances with caching"""
        cache_key = (prompt.instruction_id, prompt.exemplar_id, fidelity) # Example: (5, 12, 80) = instruction #5, exemplar #12, fidelity 80
        if cache_key in self.evaluation_cache: # if already evaluated at this fidelity
            return self.evaluation_cache[cache_key]

        # Try to extend from lower fidelity
        for prev_f in sorted([f for (i, e, f) in self.evaluation_cache.keys()
                              if i == prompt.instruction_id and e == prompt.exemplar_id and f < fidelity],
                             reverse=True): # find all lower fidelities for this specific prompt
            '''
              example: we want fidelity=80, in cache we have:
                (5, 12, 10) → 0.40
                (5, 12, 20) → 0.35
                (5, 12, 20) → 0.38
            after sorted reverse: [40, 20, 10] → We start from fidelity 40 and extend
            '''
            prev_key = (prompt.instruction_id, prompt.exemplar_id, prev_f)
            prev_error = self.evaluation_cache[prev_key]
            # Extend evaluation
            remaining = self.validation_data[prev_f:fidelity] # get remaining data to evaluate (data are shuffled once at the beginning and fixed for all prompts)
            new_error = self.llm_evaluator(prompt, remaining)
            total_error = (prev_error * prev_f + new_error * len(remaining)) / fidelity # weighted average of previous error and new error
            self.evaluation_cache[cache_key] = total_error
            return total_error

        # Full evaluation - evaluate from scratch (if not cached)
        error = self.llm_evaluator(prompt, self.validation_data[:fidelity])
        self.evaluation_cache[cache_key] = error
        return error

    def train_gp(self, fidelity: int, min_observations: int = 4) -> bool:
        """
        Train GP on design data at given fidelity level.

        Paper Section 4.2: "normalizes inputs to the unit cube and standardizes outputs"
        - Inputs (X): Unit cube normalization (min-max to [0, 1])
        - Outputs (y): Standardization (zero mean, unit variance)
        """
        # Filter data for this fidelity
        fidelity_data = [(ie, ee, ve) for _, ie, ee, ve, f in self.design_data if f == fidelity] # design_data: (prompt_idx, inst_emb, ex_emb, val_error, fidelity)

        if len(fidelity_data) < min_observations: # at least 4 prompts evaluated at given fidelity
            return False

        errors = [ve for _, _, ve in fidelity_data]
        if np.std(errors) < 1e-6:  # No variance - GP not needed - we cannot model variance
            return False

        # Prepare training tensors
        X_inst = torch.tensor([ie for ie, _, _ in fidelity_data], dtype=torch.float32, device=self.device)
        X_ex = torch.tensor([ee for _, ee, _ in fidelity_data], dtype=torch.float32, device=self.device)
        X = torch.cat([X_inst, X_ex], dim=1)
        y = torch.tensor(errors, dtype=torch.float32, device=self.device)

        # Unit cube normalization for inputs: X_norm = (X - X_min) / (X_max - X_min)
        self.X_min = X.min(dim=0)[0] # (1536,) - min for each dimension
        self.X_max = X.max(dim=0)[0] # (1536,) - max for each dimension
        denominator = self.X_max - self.X_min
        denominator[denominator == 0] = 1.0  # Avoid division by zero for constant columns
        X_norm = (X - self.X_min) / denominator

        # Standardization for outputs: y_norm = (y - mean) / std
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-6
        y_norm = (y - self.y_mean) / self.y_std

        # Create fresh GP model
        self.feature_extractor = FeatureExtractor(input_dim=768).to(self.device) # latent_dim = 10
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device) # constant noise
        self.gp_model = DeepKernelGP(X_norm, y_norm, self.likelihood, self.feature_extractor).to(self.device)

        # Training with AdamW (Section 4.2: lr=0.01, max_epochs=3000, patience=10)
        self.gp_model.train()
        self.likelihood.train()
        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss, patience = float('inf'), 0
        with gpytorch.settings.cholesky_jitter(1e-4): # numerical stability of cholesky decomposition 1) K = LLᵀ  →  K⁻¹y = L⁻ᵀ(L⁻¹y) 2) MLL: log|K| = log|LLᵀ| = 2 × Σᵢ log(Lᵢᵢ)
            for epoch in range(3000):
                try:
                    optimizer.zero_grad() # reset gradients from previous iteration
                    loss = -mll(self.gp_model(X_norm), y_norm) # we maximize MLL
                    loss.backward()
                    optimizer.step() #backprop

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
        """
        Compute Expected Improvement acquisition function (Equation 7).

        EI(p) = E[max{v_min,b - f(z_p), 0}]

        For minimization: EI = (v_min - μ) × Φ(z) + σ × φ(z)
        where z = (v_min - μ) / σ
        """
        if self.gp_model is None:
            return 0.0

        inst_emb, ex_emb = self.embed_prompt(prompt)
        X = torch.tensor(np.concatenate([inst_emb, ex_emb]), dtype=torch.float32, device=self.device).unsqueeze(0)

        # Apply same unit cube normalization as in training
        denominator = self.X_max - self.X_min
        denominator[denominator == 0] = 1.0
        X_norm = (X - self.X_min) / denominator

        # eval mode
        self.gp_model.eval()
        self.likelihood.eval()

        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
                pred = self.likelihood(self.gp_model(X_norm))
                # De-standardize predictions
                mean = pred.mean.item() * self.y_std.item() + self.y_mean.item()
                std = pred.stddev.item() * self.y_std.item()
        except Exception:
            return 0.0

        # EI formula (we minimize, so improvement = vmin - mean)
        # v_min = best observed error rate so far (lower = better)
        if std <= 0: # deterministic case
            return max(vmin_b - mean, 0)
        from scipy.stats import norm
        z = (vmin_b - mean) / std
        return (vmin_b - mean) * norm.cdf(z) + std * norm.pdf(z) # closed form solution for accuracy normally distributed

    def get_prompt_proposal(self, evaluated: List[int], vmin_b: float) -> int:
        """Propose next prompt using BO or random interleaving"""
        unevaluated = [i for i in range(len(self.prompts)) if i not in evaluated]

        # Random interleaving (Section 3.4)
        if np.random.rand() < self.random_interleaving_prob or self.gp_model is None:
            return np.random.choice(unevaluated)

        # BO proposal: maximize Expected Improvement
        return max(unevaluated, key=lambda i: self.expected_improvement(self.prompts[i], vmin_b))

    def run_hyperband(self, verbose: bool = True) -> Tuple[Prompt, float]:
        """
        Run Hyperband with BO proposals (Algorithm 1)
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Starting HbBoPs optimization")
            print("=" * 60)

        ################################################################################################################################
        # ALGORITHM 1: Hyperband with Bayesian Optimization (HbBoPs)
        ################################################################################################################################

        # Run brackets from smax down to 0
        for s in range(self.smax, -1, -1):
            if verbose:
                print(f"\nBracket s={s}")

            # Initial prompts and budget for this bracket
            n = int(np.ceil((self.B / self.nvalid) * (self.eta ** s) / (s + 1)))
            b = int(self.nvalid * (self.eta ** (-s)))

            # cap n at available prompts
            n = min(n, len(self.prompts))

            if verbose:
                print(f"  Initial: n={n} prompts, b={b} instances")

            # Sample initial prompts P = list of prompt indices, V = list of validation errors for each prompt
            P, V = [], []

            for j in range(n):
                # Train GP if enough data
                fidelities = {}
                for _, _, _, _, f in self.design_data: # design_data: (prompt_idx, inst_emb, ex_emb, val_error, fidelity)
                    fidelities[f] = fidelities.get(f, 0) + 1 # count number of observations for each fidelity
                trainable = [f for f, c in fidelities.items() if c >= 4] # filter out fidelities with less than 4 observations
                if trainable:
                    self.train_gp(max(trainable)) # train on highest fidelity with at least 4 prompts evaluated

                # Get vmin_b (champion) for acquisition
                vmin_b = float('inf')
                if trainable and self.gp_model:
                    vmin_b = min(ve for _, _, _, ve, f in self.design_data if f == max(trainable)) # best error at highest fidelity

                # Propose and evaluate prompt (proposal based on evaluated prompts P and vmin_b (champion) - best observer error rate so far)
                p_idx = self.get_prompt_proposal(P, vmin_b)
                prompt = self.prompts[p_idx]
                v = self.evaluate_prompt(prompt, b)

                P.append(p_idx)
                V.append(v)

                # Add to design data
                inst_emb, ex_emb = self.embed_prompt(prompt)
                self.design_data.append((p_idx, inst_emb, ex_emb, v, b))

                if verbose and (j + 1) % 10 == 0:
                    print(f"    Evaluated {j + 1}/{n} prompts")

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
            if final_b >= self.nvalid: # have to be evaluated at full fidelity
                for p_idx, v in zip(P, V):
                    if v < self.best_validation_error:
                        self.best_validation_error = v
                        self.best_prompt = self.prompts[p_idx]
                        if verbose:
                            print(f"  New best: error={v:.4f}")

        if verbose:
            print("\n" + "=" * 60)
            print(f"HbBoPs complete. Best error: {self.best_validation_error:.4f}")
            print("=" * 60)

        return self.best_prompt, self.best_validation_error
