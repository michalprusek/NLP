"""Hyperband for joint (instruction, exemplar_set) optimization.

Implements successive halving with BO proposals over the infinite search space
of instruction × exemplar subset combinations.

Key innovation: Two-stage proposal mechanism:
1. Sample/select instruction via EI
2. Optimize exemplar latent z_ex via gradient to maximize EI
3. Decode z_ex to discrete exemplar selection via SlotBasedExemplarDecoder
"""

import json
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from bolt.config import BOLTConfig
from bolt.encoder import GTREncoder, StructureAwareVAE
from bolt.gp import GPWithEI
from bolt.training import QAPair


@dataclass
class JointPrompt:
    """A prompt with instruction and variable-length exemplar set."""
    instruction: str
    instruction_id: int
    exemplar_ids: Tuple[int, ...]  # Sorted tuple for consistent hashing
    num_exemplars: int

    def __hash__(self) -> int:
        return hash((self.instruction_id, self.exemplar_ids))

    def __eq__(self, other) -> bool:
        if not isinstance(other, JointPrompt):
            return False
        return (self.instruction_id == other.instruction_id and
                self.exemplar_ids == other.exemplar_ids)

    def base_key(self) -> Tuple[int, FrozenSet[int]]:
        """Cache key without fidelity."""
        return (self.instruction_id, frozenset(self.exemplar_ids))


def fit_beta_prior(error_rates: List[float]) -> Tuple[float, float]:
    """Fit Beta prior using Method of Moments (Empirical Bayes).

    Args:
        error_rates: List of observed error rates [0, 1]

    Returns:
        (alpha, beta) parameters of Beta distribution
    """
    if len(error_rates) < 2:
        return 1.0, 1.0  # Uniform prior

    errors = np.array(error_rates)
    errors = np.clip(errors, 0.01, 0.99)  # Avoid edge cases

    mean = errors.mean()
    var = errors.var()

    # Method of moments
    if var < 1e-8 or var >= mean * (1 - mean):
        # Variance too small (all same) or too high for Beta - use weak prior
        return 1.0, 1.0

    common = mean * (1 - mean) / var - 1
    alpha = mean * common
    beta = (1 - mean) * common

    # Clamp to reasonable range
    alpha = np.clip(alpha, 0.5, 10.0)
    beta = np.clip(beta, 0.5, 10.0)

    return float(alpha), float(beta)


def beta_smooth_error(
    num_errors: int,
    fidelity: int,
    alpha: float,
    beta: float,
) -> float:
    """Compute Beta posterior mean (smoothed error rate).

    Args:
        num_errors: Number of incorrect answers
        fidelity: Total samples evaluated
        alpha, beta: Beta prior parameters

    Returns:
        Smoothed error rate
    """
    denominator = fidelity + alpha + beta
    if denominator <= 0:
        # Return neutral value for invalid input (uninformative prior)
        return 0.5
    return (num_errors + alpha) / denominator


class BOLTHyperband:
    """Hyperband for joint (instruction, exemplar_set) optimization.

    Key features:
    - Two-stage BO proposal: instruction selection + exemplar latent optimization
    - Fidelity extension caching (reuse lower-fidelity evaluations)
    - Beta smoothing + heteroscedastic noise from fidelity
    - Successive halving with adaptive brackets
    """

    def __init__(
        self,
        instructions: List[str],
        qa_pool: List[QAPair],
        validation_data: List[Dict],
        vae: StructureAwareVAE,
        gtr_encoder: GTREncoder,
        pool_embeddings: torch.Tensor,
        instruction_embeddings: torch.Tensor,
        llm_evaluator: Callable,
        config: BOLTConfig,
        device: str = "cuda",
        batch_llm_evaluator: Optional[Callable] = None,
    ):
        """Initialize Hyperband.

        Args:
            instructions: List of instruction texts (2000 APE)
            qa_pool: List of Q/A pairs (6154 from train.json)
            validation_data: Validation samples for evaluation
            vae: Trained StructureAwareVAE for encoding
            gtr_encoder: GTR encoder for embeddings
            pool_embeddings: Pre-computed pool embeddings (N_pool, 768)
            instruction_embeddings: Pre-computed instruction embeddings (N_inst, 768)
            llm_evaluator: Function to evaluate prompts (single candidate)
            config: Configuration
            device: Device
            batch_llm_evaluator: Optional function to evaluate multiple candidates at once
        """
        self.instructions = instructions
        self.qa_pool = qa_pool
        self.validation_data = validation_data
        self.vae = vae
        self.gtr_encoder = gtr_encoder
        self.pool_embeddings = pool_embeddings.to(device)
        self.instruction_embeddings = instruction_embeddings.to(device)
        self.llm_evaluator = llm_evaluator
        self.batch_llm_evaluator = batch_llm_evaluator
        self.config = config
        self.device = device

        # Shuffle validation data once for consistent fidelity subsets
        random.seed(config.seed)
        self.validation_data = validation_data.copy()
        random.shuffle(self.validation_data)
        self.nvalid = len(validation_data)

        # Hyperband parameters
        self.bmin = config.bmin
        self.eta = config.eta
        r = self.nvalid / self.bmin
        self.smax = int(np.floor(np.log(r) / np.log(self.eta)))
        self.B = (self.smax + 1) * self.nvalid

        print(f"\nHyperband schedule:")
        print(f"  nvalid={self.nvalid}, bmin={self.bmin}, eta={self.eta}")
        print(f"  smax={self.smax}, B={self.B}")

        # Evaluation cache: (inst_id, frozenset(exemplar_ids), fidelity) -> error_rate
        self.evaluation_cache: Dict[Tuple, float] = {}
        # Track fidelities per base key for extension
        self.fidelity_index: Dict[Tuple[int, FrozenSet[int]], List[int]] = defaultdict(list)

        # Design data for GP training: [(prompt, z_latent, error_rate, fidelity), ...]
        self.design_data: List[Tuple[JointPrompt, torch.Tensor, float, int]] = []

        # GP
        self.gp: Optional[GPWithEI] = None

        # Beta prior (updated as we observe more)
        self.alpha = 1.0
        self.beta = 1.0

        # Best result tracking (any fidelity - for debugging)
        self.best_prompt: Optional[JointPrompt] = None
        self.best_error: float = 1.0

        # Best result at full fidelity (reliable estimate)
        self.best_prompt_full_fidelity: Optional[JointPrompt] = None
        self.best_error_full_fidelity: float = 1.0

        # Stats
        self.total_llm_calls = 0

    def _make_cache_key(
        self,
        instruction_id: int,
        exemplar_ids: List[int],
        fidelity: int,
    ) -> Tuple[int, FrozenSet[int], int]:
        """Create cache key for evaluation."""
        return (instruction_id, frozenset(exemplar_ids), fidelity)

    def _index_fidelity(self, base_key: Tuple[int, FrozenSet[int]], fidelity: int) -> None:
        """Index fidelity for a base key."""
        if fidelity not in self.fidelity_index[base_key]:
            self.fidelity_index[base_key].append(fidelity)

    def _build_exemplar_text(self, prompt: JointPrompt) -> str:
        """Build exemplar text from prompt's exemplar IDs."""
        if not prompt.exemplar_ids:
            return ""
        exemplars = [self.qa_pool[i] for i in prompt.exemplar_ids]
        return "\n\n".join(qa.format() for qa in exemplars)

    def _cache_result(self, prompt: JointPrompt, error_rate: float, fidelity: int) -> None:
        """Cache evaluation result for a prompt."""
        cache_key = self._make_cache_key(
            prompt.instruction_id,
            list(prompt.exemplar_ids),
            fidelity,
        )
        self.evaluation_cache[cache_key] = error_rate
        self._index_fidelity(prompt.base_key(), fidelity)

    def evaluate_prompt(
        self,
        prompt: JointPrompt,
        fidelity: int,
    ) -> Tuple[float, int]:
        """Evaluate prompt with caching and fidelity extension.

        Args:
            prompt: Prompt to evaluate
            fidelity: Number of validation samples

        Returns:
            (error_rate, num_errors) - raw counts for Beta smoothing
        """
        cache_key = self._make_cache_key(
            prompt.instruction_id,
            list(prompt.exemplar_ids),
            fidelity,
        )

        # Check exact cache hit
        if cache_key in self.evaluation_cache:
            error_rate = self.evaluation_cache[cache_key]
            num_errors = int(error_rate * fidelity)
            return error_rate, num_errors

        # Try fidelity extension
        base_key = prompt.base_key()
        lower_fidelities = sorted(
            [f for f in self.fidelity_index.get(base_key, []) if f < fidelity],
            reverse=True,
        )

        for prev_f in lower_fidelities:
            prev_key = self._make_cache_key(
                prompt.instruction_id,
                list(prompt.exemplar_ids),
                prev_f,
            )
            if prev_key in self.evaluation_cache:
                prev_error = self.evaluation_cache[prev_key]
                prev_num_errors = int(prev_error * prev_f)

                # Evaluate only remaining samples
                remaining = self.validation_data[prev_f:fidelity]
                new_error, new_errors = self._evaluate_raw(prompt, remaining)
                self.total_llm_calls += len(remaining)

                # Combine
                total_errors = prev_num_errors + new_errors
                total_error = total_errors / fidelity

                self.evaluation_cache[cache_key] = total_error
                self._index_fidelity(base_key, fidelity)

                return total_error, total_errors

        # Full evaluation
        error_rate, num_errors = self._evaluate_raw(
            prompt,
            self.validation_data[:fidelity],
        )
        self.total_llm_calls += fidelity

        self.evaluation_cache[cache_key] = error_rate
        self._index_fidelity(base_key, fidelity)

        return error_rate, num_errors

    def _evaluate_raw(
        self,
        prompt: JointPrompt,
        samples: List[Dict],
    ) -> Tuple[float, int]:
        """Raw evaluation without caching.

        Returns:
            (error_rate, num_errors)
        """
        if not samples:
            return 0.0, 0

        error_rate = self.llm_evaluator(
            instruction=prompt.instruction,
            exemplar_text=self._build_exemplar_text(prompt),
            samples=samples,
        )
        return error_rate, int(error_rate * len(samples))

    def _encode_prompt_to_latent(self, prompt: JointPrompt) -> torch.Tensor:
        """Encode prompt to 32D joint latent."""
        inst_emb = self.instruction_embeddings[prompt.instruction_id].unsqueeze(0)

        if prompt.exemplar_ids:
            ex_embs = torch.stack([
                self.pool_embeddings[i] for i in prompt.exemplar_ids
            ]).unsqueeze(0)
            ex_mask = torch.ones(1, len(prompt.exemplar_ids), dtype=torch.bool, device=self.device)
        else:
            ex_embs = torch.zeros(1, 1, 768, device=self.device)
            ex_mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            self.vae.eval()
            z_joint = self.vae.encode_joint(inst_emb, ex_embs, ex_mask)

        return z_joint.squeeze()

    def _random_proposal(self) -> JointPrompt:
        """Generate random (instruction, exemplar_set) proposal with fixed K=8."""
        inst_id = random.randint(0, len(self.instructions) - 1)
        K = self.config.num_exemplars  # Fixed K=8

        exemplar_ids = tuple(sorted(random.sample(range(len(self.qa_pool)), K)))

        return JointPrompt(
            instruction=self.instructions[inst_id],
            instruction_id=inst_id,
            exemplar_ids=exemplar_ids,
            num_exemplars=K,
        )

    def _compute_ucb(
        self,
        z: torch.Tensor,
        beta: float = 8.0,
    ) -> torch.Tensor:
        """Compute Upper Confidence Bound for minimization.

        UCB(z) = -mu(z) + beta * sigma(z)

        We negate because GP predicts error (lower is better),
        but we want to maximize the acquisition function.
        High beta = more exploration.
        """
        if self.gp is None or self.gp.gp_model is None:
            # GP not trained yet, return neutral UCB value
            return torch.tensor(0.0, device=self.device)

        mean, std = self.gp.predict(z.unsqueeze(0))
        mean = mean.squeeze()
        std = std.squeeze()

        # UCB for minimization: prefer low mean + high uncertainty
        ucb = -mean + beta * std
        return ucb

    def _train_gp(self, min_observations: int = 4) -> bool:
        """Train GP on all design data with Beta smoothing.

        Uses all fidelity levels with heteroscedastic noise.
        """
        if len(self.design_data) < min_observations:
            return False

        # Update Beta prior from raw errors
        raw_errors = [error for _, _, error, _ in self.design_data]
        self.alpha, self.beta = fit_beta_prior(raw_errors)

        # Prepare training data with Beta smoothing
        latents = []
        smoothed_errors = []

        for _, z, error, fidelity in self.design_data:
            num_errors = int(error * fidelity)
            smoothed = beta_smooth_error(num_errors, fidelity, self.alpha, self.beta)
            latents.append(z)
            smoothed_errors.append(smoothed)

        X = torch.stack(latents).to(self.device)
        y = torch.tensor(smoothed_errors, device=self.device)
        fids = torch.tensor([d[3] for d in self.design_data], device=self.device)

        # Create and train GP (with optional Deep Kernel Learning)
        self.gp = GPWithEI(
            instruction_dim=self.config.instruction_latent_dim,
            exemplar_dim=self.config.exemplar_latent_dim,
            device=self.device,
            use_deep_kernel=self.config.use_deep_kernel,
            dkl_output_dim=self.config.dkl_output_dim,
            dkl_hidden_dim=self.config.dkl_hidden_dim,
            use_product_kernel=self.config.use_product_kernel,
        )

        # Override noise computation with our Beta variance
        self.gp.fit(
            X=X,
            y=y,
            fidelities=fids,
            epochs=self.config.gp_epochs,
            lr=self.config.gp_lr,
            patience=self.config.gp_patience,
        )

        return True

    def propose_prompt(
        self,
        evaluated_base_keys: Set[Tuple[int, FrozenSet[int]]],
    ) -> JointPrompt:
        """Propose next prompt using BO or random.

        Two-stage proposal:
        1. Random interleaving (10%): random proposal
        2. BO (90%): optimize UCB over instruction × exemplar latent
        """
        # Random interleaving
        if (random.random() < self.config.random_interleaving_prob or
            self.gp is None or
            self.gp.gp_model is None):
            return self._random_proposal()

        # BO proposal - sample candidate instructions
        candidate_inst_ids = random.sample(
            range(len(self.instructions)),
            min(50, len(self.instructions)),
        )

        best_ucb = -float('inf')
        best_prompt = None

        for inst_id in candidate_inst_ids:
            # Get instruction latent
            inst_emb = self.instruction_embeddings[inst_id].unsqueeze(0)
            with torch.no_grad():
                # InstructionEncoder.forward() returns (mu, logvar) tuple
                mu_inst, _ = self.vae.instruction_encoder(inst_emb)
            z_inst = mu_inst.squeeze()

            # Optimize exemplar latent using UCB
            z_ex_opt, exemplar_ids, ucb = self._optimize_exemplar_latent(z_inst)

            if ucb > best_ucb:
                # Check if not already evaluated
                base_key = (inst_id, frozenset(exemplar_ids))
                if base_key not in evaluated_base_keys:
                    best_ucb = ucb
                    best_prompt = JointPrompt(
                        instruction=self.instructions[inst_id],
                        instruction_id=inst_id,
                        exemplar_ids=tuple(sorted(exemplar_ids)),
                        num_exemplars=len(exemplar_ids),
                    )

        if best_prompt is None:
            # Fallback to random - all candidates were already evaluated
            print("  [BO] No unevaluated candidates found, falling back to random proposal")
            return self._random_proposal()

        return best_prompt

    def _optimize_exemplar_latent(
        self,
        z_inst: torch.Tensor,
        num_restarts: int = 5,
        num_steps: int = 20,
    ) -> Tuple[torch.Tensor, List[int], float]:
        """Optimize exemplar latent z_ex to maximize UCB.

        Uses UCB with beta=8.0 for exploration.

        Returns:
            (z_ex, exemplar_ids, ucb_value)
        """
        exemplar_dim = self.config.exemplar_latent_dim
        beta = self.config.ucb_beta  # Default 8.0

        best_z_ex = None
        best_ucb = -float('inf')
        best_exemplar_ids = []

        for _ in range(num_restarts):
            # Random initialization
            z_ex = torch.randn(exemplar_dim, device=self.device) * 0.5
            z_ex.requires_grad_(True)

            optimizer = torch.optim.Adam([z_ex], lr=0.1)

            for _ in range(num_steps):
                optimizer.zero_grad()
                z_joint = torch.cat([z_inst, z_ex])
                ucb = self._compute_ucb(z_joint, beta=beta)
                if ucb.requires_grad:
                    (-ucb).backward()  # Maximize UCB
                    optimizer.step()

            with torch.no_grad():
                z_joint = torch.cat([z_inst, z_ex])
                ucb_final = self._compute_ucb(z_joint, beta=beta).item()
                # Pass both z_inst and z_ex to decoder (scorer needs both)
                exemplar_ids = self._decode_exemplar_latent(z_inst, z_ex)

            if ucb_final > best_ucb:
                best_ucb = ucb_final
                best_z_ex = z_ex.detach()
                best_exemplar_ids = exemplar_ids

        return best_z_ex, best_exemplar_ids, best_ucb

    def _decode_exemplar_latent(
        self,
        z_inst: torch.Tensor,
        z_ex: torch.Tensor,
    ) -> List[int]:
        """Decode exemplar latent to discrete selection using scorer.

        Uses ExemplarScorer to get top-k exemplars (fixed K=8).

        Args:
            z_inst: Instruction latent vector
            z_ex: Exemplar latent vector

        Returns:
            List of K=8 exemplar pool indices
        """
        K = self.config.num_exemplars  # Fixed K=8

        with torch.no_grad():
            self.vae.eval()

            z_inst_batch = z_inst.unsqueeze(0)
            z_ex_batch = z_ex.unsqueeze(0)

            # Use scorer to select top-k exemplars
            indices, _ = self.vae.scorer.select_top_k(
                z_inst_batch,
                z_ex_batch,
                self.pool_embeddings,
                k=K,
            )

            exemplar_ids = indices.squeeze(0).tolist()

        return exemplar_ids

    def _evaluate_batch(
        self,
        prompts: List[JointPrompt],
        fidelity: int,
    ) -> List[Tuple[float, int]]:
        """Batch evaluate multiple prompts at once.

        Uses batch_llm_evaluator if available, otherwise falls back to sequential.
        """
        if not prompts:
            return []

        # Check cache and partition prompts
        results: List[Optional[Tuple[float, int]]] = [None] * len(prompts)
        uncached: List[Tuple[int, JointPrompt]] = []

        for i, prompt in enumerate(prompts):
            cache_key = self._make_cache_key(
                prompt.instruction_id,
                list(prompt.exemplar_ids),
                fidelity,
            )
            if cache_key in self.evaluation_cache:
                error_rate = self.evaluation_cache[cache_key]
                results[i] = (error_rate, int(error_rate * fidelity))
            else:
                uncached.append((i, prompt))

        if not uncached:
            return results

        # Evaluate uncached prompts
        if self.batch_llm_evaluator is not None:
            candidates = [
                (prompt.instruction, self._build_exemplar_text(prompt))
                for _, prompt in uncached
            ]
            samples = self.validation_data[:fidelity]

            try:
                error_rates = self.batch_llm_evaluator(candidates, samples)
                self.total_llm_calls += len(uncached) * fidelity

                for (idx, prompt), error_rate in zip(uncached, error_rates):
                    results[idx] = (error_rate, int(error_rate * fidelity))
                    self._cache_result(prompt, error_rate, fidelity)
            except RuntimeError as e:
                # Batch evaluation failed - fall back to sequential
                print(f"[WARNING] Batch evaluation failed: {e}")
                print("  Falling back to sequential evaluation...")
                for idx, prompt in uncached:
                    error, num_errors = self.evaluate_prompt(prompt, fidelity)
                    results[idx] = (error, num_errors)
        else:
            for idx, prompt in uncached:
                error, num_errors = self.evaluate_prompt(prompt, fidelity)
                results[idx] = (error, num_errors)

        return results

    def run_hyperband(self, verbose: bool = True) -> Tuple[JointPrompt, float]:
        """Run Hyperband with BO proposals.

        Returns:
            (best_prompt, best_error)
        """
        evaluated_base_keys: Set[Tuple[int, FrozenSet[int]]] = set()
        batch_size = 5  # Number of candidates to batch together

        for s in range(self.smax, -1, -1):
            n = int(np.ceil((self.B / self.nvalid) * (self.eta ** s) / (s + 1)))
            b = int(self.nvalid * (self.eta ** (-s)))

            # Cap n at reasonable number
            n = min(n, 100)

            if verbose:
                print(f"\n[Bracket s={s}] n={n} candidates, initial fidelity b={b}")

            P: List[JointPrompt] = []
            V: List[float] = []

            # Initial sampling with batched evaluation
            pbar = tqdm(total=n, desc=f"Bracket {s} init", disable=not verbose)
            j = 0
            while j < n:
                # Determine batch size for this iteration
                current_batch_size = min(batch_size, n - j)

                # Generate proposals
                batch_prompts = []
                for _ in range(current_batch_size):
                    prompt = self.propose_prompt(evaluated_base_keys)
                    batch_prompts.append(prompt)
                    evaluated_base_keys.add(prompt.base_key())

                # Batch evaluate
                batch_results = self._evaluate_batch(batch_prompts, b)

                # Process results
                for prompt, (error, num_errors) in zip(batch_prompts, batch_results):
                    P.append(prompt)
                    V.append(error)

                    # Add to design data
                    z = self._encode_prompt_to_latent(prompt)
                    self.design_data.append((prompt, z, error, b))

                    # Update best
                    if error < self.best_error:
                        self.best_error = error
                        self.best_prompt = prompt
                        if verbose:
                            pbar.write(f"  New best (fid={b}): error={error:.4f}, inst={prompt.instruction_id}, K={prompt.num_exemplars}")

                    if b >= self.nvalid and error < self.best_error_full_fidelity:
                        self.best_error_full_fidelity = error
                        self.best_prompt_full_fidelity = prompt
                        if verbose:
                            pbar.write(f"  New best [FULL FIDELITY]: error={error:.4f}")

                j += current_batch_size
                pbar.update(current_batch_size)

                # Train GP after each batch
                if j < n:
                    self._train_gp()

            pbar.close()

            # Successive halving with batched evaluation
            for i in range(1, s + 1):
                ni = int(np.floor(n * (self.eta ** (-i))))
                bi = int(b * (self.eta ** i))

                if ni == 0:
                    break

                if verbose:
                    print(f"  [Stage {i}] Keep top {ni}, extend to fidelity {bi}")

                # Keep top prompts
                sorted_idx = np.argsort(V)[:ni]
                P = [P[idx] for idx in sorted_idx]

                # Batch evaluate all survivors at once
                batch_results = self._evaluate_batch(P, bi)

                V = []
                for prompt, (error, num_errors) in zip(P, batch_results):
                    V.append(error)

                    # Add extended observation to design data
                    z = self._encode_prompt_to_latent(prompt)
                    self.design_data.append((prompt, z, error, bi))

                    # Update best
                    if error < self.best_error:
                        self.best_error = error
                        self.best_prompt = prompt
                        if verbose:
                            print(f"    New best (fid={bi}): error={error:.4f}")

                    if bi >= self.nvalid and error < self.best_error_full_fidelity:
                        self.best_error_full_fidelity = error
                        self.best_prompt_full_fidelity = prompt
                        if verbose:
                            print(f"    New best [FULL FIDELITY]: error={error:.4f}")

                # Retrain GP
                self._train_gp()

        if verbose:
            print(f"\nHyperband complete!")
            print(f"  Total LLM calls: {self.total_llm_calls}")
            print(f"  Best error (any fidelity): {self.best_error:.4f}")
            print(f"  Best error [FULL FIDELITY]: {self.best_error_full_fidelity:.4f}")
            if self.best_prompt_full_fidelity:
                print(f"  Best instruction [FULL FIDELITY]: {self.best_prompt_full_fidelity.instruction_id}")
                print(f"  Best num_exemplars [FULL FIDELITY]: {self.best_prompt_full_fidelity.num_exemplars}")
            else:
                print(f"  WARNING: No full fidelity evaluations completed!")

        # Return full fidelity best if available, otherwise any-fidelity best
        if self.best_prompt_full_fidelity is not None:
            return self.best_prompt_full_fidelity, self.best_error_full_fidelity
        else:
            # Fallback to any-fidelity best (with warning)
            print("  WARNING: Returning best at partial fidelity (full fidelity best not available)")
            return self.best_prompt, self.best_error

    def get_design_data_for_vae(self) -> List[Dict]:
        """Export design data for VAE training.

        Returns list of dicts with instruction, exemplars, error, fidelity.
        """
        data = []
        for prompt, z, error, fidelity in self.design_data:
            data.append({
                "instruction_id": prompt.instruction_id,
                "instruction": prompt.instruction,
                "exemplar_ids": list(prompt.exemplar_ids),
                "num_exemplars": prompt.num_exemplars,
                "error_rate": error,
                "fidelity": fidelity,
            })
        return data

    def _prompt_to_dict(self, prompt: Optional[JointPrompt]) -> Optional[Dict]:
        """Convert JointPrompt to serializable dict."""
        if prompt is None:
            return None
        return {
            "instruction_id": prompt.instruction_id,
            "instruction": prompt.instruction,
            "exemplar_ids": list(prompt.exemplar_ids),
            "num_exemplars": prompt.num_exemplars,
        }

    def save_results(self, output_path: str) -> None:
        """Save Hyperband results to JSON."""
        results = {
            "best_prompt": self._prompt_to_dict(self.best_prompt_full_fidelity),
            "best_error": self.best_error_full_fidelity,
            "best_prompt_any_fidelity": self._prompt_to_dict(self.best_prompt),
            "best_error_any_fidelity": self.best_error,
            "total_llm_calls": self.total_llm_calls,
            "num_evaluations": len(self.design_data),
            "beta_prior": {"alpha": self.alpha, "beta": self.beta},
            "design_data": self.get_design_data_for_vae(),
        }

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        except (OSError, IOError) as e:
            print(f"ERROR: Failed to save results to {output_path}: {e}")
            backup_path = os.path.join(
                tempfile.gettempdir(),
                f"hyperband_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            try:
                with open(backup_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to backup location: {backup_path}")
            except Exception as backup_e:
                raise RuntimeError(
                    f"Failed to save results. Original: {e}. Backup: {backup_e}"
                ) from e
