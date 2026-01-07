"""Hyperband for joint (instruction, exemplar_set) optimization.

Implements successive halving with BO proposals over the infinite search space
of instruction × exemplar subset combinations.

Key innovation: Two-stage proposal mechanism:
1. Sample/select instruction via EI
2. Optimize exemplar latent z_ex via gradient to maximize EI
3. Decode z_ex to discrete exemplar selection via SlotBasedExemplarDecoder
"""

import torch
import numpy as np
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Set, Callable, FrozenSet
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import norm

from lipo_e.config import LIPOEConfig
from lipo_e.encoder import GTREncoder, StructureAwareVAE
from lipo_e.gp import GPWithEI
from lipo_e.training import QAPair


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
    return (num_errors + alpha) / (fidelity + alpha + beta)


def beta_variance(
    smoothed_error: float,
    fidelity: int,
    alpha: float,
    beta: float,
) -> float:
    """Compute Beta posterior variance (heteroscedastic noise).

    High fidelity → low variance → GP trusts observation more.

    Args:
        smoothed_error: Beta posterior mean
        fidelity: Total samples evaluated
        alpha, beta: Beta prior parameters

    Returns:
        Variance estimate
    """
    return smoothed_error * (1 - smoothed_error) / (fidelity + alpha + beta + 1)


class LIPOEHyperband:
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
        config: LIPOEConfig,
        device: str = "cuda",
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
            llm_evaluator: Function to evaluate prompts
            config: Configuration
            device: Device
        """
        self.instructions = instructions
        self.qa_pool = qa_pool
        self.validation_data = validation_data
        self.vae = vae
        self.gtr_encoder = gtr_encoder
        self.pool_embeddings = pool_embeddings.to(device)
        self.instruction_embeddings = instruction_embeddings.to(device)
        self.llm_evaluator = llm_evaluator
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

        # Best result tracking
        self.best_prompt: Optional[JointPrompt] = None
        self.best_error: float = 1.0

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

    def _index_fidelity(self, base_key: Tuple[int, FrozenSet[int]], fidelity: int):
        """Index fidelity for a base key."""
        if fidelity not in self.fidelity_index[base_key]:
            self.fidelity_index[base_key].append(fidelity)

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

        # Build exemplar text
        exemplars = [self.qa_pool[i] for i in prompt.exemplar_ids]
        exemplar_text = "\n\n".join(qa.format() for qa in exemplars) if exemplars else ""

        # Call LLM evaluator
        error_rate = self.llm_evaluator(
            instruction=prompt.instruction,
            exemplar_text=exemplar_text,
            samples=samples,
        )

        num_errors = int(error_rate * len(samples))
        return error_rate, num_errors

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
        """Generate random (instruction, exemplar_set) proposal."""
        inst_id = random.randint(0, len(self.instructions) - 1)
        K = random.randint(self.config.min_exemplars, self.config.num_slots)

        if K > 0:
            exemplar_ids = tuple(sorted(random.sample(range(len(self.qa_pool)), K)))
        else:
            exemplar_ids = ()

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

    def _get_vmin_b(self) -> float:
        """Get best error rate for EI calculation."""
        if not self.design_data:
            return 1.0

        # Use smoothed errors
        smoothed_errors = []
        for _, _, error, fidelity in self.design_data:
            num_errors = int(error * fidelity)
            smoothed = beta_smooth_error(num_errors, fidelity, self.alpha, self.beta)
            smoothed_errors.append(smoothed)

        return min(smoothed_errors)

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
        noise_vars = []

        for prompt, z, error, fidelity in self.design_data:
            num_errors = int(error * fidelity)
            smoothed = beta_smooth_error(num_errors, fidelity, self.alpha, self.beta)
            variance = beta_variance(smoothed, fidelity, self.alpha, self.beta)

            latents.append(z)
            smoothed_errors.append(smoothed)
            noise_vars.append(variance)

        X = torch.stack(latents).to(self.device)
        y = torch.tensor(smoothed_errors, device=self.device)
        fids = torch.tensor([d[3] for d in self.design_data], device=self.device)

        # Create and train GP
        self.gp = GPWithEI(
            instruction_dim=self.config.instruction_latent_dim,
            exemplar_dim=self.config.exemplar_latent_dim,
            device=self.device,
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
                exemplar_ids = self._decode_exemplar_latent(z_ex)

            if ucb_final > best_ucb:
                best_ucb = ucb_final
                best_z_ex = z_ex.detach()
                best_exemplar_ids = exemplar_ids

        return best_z_ex, best_exemplar_ids, best_ucb

    def _decode_exemplar_latent(self, z_ex: torch.Tensor) -> List[int]:
        """Decode exemplar latent to discrete selection.

        Uses slot-based attention over pool embeddings.
        """
        with torch.no_grad():
            self.vae.eval()

            # Decode z_ex to selection
            z_ex_batch = z_ex.unsqueeze(0)

            # SlotBasedExemplarDecoder returns (selection_probs, num_ex_logits, selected_indices)
            # pool_embeddings should be (N_pool, 768), not unsqueezed
            selection_probs, num_ex_logits, _ = self.vae.exemplar_decoder(
                z_ex_batch,
                self.pool_embeddings,  # Shape: (N_pool, 768)
                hard=True,  # Use hard selection for discrete indices
            )

            # Number of exemplars to select
            num_ex = num_ex_logits.argmax(dim=-1).item()
            num_ex = min(num_ex, self.config.num_slots)
            num_ex = max(num_ex, self.config.min_exemplars)

            if num_ex == 0:
                return []

            # selection_probs is already softmaxed/gumbel-softmaxed from decoder
            # Shape: (batch=1, num_slots, N_pool)
            # Get top-1 index from each of the first num_ex slots
            selection_probs_squeezed = selection_probs.squeeze(0)  # (num_slots, N_pool)
            top_indices = selection_probs_squeezed[:num_ex].argmax(dim=-1).tolist()

            # Remove duplicates while preserving order
            unique_indices = list(dict.fromkeys(top_indices))

        return unique_indices

    def run_hyperband(self, verbose: bool = True) -> Tuple[JointPrompt, float]:
        """Run Hyperband with BO proposals.

        Returns:
            (best_prompt, best_error)
        """
        evaluated_base_keys: Set[Tuple[int, FrozenSet[int]]] = set()

        for s in range(self.smax, -1, -1):
            n = int(np.ceil((self.B / self.nvalid) * (self.eta ** s) / (s + 1)))
            b = int(self.nvalid * (self.eta ** (-s)))

            # Cap n at reasonable number
            n = min(n, 100)

            if verbose:
                print(f"\n[Bracket s={s}] n={n} candidates, initial fidelity b={b}")

            P: List[JointPrompt] = []
            V: List[float] = []

            # Initial sampling with BO proposals (UCB with beta=8.0)
            for j in tqdm(range(n), desc=f"Bracket {s} init", disable=not verbose):
                # Train GP periodically
                if j > 0 and j % 5 == 0:
                    self._train_gp()

                prompt = self.propose_prompt(evaluated_base_keys)

                error, num_errors = self.evaluate_prompt(prompt, b)

                P.append(prompt)
                V.append(error)
                evaluated_base_keys.add(prompt.base_key())

                # Add to design data
                z = self._encode_prompt_to_latent(prompt)
                self.design_data.append((prompt, z, error, b))

                # Update best
                if error < self.best_error:
                    self.best_error = error
                    self.best_prompt = prompt
                    if verbose:
                        print(f"  New best: error={error:.4f}, inst={prompt.instruction_id}, K={prompt.num_exemplars}")

            # Successive halving
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

                # Extend fidelity
                V = []
                for prompt in tqdm(P, desc=f"Stage {i}", disable=not verbose):
                    error, num_errors = self.evaluate_prompt(prompt, bi)
                    V.append(error)

                    # Add extended observation to design data
                    z = self._encode_prompt_to_latent(prompt)
                    self.design_data.append((prompt, z, error, bi))

                    # Update best
                    if error < self.best_error:
                        self.best_error = error
                        self.best_prompt = prompt
                        if verbose:
                            print(f"    New best: error={error:.4f}")

                # Retrain GP
                self._train_gp()

        if verbose:
            print(f"\nHyperband complete!")
            print(f"  Total LLM calls: {self.total_llm_calls}")
            print(f"  Best error: {self.best_error:.4f}")
            print(f"  Best instruction: {self.best_prompt.instruction_id if self.best_prompt else None}")
            print(f"  Best num_exemplars: {self.best_prompt.num_exemplars if self.best_prompt else None}")

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

    def save_results(self, output_path: str):
        """Save Hyperband results to JSON."""
        results = {
            "best_prompt": {
                "instruction_id": self.best_prompt.instruction_id,
                "instruction": self.best_prompt.instruction,
                "exemplar_ids": list(self.best_prompt.exemplar_ids),
                "num_exemplars": self.best_prompt.num_exemplars,
            } if self.best_prompt else None,
            "best_error": self.best_error,
            "total_llm_calls": self.total_llm_calls,
            "num_evaluations": len(self.design_data),
            "beta_prior": {"alpha": self.alpha, "beta": self.beta},
            "design_data": self.get_design_data_for_vae(),
        }

        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        except (OSError, IOError) as e:
            print(f"ERROR: Failed to save results to {output_path}: {e}")
            # Try backup location
            import tempfile
            from datetime import datetime
            backup_path = os.path.join(
                tempfile.gettempdir(),
                f"hyperband_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            try:
                with open(backup_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to backup location: {backup_path}")
            except Exception as backup_e:
                print(f"CRITICAL: Could not save results anywhere: {backup_e}")
                raise RuntimeError(
                    f"Failed to save results. Original error: {e}. Backup error: {backup_e}"
                ) from e
