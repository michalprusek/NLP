#!/usr/bin/env python3
"""
NFBO for GSM8K - Adapted from official GitHub repository.

Original: https://github.com/mlvlab/NFBO
Paper: "Normalizing Flow-Based Bayesian Optimization"

This implementation combines:
1. TURBO trust region from official NFBO
2. z → instruction mapping (similar to InstructZero)
3. Deep kernel GP surrogate

Key differences from original NFBO:
- No TextFlow/SeqFlow VAE (prompts are generated directly from z)
- Uses vLLM for evaluation instead of molecular oracles
- Latent space is lower-dimensional (10D instead of 40D)

Usage:
    uv run python -m nfbo_gsm8k.run --max-calls 50000
"""

import sys
import os
import random
import time
import json
import re
import copy
import logging
from pathlib import Path
from dataclasses import dataclass, field

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import from botorch/gpytorch
try:
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
    from gpytorch.priors import GammaPrior
    from torch.quasirandom import SobolEngine
except ImportError as e:
    logger.error(f"Failed to import BO libraries: {e}")
    sys.exit(1)


# ============================================================================
# TURBO Trust Region (from official NFBO)
# ============================================================================

@dataclass
class TurboState:
    """Trust region state from NFBO's turbo.py."""
    dim: int
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5 ** 7  # ~0.0078
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def update(self, Y_next: torch.Tensor) -> None:
        """Update trust region based on new observations."""
        if max(Y_next) > self.best_value + 1e-3 * abs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
            self.best_value = max(Y_next).item()
        else:
            self.success_counter = 0
            self.failure_counter += 1

        # Expand trust region on success
        if self.success_counter >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
            logger.info(f"TR expanded to {self.length:.4f}")

        # Shrink trust region on failure
        elif self.failure_counter >= self.failure_tolerance:
            self.length = max(self.length / 2.0, self.length_min)
            self.failure_counter = 0
            logger.info(f"TR shrunk to {self.length:.4f}")

        # Restart if trust region too small
        if self.length <= self.length_min:
            self.restart_triggered = True
            logger.info("TR restart triggered")


def generate_candidates_turbo(
    X: torch.Tensor,
    Y: torch.Tensor,
    state: TurboState,
    n_candidates: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generate candidates using TURBO trust region.

    From NFBO's turbo.py generate_batch function.
    """
    dim = X.shape[-1]

    # Find best point as trust region center
    best_idx = Y.argmax()
    x_center = X[best_idx].clone()

    # Trust region bounds
    weights = torch.ones(dim, device=device, dtype=dtype) * 8.0  # Default weight
    tr_lb = torch.clamp(x_center - weights * state.length / 2, -1.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2, -1.0, 1.0)

    # Sobol sampling within trust region
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(device=device, dtype=dtype)

    # Scale to trust region
    X_cand = tr_lb + (tr_ub - tr_lb) * pert

    return X_cand


# ============================================================================
# GSM8K Evaluator
# ============================================================================

class GSM8KEvaluator:
    """GSM8K evaluator using vLLM."""

    def __init__(self, llm_client, dataset_path: str = "datasets/gsm8k", split: str = "train"):
        self.llm_client = llm_client
        self.dataset_path = dataset_path
        self.split = split
        self._load_dataset()

    def _load_dataset(self):
        from datasets import load_from_disk
        ds = load_from_disk(self.dataset_path)
        self.dataset = ds[self.split]
        logger.info(f"Loaded {len(self.dataset)} GSM8K {self.split} examples")

    def extract_answer(self, text: str) -> str:
        match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
        numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if numbers:
            return numbers[-1].replace(',', '')
        return ""

    def evaluate_batch(self, instruction: str, indices: list[int]) -> tuple[float, np.ndarray]:
        prompts = []
        for idx in indices:
            question = self.dataset[idx]["question"]
            prompt = f"Q: {question}\n{instruction}\nA:"
            prompts.append(prompt)

        responses = self.llm_client.generate_batch(prompts)

        correct = 0
        scores = []
        for i, idx in enumerate(indices):
            gold_answer = self.dataset[idx]["answer"]
            gold_num = self.extract_answer(gold_answer)
            pred_num = self.extract_answer(responses[i])
            is_correct = (gold_num == pred_num) if gold_num and pred_num else False
            scores.append(1.0 if is_correct else 0.0)
            correct += int(is_correct)

        accuracy = correct / len(indices)
        return accuracy, np.array(scores)


# ============================================================================
# NFBO GSM8K Optimizer
# ============================================================================

class NFBOOptimizer:
    """
    NFBO-style optimizer for GSM8K prompt optimization.

    Combines:
    - TURBO trust region for adaptive exploration
    - z → instruction mapping
    - Deep kernel GP surrogate
    """

    def __init__(
        self,
        llm_client,
        latent_dim: int = 10,
        n_initial: int = 20,
        batch_size: int = 10,
        minibatch_size: int = 150,
        results_dir: str = "nfbo_gsm8k/results",
        seed: int = 42,
        device: str = "cuda",
    ):
        self.llm_client = llm_client
        self.latent_dim = latent_dim
        self.n_initial = n_initial
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

        self.evaluator = GSM8KEvaluator(llm_client)

        # Tracking
        self.num_calls = 0
        self.best_score = -float("inf")
        self.best_instruction = None
        self.best_z = None
        self.prompts_cache = {}
        self.history = []

        # Instruction generation components
        self.instruction_templates = [
            "Solve step by step: {focus}",
            "Think carefully and {focus}",
            "Break this into parts: {focus}",
            "Work through systematically: {focus}",
            "Calculate carefully: {focus}",
        ]

        self.focus_options = [
            "show all your work",
            "identify the key numbers first",
            "set up an equation",
            "work backwards from what you need",
            "check your answer makes sense",
            "explain each step",
            "use clear notation",
            "break into smaller problems",
        ]

    def _z_to_instruction(self, z: np.ndarray) -> str:
        """Map latent z to instruction string."""
        # Use z components to select instruction components
        template_idx = int((z[0] + 1) / 2 * len(self.instruction_templates)) % len(self.instruction_templates)
        focus_idx = int((z[1] + 1) / 2 * len(self.focus_options)) % len(self.focus_options)

        template = self.instruction_templates[template_idx]
        focus = self.focus_options[focus_idx]

        # Additional modifiers based on other z components
        if z[2] > 0.3:
            focus += " and double-check your calculations"
        if z[3] > 0.3:
            focus += ", defining variables as needed"
        if z[4] < -0.3:
            focus = "clearly " + focus

        instruction = template.format(focus=focus)

        # Add variety using remaining z components
        if z[5] > 0.5:
            instruction = "Let's think step by step. " + instruction
        elif z[5] < -0.5:
            instruction += " Show your reasoning."

        return instruction

    def _evaluate(self, z: np.ndarray) -> float:
        """Evaluate latent z by mapping to instruction and scoring."""
        instruction = self._z_to_instruction(z)

        # Check cache
        if instruction in self.prompts_cache:
            return self.prompts_cache[instruction]

        # Sample random batch
        n_examples = len(self.evaluator.dataset)
        indices = random.sample(range(n_examples), min(self.minibatch_size, n_examples))

        accuracy, _ = self.evaluator.evaluate_batch(instruction, indices)
        self.num_calls += len(indices)

        self.prompts_cache[instruction] = accuracy

        # Track best
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_instruction = instruction
            self.best_z = z.copy()
            logger.info(f"*** New best: {accuracy:.4f} ***")
            logger.info(f"  Instruction: {instruction}")

        return accuracy

    def run(self, max_calls: int = 50000) -> dict:
        """Run NFBO optimization."""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("NFBO for GSM8K (TURBO + z→Instruction)")
        logger.info("=" * 60)
        logger.info(f"Latent dim: {self.latent_dim}")
        logger.info(f"N initial: {self.n_initial}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Max calls: {max_calls}")
        logger.info("=" * 60)

        # Phase 1: Initial Sobol sampling
        logger.info("Phase 1: Initial Sobol sampling...")
        sobol = SobolEngine(self.latent_dim, scramble=True, seed=self.seed)
        X_init = 2 * sobol.draw(self.n_initial) - 1  # Scale to [-1, 1]

        X = []
        Y = []

        for i, z in enumerate(X_init):
            if self.num_calls >= max_calls:
                break

            z_np = z.numpy()
            score = self._evaluate(z_np)

            X.append(z_np)
            Y.append(score)

            logger.info(f"Init {i+1}/{self.n_initial}: score={score:.4f}, calls={self.num_calls}")

        X = torch.tensor(np.array(X), dtype=self.dtype, device=self.device)
        Y = torch.tensor(Y, dtype=self.dtype, device=self.device).unsqueeze(-1)

        logger.info(f"Initial best: {self.best_score:.4f}")

        # Initialize TURBO state
        tr_state = TurboState(dim=self.latent_dim, batch_size=self.batch_size)
        tr_state.best_value = Y.max().item()

        self.history.append({
            "iteration": 0,
            "num_calls": self.num_calls,
            "best_score": self.best_score,
            "tr_length": tr_state.length,
        })

        # Phase 2: TURBO Bayesian Optimization
        logger.info("Phase 2: TURBO Bayesian Optimization...")
        iteration = 0

        while self.num_calls < max_calls:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"TURBO Iteration {iteration} | Calls: {self.num_calls}/{max_calls} | TR: {tr_state.length:.4f}")
            logger.info(f"{'='*60}")

            # Check for restart
            if tr_state.restart_triggered:
                logger.info("Restarting trust region...")
                tr_state = TurboState(dim=self.latent_dim, batch_size=self.batch_size)
                tr_state.best_value = Y.max().item()

            # Fit GP
            try:
                # Standardize Y
                Y_mean, Y_std = Y.mean(), Y.std()
                Y_std = Y_std if Y_std > 0 else 1.0
                Y_norm = (Y - Y_mean) / Y_std

                # Define kernel
                covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=self.latent_dim)
                )

                gp_model = SingleTaskGP(X, Y_norm, covar_module=covar_module)
                gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
                fit_gpytorch_mll(gp_mll)
            except Exception as e:
                logger.warning(f"GP fitting failed: {e}")
                # Fallback: random candidates
                X_cand = 2 * torch.rand(self.batch_size, self.latent_dim) - 1
                X_cand = X_cand.to(device=self.device, dtype=self.dtype)

            else:
                # Generate candidates using TURBO
                X_cand = generate_candidates_turbo(
                    X, Y, tr_state, n_candidates=100, device=self.device, dtype=self.dtype
                )

                # Use LogEI to select best candidates
                try:
                    acq = LogExpectedImprovement(gp_model, best_f=Y_norm.max().item())
                    with torch.no_grad():
                        acq_values = acq(X_cand.unsqueeze(1))
                    top_indices = acq_values.argsort(descending=True)[:self.batch_size]
                    X_cand = X_cand[top_indices]
                except Exception as e:
                    logger.warning(f"Acquisition failed: {e}")
                    X_cand = X_cand[:self.batch_size]

            # Evaluate candidates
            Y_new = []
            for z in X_cand:
                if self.num_calls >= max_calls:
                    break

                z_np = z.cpu().numpy()
                score = self._evaluate(z_np)
                Y_new.append(score)

            if Y_new:
                Y_new_tensor = torch.tensor(Y_new, dtype=self.dtype, device=self.device)

                # Update trust region
                tr_state.update(Y_new_tensor)

                # Append to data
                X = torch.cat([X, X_cand[:len(Y_new)]])
                Y = torch.cat([Y, Y_new_tensor.unsqueeze(-1)])

            # Record history
            self.history.append({
                "iteration": iteration,
                "num_calls": self.num_calls,
                "best_score": self.best_score,
                "tr_length": tr_state.length,
            })

            logger.info(f"Best so far: {self.best_score:.4f}")

            # Save intermediate
            if iteration % 10 == 0:
                self._save_results()

        # Final results
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("NFBO Optimization Complete")
        logger.info(f"Total calls: {self.num_calls}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best instruction:\n{self.best_instruction}")
        logger.info("=" * 60)

        results = {
            "method": "nfbo_turbo",
            "best_instruction": self.best_instruction,
            "best_score": float(self.best_score),
            "total_calls": self.num_calls,
            "total_time": elapsed,
            "history": self.history,
        }

        self._save_results(results)
        return results

    def _save_results(self, results: dict = None) -> None:
        if results is None:
            results = {
                "method": "nfbo_turbo",
                "best_instruction": self.best_instruction,
                "best_score": float(self.best_score) if self.best_score else 0.0,
                "total_calls": self.num_calls,
                "history": self.history,
            }

        filepath = self.results_dir / f"nfbo_{int(time.time())}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

    def evaluate_on_test(self, instruction: str = None) -> float:
        """Evaluate on GSM8K test set."""
        if instruction is None:
            instruction = self.best_instruction

        logger.info("Evaluating on GSM8K test set...")
        logger.info(f"Instruction: {instruction}")

        from datasets import load_from_disk
        ds = load_from_disk("datasets/gsm8k")
        test_data = ds["test"]

        prompts = []
        for i in range(len(test_data)):
            question = test_data[i]["question"]
            prompt = f"Q: {question}\n{instruction}\nA:"
            prompts.append(prompt)

        all_responses = []
        batch_size = 100
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            responses = self.llm_client.generate_batch(batch)
            all_responses.extend(responses)
            logger.info(f"Evaluated {len(all_responses)}/{len(prompts)} test examples")

        correct = 0
        for i, response in enumerate(all_responses):
            gold_answer = test_data[i]["answer"]
            gold_num = self.evaluator.extract_answer(gold_answer)
            pred_num = self.evaluator.extract_answer(response)
            if gold_num and pred_num and gold_num == pred_num:
                correct += 1

        accuracy = correct / len(test_data)
        logger.info(f"Test accuracy: {accuracy:.4f} ({correct}/{len(test_data)})")
        return accuracy


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NFBO for GSM8K")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--backend", type=str, default="vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-calls", type=int, default=50000)
    parser.add_argument("--minibatch-size", type=int, default=150)
    parser.add_argument("--results-dir", type=str, default="nfbo_gsm8k/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-test-eval", action="store_true")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NFBO for GSM8K (TURBO Trust Region)")
    logger.info("Original: https://github.com/mlvlab/NFBO")
    logger.info("=" * 60)

    from shared.llm_client import create_llm_client
    llm_client = create_llm_client(
        args.model,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size
    )

    optimizer = NFBOOptimizer(
        llm_client=llm_client,
        latent_dim=args.latent_dim,
        n_initial=args.n_initial,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    results = optimizer.run(max_calls=args.max_calls)

    if not args.skip_test_eval:
        logger.info("\nEvaluating on GSM8K test set...")
        test_accuracy = optimizer.evaluate_on_test()
        results["test_accuracy"] = test_accuracy

        results_path = Path(args.results_dir) / f"nfbo_final_{int(time.time())}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Final results saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
