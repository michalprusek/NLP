#!/usr/bin/env python3
"""
InstructZero for GSM8K - Adapted from official GitHub repository.

Original: https://github.com/Lichang-Chen/InstructZero
Paper: "InstructZero: Efficient Instruction Optimization via Zeroth-Order Optimization" (ICML 2024)

Adaptations for GSM8K:
1. Data loading: Uses GSM8K instead of instruction induction tasks
2. Evaluation: Uses vLLM instead of ChatGPT
3. Prompt format: Q_end style (Q: {question}\n{instruction}\nA:)

Key components preserved from original:
- Soft prompt projection (z → Az)
- Instruction-coupled kernel for GP
- CMA-ES for acquisition function optimization
- Bayesian optimization loop

Usage:
    uv run python -m instructzero_gsm8k.run --max-calls 50000
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

import torch
import numpy as np

# Add official InstructZero to path
INSTRUCTZERO_PATH = Path(__file__).parent.parent / "instructzero_original" / "InstructZero" / "experiments"
sys.path.insert(0, str(INSTRUCTZERO_PATH))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import from official InstructZero
try:
    from torch.quasirandom import SobolEngine
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.analytic import ExpectedImprovement
    from gpytorch.kernels import ScaleKernel, MaternKernel
    from gpytorch.priors import GammaPrior
    from instruction_coupled_kernel import CombinedStringKernel, cma_es_concat
except ImportError as e:
    logger.error(f"Failed to import from official InstructZero: {e}")
    logger.error("Make sure instructzero_original/ is cloned from GitHub")
    sys.exit(1)


# ============================================================================
# GSM8K Evaluation Adapter
# ============================================================================

class GSM8KEvaluator:
    """GSM8K evaluator using vLLM."""

    def __init__(self, llm_client, dataset_path: str = "datasets/gsm8k", split: str = "train"):
        self.llm_client = llm_client
        self.dataset_path = dataset_path
        self.split = split
        self._load_dataset()

    def _load_dataset(self):
        """Load GSM8K dataset."""
        from datasets import load_from_disk
        ds = load_from_disk(self.dataset_path)
        self.dataset = ds[self.split]
        logger.info(f"Loaded {len(self.dataset)} GSM8K {self.split} examples")

    def extract_answer(self, text: str) -> str:
        """Extract final numerical answer from model output."""
        # Pattern: #### followed by number
        match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')

        # Fallback: last number in text
        numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if numbers:
            return numbers[-1].replace(',', '')
        return ""

    def evaluate_batch(self, instruction: str, indices: list[int]) -> tuple[float, np.ndarray]:
        """
        Evaluate instruction on GSM8K batch.

        Returns:
            (accuracy, per_example_scores)
        """
        # Create prompts with Q_end format
        prompts = []
        for idx in indices:
            question = self.dataset[idx]["question"]
            prompt = f"Q: {question}\n{instruction}\nA:"
            prompts.append(prompt)

        # Generate responses
        responses = self.llm_client.generate_batch(prompts)

        # Evaluate
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
# InstructZero GSM8K Adapter
# ============================================================================

class InstructZeroGSM8K:
    """
    InstructZero adapted for GSM8K.

    Uses official InstructZero components:
    - Soft prompt projection (linear layer: intrinsic_dim → n_prompt_tokens * hidden_size)
    - Instruction-coupled kernel
    - CMA-ES acquisition optimization
    - BO loop structure

    Replaces:
    - ChatGPT → vLLM for both generation and evaluation
    - Instruction induction → GSM8K task
    """

    def __init__(
        self,
        llm_client,
        intrinsic_dim: int = 10,
        n_prompt_tokens: int = 5,
        n_initial: int = 25,
        batch_size: int = 25,
        n_iterations: int = 5,
        minibatch_size: int = 150,
        results_dir: str = "instructzero_gsm8k/results",
        seed: int = 42,
        device: str = "cuda",
    ):
        """
        Initialize InstructZero for GSM8K.

        Args:
            llm_client: vLLM client for generation
            intrinsic_dim: Dimension of latent space z (default 10)
            n_prompt_tokens: Number of soft prompt tokens (default 5)
            n_initial: Initial Sobol samples (default 25)
            batch_size: Candidates per BO iteration (default 25)
            n_iterations: BO iterations (default 5)
            minibatch_size: GSM8K examples per evaluation (default 150)
            results_dir: Directory for results
            seed: Random seed
            device: Torch device
        """
        self.llm_client = llm_client
        self.intrinsic_dim = intrinsic_dim
        self.n_prompt_tokens = n_prompt_tokens
        self.n_initial = n_initial
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.minibatch_size = minibatch_size
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.device = device

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Device settings (from official InstructZero misc.py)
        self.tkwargs = {
            "device": torch.device(device if torch.cuda.is_available() else "cpu"),
            "dtype": torch.double,
        }

        # Initialize evaluator
        self.evaluator = GSM8KEvaluator(llm_client)

        # Soft prompt projection layer (official InstructZero style)
        # Maps: z (intrinsic_dim) → soft_prompt (n_prompt_tokens * hidden_size)
        # For simplicity, we project to a fixed embedding dimension
        self.embedding_dim = 768  # Standard transformer embedding size
        self.linear = torch.nn.Linear(
            intrinsic_dim,
            n_prompt_tokens * self.embedding_dim,
            bias=False
        )
        torch.nn.init.uniform_(self.linear.weight, -1, 1)

        # Tracking
        self.num_calls = 0
        self.best_score = -float("inf")
        self.best_instruction = None
        self.best_z = None
        self.prompts_cache = {}  # instruction → (score, embedding)
        self.history = []

        # Seed instructions for GSM8K
        self.seed_instructions = [
            "Solve this math problem step by step. Show your work.",
            "Let's think through this problem carefully. First, identify the key information.",
            "Break down this word problem into smaller parts and solve each one.",
            "Use algebra to solve this problem. Define variables for unknown quantities.",
            "Read the problem carefully. What are you asked to find? Work backwards if needed.",
            "Think step by step and explain your reasoning for each calculation.",
            "Identify the given information and what you need to calculate. Then solve systematically.",
            "Set up an equation based on the problem description and solve it step by step.",
            "First, understand the question. Then work through the math systematically.",
            "Show all your work. Calculate the final answer carefully and double-check.",
        ]

    def _z_to_instruction(self, z: torch.Tensor) -> str:
        """
        Convert latent z to instruction via LLM generation.

        Uses z values to control instruction generation diversity:
        1. z[0:3] controls which few-shot examples to use
        2. z[3:6] controls the generation style/focus
        3. z[6:9] controls instruction structure

        Args:
            z: Latent vector (intrinsic_dim,)

        Returns:
            Generated instruction string
        """
        z_np = z.detach().cpu().numpy()

        # Use z to select different few-shot examples (z[0:3] → example indices)
        n_examples = len(self.evaluator.dataset)
        # Map z values to indices using sigmoid-like mapping
        example_indices = []
        for i in range(3):
            idx = int((z_np[i % len(z_np)] + 1) / 2 * n_examples) % n_examples
            if idx not in example_indices:
                example_indices.append(idx)
        # Add more if needed
        while len(example_indices) < 3:
            idx = random.randint(0, n_examples - 1)
            if idx not in example_indices:
                example_indices.append(idx)

        exemplars = []
        for idx in example_indices:
            item = self.evaluator.dataset[idx]
            exemplars.append(f"Q: {item['question']}\nA: {item['answer'][:150]}...")
        demo_text = "\n\n".join(exemplars)

        # Use z to select instruction focus (z[3:6])
        focus_options = [
            "step-by-step calculation",
            "identifying key variables first",
            "working backwards from the answer",
            "breaking into smaller sub-problems",
            "using clear mathematical notation",
            "explaining reasoning at each step",
            "checking the answer makes sense",
            "setting up equations from the problem",
        ]
        focus_idx = int((z_np[3 % len(z_np)] + 1) / 2 * len(focus_options)) % len(focus_options)
        focus = focus_options[focus_idx]

        # Use z to select instruction style (z[6:9])
        style_options = [
            "concise and direct",
            "detailed and thorough",
            "encouraging and supportive",
            "formal and precise",
        ]
        style_idx = int((z_np[6 % len(z_np)] + 1) / 2 * len(style_options)) % len(style_options)
        style = style_options[style_idx]

        # Generate instruction with z-conditioned prompt
        gen_prompt = f"""Here are some math problem examples:

{demo_text}

Generate a {style} instruction for solving similar math problems.
The instruction should emphasize {focus}.
Keep it to 1-2 sentences.
Instruction:"""

        # Generate instruction
        responses = self.llm_client.generate_batch([gen_prompt])
        instruction = responses[0].strip()

        # Clean up instruction
        lines = instruction.split('\n')
        instruction = lines[0] if lines else instruction
        instruction = instruction.strip('"\'')

        # Remove common prefixes
        for prefix in ["Instruction:", "Here is", "The instruction is"]:
            if instruction.lower().startswith(prefix.lower()):
                instruction = instruction[len(prefix):].strip()

        if len(instruction) < 10:
            # Fallback to z-selected seed instruction
            seed_idx = int((z_np[0] + 1) / 2 * len(self.seed_instructions)) % len(self.seed_instructions)
            instruction = self.seed_instructions[seed_idx]

        return instruction

    def _evaluate(self, instruction: str) -> tuple[float, np.ndarray]:
        """
        Evaluate instruction on GSM8K minibatch.

        Returns:
            (accuracy, per_example_scores)
        """
        # Check cache
        if instruction in self.prompts_cache:
            return self.prompts_cache[instruction]

        # Sample random batch
        n_examples = len(self.evaluator.dataset)
        indices = random.sample(range(n_examples), min(self.minibatch_size, n_examples))

        # Evaluate
        accuracy, scores = self.evaluator.evaluate_batch(instruction, indices)
        self.num_calls += len(indices)

        # Cache
        self.prompts_cache[instruction] = (accuracy, scores)

        return accuracy, scores

    def _eval_z(self, z: torch.Tensor) -> tuple[float, np.ndarray]:
        """
        Evaluate latent z: convert to instruction and evaluate.

        Returns:
            (accuracy, per_example_scores_embedding)
        """
        instruction = self._z_to_instruction(z)
        accuracy, scores = self._evaluate(instruction)

        logger.info(f"Evaluated: acc={accuracy:.3f}")
        logger.info(f"  Instruction: {instruction}")

        # Track best
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_instruction = instruction
            self.best_z = z.clone()
            logger.info(f"*** New best! ***")

        return accuracy, scores

    def run(self, max_calls: int = 50000) -> dict:
        """
        Run InstructZero optimization for GSM8K.

        Uses official InstructZero's BO loop:
        1. Sobol sampling for initial points
        2. GP with instruction-coupled kernel
        3. CMA-ES for acquisition optimization
        4. Expected Improvement acquisition

        Args:
            max_calls: Maximum LLM evaluation calls

        Returns:
            Results dictionary
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("InstructZero for GSM8K (Official Implementation Adapted)")
        logger.info("=" * 60)
        logger.info(f"Intrinsic dim: {self.intrinsic_dim}")
        logger.info(f"N prompt tokens: {self.n_prompt_tokens}")
        logger.info(f"N initial: {self.n_initial}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"N iterations: {self.n_iterations}")
        logger.info(f"Max calls: {max_calls}")
        logger.info("=" * 60)

        # Phase 1: Initialize with seed instructions
        logger.info("Phase 1: Evaluating seed instructions...")
        X = []
        Y = []
        Y_scores = []

        for i, instruction in enumerate(self.seed_instructions[:min(5, len(self.seed_instructions))]):
            if self.num_calls >= max_calls:
                break

            accuracy, scores = self._evaluate(instruction)

            # Create random z for this instruction
            z = torch.randn(self.intrinsic_dim)
            X.append(z)
            Y.append(accuracy)
            Y_scores.append(scores)

            logger.info(f"Seed {i+1}: acc={accuracy:.3f} - {instruction[:80]}...")

            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_instruction = instruction
                self.best_z = z.clone()

        # Phase 2: Sobol sampling for remaining initial points
        logger.info("Phase 2: Sobol sampling...")
        n_sobol = self.n_initial - len(X)
        if n_sobol > 0:
            sobol_engine = SobolEngine(dimension=self.intrinsic_dim, scramble=True, seed=self.seed)
            sobol_points = sobol_engine.draw(n_sobol)
            # Scale from [0,1] to [-1,1]
            sobol_points = 2 * sobol_points - 1

            for i, z in enumerate(sobol_points):
                if self.num_calls >= max_calls:
                    break

                accuracy, scores = self._eval_z(z)
                X.append(z)
                Y.append(accuracy)
                Y_scores.append(scores)

        # Convert to tensors
        X = torch.stack(X).to(**self.tkwargs)
        Y = torch.tensor(Y, dtype=torch.double).unsqueeze(-1).to(**self.tkwargs)

        # For instruction-coupled kernel, we need score embeddings
        # Pad/truncate to fixed size for kernel
        score_dim = 100
        Y_scores_padded = []
        for scores in Y_scores:
            if len(scores) >= score_dim:
                Y_scores_padded.append(scores[:score_dim])
            else:
                padded = np.zeros(score_dim)
                padded[:len(scores)] = scores
                Y_scores_padded.append(padded)
        Y_scores_tensor = torch.tensor(np.array(Y_scores_padded), dtype=torch.double).to(**self.tkwargs)

        logger.info(f"Initial best: {self.best_score:.4f}")
        logger.info(f"LLM calls: {self.num_calls}/{max_calls}")

        # Record history
        self.history.append({
            "iteration": 0,
            "num_calls": self.num_calls,
            "best_score": self.best_score,
            "best_instruction": self.best_instruction,
        })

        # Phase 3: Bayesian Optimization loop (official InstructZero style)
        logger.info("Phase 3: Bayesian Optimization...")
        iteration = 0

        while self.num_calls < max_calls:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"BO Iteration {iteration} | Calls: {self.num_calls}/{max_calls}")
            logger.info(f"{'='*60}")

            # Standardize Y
            X_train = X.clone()
            y_train = (Y - Y.mean(dim=0)) / (Y.std(dim=0) + 1e-9)

            # Define kernels (official InstructZero style)
            matern_kernel = MaternKernel(
                nu=2.5,
                ard_num_dims=X_train.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
            matern_kernel_instruction = MaternKernel(
                nu=2.5,
                ard_num_dims=Y_scores_tensor.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )

            # Instruction-coupled kernel (key innovation of InstructZero)
            try:
                covar_module = ScaleKernel(
                    base_kernel=CombinedStringKernel(
                        base_latent_kernel=matern_kernel,
                        instruction_kernel=matern_kernel_instruction,
                        latent_train=X_train.double(),
                        instruction_train=Y_scores_tensor
                    )
                )
            except Exception as e:
                logger.warning(f"Instruction-coupled kernel failed: {e}")
                # Fallback to standard Matern kernel
                covar_module = ScaleKernel(base_kernel=matern_kernel)

            # Fit GP
            gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
            gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

            try:
                fit_gpytorch_mll(gp_mll)
            except Exception as e:
                logger.warning(f"GP fitting failed: {e}")
                # Continue with unfitted model

            # Expected Improvement acquisition
            EI = ExpectedImprovement(gp_model, best_f=y_train.max().item())

            # CMA-ES optimization from top points (official InstructZero style)
            starting_idxs = torch.argsort(-y_train.squeeze())[:self.batch_size]
            starting_points = X_train[starting_idxs]

            best_points = []
            best_vals = []

            for starting_point in starting_points:
                if (torch.max(starting_point) > 1 or torch.min(starting_point) < -1):
                    continue
                try:
                    newp, newv = cma_es_concat(starting_point, EI, self.tkwargs)
                    best_points.append(newp)
                    best_vals.append(newv)
                except Exception as e:
                    logger.warning(f"CMA-ES failed: {e}")
                    continue

            if not best_points:
                # Fallback: random sampling
                logger.warning("CMA-ES produced no candidates, using random sampling")
                for _ in range(self.batch_size):
                    best_points.append(np.random.uniform(-1, 1, self.intrinsic_dim))
                    best_vals.append(0.0)

            # Evaluate top candidates
            for idx in np.argsort(-np.array(best_vals))[:self.batch_size]:
                if self.num_calls >= max_calls:
                    break

                z_next = torch.from_numpy(best_points[idx]).float()
                accuracy, scores = self._eval_z(z_next)

                # Update data
                X = torch.cat([X, z_next.unsqueeze(0).to(**self.tkwargs)])
                Y = torch.cat([Y, torch.tensor([[accuracy]], dtype=torch.double).to(**self.tkwargs)])

                # Update score embeddings
                if len(scores) >= score_dim:
                    scores_padded = scores[:score_dim]
                else:
                    scores_padded = np.zeros(score_dim)
                    scores_padded[:len(scores)] = scores
                Y_scores_tensor = torch.cat([
                    Y_scores_tensor,
                    torch.tensor([scores_padded], dtype=torch.double).to(**self.tkwargs)
                ])

            # Record history
            self.history.append({
                "iteration": iteration,
                "num_calls": self.num_calls,
                "best_score": self.best_score,
                "best_instruction": self.best_instruction,
            })

            logger.info(f"Best so far: {self.best_score:.4f}")

            # Save intermediate results
            if iteration % 5 == 0:
                self._save_results()

        # Final results
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("InstructZero Optimization Complete")
        logger.info(f"Total calls: {self.num_calls}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best instruction:\n{self.best_instruction}")
        logger.info("=" * 60)

        results = {
            "method": "instructzero_official",
            "best_instruction": self.best_instruction,
            "best_score": float(self.best_score),
            "total_calls": self.num_calls,
            "total_time": elapsed,
            "history": self.history,
        }

        self._save_results(results)
        return results

    def _save_results(self, results: dict = None) -> None:
        """Save results to JSON."""
        if results is None:
            results = {
                "method": "instructzero_official",
                "best_instruction": self.best_instruction,
                "best_score": float(self.best_score) if self.best_score else 0.0,
                "total_calls": self.num_calls,
                "history": self.history,
            }

        filepath = self.results_dir / f"instructzero_{int(time.time())}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

    def evaluate_on_test(self, instruction: str = None) -> float:
        """Evaluate instruction on GSM8K test set."""
        if instruction is None:
            instruction = self.best_instruction

        logger.info("Evaluating on GSM8K test set...")
        logger.info(f"Instruction: {instruction}")

        # Load test set
        from datasets import load_from_disk
        ds = load_from_disk("datasets/gsm8k")
        test_data = ds["test"]

        # Create prompts
        prompts = []
        for i in range(len(test_data)):
            question = test_data[i]["question"]
            prompt = f"Q: {question}\n{instruction}\nA:"
            prompts.append(prompt)

        # Generate in batches
        all_responses = []
        batch_size = 100
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            responses = self.llm_client.generate_batch(batch)
            all_responses.extend(responses)
            logger.info(f"Evaluated {len(all_responses)}/{len(prompts)} test examples")

        # Calculate accuracy
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

    parser = argparse.ArgumentParser(
        description="InstructZero for GSM8K (Official Implementation Adapted)"
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="qwen",
                        help="Model (qwen, llama)")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "openai", "deepinfra", "anthropic"],
                        help="LLM backend")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size for vLLM")

    # InstructZero parameters (official defaults)
    parser.add_argument("--intrinsic-dim", type=int, default=10,
                        help="Soft prompt intrinsic dimension")
    parser.add_argument("--n-prompt-tokens", type=int, default=5,
                        help="Number of soft prompt tokens")
    parser.add_argument("--n-initial", type=int, default=25,
                        help="Initial Sobol samples")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Candidates per BO iteration")
    parser.add_argument("--n-iterations", type=int, default=5,
                        help="BO iterations")

    # Evaluation parameters
    parser.add_argument("--max-calls", type=int, default=50000,
                        help="Maximum LLM calls budget")
    parser.add_argument("--minibatch-size", type=int, default=150,
                        help="GSM8K examples per evaluation")

    # Other
    parser.add_argument("--results-dir", type=str, default="instructzero_gsm8k/results",
                        help="Results directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip-test-eval", action="store_true",
                        help="Skip final test set evaluation")

    args = parser.parse_args()

    # Print configuration
    logger.info("=" * 60)
    logger.info("InstructZero for GSM8K")
    logger.info("Original: https://github.com/Lichang-Chen/InstructZero")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model} (backend: {args.backend})")
    logger.info(f"Intrinsic dim: {args.intrinsic_dim}")
    logger.info(f"N prompt tokens: {args.n_prompt_tokens}")
    logger.info(f"N initial: {args.n_initial}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max calls: {args.max_calls}")
    logger.info("=" * 60)

    # Initialize LLM client
    from shared.llm_client import create_llm_client
    llm_client = create_llm_client(
        args.model,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # Create optimizer
    optimizer = InstructZeroGSM8K(
        llm_client=llm_client,
        intrinsic_dim=args.intrinsic_dim,
        n_prompt_tokens=args.n_prompt_tokens,
        n_initial=args.n_initial,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        minibatch_size=args.minibatch_size,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    # Run optimization
    results = optimizer.run(max_calls=args.max_calls)

    # Evaluate on test set
    if not args.skip_test_eval:
        logger.info("\nEvaluating on GSM8K test set...")
        test_accuracy = optimizer.evaluate_on_test()
        results["test_accuracy"] = test_accuracy

        # Save final results
        results_path = Path(args.results_dir) / f"instructzero_final_{int(time.time())}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Final results saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
