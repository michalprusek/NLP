"""
InstructZero-style optimization loop for GSM8K prompt optimization.

Main loop that coordinates:
- Soft prompt space and projection
- LLM-based instruction generation
- Bayesian optimization with GP
- GSM8K evaluation
"""

import torch
import time
import json
import random
from pathlib import Path
from typing import Optional
import logging

from .soft_prompt import SoftPromptSpace, SoftPromptToText
from .gp_optimizer import GPBayesianOptimizer, TurboGPOptimizer

logger = logging.getLogger(__name__)

# Default seed instructions for initialization
DEFAULT_SEED_INSTRUCTIONS = [
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


class InstructZeroLoop:
    """
    InstructZero-style Bayesian Optimization for prompt optimization.

    Algorithm:
    1. Initialize with seed instructions (evaluated on GSM8K)
    2. Fit GP surrogate on (soft_prompt, score) pairs
    3. Use acquisition function to suggest new soft prompts
    4. Decode soft prompts to instructions via LLM
    5. Evaluate instructions on GSM8K
    6. Update GP and repeat
    """

    def __init__(
        self,
        llm_client,
        evaluator,
        intrinsic_dim: int = 10,
        projection_dim: int = 50,
        n_initial: int = 10,
        batch_size: int = 4,
        minibatch_size: int = 150,
        use_turbo: bool = False,
        temperature: float = 0.7,
        results_dir: str = "instructzero/results",
        device: str = "cuda",
        verbose: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize InstructZero optimization loop.

        Args:
            llm_client: LLM client for generation and evaluation
            evaluator: GSM8K evaluator
            intrinsic_dim: Soft prompt dimension (default 10)
            projection_dim: Projection dimension (default 50)
            n_initial: Number of initial random evaluations
            batch_size: Candidates per iteration
            minibatch_size: GSM8K examples per evaluation
            use_turbo: Use TuRBO instead of standard BO
            temperature: LLM generation temperature
            results_dir: Directory for results
            device: Torch device
            verbose: Print progress
            seed: Random seed
        """
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.n_initial = n_initial
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.temperature = temperature
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.verbose = verbose

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        # Initialize soft prompt space
        self.prompt_space = SoftPromptSpace(
            intrinsic_dim=intrinsic_dim,
            projection_dim=projection_dim,
            device=device,
            seed=seed
        )

        # Load exemplars for instruction generation
        self.exemplars = self._load_exemplars()

        # Initialize instruction generator
        self.instruction_generator = SoftPromptToText(
            llm_client=llm_client,
            exemplars=self.exemplars,
            n_exemplars=3,
            temperature=temperature
        )

        # Initialize Bayesian optimizer
        bounds = self.prompt_space.get_bounds()
        if use_turbo:
            self.optimizer = TurboGPOptimizer(bounds=bounds, device=device)
        else:
            self.optimizer = GPBayesianOptimizer(bounds=bounds, device=device)

        # Tracking
        self.num_calls = 0
        self.iteration = 0
        self.history = []
        self.instruction_cache = {}  # soft_prompt_hash -> instruction
        self.best_instruction = ""
        self.best_score = -float("inf")
        self.best_soft_prompt = None
        self.start_time = None

    def _load_exemplars(self) -> list[dict]:
        """Load exemplars from evaluator for instruction generation."""
        exemplars = []
        # Get first 10 examples from train set
        for i in range(min(10, len(self.evaluator.dataset))):
            item = self.evaluator.dataset[i]
            exemplars.append({
                "question": item["question"],
                "answer": item["answer"]
            })
        return exemplars

    def _evaluate_instruction(self, instruction: str) -> float:
        """
        Evaluate an instruction on GSM8K minibatch.

        Args:
            instruction: Instruction text to evaluate

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        # Sample random batch of examples
        n_examples = len(self.evaluator.dataset)
        indices = random.sample(range(n_examples), min(self.minibatch_size, n_examples))

        # Create prompts with instruction
        prompts = []
        for idx in indices:
            question = self.evaluator.dataset[idx]["question"]
            # Q_end format: Q: {question}\n{instruction}\nA:
            prompt = f"Q: {question}\n{instruction}\nA:"
            prompts.append(prompt)

        # Generate responses
        responses = self.llm_client.generate_batch(prompts)
        self.num_calls += len(responses)

        # Evaluate
        result = self.evaluator.evaluate_batch(responses, indices)
        return result["accuracy"]

    def _initialize_with_seeds(self) -> None:
        """Initialize with seed instructions and random soft prompts."""
        logger.info(f"Initializing with {self.n_initial} seed evaluations...")

        soft_prompts = []
        scores = []
        instructions = []

        # First, evaluate default seed instructions
        n_seeds = min(len(DEFAULT_SEED_INSTRUCTIONS), self.n_initial // 2)
        for i, instruction in enumerate(DEFAULT_SEED_INSTRUCTIONS[:n_seeds]):
            score = self._evaluate_instruction(instruction)
            scores.append(score)
            instructions.append(instruction)

            # Create a random soft prompt and associate it with this score
            soft_prompt = self.prompt_space.sample_initial(1).squeeze(0)
            soft_prompts.append(soft_prompt)

            if self.verbose:
                logger.info(f"Seed {i+1}/{n_seeds}: acc={score:.3f}")
                logger.info(f"  {instruction}")

            if score > self.best_score:
                self.best_score = score
                self.best_instruction = instruction
                self.best_soft_prompt = soft_prompt

        # Generate and evaluate random soft prompt-based instructions
        n_random = self.n_initial - n_seeds
        if n_random > 0:
            random_prompts = self.prompt_space.sample_initial(n_random)
            for i in range(n_random):
                soft_prompt = random_prompts[i]
                projected = self.prompt_space.project(soft_prompt)
                instruction = self.instruction_generator.decode(projected)

                score = self._evaluate_instruction(instruction)

                soft_prompts.append(soft_prompt)
                scores.append(score)
                instructions.append(instruction)

                if self.verbose:
                    logger.info(f"Random {i+1}/{n_random}: acc={score:.3f}")
                    logger.info(f"  {instruction}")

                if score > self.best_score:
                    self.best_score = score
                    self.best_instruction = instruction
                    self.best_soft_prompt = soft_prompt

        # Update optimizer with initial data
        X = torch.stack(soft_prompts)
        Y = torch.tensor(scores, device=self.device)
        self.optimizer.update(X, Y)

        # Record history
        self.history.append({
            "iteration": 0,
            "num_calls": self.num_calls,
            "best_score": self.best_score,
            "best_instruction": self.best_instruction,
            "time_elapsed": time.time() - self.start_time
        })

        logger.info(f"Initialization complete. Best score: {self.best_score:.4f}")

    def run(self, max_calls: int = 50000) -> dict:
        """
        Run InstructZero optimization loop.

        Args:
            max_calls: Maximum number of LLM calls

        Returns:
            Dictionary with optimization results
        """
        self.start_time = time.time()
        logger.info(f"Starting InstructZero optimization with budget of {max_calls} calls...")

        # Initialize
        self._initialize_with_seeds()

        # Main optimization loop
        while self.num_calls < max_calls:
            self.iteration += 1

            if self.verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {self.iteration} | Calls: {self.num_calls}/{max_calls}")
                logger.info(f"Best so far: {self.best_score:.4f}")
                logger.info(f"{'='*60}")

            # Suggest new soft prompts
            suggested_prompts = self.optimizer.suggest_diverse(self.batch_size)

            # Decode to instructions and evaluate
            for i in range(len(suggested_prompts)):
                soft_prompt = suggested_prompts[i]
                projected = self.prompt_space.project(soft_prompt)

                # Generate instruction
                instruction = self.instruction_generator.decode(projected)

                # Skip if we've seen very similar instruction
                if instruction in self.instruction_cache.values():
                    logger.debug(f"Skipping duplicate instruction")
                    continue

                # Evaluate
                score = self._evaluate_instruction(instruction)

                # Update optimizer
                self.optimizer.update(
                    soft_prompt.unsqueeze(0),
                    torch.tensor([score], device=self.device)
                )

                if self.verbose:
                    logger.info(f"Candidate {i+1}/{self.batch_size}: acc={score:.3f}")
                    logger.info(f"  {instruction}")

                # Track best
                if score > self.best_score:
                    self.best_score = score
                    self.best_instruction = instruction
                    self.best_soft_prompt = soft_prompt
                    logger.info(f"*** New best! Score: {score:.4f} ***")

                # Cache instruction
                prompt_hash = hash(tuple(soft_prompt.cpu().numpy().tolist()))
                self.instruction_cache[prompt_hash] = instruction

                # Check budget
                if self.num_calls >= max_calls:
                    break

            # Record history
            self.history.append({
                "iteration": self.iteration,
                "num_calls": self.num_calls,
                "best_score": self.best_score,
                "best_instruction": self.best_instruction,
                "time_elapsed": time.time() - self.start_time
            })

            # Save intermediate results
            if self.iteration % 10 == 0:
                self._save_results()

        # Final results
        elapsed = time.time() - self.start_time
        logger.info(f"\n{'='*60}")
        logger.info("InstructZero Optimization Complete")
        logger.info(f"Total calls: {self.num_calls}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best instruction: {self.best_instruction}")
        logger.info(f"{'='*60}")

        results = {
            "method": "instructzero",
            "best_instruction": self.best_instruction,
            "best_score": self.best_score,
            "total_calls": self.num_calls,
            "total_time": elapsed,
            "history": self.history
        }

        self._save_results(results)
        return results

    def _save_results(self, results: Optional[dict] = None) -> None:
        """Save results to JSON file."""
        if results is None:
            results = {
                "method": "instructzero",
                "best_instruction": self.best_instruction,
                "best_score": self.best_score,
                "total_calls": self.num_calls,
                "iteration": self.iteration,
                "history": self.history
            }

        filepath = self.results_dir / f"instructzero_{int(time.time())}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.debug(f"Results saved to {filepath}")

    def evaluate_on_test(self, instruction: Optional[str] = None) -> float:
        """
        Evaluate instruction on full GSM8K test set.

        Args:
            instruction: Instruction to evaluate (default: best found)

        Returns:
            Test accuracy
        """
        if instruction is None:
            instruction = self.best_instruction

        logger.info(f"Evaluating on test set...")
        logger.info(f"Instruction: {instruction}")

        # Load test set
        from datasets import load_from_disk
        ds = load_from_disk("datasets/gsm8k")
        test_data = ds["test"]

        # Create prompts
        prompts = []
        indices = list(range(len(test_data)))
        for idx in indices:
            question = test_data[idx]["question"]
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

        # Calculate accuracy using test evaluator
        from shared.gsm8k_evaluator import GSM8KEvaluator
        test_evaluator = GSM8KEvaluator(dataset_path="datasets/gsm8k", split="test")
        result = test_evaluator.evaluate_batch(all_responses, indices)

        accuracy = result["accuracy"]
        logger.info(f"Test accuracy: {accuracy:.4f} ({result['correct']}/{result['total']})")

        return accuracy
