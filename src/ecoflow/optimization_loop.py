"""
Bayesian optimization loop for prompt optimization in SONAR embedding space.

This module implements the core optimization loop that orchestrates:
1. Guided sampling from flow model using LCB acquisition
2. Decoding SONAR embeddings to text prompts
3. Evaluating prompts on GSM8K via LLM
4. Updating GP surrogate with new observations

Key classes:
- BOOptimizationLoop: Main orchestrator connecting all Phase 1-2 components
- OptimizationState: Checkpoint-able state for resumable optimization
- MetricsTracker: Sample efficiency metrics tracking

Usage:
    >>> loop = BOOptimizationLoop(flow_model, gp, sampler, decoder, evaluator, llm_client)
    >>> loop.initialize()  # Generate initial samples, fit GP
    >>> for _ in range(100):
    ...     result = loop.step()  # One BO iteration
    ...     print(f"Iteration {result['iteration']}: best={result['best_so_far']:.3f}")
"""

from dataclasses import dataclass, field
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.ecoflow.decoder import SonarDecoder
from src.ecoflow.flow_model import FlowMatchingModel
from src.ecoflow.gp_surrogate import SonarGPSurrogate
from src.ecoflow.guided_flow import GuidedFlowSampler
from src.gsm8k_evaluator import GSM8KEvaluator
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track optimization metrics for sample efficiency analysis.

    Tracks iteration-level metrics including:
    - best_so_far: Running maximum score
    - simple_regret: Gap to optimal (1.0 - best_so_far)
    - batch_means/maxes: Per-batch statistics
    - n_observations: Cumulative observation count

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.log_iteration(1, [0.5, 0.6, 0.4], 0.6, 10)
        >>> print(tracker.to_dict())
    """

    def __init__(self):
        """Initialize empty metrics tracker."""
        self.iterations: List[int] = []
        self.best_so_far: List[float] = []
        self.simple_regret: List[float] = []  # 1.0 - best_so_far
        self.batch_means: List[float] = []
        self.batch_maxes: List[float] = []
        self.n_observations: List[int] = []

    def log_iteration(
        self,
        iteration: int,
        batch_scores: List[float],
        best_so_far: float,
        n_observations: int,
        optimal: float = 1.0,  # GSM8K max accuracy
    ) -> None:
        """
        Log metrics for one optimization iteration.

        Args:
            iteration: Current iteration number
            batch_scores: Scores from current batch
            best_so_far: Running maximum score
            n_observations: Total observations so far
            optimal: Maximum achievable score (default 1.0 for accuracy)
        """
        self.iterations.append(iteration)
        self.best_so_far.append(best_so_far)
        self.simple_regret.append(optimal - best_so_far)

        if batch_scores:
            self.batch_means.append(sum(batch_scores) / len(batch_scores))
            self.batch_maxes.append(max(batch_scores))
        else:
            self.batch_means.append(0.0)
            self.batch_maxes.append(0.0)

        self.n_observations.append(n_observations)

    def to_dict(self) -> dict:
        """
        Export all metrics as dict for JSON serialization.

        Returns:
            Dictionary with all tracked metrics
        """
        return {
            "iterations": self.iterations,
            "best_so_far": self.best_so_far,
            "simple_regret": self.simple_regret,
            "batch_means": self.batch_means,
            "batch_maxes": self.batch_maxes,
            "n_observations": self.n_observations,
        }

    def save(self, path: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            path: Path to save metrics (should end with .json)
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved metrics to {path}")

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsTracker":
        """
        Create MetricsTracker from dict (for checkpoint loading).

        Args:
            data: Dictionary from to_dict()

        Returns:
            MetricsTracker with restored state
        """
        tracker = cls()
        tracker.iterations = data.get("iterations", [])
        tracker.best_so_far = data.get("best_so_far", [])
        tracker.simple_regret = data.get("simple_regret", [])
        tracker.batch_means = data.get("batch_means", [])
        tracker.batch_maxes = data.get("batch_maxes", [])
        tracker.n_observations = data.get("n_observations", [])
        return tracker


@dataclass
class OptimizationState:
    """
    Complete state for checkpointing and resumable optimization.

    Contains all data needed to resume optimization from any point:
    - Training data (X, Y tensors)
    - Best results found so far
    - Iteration counter
    - Evaluation indices for reproducibility
    - Metrics history

    Example:
        >>> state = loop.get_state()
        >>> state.save("checkpoint.pt")
        >>> loaded = OptimizationState.load("checkpoint.pt")
        >>> loop.load_checkpoint("checkpoint.pt")
    """

    train_X: torch.Tensor  # [N, 1024]
    train_Y: torch.Tensor  # [N]
    best_so_far: List[float]
    iteration: int
    best_prompt: str
    best_score: float
    eval_indices: List[int]
    metrics: Optional[dict] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """
        Save state to .pt file using torch.save().

        Tensors are saved on CPU to reduce file size.

        Args:
            path: Path to save checkpoint (should end with .pt)
        """
        state_dict = {
            "train_X": self.train_X.cpu(),
            "train_Y": self.train_Y.cpu(),
            "best_so_far": self.best_so_far,
            "iteration": self.iteration,
            "best_prompt": self.best_prompt,
            "best_score": self.best_score,
            "eval_indices": self.eval_indices,
            "metrics": self.metrics,
        }
        torch.save(state_dict, path)
        logger.info(f"Saved checkpoint to {path} (iteration={self.iteration}, best={self.best_score:.4f})")

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "OptimizationState":
        """
        Load state from .pt file.

        Args:
            path: Path to checkpoint file
            device: Device to load tensors to

        Returns:
            OptimizationState with restored values
        """
        state_dict = torch.load(path, map_location=device, weights_only=False)

        return cls(
            train_X=state_dict["train_X"].to(device),
            train_Y=state_dict["train_Y"].to(device),
            best_so_far=state_dict["best_so_far"],
            iteration=state_dict["iteration"],
            best_prompt=state_dict["best_prompt"],
            best_score=state_dict["best_score"],
            eval_indices=state_dict["eval_indices"],
            metrics=state_dict.get("metrics", {}),
        )


class BOOptimizationLoop:
    """
    Bayesian optimization loop for prompt optimization.

    Orchestrates the full BO pipeline:
    1. Generate samples using LCB-guided flow sampling
    2. Decode SONAR embeddings to text prompts
    3. Evaluate prompts on GSM8K subset via LLM
    4. Update GP surrogate with observations
    5. Track metrics for sample efficiency analysis

    The loop supports checkpointing for resumable long-running optimization.

    Attributes:
        flow_model: FlowMatchingModel for generating samples
        gp: SonarGPSurrogate for acquisition function
        sampler: GuidedFlowSampler for LCB-guided sampling
        decoder: SonarDecoder for embedding-to-text
        evaluator: GSM8KEvaluator for scoring prompts
        llm_client: LLMClient for answer generation
        n_initial: Number of initial random samples
        batch_size: Samples per BO iteration
        eval_subset_size: GSM8K questions per evaluation
        device: Computation device

    Example:
        >>> loop = BOOptimizationLoop(flow_model, gp, sampler, decoder, evaluator, llm)
        >>> init_result = loop.initialize()
        >>> print(f"Initial best: {init_result['best_score']:.3f}")
        >>> for _ in range(100):
        ...     result = loop.step()
        ...     loop.save_checkpoint(f"checkpoint_iter{result['iteration']}.pt")
    """

    def __init__(
        self,
        flow_model: FlowMatchingModel,
        gp: SonarGPSurrogate,
        sampler: GuidedFlowSampler,
        decoder: SonarDecoder,
        evaluator: GSM8KEvaluator,
        llm_client: LLMClient,
        n_initial: int = 10,
        batch_size: int = 4,
        eval_subset_size: int = 150,
        device: str = "cuda",
    ):
        """
        Initialize the optimization loop.

        Args:
            flow_model: Trained FlowMatchingModel for sample generation
            gp: SonarGPSurrogate for GP fitting and LCB computation
            sampler: GuidedFlowSampler with GP reference
            decoder: SonarDecoder for embedding-to-text conversion
            evaluator: GSM8KEvaluator for prompt scoring
            llm_client: LLMClient for generating answers
            n_initial: Number of initial random samples (default 10)
            batch_size: Samples per iteration (default 4)
            eval_subset_size: GSM8K subset size for evaluation (default 150)
            device: Computation device (default "cuda")
        """
        self.flow_model = flow_model
        self.gp = gp
        self.sampler = sampler
        self.decoder = decoder
        self.evaluator = evaluator
        self.llm_client = llm_client

        self.n_initial = n_initial
        self.batch_size = batch_size
        self.eval_subset_size = eval_subset_size
        self.device = device

        # State variables (initialized by initialize() or load_checkpoint())
        self.train_X: Optional[torch.Tensor] = None  # [N, 1024]
        self.train_Y: Optional[torch.Tensor] = None  # [N]
        self.best_so_far_list: List[float] = []
        self.iteration: int = 0
        self.best_prompt: str = ""
        self.best_score: float = 0.0
        self._prompts: Dict[int, str] = {}  # idx -> prompt text for best tracking

        # Fixed evaluation indices for fair comparison
        dataset_size = len(evaluator)
        self.eval_indices = random.sample(range(dataset_size), min(eval_subset_size, dataset_size))
        logger.info(f"Fixed evaluation subset: {len(self.eval_indices)} questions")

        # Metrics tracker
        self.metrics = MetricsTracker()

    def _is_valid_prompt(self, text: str) -> bool:
        """
        Check if prompt is valid for evaluation.

        A valid prompt:
        - Has at least 10 characters
        - Contains at least 3 words
        - Does not have excessive repetition

        Args:
            text: Decoded prompt text

        Returns:
            True if prompt is valid
        """
        if not text or len(text) < 10:
            return False

        words = text.split()
        if len(words) < 3:
            return False

        # Check for excessive repetition (same word > 50% of text)
        if words:
            word_counts = {}
            for word in words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.5 and len(words) > 5:
                return False

        return True

    def _decode_safe(self, embeddings: torch.Tensor) -> List[str]:
        """
        Safely decode embeddings to text with error handling.

        Wraps decoder.decode() with try/except for OOD samples that
        may fail to decode. Returns empty string for failed decodings.

        Args:
            embeddings: SONAR embeddings [N, 1024]

        Returns:
            List of decoded texts (empty string for failures)
        """
        try:
            texts = self.decoder.decode(embeddings)
            return texts
        except Exception as e:
            logger.warning(f"Batch decode failed: {e}. Attempting individual decoding.")
            # Fall back to individual decoding
            texts = []
            for i in range(embeddings.shape[0]):
                try:
                    text = self.decoder.decode(embeddings[i:i+1])[0]
                    texts.append(text)
                except Exception as e2:
                    logger.warning(f"Individual decode failed for sample {i}: {e2}")
                    texts.append("")
            return texts

    def _evaluate_prompts(self, prompts: List[str]) -> List[float]:
        """
        Evaluate prompts on fixed GSM8K subset.

        Uses Q_end format from OPRO paper:
        Q: {question}
        {prompt}
        A:

        Args:
            prompts: List of instruction prompts to evaluate

        Returns:
            List of accuracy scores (0.0-1.0) for each prompt
        """
        scores = []

        for prompt_idx, prompt in enumerate(prompts):
            # Check validity first
            if not self._is_valid_prompt(prompt):
                logger.info(f"  Prompt {prompt_idx}: INVALID (too short or repetitive)")
                scores.append(0.0)
                continue

            # Format all questions with this prompt using Q_end style
            formatted_questions = []
            for idx in self.eval_indices:
                example = self.evaluator.dataset[idx]
                question = example["question"]
                # Q_end format: question first, then instruction
                formatted = f"Q: {question}\n{prompt}\nA:"
                formatted_questions.append(formatted)

            # Generate answers via LLM
            try:
                outputs = self.llm_client.generate_batch(
                    formatted_questions,
                    max_new_tokens=512,
                    temperature=0.0,
                )
            except Exception as e:
                logger.error(f"LLM generation failed for prompt {prompt_idx}: {e}")
                scores.append(0.0)
                continue

            # Evaluate outputs
            result = self.evaluator.evaluate_batch(outputs, self.eval_indices)
            accuracy = result["accuracy"]
            scores.append(accuracy)

            # Log full prompt (never truncate per CLAUDE.md)
            logger.info(f"  Prompt {prompt_idx} (acc={accuracy:.3f}):\n{prompt}")

        return scores

    def initialize(self) -> dict:
        """
        Generate initial random samples and fit GP.

        Generates n_initial samples from the flow model (no guidance),
        decodes them to text, evaluates on GSM8K, and fits the GP.

        Returns:
            Dict with:
            - n_samples: Number of initial samples
            - scores: List of initial scores
            - best_score: Best initial score
            - best_prompt: Best initial prompt text
        """
        logger.info(f"Initializing with {self.n_initial} random samples...")

        # Generate random samples (no guidance - use flow_model directly)
        with torch.no_grad():
            embeddings = self.flow_model.sample(
                n_samples=self.n_initial,
                device=self.device,
                method="euler",
                num_steps=50,
                denormalize=True,
            )

        # Decode to text
        prompts = self._decode_safe(embeddings)

        # Evaluate all prompts
        scores = self._evaluate_prompts(prompts)

        # Store initial training data
        self.train_X = embeddings.to(self.device)
        self.train_Y = torch.tensor(scores, device=self.device, dtype=torch.float32)

        # Track best
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        self.best_score = scores[best_idx]
        self.best_prompt = prompts[best_idx]
        self.best_so_far_list = [self.best_score]

        # Store prompts for tracking
        for i, (prompt, score) in enumerate(zip(prompts, scores)):
            self._prompts[i] = prompt

        # Fit GP
        logger.info(f"Fitting GP on {len(scores)} initial observations...")
        self.gp.fit(self.train_X, self.train_Y)

        # Update sampler's GP reference
        self.sampler.update_gp(self.gp)

        # Log metrics
        self.metrics.log_iteration(
            iteration=0,
            batch_scores=scores,
            best_so_far=self.best_score,
            n_observations=len(scores),
        )

        # Clear GPU cache
        torch.cuda.empty_cache()

        logger.info(f"Initialization complete: best_score={self.best_score:.4f}")
        logger.info(f"Best initial prompt:\n{self.best_prompt}")

        return {
            "n_samples": self.n_initial,
            "scores": scores,
            "best_score": self.best_score,
            "best_prompt": self.best_prompt,
        }

    def step(self) -> dict:
        """
        Execute one BO iteration.

        Steps:
        1. Generate guided samples via sampler.sample()
        2. Decode to text via decoder.decode()
        3. Evaluate prompts on GSM8K subset
        4. Update GP with new observations
        5. Update sampler's GP reference
        6. Clear GPU cache

        Returns:
            Dict with:
            - iteration: Current iteration number
            - batch_scores: Scores for this batch
            - best_so_far: Running maximum score
            - best_prompt: Current best prompt
            - n_observations: Total observations so far
        """
        self.iteration += 1
        logger.info(f"\n=== Iteration {self.iteration} ===")

        # 1. Generate guided samples
        logger.info(f"Generating {self.batch_size} guided samples...")
        embeddings = self.sampler.sample(
            n_samples=self.batch_size,
            device=self.device,
            num_steps=50,
            method="euler",
        )

        # 2. Decode to text
        logger.info("Decoding embeddings to text...")
        prompts = self._decode_safe(embeddings)

        # 3. Evaluate prompts
        logger.info(f"Evaluating {len(prompts)} prompts on {len(self.eval_indices)} questions...")
        scores = self._evaluate_prompts(prompts)

        # 4. Update GP with new observations
        new_X = embeddings.to(self.device)
        new_Y = torch.tensor(scores, device=self.device, dtype=torch.float32)

        logger.info(f"Updating GP with {len(scores)} new observations...")
        self.gp.update(new_X, new_Y)

        # Update training data
        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)

        # 5. Update sampler's GP reference
        self.sampler.update_gp(self.gp)

        # Track best
        batch_best_idx = max(range(len(scores)), key=lambda i: scores[i])
        batch_best_score = scores[batch_best_idx]

        if batch_best_score > self.best_score:
            self.best_score = batch_best_score
            self.best_prompt = prompts[batch_best_idx]
            logger.info(f"NEW BEST: {self.best_score:.4f}")
            logger.info(f"Best prompt:\n{self.best_prompt}")

        self.best_so_far_list.append(self.best_score)

        # Store prompts
        base_idx = len(self._prompts)
        for i, prompt in enumerate(prompts):
            self._prompts[base_idx + i] = prompt

        # Log metrics
        self.metrics.log_iteration(
            iteration=self.iteration,
            batch_scores=scores,
            best_so_far=self.best_score,
            n_observations=self.train_X.shape[0],
        )

        # 6. Clear GPU cache
        torch.cuda.empty_cache()

        result = {
            "iteration": self.iteration,
            "batch_scores": scores,
            "best_so_far": self.best_score,
            "best_prompt": self.best_prompt,
            "n_observations": self.train_X.shape[0],
        }

        logger.info(
            f"Iteration {self.iteration} complete: "
            f"batch_mean={sum(scores)/len(scores):.3f}, "
            f"batch_max={max(scores):.3f}, "
            f"best_so_far={self.best_score:.3f}"
        )

        return result

    def get_state(self) -> OptimizationState:
        """
        Get current optimization state for checkpointing.

        Returns:
            OptimizationState with all current values
        """
        return OptimizationState(
            train_X=self.train_X,
            train_Y=self.train_Y,
            best_so_far=self.best_so_far_list.copy(),
            iteration=self.iteration,
            best_prompt=self.best_prompt,
            best_score=self.best_score,
            eval_indices=self.eval_indices.copy(),
            metrics=self.metrics.to_dict(),
        )

    def save_checkpoint(self, path: str) -> None:
        """
        Save current state to checkpoint file.

        Args:
            path: Path to save checkpoint (should end with .pt)
        """
        state = self.get_state()
        state.save(path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load state from checkpoint file.

        Restores all state variables including:
        - Training data (X, Y)
        - Iteration counter
        - Best results
        - Evaluation indices
        - Metrics history

        Also re-fits GP on loaded data.

        Args:
            path: Path to checkpoint file
        """
        state = OptimizationState.load(path, device=self.device)

        # Restore state
        self.train_X = state.train_X
        self.train_Y = state.train_Y
        self.best_so_far_list = state.best_so_far
        self.iteration = state.iteration
        self.best_prompt = state.best_prompt
        self.best_score = state.best_score
        self.eval_indices = state.eval_indices

        # Restore metrics
        if state.metrics:
            self.metrics = MetricsTracker.from_dict(state.metrics)

        # Re-fit GP on loaded data
        logger.info(f"Re-fitting GP on {self.train_X.shape[0]} observations...")
        self.gp.fit(self.train_X, self.train_Y)
        self.sampler.update_gp(self.gp)

        logger.info(
            f"Loaded checkpoint from {path}: "
            f"iteration={self.iteration}, "
            f"n_obs={self.train_X.shape[0]}, "
            f"best_score={self.best_score:.4f}"
        )

    @property
    def n_observations(self) -> int:
        """Total number of observations collected."""
        return 0 if self.train_X is None else self.train_X.shape[0]
