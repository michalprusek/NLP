"""
Bayesian optimization loop for prompt optimization in SONAR embedding space.

This module implements the core optimization loop that orchestrates:
1. GP-UCB optimization to find optimal embedding
2. Flow projection to keep samples on-manifold
3. Decoding SONAR embeddings to text prompts
4. Evaluating prompts on GSM8K via LLM
5. Updating GP surrogate with new observations

Key classes:
- BOOptimizationLoop: Main orchestrator connecting all components
- OptimizationState: Checkpoint-able state for resumable optimization
- MetricsTracker: Sample efficiency metrics tracking

Usage:
    >>> loop = BOOptimizationLoop(flow_model, gp, sampler, decoder, evaluator, llm_client)
    >>> loop.initialize()
    >>> for _ in range(100):
    ...     result = loop.step()
    ...     print(f"Iteration {result['iteration']}: best={result['best_so_far']:.3f}")
"""

from dataclasses import dataclass, field
import json
import logging
import random
from typing import Any, Optional

import torch

from ecoflow.decoder import SonarDecoder
from ecoflow.flow_model import FlowMatchingModel
from ecoflow.gp_surrogate import SonarGPSurrogate
from ecoflow.guided_flow import GuidedFlowSampler
from shared.gsm8k_evaluator import GSM8KEvaluator
from shared.llm_client import LLMClient

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track optimization metrics for sample efficiency analysis."""

    def __init__(self):
        self.iterations: list[int] = []
        self.best_so_far: list[float] = []
        self.simple_regret: list[float] = []
        self.scores: list[float] = []
        self.n_observations: list[int] = []

    def log_iteration(
        self,
        iteration: int,
        score: float,
        best_so_far: float,
        n_observations: int,
        optimal: float = 1.0,
    ) -> None:
        """Log metrics for one optimization iteration."""
        self.iterations.append(iteration)
        self.scores.append(score)
        self.best_so_far.append(best_so_far)
        self.simple_regret.append(optimal - best_so_far)
        self.n_observations.append(n_observations)

    def to_dict(self) -> dict:
        """Export all metrics as dict for JSON serialization."""
        return {
            "iterations": self.iterations,
            "scores": self.scores,
            "best_so_far": self.best_so_far,
            "simple_regret": self.simple_regret,
            "n_observations": self.n_observations,
        }

    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsTracker":
        """Create MetricsTracker from dict (for checkpoint loading)."""
        tracker = cls()
        tracker.iterations = data.get("iterations", [])
        tracker.scores = data.get("scores", [])
        tracker.best_so_far = data.get("best_so_far", [])
        tracker.simple_regret = data.get("simple_regret", [])
        tracker.n_observations = data.get("n_observations", [])
        return tracker


@dataclass
class OptimizationState:
    """Complete state for checkpointing and resumable optimization."""

    train_X: torch.Tensor
    train_Y: torch.Tensor
    best_so_far: list[float]
    iteration: int
    best_prompt: str
    best_score: float
    eval_indices: list[int]
    metrics: Optional[dict] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save state to .pt file using torch.save()."""
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
        """Load state from .pt file."""
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

    Orchestrates the simple BO pipeline:
    1. GP-UCB optimization to find optimal embedding
    2. Flow projection to keep samples on-manifold
    3. Decode SONAR embedding to text prompt
    4. Evaluate prompt on GSM8K subset via LLM
    5. Update GP surrogate with observation
    6. Track metrics for sample efficiency analysis

    Example:
        >>> loop = BOOptimizationLoop(flow_model, gp, sampler, decoder, evaluator, llm)
        >>> init_result = loop.initialize()
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
        eval_subset_size: int = 150,
        device: str = "cuda",
        encoder: Optional[Any] = None,
        l2r_threshold: float = 0.5,
        l2r_filter_enabled: bool = True,
    ):
        """
        Initialize the optimization loop.

        Args:
            flow_model: Trained FlowMatchingModel for sample generation
            gp: SonarGPSurrogate for GP fitting and UCB computation
            sampler: GuidedFlowSampler with GP reference
            decoder: SonarDecoder for embedding-to-text conversion
            evaluator: GSM8KEvaluator for prompt scoring
            llm_client: LLMClient for generating answers
            n_initial: Number of initial random samples (default 10)
            eval_subset_size: GSM8K subset size for evaluation (default 150)
            device: Computation device (default "cuda")
            encoder: Optional SONAR encoder for round-trip fidelity computation
            l2r_threshold: L2-r threshold for on-manifold filtering (default 0.5)
            l2r_filter_enabled: Whether to enable L2-r filtering (default True)
        """
        self.flow_model = flow_model
        self.gp = gp
        self.sampler = sampler
        self.decoder = decoder
        self.evaluator = evaluator
        self.llm_client = llm_client

        self.n_initial = n_initial
        self.eval_subset_size = eval_subset_size
        self.device = device

        self.encoder = encoder
        self.l2r_threshold = l2r_threshold
        self.l2r_filter_enabled = l2r_filter_enabled

        # State variables
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None
        self.best_so_far_list: list[float] = []
        self.iteration: int = 0
        self.best_prompt: str = ""
        self.best_score: float = 0.0
        self._prompts: dict[int, str] = {}

        # Fixed evaluation indices for fair comparison
        dataset_size = len(evaluator)
        self.eval_indices = random.sample(range(dataset_size), min(eval_subset_size, dataset_size))

        self.metrics = MetricsTracker()

    def _is_valid_prompt(self, text: str) -> bool:
        """Check if prompt is valid for evaluation."""
        if not text or len(text) < 10:
            return False

        words = text.split()
        if len(words) < 3:
            return False

        # Check for excessive repetition
        if words:
            word_counts: dict[str, int] = {}
            for word in words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.5 and len(words) > 5:
                return False

        return True

    def _compute_round_trip_fidelity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute round-trip fidelity (L2-r) for embeddings.

        L2-r = ||emb - encode(decode(emb))||_2

        Low L2-r indicates embedding is on valid SONAR manifold.
        """
        if self.encoder is None:
            return torch.zeros(embeddings.shape[0], device=embeddings.device)

        texts = self._decode_safe(embeddings)

        try:
            re_embeddings = self.encoder.predict(
                texts,
                source_lang="eng_Latn",
                batch_size=len(texts),
            )
            if not isinstance(re_embeddings, torch.Tensor):
                re_embeddings = torch.tensor(re_embeddings, device=embeddings.device)
            re_embeddings = re_embeddings.to(embeddings.device)
        except Exception as e:
            logger.error(
                f"SONAR re-encoding failed: {e}. "
                "L2-r filtering is DISABLED for this batch - returning inf to reject."
            )
            # Return high L2-r values to REJECT these embeddings rather than silently accept
            return torch.full((embeddings.shape[0],), float('inf'), device=embeddings.device)

        return (embeddings - re_embeddings).norm(dim=-1)

    def _decode_safe(self, embeddings: torch.Tensor) -> list[str]:
        """Safely decode embeddings to text with error handling."""
        try:
            return self.decoder.decode(embeddings)
        except Exception as e:
            logger.warning(f"Batch decode failed: {e}. Attempting individual decoding.")
            texts = []
            for i in range(embeddings.shape[0]):
                try:
                    text = self.decoder.decode(embeddings[i : i + 1])[0]
                    texts.append(text)
                except Exception as inner_e:
                    logger.warning(f"Individual decode failed for embedding {i}: {inner_e}")
                    texts.append("")
            n_failed = sum(1 for t in texts if t == "")
            if n_failed > 0:
                logger.warning(f"Decoding: {n_failed}/{len(texts)} embeddings failed to decode")
            return texts

    def _evaluate_prompt(self, prompt: str) -> float:
        """
        Evaluate a single prompt on fixed GSM8K subset.

        Uses Q_end format from OPRO paper: question first, then instruction.
        """
        if not self._is_valid_prompt(prompt):
            return 0.0

        formatted_questions = []
        for idx in self.eval_indices:
            example = self.evaluator.dataset[idx]
            question = example["question"]
            formatted = f"Q: {question}\n{prompt}\nA:"
            formatted_questions.append(formatted)

        try:
            outputs = self.llm_client.generate_batch(
                formatted_questions,
                max_new_tokens=512,
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return 0.0

        result = self.evaluator.evaluate_batch(outputs, self.eval_indices)
        return result["accuracy"]

    def warm_start(self, embeddings_path: str, top_k: int = 100) -> dict:
        """
        Initialize GP with pre-evaluated embeddings (warm start).

        Args:
            embeddings_path: Path to .pt file with 'embeddings' and 'accuracies'
            top_k: Use top K embeddings by accuracy

        Returns:
            Dict with initialization results
        """
        logger.info(f"Warm-starting from {embeddings_path}...")

        data = torch.load(embeddings_path, map_location=self.device, weights_only=False)
        embeddings = data["embeddings"].to(self.device)
        scores = data["accuracies"].to(self.device).float()
        instructions = data.get("instructions", [])

        if top_k < len(scores):
            top_indices = scores.argsort(descending=True)[:top_k]
            embeddings = embeddings[top_indices]
            scores = scores[top_indices]
            if instructions:
                instructions = [instructions[i] for i in top_indices.cpu().tolist()]

        self.train_X = embeddings
        self.train_Y = scores

        best_idx = scores.argmax().item()
        self.best_score = scores[best_idx].item()
        self.best_prompt = instructions[best_idx] if instructions else f"[embedding {best_idx}]"
        self.best_so_far_list = [self.best_score]

        for i, instr in enumerate(instructions):
            self._prompts[i] = instr

        self.gp.fit(self.train_X, self.train_Y)
        self.sampler.update_gp(self.gp)

        self.metrics.log_iteration(
            iteration=0,
            score=self.best_score,
            best_so_far=self.best_score,
            n_observations=len(scores),
        )

        torch.cuda.empty_cache()

        logger.info(f"Warm-start complete: {len(scores)} observations, best={self.best_score:.4f}")

        return {
            "n_samples": len(scores),
            "best_score": self.best_score,
            "best_prompt": self.best_prompt,
            "score_range": (scores.min().item(), scores.max().item()),
        }

    def initialize(self) -> dict:
        """
        Generate initial random samples and fit GP.

        Returns:
            Dict with n_samples, scores, best_score, best_prompt
        """
        logger.info(f"Initializing with {self.n_initial} random samples...")

        with torch.no_grad():
            embeddings = self.flow_model.sample(
                n_samples=self.n_initial,
                device=self.device,
                method="heun",
                num_steps=50,
                denormalize=True,
            )

        prompts = self._decode_safe(embeddings)

        scores = []
        for prompt in prompts:
            score = self._evaluate_prompt(prompt)
            scores.append(score)
            logger.info(f"Initial prompt (acc={score:.3f}):\n{prompt}")

        self.train_X = embeddings.to(self.device)
        self.train_Y = torch.tensor(scores, device=self.device, dtype=torch.float32)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        self.best_score = scores[best_idx]
        self.best_prompt = prompts[best_idx]
        self.best_so_far_list = [self.best_score]

        for i, prompt in enumerate(prompts):
            self._prompts[i] = prompt

        self.gp.fit(self.train_X, self.train_Y)
        self.sampler.update_gp(self.gp)

        self.metrics.log_iteration(
            iteration=0,
            score=self.best_score,
            best_so_far=self.best_score,
            n_observations=len(scores),
        )

        torch.cuda.empty_cache()

        logger.info(f"Initialization complete: best_score={self.best_score:.4f}")

        return {
            "n_samples": self.n_initial,
            "scores": scores,
            "best_score": self.best_score,
            "best_prompt": self.best_prompt,
        }

    def step(
        self,
        ucb_alpha: float = 1.96,
        n_restarts: int = 5,
        n_opt_steps: int = 100,
        lr: float = 0.1,
        n_candidates: int = 512,
        use_guided_sampling: bool = True,
    ) -> dict:
        """
        Execute one BO iteration using guided flow sampling.

        Algorithm (use_guided_sampling=True, recommended):
        1. Generate candidates via guided flow (stays on manifold)
        2. Rank candidates using GP-UCB acquisition function
        3. Select best candidate by UCB score
        4. Decode to text and evaluate on GSM8K
        5. Update GP with new observation

        Algorithm (use_guided_sampling=False, legacy):
        1. Find z* = argmax UCB(z) via gradient ascent
        2. Project z* through flow (encode/decode) to stay on-manifold
        3. Decode and evaluate

        Args:
            ucb_alpha: UCB exploration weight (default 1.96 for 95% CI)
            n_restarts: GP optimization restarts (for legacy method)
            n_opt_steps: Gradient steps per restart (for legacy method)
            lr: Learning rate (for legacy method)
            n_candidates: Number of guided flow candidates to generate
            use_guided_sampling: Use guided sampling (True) or direct UCB opt (False)

        Returns:
            Dict with iteration results
        """
        self.iteration += 1

        if use_guided_sampling:
            # Generate candidates via guided flow (stays on manifold!)
            with torch.no_grad():
                candidates = self.sampler.sample(
                    n_samples=n_candidates,
                    device=self.device,
                    num_steps=50,
                    method="heun",
                )

            # Rank candidates using GP-UCB
            ucb_values = self.gp.compute_acquisition(
                candidates,
                acquisition="ucb",
                alpha=ucb_alpha,
            )

            # Select best candidate
            best_idx = ucb_values.argmax().item()
            embedding = candidates[best_idx:best_idx+1]
            ucb_value = ucb_values[best_idx].item()
            l2_projection = 0.0  # Already on manifold

            logger.debug(f"Generated {n_candidates} candidates, best UCB={ucb_value:.4f}")

        else:
            # Legacy: GP-UCB optimization + flow projection
            embedding, info = self.sampler.sample_optimal(
                device=self.device,
                num_steps=50,
                method="heun",
                ucb_alpha=ucb_alpha,
                n_restarts=n_restarts,
                n_opt_steps=n_opt_steps,
                lr=lr,
            )
            ucb_value = info["ucb_value"]
            l2_projection = info["l2_projection"]

        # Optional L2-r check
        l2r_value = None
        if self.l2r_filter_enabled and self.encoder is not None:
            l2_r = self._compute_round_trip_fidelity(embedding)
            l2r_value = l2_r[0].item()

        # 3. Decode to text
        prompts = self._decode_safe(embedding)
        prompt = prompts[0]

        # 4. Evaluate
        score = self._evaluate_prompt(prompt)

        logger.info(
            f"Iter {self.iteration}: score={score:.3f}, UCB={ucb_value:.3f}, "
            f"L2_proj={l2_projection:.4f}, best={max(self.best_score, score):.3f}"
        )
        logger.info(f"Prompt:\n{prompt}")

        # 5. Update GP
        new_X = embedding.to(self.device)
        new_Y = torch.tensor([score], device=self.device, dtype=torch.float32)

        self.gp.update(new_X, new_Y)
        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)
        self.sampler.update_gp(self.gp)

        # Track best
        if score > self.best_score:
            self.best_score = score
            self.best_prompt = prompt
            logger.info(f"NEW BEST: {self.best_score:.4f}")

        self.best_so_far_list.append(self.best_score)
        self._prompts[len(self._prompts)] = prompt

        self.metrics.log_iteration(
            iteration=self.iteration,
            score=score,
            best_so_far=self.best_score,
            n_observations=self.train_X.shape[0],
        )

        torch.cuda.empty_cache()

        result = {
            "iteration": self.iteration,
            "score": score,
            "ucb_value": ucb_value,
            "l2_projection": l2_projection,
            "best_so_far": self.best_score,
            "best_prompt": self.best_prompt,
            "n_observations": self.train_X.shape[0],
            "prompt": prompt,
        }

        if l2r_value is not None:
            result["l2r"] = l2r_value

        return result

    def get_state(self) -> OptimizationState:
        """Get current optimization state for checkpointing."""
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
        """Save current state to checkpoint file."""
        state = self.get_state()
        state.save(path)

    def load_checkpoint(self, path: str) -> None:
        """Load state from checkpoint file and re-fit GP."""
        state = OptimizationState.load(path, device=self.device)

        self.train_X = state.train_X
        self.train_Y = state.train_Y
        self.best_so_far_list = state.best_so_far
        self.iteration = state.iteration
        self.best_prompt = state.best_prompt
        self.best_score = state.best_score
        self.eval_indices = state.eval_indices

        if state.metrics:
            self.metrics = MetricsTracker.from_dict(state.metrics)

        self.gp.fit(self.train_X, self.train_Y)
        self.sampler.update_gp(self.gp)

        logger.info(
            f"Loaded checkpoint from {path}: iteration={self.iteration}, "
            f"n_obs={self.train_X.shape[0]}, best_score={self.best_score:.4f}"
        )

    @property
    def n_observations(self) -> int:
        """Total number of observations collected."""
        if self.train_X is None:
            return 0
        return self.train_X.shape[0]
