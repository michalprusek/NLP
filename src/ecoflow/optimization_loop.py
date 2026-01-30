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

Features:
- Round-trip fidelity (L2-r) filtering to reject off-manifold samples
- UCB-based candidate selection for sample efficiency
- Checkpoint/resume support for long-running optimization

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
from typing import Any, Dict, List, Optional

import torch

from src.ecoflow.batch_selection import select_batch_candidates
from src.ecoflow.decoder import SonarDecoder
from src.ecoflow.flow_density import filter_by_flow_density
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
        encoder: Optional[Any] = None,
        l2r_threshold: float = 0.5,
        l2r_filter_enabled: bool = True,
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
        self.batch_size = batch_size
        self.eval_subset_size = eval_subset_size
        self.device = device

        # L2-r filtering parameters
        self.encoder = encoder
        self.l2r_threshold = l2r_threshold
        self.l2r_filter_enabled = l2r_filter_enabled

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

    def _compute_round_trip_fidelity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute round-trip fidelity (L2-r) for embeddings.

        L2-r = ||emb - encode(decode(emb))||_2

        Low L2-r indicates embedding is on valid SONAR manifold.
        High L2-r indicates off-manifold (will decode poorly).

        From Phase 7 diagnosis:
        - Flow samples: L2-r avg 0.3365 (off-manifold)
        - Instruction embeddings: L2-r avg 0.0548 (on-manifold)
        - Threshold of 0.5 should reject most off-manifold samples

        Args:
            embeddings: SONAR embeddings [B, 1024]

        Returns:
            L2-r distances [B]
        """
        if self.encoder is None:
            logger.debug("No encoder provided, skipping L2-r computation")
            return torch.zeros(embeddings.shape[0], device=embeddings.device)

        # Decode to text
        texts = self._decode_safe(embeddings)

        # Re-encode
        try:
            re_embeddings = self.encoder.predict(
                texts,
                source_lang="eng_Latn",
                batch_size=len(texts),
            )
            # Convert to tensor if needed
            if not isinstance(re_embeddings, torch.Tensor):
                re_embeddings = torch.tensor(re_embeddings, device=embeddings.device)
            re_embeddings = re_embeddings.to(embeddings.device)
        except Exception as e:
            logger.warning(f"Re-encoding failed: {e}")
            return torch.zeros(embeddings.shape[0], device=embeddings.device)

        # Compute L2 distance (round-trip fidelity)
        l2_r = (embeddings - re_embeddings).norm(dim=-1)

        return l2_r

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

    def warm_start(self, embeddings_path: str, top_k: int = 100) -> dict:
        """
        Initialize GP with pre-evaluated embeddings (warm start).

        Loads embeddings and scores from file, skipping expensive LLM evaluation.
        Much more efficient than random initialization.

        Args:
            embeddings_path: Path to .pt file with 'embeddings' and 'accuracies'
            top_k: Use top K embeddings by accuracy (default: all)

        Returns:
            Dict with initialization results
        """
        logger.info(f"Warm-starting from {embeddings_path}...")

        # Load pre-evaluated data
        data = torch.load(embeddings_path, map_location=self.device, weights_only=False)
        embeddings = data["embeddings"].to(self.device)
        scores = data["accuracies"].to(self.device).float()
        instructions = data.get("instructions", [])

        # Select top K by accuracy
        if top_k < len(scores):
            top_indices = scores.argsort(descending=True)[:top_k]
            embeddings = embeddings[top_indices]
            scores = scores[top_indices]
            if instructions:
                instructions = [instructions[i] for i in top_indices.cpu().tolist()]
            logger.info(f"Selected top {top_k} embeddings (acc range: {scores.min():.4f} - {scores.max():.4f})")
        else:
            logger.info(f"Using all {len(scores)} embeddings")

        # Store training data
        self.train_X = embeddings
        self.train_Y = scores

        # Track best
        best_idx = scores.argmax().item()
        self.best_score = scores[best_idx].item()
        self.best_prompt = instructions[best_idx] if instructions else f"[embedding {best_idx}]"
        self.best_so_far_list = [self.best_score]

        # Store prompts
        for i, instr in enumerate(instructions):
            self._prompts[i] = instr

        # Fit GP
        logger.info(f"Fitting GP on {len(scores)} pre-evaluated observations...")
        self.gp.fit(self.train_X, self.train_Y)

        # Update sampler's GP reference
        self.sampler.update_gp(self.gp)

        # Log metrics
        self.metrics.log_iteration(
            iteration=0,
            batch_scores=scores.cpu().tolist(),
            best_so_far=self.best_score,
            n_observations=len(scores),
        )

        torch.cuda.empty_cache()

        logger.info(f"Warm-start complete: {len(scores)} observations, best={self.best_score:.4f}")
        logger.info(f"Best prompt:\n{self.best_prompt}")

        return {
            "n_samples": len(scores),
            "best_score": self.best_score,
            "best_prompt": self.best_prompt,
            "score_range": (scores.min().item(), scores.max().item()),
        }

    def initialize(self) -> dict:
        """
        Generate initial random samples and fit GP.

        Generates n_initial samples from the flow model (no guidance),
        decodes them to text, evaluates on GSM8K, and fits the GP.

        NOTE: Consider using warm_start() with pre-evaluated embeddings instead.

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
                method="heun",
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

    def step(self, n_candidates: int = 64, ucb_alpha: float = 1.96) -> dict:
        """
        Execute one BO iteration with UCB-based candidate selection.

        Efficient approach:
        1. Generate many candidates via guided sampling
        2. Select best candidate by UCB acquisition (no LLM needed)
        3. Decode and evaluate only the selected candidate
        4. Update GP with the single new observation

        Args:
            n_candidates: Number of candidates to generate (default: 64)
            ucb_alpha: UCB exploration weight (default: 1.96 for 95% CI)

        Returns:
            Dict with iteration results
        """
        self.iteration += 1
        logger.info(f"\n=== Iteration {self.iteration} ===")

        # 1. Generate many guided candidates
        logger.info(f"Generating {n_candidates} guided candidates...")
        embeddings = self.sampler.sample(
            n_samples=n_candidates,
            device=self.device,
            num_steps=50,
            method="heun",
        )

        # 1.5. L2-r filtering (if enabled and encoder available)
        l2r_stats = {}
        if self.l2r_filter_enabled and self.encoder is not None:
            l2_r = self._compute_round_trip_fidelity(embeddings)
            l2r_stats = {
                "l2r_mean": l2_r.mean().item(),
                "l2r_max": l2_r.max().item(),
                "l2r_min": l2_r.min().item(),
            }
            logger.info(f"  L2-r stats: mean={l2_r.mean():.4f}, max={l2_r.max():.4f}, min={l2_r.min():.4f}")

            on_manifold_mask = l2_r <= self.l2r_threshold
            n_on_manifold = on_manifold_mask.sum().item()
            l2r_stats["n_on_manifold"] = n_on_manifold
            l2r_stats["n_candidates"] = len(embeddings)
            logger.info(f"  On-manifold: {n_on_manifold}/{len(embeddings)} (threshold={self.l2r_threshold})")

            if n_on_manifold >= 1:
                # Filter to on-manifold candidates for UCB selection
                embeddings = embeddings[on_manifold_mask]
                logger.info(f"  Filtered to {len(embeddings)} on-manifold candidates")
            else:
                logger.warning(
                    f"No candidates on-manifold (L2-r <= {self.l2r_threshold}). "
                    "Using all candidates but quality may be poor."
                )

        # 2. Select best candidate by UCB (μ + α·σ for maximization)
        logger.info("Selecting best candidate by UCB...")
        with torch.no_grad():
            mean, std = self.gp.predict(embeddings)
            ucb = mean + ucb_alpha * std
            best_idx = ucb.argmax().item()

        ucb_value = ucb[best_idx].item()
        gp_mean = mean[best_idx].item()
        gp_std = std[best_idx].item()
        logger.info(f"  Best UCB: {ucb_value:.4f} (μ={gp_mean:.4f}, σ={gp_std:.4f})")

        # 3. Decode only the selected candidate
        selected_embedding = embeddings[best_idx:best_idx+1]
        logger.info("Decoding selected embedding...")
        prompts = self._decode_safe(selected_embedding)
        prompt = prompts[0]

        # 4. Evaluate only the selected prompt
        if self._is_valid_prompt(prompt):
            logger.info(f"Evaluating on {len(self.eval_indices)} questions...")
            scores = self._evaluate_prompts([prompt])
            score = scores[0]
        else:
            logger.info("  Selected prompt INVALID, assigning score=0")
            score = 0.0

        logger.info(f"  Prompt (acc={score:.3f}):\n{prompt[:200]}...")

        # 5. Update GP with the single new observation
        new_X = selected_embedding.to(self.device)
        new_Y = torch.tensor([score], device=self.device, dtype=torch.float32)

        self.gp.update(new_X, new_Y)

        # Update training data
        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)

        # 6. Update sampler's GP reference
        self.sampler.update_gp(self.gp)

        # Track best
        if score > self.best_score:
            self.best_score = score
            self.best_prompt = prompt
            logger.info(f"NEW BEST: {self.best_score:.4f}")

        self.best_so_far_list.append(self.best_score)

        # Store prompt
        self._prompts[len(self._prompts)] = prompt

        # Log metrics
        self.metrics.log_iteration(
            iteration=self.iteration,
            batch_scores=[score],
            best_so_far=self.best_score,
            n_observations=self.train_X.shape[0],
        )

        # Clear GPU cache
        torch.cuda.empty_cache()

        result = {
            "iteration": self.iteration,
            "score": score,
            "ucb_value": ucb_value,
            "gp_mean": gp_mean,
            "gp_std": gp_std,
            "best_so_far": self.best_score,
            "best_prompt": self.best_prompt,
            "n_observations": self.train_X.shape[0],
            "prompt": prompt,
            **l2r_stats,  # Include L2-r filtering stats if computed
        }

        logger.info(
            f"Iteration {self.iteration} complete: "
            f"score={score:.3f}, UCB={ucb_value:.3f}, "
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

    def batch_step(
        self,
        batch_size: int = 4,
        n_candidates: int = 64,
        use_local_penalization: bool = True,
        use_density_filter: bool = False,
        density_percentile: float = 25.0,
        ucb_alpha: float = 1.96,
    ) -> dict:
        """
        Execute one batch BO iteration with parallel evaluation.

        More efficient than sequential step() when LLM supports batch inference.
        Evaluates multiple candidates per iteration for better throughput.

        Algorithm:
        1. Generate candidates via guided sampling
        2. Optional: filter by flow density (keeps high-density samples)
        3. Optional: filter by L2-r (keeps on-manifold samples)
        4. Select diverse batch using Local Penalization
        5. Decode all batch candidates to text
        6. Evaluate all prompts (parallel LLM batch inference)
        7. Update GP with all observations

        Args:
            batch_size: Number of candidates to evaluate per iteration (default 4)
            n_candidates: Number of initial candidates to generate (default 64)
            use_local_penalization: Use LP for diverse batch selection (default True)
            use_density_filter: Filter by flow density before selection (default False)
            density_percentile: Reject samples below this density percentile (default 25.0)
            ucb_alpha: UCB exploration weight for batch selection (default 1.96)

        Returns:
            Dict with batch results including:
            - iteration: Current iteration number
            - scores: List of scores for each batch candidate
            - prompts: List of decoded prompts
            - best_so_far: Running best score
            - n_observations: Total observation count
            - stats: Dict with filtering and selection statistics

        Example:
            >>> loop = BOOptimizationLoop(...)
            >>> loop.initialize()
            >>> for _ in range(25):  # 25 batch iterations = 100 evaluations with batch_size=4
            ...     result = loop.batch_step(batch_size=4, use_density_filter=True)
            ...     print(f"Best: {result['best_so_far']:.3f}")
        """
        self.iteration += 1
        logger.info(f"\n=== Batch Iteration {self.iteration} (batch_size={batch_size}) ===")

        stats = {
            "n_candidates_initial": n_candidates,
            "batch_size": batch_size,
        }

        # 1. Generate guided candidates
        logger.info(f"Generating {n_candidates} guided candidates...")
        embeddings = self.sampler.sample(
            n_samples=n_candidates,
            device=self.device,
            num_steps=50,
            method="heun",
        )
        n_after_gen = len(embeddings)

        # 2. Optional: Filter by flow density
        if use_density_filter:
            logger.info(f"Filtering by flow density (percentile={density_percentile})...")
            embeddings, log_densities = filter_by_flow_density(
                self.flow_model,
                embeddings,
                percentile=density_percentile,
                min_samples=batch_size,
            )
            stats["density_filter"] = {
                "n_before": n_after_gen,
                "n_after": len(embeddings),
                "log_density_min": log_densities.min().item(),
                "log_density_max": log_densities.max().item(),
                "log_density_mean": log_densities.mean().item(),
            }
            logger.info(f"  Density filter: {n_after_gen} -> {len(embeddings)}")

        # 3. Optional: L2-r filtering (if enabled and encoder available)
        if self.l2r_filter_enabled and self.encoder is not None:
            n_before_l2r = len(embeddings)
            l2_r = self._compute_round_trip_fidelity(embeddings)
            stats["l2r_filter"] = {
                "l2r_mean": l2_r.mean().item(),
                "l2r_max": l2_r.max().item(),
                "l2r_min": l2_r.min().item(),
                "threshold": self.l2r_threshold,
            }
            logger.info(
                f"  L2-r stats: mean={l2_r.mean():.4f}, "
                f"max={l2_r.max():.4f}, min={l2_r.min():.4f}"
            )

            on_manifold_mask = l2_r <= self.l2r_threshold
            n_on_manifold = on_manifold_mask.sum().item()
            stats["l2r_filter"]["n_on_manifold"] = n_on_manifold

            if n_on_manifold >= batch_size:
                embeddings = embeddings[on_manifold_mask]
                logger.info(
                    f"  L2-r filter: {n_before_l2r} -> {len(embeddings)} "
                    f"(threshold={self.l2r_threshold})"
                )
            else:
                logger.warning(
                    f"Only {n_on_manifold} candidates on-manifold (need {batch_size}). "
                    "Keeping all candidates."
                )

        # 4. Select diverse batch
        n_available = len(embeddings)
        actual_batch_size = min(batch_size, n_available)

        if use_local_penalization and actual_batch_size < n_available:
            logger.info(f"Selecting {actual_batch_size} candidates via Local Penalization...")
            selected_embeddings, selected_indices = select_batch_candidates(
                self.gp,
                embeddings,
                batch_size=actual_batch_size,
                method="local_penalization",
                alpha=ucb_alpha,
            )
            stats["selection_method"] = "local_penalization"
        else:
            # Simple greedy selection by UCB
            logger.info(f"Selecting {actual_batch_size} candidates via greedy UCB...")
            with torch.no_grad():
                mean, std = self.gp.predict(embeddings)
                ucb = mean + ucb_alpha * std
                _, selected_indices = ucb.topk(actual_batch_size)
            selected_embeddings = embeddings[selected_indices]
            stats["selection_method"] = "greedy"

        stats["n_selected"] = len(selected_embeddings)

        # Compute batch diversity (mean pairwise distance)
        if len(selected_embeddings) > 1:
            pairwise_dists = torch.cdist(selected_embeddings, selected_embeddings)
            # Get upper triangle (excluding diagonal)
            triu_indices = torch.triu_indices(len(selected_embeddings), len(selected_embeddings), offset=1)
            mean_dist = pairwise_dists[triu_indices[0], triu_indices[1]].mean().item()
            stats["batch_diversity"] = mean_dist
            logger.info(f"  Batch diversity (mean pairwise dist): {mean_dist:.4f}")

        # 5. Decode all batch candidates
        logger.info(f"Decoding {len(selected_embeddings)} selected candidates...")
        prompts = self._decode_safe(selected_embeddings)

        # 6. Evaluate all prompts
        logger.info(f"Evaluating {len(prompts)} prompts on {len(self.eval_indices)} questions...")
        scores = self._evaluate_prompts(prompts)

        # 7. Update GP with all new observations
        new_X = selected_embeddings.to(self.device)
        new_Y = torch.tensor(scores, device=self.device, dtype=torch.float32)

        # Update GP with batch
        self.gp.update(new_X, new_Y)

        # Update training data
        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)

        # Update sampler's GP reference
        self.sampler.update_gp(self.gp)

        # Track best
        batch_best_idx = max(range(len(scores)), key=lambda i: scores[i])
        batch_best_score = scores[batch_best_idx]
        batch_best_prompt = prompts[batch_best_idx]

        if batch_best_score > self.best_score:
            self.best_score = batch_best_score
            self.best_prompt = batch_best_prompt
            logger.info(f"NEW BEST: {self.best_score:.4f}")
            logger.info(f"Best prompt:\n{self.best_prompt}")

        self.best_so_far_list.append(self.best_score)

        # Store prompts
        for i, (prompt, score) in enumerate(zip(prompts, scores)):
            self._prompts[len(self._prompts)] = prompt

        # Log metrics
        self.metrics.log_iteration(
            iteration=self.iteration,
            batch_scores=scores,
            best_so_far=self.best_score,
            n_observations=self.train_X.shape[0],
        )

        # Clear GPU cache
        torch.cuda.empty_cache()

        result = {
            "iteration": self.iteration,
            "scores": scores,
            "prompts": prompts,
            "batch_best_score": batch_best_score,
            "batch_best_prompt": batch_best_prompt,
            "best_so_far": self.best_score,
            "best_prompt": self.best_prompt,
            "n_observations": self.train_X.shape[0],
            "stats": stats,
        }

        logger.info(
            f"Batch iteration {self.iteration} complete: "
            f"batch_scores={[f'{s:.3f}' for s in scores]}, "
            f"best_so_far={self.best_score:.3f}, "
            f"n_obs={self.train_X.shape[0]}"
        )

        return result

    @property
    def n_observations(self) -> int:
        """Total number of observations collected."""
        return 0 if self.train_X is None else self.train_X.shape[0]
