"""
Multi-Model Evaluator Pool for LLM evaluation.

Supports two modes:
1. Multi-GPU parallel: Each model on separate GPU (fast, needs 4 GPUs)
2. Single-GPU sequential: Models loaded/unloaded one at a time (memory efficient)
"""
import gc
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

# Add parent directory for imports
sys.path.insert(0, "/home/prusek/NLP")

from multi_model_optimizer.config import MultiModelConfig


@dataclass
class EvaluationResult:
    """Result of evaluating a prompt on one model."""

    model_name: str
    num_correct: int
    num_total: int
    error_rate: float
    responses: Optional[List[str]] = None


def extract_answer(text: str) -> Optional[str]:
    """
    Extract numerical answer from model response.

    Supports formats:
        - "final_answer: NUMBER"
        - "#### NUMBER"
        - Last number in the response
    """
    if not text:
        return None

    # Try "final_answer: NUMBER" format
    match = re.search(r"final_answer:\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try "#### NUMBER" format (GSM8K standard)
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", text)
    if match:
        return match.group(1)

    # Fallback: last number in text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1]

    return None


def compare_answers(extracted: Optional[str], expected: str) -> bool:
    """Compare extracted answer with expected answer."""
    if extracted is None:
        return False

    try:
        ext_num = float(extracted.replace(",", ""))
        exp_num = float(expected.replace(",", ""))

        if ext_num == int(ext_num) and exp_num == int(exp_num):
            return int(ext_num) == int(exp_num)

        return abs(ext_num - exp_num) < 1e-6

    except ValueError:
        return extracted.strip() == expected.strip()


def _evaluate_single_model_subprocess(
    model_name: str,
    gpu_id: int,
    prompts: List[str],
    answers: List[str],
    max_tokens: int = 1024,
) -> EvaluationResult:
    """
    Evaluate prompts on a single model in a subprocess.
    Used for multi-GPU parallel mode.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from src.llm_client import VLLMClient

    client = VLLMClient(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )

    responses = client.generate_batch(
        prompts,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    num_correct = 0
    for response, expected in zip(responses, answers):
        extracted = extract_answer(response)
        if compare_answers(extracted, expected):
            num_correct += 1

    return EvaluationResult(
        model_name=model_name,
        num_correct=num_correct,
        num_total=len(prompts),
        error_rate=1.0 - (num_correct / len(prompts)) if prompts else 0.0,
        responses=responses,
    )


class SingleGPUModelManager:
    """
    Manages loading/unloading of models on a single GPU.

    Intelligently switches between models by:
    1. Keeping track of currently loaded model
    2. Only reloading when switching to different model
    3. Properly cleaning up GPU memory between loads
    """

    def __init__(self, gpu_id: int = 0, gpu_memory_utilization: float = 0.85):
        """
        Initialize single-GPU model manager.

        Args:
            gpu_id: GPU device ID to use
            gpu_memory_utilization: vLLM memory utilization (lower = safer swaps)
        """
        self.gpu_id = gpu_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.current_model_name: Optional[str] = None
        self.current_client: Optional[Any] = None

        # Set GPU visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def _cleanup_gpu(self):
        """Clean up GPU memory after unloading a model."""
        if self.current_client is not None:
            # Delete the client
            del self.current_client
            self.current_client = None
            self.current_model_name = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("  GPU memory cleaned up")

    def load_model(self, model_name: str, max_tokens: int = 1024) -> Any:
        """
        Load a model, switching from current if different.

        Args:
            model_name: HuggingFace model identifier
            max_tokens: Max tokens for generation

        Returns:
            Loaded VLLMClient
        """
        # If same model already loaded, reuse it
        if model_name == self.current_model_name and self.current_client is not None:
            print(f"  Reusing loaded model: {model_name}")
            return self.current_client

        # Unload current model if different
        if self.current_model_name is not None:
            print(f"  Unloading model: {self.current_model_name}")
            self._cleanup_gpu()

        # Load new model
        print(f"  Loading model: {model_name}")
        from src.llm_client import VLLMClient

        self.current_client = VLLMClient(
            model_name=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.current_model_name = model_name

        return self.current_client

    def evaluate(
        self,
        model_name: str,
        prompts: List[str],
        answers: List[str],
        max_tokens: int = 1024,
    ) -> EvaluationResult:
        """
        Evaluate prompts on a model (loading if needed).

        Args:
            model_name: Model to evaluate on
            prompts: List of formatted prompts
            answers: List of expected answers
            max_tokens: Max tokens for generation

        Returns:
            EvaluationResult with accuracy
        """
        client = self.load_model(model_name, max_tokens)

        responses = client.generate_batch(
            prompts,
            max_new_tokens=max_tokens,
            temperature=0.0,
        )

        num_correct = 0
        for response, expected in zip(responses, answers):
            extracted = extract_answer(response)
            if compare_answers(extracted, expected):
                num_correct += 1

        return EvaluationResult(
            model_name=model_name,
            num_correct=num_correct,
            num_total=len(prompts),
            error_rate=1.0 - (num_correct / len(prompts)) if prompts else 0.0,
            responses=responses,
        )

    def unload(self):
        """Explicitly unload current model and free GPU memory."""
        if self.current_model_name is not None:
            print(f"  Unloading model: {self.current_model_name}")
            self._cleanup_gpu()


class ModelEvaluatorPool:
    """
    Pool of LLM evaluators for multi-model evaluation.

    Supports two modes:
    1. parallel_models=True, single_gpu=False: Multi-GPU parallel (needs 4 GPUs)
    2. single_gpu=True: Sequential model switching on one GPU (memory efficient)

    Example:
        >>> config = MultiModelConfig(single_gpu=True)
        >>> pool = ModelEvaluatorPool(config)
        >>> results = pool.evaluate_prompt_all_models(
        ...     instruction="Solve step by step:",
        ...     exemplar="Q: 2+2=? A: 4",
        ...     validation_data=data,
        ...     fidelity=100
        ... )
    """

    def __init__(self, config: MultiModelConfig):
        """
        Initialize the evaluator pool.

        Args:
            config: Multi-model configuration
        """
        self.config = config
        self.models = config.target_models
        self.gpu_assignment = config.gpu_assignment
        self.max_tokens = config.task_max_tokens

        # Initialize single-GPU manager if needed
        self.single_gpu_manager: Optional[SingleGPUModelManager] = None
        if getattr(config, "single_gpu", False):
            gpu_id = getattr(config, "single_gpu_id", 0)
            self.single_gpu_manager = SingleGPUModelManager(
                gpu_id=gpu_id,
                gpu_memory_utilization=0.85,  # Lower for safer swaps
            )
            print(f"Single-GPU mode enabled on GPU {gpu_id}")

    def format_prompt(
        self,
        question: str,
        instruction: str,
        exemplar: str,
    ) -> str:
        """Format a prompt for evaluation."""
        return f"Question: {question}\n\n{instruction}\n\n{exemplar}\n\nAnswer:"

    def evaluate_prompt_all_models(
        self,
        instruction: str,
        exemplar: str,
        validation_data: List[Dict[str, str]],
        fidelity: int,
    ) -> Dict[str, float]:
        """
        Evaluate a prompt on all models.

        Args:
            instruction: System instruction text
            exemplar: Few-shot examples text
            validation_data: List of {"question": str, "answer": str}
            fidelity: Number of validation examples to use

        Returns:
            Dict mapping model_name -> error_rate
        """
        data = validation_data[:fidelity]
        prompts = [
            self.format_prompt(d["question"], instruction, exemplar)
            for d in data
        ]
        answers = [d["answer"] for d in data]

        # Single-GPU mode: use model manager
        if self.single_gpu_manager is not None:
            return self._evaluate_single_gpu(prompts, answers)

        # Multi-GPU mode
        if self.config.parallel_models:
            return self._evaluate_parallel(prompts, answers)
        else:
            return self._evaluate_sequential(prompts, answers)

    def _evaluate_single_gpu(
        self,
        prompts: List[str],
        answers: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate on all models using single GPU with model switching.
        """
        results = {}

        for model_name in self.models:
            print(f"\n[Single-GPU] Evaluating on {model_name}")
            try:
                result = self.single_gpu_manager.evaluate(
                    model_name,
                    prompts,
                    answers,
                    self.max_tokens,
                )
                results[model_name] = result.error_rate
                print(f"  Accuracy: {1 - result.error_rate:.2%}")
            except Exception as e:
                print(f"  Error: {e}")
                results[model_name] = 1.0

        return results

    def _evaluate_parallel(
        self,
        prompts: List[str],
        answers: List[str],
    ) -> Dict[str, float]:
        """Evaluate on all models in parallel using ProcessPoolExecutor."""
        results = {}

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_single_model_subprocess,
                    model_name,
                    self.gpu_assignment[model_name],
                    prompts,
                    answers,
                    self.max_tokens,
                ): model_name
                for model_name in self.models
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result(timeout=self.config.model_timeout)
                    results[model_name] = result.error_rate
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    results[model_name] = 1.0

        return results

    def _evaluate_sequential(
        self,
        prompts: List[str],
        answers: List[str],
    ) -> Dict[str, float]:
        """Evaluate on all models sequentially (multi-GPU, but not parallel)."""
        results = {}

        for model_name in self.models:
            try:
                result = _evaluate_single_model_subprocess(
                    model_name,
                    self.gpu_assignment[model_name],
                    prompts,
                    answers,
                    self.max_tokens,
                )
                results[model_name] = result.error_rate
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = 1.0

        return results

    def evaluate_batch_all_models(
        self,
        instruction: str,
        exemplar: str,
        validation_data: List[Dict[str, str]],
        start_idx: int,
        end_idx: int,
    ) -> Dict[str, int]:
        """
        Evaluate a batch and return success counts.
        Used by sequential tester for incremental evaluation.
        """
        data = validation_data[start_idx:end_idx]
        prompts = [
            self.format_prompt(d["question"], instruction, exemplar)
            for d in data
        ]
        answers = [d["answer"] for d in data]

        results = {}

        # Single-GPU mode
        if self.single_gpu_manager is not None:
            for model_name in self.models:
                try:
                    result = self.single_gpu_manager.evaluate(
                        model_name,
                        prompts,
                        answers,
                        self.max_tokens,
                    )
                    results[model_name] = result.num_correct
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    results[model_name] = 0
            return results

        # Multi-GPU mode
        if self.config.parallel_models:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(
                        _evaluate_single_model_subprocess,
                        model_name,
                        self.gpu_assignment[model_name],
                        prompts,
                        answers,
                        self.max_tokens,
                    ): model_name
                    for model_name in self.models
                }

                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        result = future.result(timeout=self.config.model_timeout)
                        results[model_name] = result.num_correct
                    except Exception as e:
                        print(f"Error evaluating {model_name}: {e}")
                        results[model_name] = 0
        else:
            for model_name in self.models:
                try:
                    result = _evaluate_single_model_subprocess(
                        model_name,
                        self.gpu_assignment[model_name],
                        prompts,
                        answers,
                        self.max_tokens,
                    )
                    results[model_name] = result.num_correct
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    results[model_name] = 0

        return results

    def evaluate_candidates_batch_per_model(
        self,
        candidates: List[Dict],
        validation_data: List[Dict[str, str]],
        fidelity: int,
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple candidates with minimal model switching.

        Strategy: Load each model once, evaluate ALL candidates, then switch.
        This minimizes GPU memory operations.

        Args:
            candidates: List of {"instruction": str, "exemplar": str, ...}
            validation_data: Full validation dataset
            fidelity: Number of validation examples per candidate

        Returns:
            List of {model_name -> error_rate} for each candidate
        """
        if not candidates:
            return []

        data = validation_data[:fidelity]
        answers = [d["answer"] for d in data]

        # Pre-format all prompts for all candidates
        all_candidate_prompts = []
        for cand in candidates:
            prompts = [
                self.format_prompt(d["question"], cand["instruction"], cand["exemplar"])
                for d in data
            ]
            all_candidate_prompts.append(prompts)

        # Initialize results: list of dicts, one per candidate
        results = [{} for _ in candidates]

        print(f"\n[Batch Evaluation] {len(candidates)} candidates × {len(self.models)} models")
        print(f"  Fidelity: {fidelity} samples per candidate")

        # Single-GPU mode: evaluate all candidates per model before switching
        if self.single_gpu_manager is not None:
            for model_name in self.models:
                print(f"\n  Loading {model_name}...")
                try:
                    client = self.single_gpu_manager.load_model(model_name, self.max_tokens)

                    # Evaluate all candidates on this model
                    for cand_idx, prompts in enumerate(all_candidate_prompts):
                        responses = client.generate_batch(
                            prompts,
                            max_new_tokens=self.max_tokens,
                            temperature=0.0,
                        )

                        num_correct = 0
                        for response, expected in zip(responses, answers):
                            extracted = extract_answer(response)
                            if compare_answers(extracted, expected):
                                num_correct += 1

                        error_rate = 1.0 - (num_correct / len(prompts)) if prompts else 0.0
                        results[cand_idx][model_name] = error_rate

                    print(f"    Evaluated {len(candidates)} candidates")

                except Exception as e:
                    print(f"    Error: {e}")
                    # Set error rate to 1.0 for all candidates on this model
                    for cand_idx in range(len(candidates)):
                        results[cand_idx][model_name] = 1.0
        else:
            # Multi-GPU mode: still batch per model for consistency
            for model_name in self.models:
                gpu_id = self.gpu_assignment[model_name]
                print(f"\n  Evaluating on {model_name} (GPU {gpu_id})...")

                for cand_idx, prompts in enumerate(all_candidate_prompts):
                    try:
                        result = _evaluate_single_model_subprocess(
                            model_name,
                            gpu_id,
                            prompts,
                            answers,
                            self.max_tokens,
                        )
                        results[cand_idx][model_name] = result.error_rate
                    except Exception as e:
                        print(f"    Error on candidate {cand_idx}: {e}")
                        results[cand_idx][model_name] = 1.0

        return results

    def cleanup(self):
        """Clean up resources (unload models in single-GPU mode)."""
        if self.single_gpu_manager is not None:
            self.single_gpu_manager.unload()
