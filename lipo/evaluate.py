"""GSM8K evaluation for LIPO.

Provides LLM-based evaluation of instructions with call counting.
Self-contained - no imports from other modules outside lipo/.
"""

import re
from typing import List, Dict, Optional, Callable
from lipo.instruction import InstructionOnlyPrompt


def extract_answer(text: str) -> Optional[float]:
    """Extract the last number from model output.

    Always extracts the last number only - no format-specific patterns.
    This is robust to different output formats (####, boxed{}, plain numbers).

    Args:
        text: Model output text

    Returns:
        Extracted number or None if no number found
    """
    if not text:
        return None

    # Find all numbers (including negative and decimal)
    numbers = re.findall(r'[-+]?\d+(?:[.,]\d+)?', text)

    if not numbers:
        return None

    # Get the last number
    last_num = numbers[-1].replace(',', '')

    try:
        return float(last_num)
    except ValueError:
        return None


def extract_gold_answer(answer_text: str) -> Optional[float]:
    """Extract gold answer from GSM8K answer field.

    GSM8K answers typically end with #### NUMBER.

    Args:
        answer_text: Full answer text from GSM8K

    Returns:
        Gold answer number
    """
    # Try #### format first
    if "####" in answer_text:
        after_marker = answer_text.split("####")[-1].strip()
        numbers = re.findall(r'[-+]?\d+(?:[.,]\d+)?', after_marker)
        if numbers:
            try:
                return float(numbers[0].replace(',', ''))
            except ValueError:
                pass

    # Fallback to last number
    return extract_answer(answer_text)


class GSM8KEvaluator:
    """GSM8K instruction evaluator with LLM call counting.

    Uses Q_end format (instruction comes AFTER the question):
        Q: {question}
        {instruction}
        A:
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        backend: str = "vllm",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        """Initialize evaluator.

        Args:
            model: Model name for evaluation
            backend: LLM backend (vllm, openai, etc.)
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature (0 = greedy)
        """
        self.model = model
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self.total_calls = 0

    def _get_client(self):
        """Lazy load LLM client."""
        if self._client is None:
            from src.llm_client import create_llm_client
            print(f"Initializing evaluation LLM: {self.model}")
            self._client = create_llm_client(self.model, self.backend)
        return self._client

    def format_prompt(self, instruction: str, question: str) -> str:
        """Format prompt in Q_end style.

        Args:
            instruction: Instruction text
            question: Question to solve

        Returns:
            Formatted prompt
        """
        return f"Q: {question}\n{instruction}\nA:"

    def evaluate_single(
        self,
        instruction: str,
        question: str,
        gold_answer: float,
    ) -> bool:
        """Evaluate single Q/A pair.

        Args:
            instruction: Instruction text
            question: Question to solve
            gold_answer: Correct answer

        Returns:
            True if model got the correct answer
        """
        client = self._get_client()
        prompt = self.format_prompt(instruction, question)

        response = client.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        self.total_calls += 1

        model_answer = extract_answer(response)

        if model_answer is None:
            return False

        # Check with tolerance for floats
        return abs(model_answer - gold_answer) < 1e-6

    def evaluate_batch(
        self,
        instruction: str,
        data: List[Dict],
        verbose: bool = False,
    ) -> float:
        """Evaluate instruction on batch of Q/A pairs.

        Args:
            instruction: Instruction text
            data: List of {"question": str, "answer": str} dicts
            verbose: Print progress

        Returns:
            Error rate (fraction of incorrect answers)
        """
        client = self._get_client()

        # Format all prompts
        prompts = []
        gold_answers = []
        for item in data:
            prompts.append(self.format_prompt(instruction, item["question"]))
            gold = extract_gold_answer(item["answer"])
            gold_answers.append(gold)

        # Generate all responses
        responses = client.generate_batch(
            prompts,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        self.total_calls += len(prompts)

        # Count correct
        correct = 0
        for response, gold in zip(responses, gold_answers):
            model_answer = extract_answer(response)
            if model_answer is not None and gold is not None:
                if abs(model_answer - gold) < 1e-6:
                    correct += 1

        error_rate = 1.0 - (correct / len(data))

        if verbose:
            print(f"  Evaluated {len(data)} samples: {correct}/{len(data)} correct, error={error_rate:.4f}")

        return error_rate

    def __call__(
        self,
        prompt: InstructionOnlyPrompt,
        data: List[Dict],
    ) -> float:
        """Callable interface for Hyperband.

        Args:
            prompt: InstructionOnlyPrompt to evaluate
            data: Validation data subset

        Returns:
            Error rate
        """
        return self.evaluate_batch(prompt.instruction, data)


def create_evaluator_function(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    backend: str = "vllm",
) -> Callable[[InstructionOnlyPrompt, List[Dict]], float]:
    """Create evaluator function for Hyperband.

    Returns a callable that evaluates prompts and tracks LLM calls.

    Args:
        model: Model name
        backend: LLM backend

    Returns:
        Evaluator function with .total_calls attribute
    """
    evaluator = GSM8KEvaluator(model=model, backend=backend)

    def evaluate(prompt: InstructionOnlyPrompt, data: List[Dict]) -> float:
        return evaluator(prompt, data)

    # Attach evaluator for call counting access
    evaluate.evaluator = evaluator

    return evaluate
