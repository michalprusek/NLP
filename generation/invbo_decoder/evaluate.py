"""Instruction evaluation utilities for InvBO.

Provides functions to evaluate instruction prompts on GSM8K test data.
"""

from typing import Any, Dict, List, Tuple

from src.gsm8k_evaluator import compare_numbers, extract_answer, extract_ground_truth
from src.llm_client import LLMClient


def evaluate_instruction(
    instruction: str,
    test_data: List[Dict[str, Any]],
    client: LLMClient,
    max_tokens: int = 512,
) -> Tuple[float, int, int]:
    """Evaluate a single instruction on the test set.

    Args:
        instruction: The instruction/system prompt to evaluate
        test_data: List of test examples with 'question' and 'answer' keys
        client: LLM client for generation
        max_tokens: Maximum tokens for generation

    Returns:
        Tuple of (error_rate, correct_count, total_count)
    """
    # Build prompts: instruction + question (Q_end style from OPRO paper)
    prompts = []
    for item in test_data:
        prompt = f"Q: {item['question']}\n{instruction}\nA:"
        prompts.append(prompt)

    # Batch generate with deterministic sampling
    responses = client.generate_batch(
        prompts,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    # Evaluate responses
    correct = 0
    evaluated = 0
    skipped = 0

    for response, item in zip(responses, test_data):
        predicted = extract_answer(response)
        try:
            expected = extract_ground_truth(item["answer"])
        except ValueError:
            # Skip if we can't extract ground truth
            skipped += 1
            continue

        evaluated += 1
        if predicted is not None and compare_numbers(predicted, expected):
            correct += 1

    # Log warning if items were skipped
    if skipped > 0:
        print(
            f"  WARNING: Skipped {skipped}/{len(test_data)} items "
            f"(ground truth extraction failed)"
        )

    error_rate = 1.0 - (correct / evaluated) if evaluated > 0 else 1.0

    return error_rate, correct, evaluated
