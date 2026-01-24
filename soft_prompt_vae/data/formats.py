"""Format converters for different instruction-tuning dataset formats.

Supports:
- ShareGPT format (FineTome-100k, OpenHermes-2.5)
- Alpaca format (WizardLM, Magicoder)
- Messages format (no_robots)
"""

from typing import Dict, List, Optional, Iterator, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class InstructionResponse:
    """Normalized instruction-response pair."""

    instruction: str
    response: str
    source: str  # Original dataset name


def _extract_sharegpt(
    example: Dict[str, Any], source: str
) -> Optional[InstructionResponse]:
    """Extract instruction-response from ShareGPT format.

    ShareGPT format:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }
    """
    conversations = example.get("conversations", [])
    if len(conversations) < 2:
        return None

    # Find first human-gpt pair
    instruction = None
    response = None

    for i, turn in enumerate(conversations):
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))

        if role in ("human", "user") and instruction is None:
            instruction = value
        elif role in ("gpt", "assistant") and instruction is not None:
            response = value
            break

    if instruction and response:
        return InstructionResponse(
            instruction=instruction.strip(),
            response=response.strip(),
            source=source,
        )
    return None


def _extract_alpaca(
    example: Dict[str, Any], source: str
) -> Optional[InstructionResponse]:
    """Extract instruction-response from Alpaca format.

    Alpaca format:
    {
        "instruction": "...",
        "input": "...",  # optional
        "output": "..."
    }
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", example.get("response", ""))

    if not instruction or not output:
        return None

    # Combine instruction and input if present
    if input_text:
        full_instruction = f"{instruction}\n\nInput: {input_text}"
    else:
        full_instruction = instruction

    return InstructionResponse(
        instruction=full_instruction.strip(),
        response=output.strip(),
        source=source,
    )


def _extract_messages(
    example: Dict[str, Any], source: str
) -> Optional[InstructionResponse]:
    """Extract instruction-response from messages format.

    Messages format (used by no_robots):
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }
    """
    messages = example.get("messages", [])
    if len(messages) < 2:
        return None

    instruction = None
    response = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user" and instruction is None:
            instruction = content
        elif role == "assistant" and instruction is not None:
            response = content
            break

    if instruction and response:
        return InstructionResponse(
            instruction=instruction.strip(),
            response=response.strip(),
            source=source,
        )
    return None


def _extract_finetome(
    example: Dict[str, Any], source: str
) -> Optional[InstructionResponse]:
    """Extract from FineTome-100k specific format.

    FineTome uses ShareGPT but may have system messages.
    """
    conversations = example.get("conversations", [])
    if not conversations:
        return None

    # Skip system messages, find first user-assistant pair
    instruction = None
    response = None

    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))

        if role == "system":
            continue  # Skip system prompts
        elif role in ("human", "user") and instruction is None:
            instruction = value
        elif role in ("gpt", "assistant") and instruction is not None:
            response = value
            break

    if instruction and response:
        return InstructionResponse(
            instruction=instruction.strip(),
            response=response.strip(),
            source=source,
        )
    return None


def _extract_magicoder(
    example: Dict[str, Any], source: str
) -> Optional[InstructionResponse]:
    """Extract from Magicoder-OSS-Instruct format.

    Magicoder format:
    {
        "problem": "...",
        "solution": "..."
    }
    """
    problem = example.get("problem", example.get("instruction", ""))
    solution = example.get("solution", example.get("output", ""))

    if not problem or not solution:
        return None

    return InstructionResponse(
        instruction=problem.strip(),
        response=solution.strip(),
        source=source,
    )


# Dataset name to extractor mapping
EXTRACTORS = {
    "mlabonne/FineTome-100k": _extract_finetome,
    "teknium/OpenHermes-2.5": _extract_sharegpt,
    "WizardLMTeam/WizardLM_evol_instruct_V2_196k": _extract_alpaca,
    "HuggingFaceH4/no_robots": _extract_messages,
    "ise-uiuc/Magicoder-OSS-Instruct-75K": _extract_magicoder,
}


def detect_format(example: Dict[str, Any]) -> str:
    """Auto-detect format from example structure."""
    if "conversations" in example:
        return "sharegpt"
    elif "messages" in example:
        return "messages"
    elif "problem" in example and "solution" in example:
        return "magicoder"
    elif "instruction" in example:
        return "alpaca"
    else:
        return "unknown"


def convert_example(
    example: Dict[str, Any], dataset_name: str
) -> Optional[InstructionResponse]:
    """Convert a single example to InstructionResponse.

    Args:
        example: Raw example from dataset
        dataset_name: Name of the source dataset

    Returns:
        InstructionResponse or None if conversion fails
    """
    # Try dataset-specific extractor first
    if dataset_name in EXTRACTORS:
        return EXTRACTORS[dataset_name](example, dataset_name)

    # Fall back to format detection
    format_type = detect_format(example)

    if format_type == "sharegpt":
        return _extract_sharegpt(example, dataset_name)
    elif format_type == "messages":
        return _extract_messages(example, dataset_name)
    elif format_type == "magicoder":
        return _extract_magicoder(example, dataset_name)
    elif format_type == "alpaca":
        return _extract_alpaca(example, dataset_name)
    else:
        logger.warning(f"Unknown format for dataset {dataset_name}: {list(example.keys())}")
        return None


def convert_to_instruction_response(
    dataset: Iterator[Dict[str, Any]], dataset_name: str
) -> Iterator[InstructionResponse]:
    """Convert an entire dataset to InstructionResponse format.

    Args:
        dataset: Iterator over raw examples
        dataset_name: Name of the source dataset

    Yields:
        InstructionResponse objects
    """
    converted = 0
    failed = 0

    for example in dataset:
        result = convert_example(example, dataset_name)
        if result is not None:
            converted += 1
            yield result
        else:
            failed += 1

    total = converted + failed
    if total == 0:
        logger.warning(
            f"Dataset {dataset_name}: No examples processed. "
            f"Check if dataset is empty or format detection failed."
        )
    else:
        logger.info(
            f"Dataset {dataset_name}: converted {converted}, failed {failed} "
            f"({100 * converted / total:.1f}% success)"
        )
