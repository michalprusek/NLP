"""Data quality filters for instruction-response pairs.

Implements:
- Length filtering (token-based)
- Language detection (FastText)
- Quality filtering (control chars, repetition)
"""

import re
import logging
from typing import Iterator, Optional, Callable, List
from dataclasses import dataclass
from pathlib import Path

from soft_prompt_vae.data.formats import InstructionResponse
from soft_prompt_vae.config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Statistics for filtering operations."""

    total: int = 0
    passed: int = 0
    failed_length: int = 0
    failed_language: int = 0
    failed_quality: int = 0

    def __str__(self) -> str:
        if self.total == 0:
            return (
                "No samples processed - possible causes: "
                "(1) empty input dataset, "
                "(2) upstream loading failure, "
                "(3) all samples filtered by previous stage"
            )
        pass_rate = 100 * self.passed / self.total
        return (
            f"Total: {self.total}, Passed: {self.passed} ({pass_rate:.1f}%), "
            f"Failed length: {self.failed_length}, "
            f"Failed language: {self.failed_language}, "
            f"Failed quality: {self.failed_quality}"
        )


class LengthFilter:
    """Filter based on token counts."""

    def __init__(
        self,
        tokenizer,
        min_instruction: int = 10,
        max_instruction: int = 512,
        min_response: int = 20,
        max_response: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.min_instruction = min_instruction
        self.max_instruction = max_instruction
        self.min_response = min_response
        self.max_response = max_response

    def __call__(self, item: InstructionResponse) -> bool:
        """Check if item passes length filter."""
        # Tokenize without special tokens for accurate count
        instr_tokens = len(
            self.tokenizer.encode(item.instruction, add_special_tokens=False)
        )
        resp_tokens = len(
            self.tokenizer.encode(item.response, add_special_tokens=False)
        )

        return (
            self.min_instruction <= instr_tokens <= self.max_instruction
            and self.min_response <= resp_tokens <= self.max_response
        )


class LanguageFilter:
    """Filter based on language detection using FastText."""

    def __init__(
        self,
        model_path: Path,
        language: str = "en",
        threshold: float = 0.8,
    ):
        self.language = language
        self.threshold = threshold
        self._model = None
        self._model_path = model_path

    @property
    def model(self):
        """Lazy load FastText model."""
        if self._model is None:
            try:
                import fasttext

                # Suppress FastText warnings (non-critical)
                fasttext.FastText.eprint = lambda x: None
                self._model = fasttext.load_model(str(self._model_path))
                logger.info(f"Loaded FastText model from {self._model_path}")
            except ImportError as e:
                logger.error(
                    f"FastText not installed: {e}. "
                    f"Install with: pip install fasttext-wheel"
                )
                logger.warning(
                    "Language filtering DISABLED - non-English samples may be included"
                )
                self._model = "disabled"
            except FileNotFoundError:
                logger.error(
                    f"FastText model not found at {self._model_path}. "
                    f"Download with: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
                )
                logger.warning(
                    "Language filtering DISABLED - non-English samples may be included"
                )
                self._model = "disabled"
            except Exception as e:
                logger.error(f"Failed to load FastText model: {e}")
                logger.warning(
                    "Language filtering DISABLED - non-English samples may be included"
                )
                self._model = "disabled"
        return self._model

    def __call__(self, item: InstructionResponse) -> bool:
        """Check if item passes language filter."""
        if self.model == "disabled":
            return True

        # Combine instruction and response for language detection
        text = f"{item.instruction} {item.response}"
        # Remove newlines (FastText doesn't handle them well)
        text = text.replace("\n", " ").strip()

        if not text:
            return False

        try:
            predictions = self.model.predict(text, k=1)
            label = predictions[0][0].replace("__label__", "")
            confidence = predictions[1][0]

            return label == self.language and confidence >= self.threshold
        except (ValueError, RuntimeError) as e:
            # FastText can fail on unusual inputs; log at WARNING level and pass through
            logger.warning(
                f"FastText prediction failed (text len={len(text)}): {e}. "
                f"Sample will be INCLUDED in training."
            )
            return True


class QualityFilter:
    """Filter based on text quality heuristics."""

    # Control characters (except newline, tab)
    CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    # Repeated characters (10+ of same char)
    REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{9,}")

    # Repeated words (5+ of same word)
    REPEATED_WORD_PATTERN = re.compile(r"\b(\w+)(?:\s+\1){4,}\b", re.IGNORECASE)

    # Too many special characters (>30% non-alphanumeric)
    SPECIAL_CHAR_THRESHOLD = 0.3

    def __call__(self, item: InstructionResponse) -> bool:
        """Check if item passes quality filter."""
        for text in [item.instruction, item.response]:
            if self.CONTROL_CHAR_PATTERN.search(text):
                return False

            if self.REPEATED_CHAR_PATTERN.search(text):
                return False

            if self.REPEATED_WORD_PATTERN.search(text):
                return False

            # Check special character ratio (reject if >30% special chars)
            if text:
                normal_char_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
                if normal_char_ratio < (1 - self.SPECIAL_CHAR_THRESHOLD):
                    return False

        return True


class CompositeFilter:
    """Combine multiple filters with statistics tracking."""

    def __init__(
        self,
        filters: List[tuple],  # List of (name, filter_fn) tuples
    ):
        self.filters = filters
        self.stats = FilterStats()

    def __call__(self, item: InstructionResponse) -> bool:
        """Apply all filters and track statistics."""
        self.stats.total += 1

        for name, filter_fn in self.filters:
            if not filter_fn(item):
                if name == "length":
                    self.stats.failed_length += 1
                elif name == "language":
                    self.stats.failed_language += 1
                elif name == "quality":
                    self.stats.failed_quality += 1
                return False

        self.stats.passed += 1
        return True

    def get_stats(self) -> FilterStats:
        """Get filtering statistics."""
        return self.stats


def create_filter_pipeline(
    tokenizer,
    config: DataConfig,
) -> CompositeFilter:
    """Create a complete filter pipeline from config.

    Args:
        tokenizer: HuggingFace tokenizer for length filtering
        config: Data configuration

    Returns:
        CompositeFilter ready to apply
    """
    filters = []

    # Length filter
    length_filter = LengthFilter(
        tokenizer=tokenizer,
        min_instruction=config.min_instruction_length,
        max_instruction=config.max_instruction_length,
        min_response=config.min_response_length,
        max_response=config.max_response_length,
    )
    filters.append(("length", length_filter))

    # Language filter (optional, requires FastText model)
    if config.fasttext_model.exists():
        lang_filter = LanguageFilter(
            model_path=config.fasttext_model,
            language=config.language,
            threshold=config.language_threshold,
        )
        filters.append(("language", lang_filter))
    else:
        logger.warning(
            f"FastText model not found at {config.fasttext_model}, "
            "skipping language filtering"
        )

    # Quality filter
    quality_filter = QualityFilter()
    filters.append(("quality", quality_filter))

    return CompositeFilter(filters)


def apply_filters(
    data: Iterator[InstructionResponse],
    filter_pipeline: CompositeFilter,
) -> Iterator[InstructionResponse]:
    """Apply filter pipeline to data stream.

    Args:
        data: Iterator of InstructionResponse
        filter_pipeline: CompositeFilter to apply

    Yields:
        Filtered InstructionResponse items
    """
    for item in data:
        if filter_pipeline(item):
            yield item
