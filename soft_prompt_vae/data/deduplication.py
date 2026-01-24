"""Deduplication using exact hash and MinHash LSH.

Implements:
- Exact hash deduplication (MD5)
- MinHash LSH for near-duplicate detection
"""

import hashlib
import logging
from typing import Iterator, Set, List
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from soft_prompt_vae.data.formats import InstructionResponse
from soft_prompt_vae.config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationStats:
    """Statistics for deduplication."""

    total: int = 0
    unique: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0

    def __str__(self) -> str:
        if self.total == 0:
            return "No samples processed"
        unique_rate = 100 * self.unique / self.total
        return (
            f"Total: {self.total}, Unique: {self.unique} ({unique_rate:.1f}%), "
            f"Exact duplicates: {self.exact_duplicates}, "
            f"Near duplicates: {self.near_duplicates}"
        )


def _compute_hash(text: str) -> str:
    """Compute MD5 hash of text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _compute_minhash(text: str, num_perm: int = 128, ngram: int = 5) -> MinHash:
    """Compute MinHash signature for text.

    Args:
        text: Input text
        num_perm: Number of permutations (hash functions)
        ngram: N-gram size for shingling

    Returns:
        MinHash object
    """
    minhash = MinHash(num_perm=num_perm)

    # Create character n-grams (shingles)
    text = text.lower()
    for i in range(len(text) - ngram + 1):
        shingle = text[i : i + ngram]
        minhash.update(shingle.encode("utf-8"))

    return minhash


class Deduplicator:
    """Deduplication using exact hash and MinHash LSH."""

    def __init__(
        self,
        threshold: float = 0.85,
        num_perm: int = 128,
        ngram: int = 5,
    ):
        """Initialize deduplicator.

        Args:
            threshold: Jaccard similarity threshold for near-duplicates
            num_perm: Number of permutations for MinHash
            ngram: N-gram size for shingling
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram = ngram

        # Exact hash set
        self._exact_hashes: Set[str] = set()

        # MinHash LSH index
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._minhash_count = 0

        # Statistics
        self.stats = DeduplicationStats()

    def _get_combined_text(self, item: InstructionResponse) -> str:
        """Get combined text for hashing."""
        return f"{item.instruction}\n{item.response}"

    def is_duplicate(self, item: InstructionResponse) -> bool:
        """Check if item is a duplicate.

        Args:
            item: InstructionResponse to check

        Returns:
            True if duplicate, False if unique
        """
        self.stats.total += 1
        text = self._get_combined_text(item)

        # Check exact hash
        exact_hash = _compute_hash(text)
        if exact_hash in self._exact_hashes:
            self.stats.exact_duplicates += 1
            return True

        # Check MinHash LSH for near-duplicates
        minhash = _compute_minhash(text, self.num_perm, self.ngram)
        similar = self._lsh.query(minhash)
        if similar:
            self.stats.near_duplicates += 1
            return True

        # Add to indices
        self._exact_hashes.add(exact_hash)
        self._lsh.insert(f"doc_{self._minhash_count}", minhash)
        self._minhash_count += 1

        self.stats.unique += 1
        return False

    def get_stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        return self.stats

    def reset(self) -> None:
        """Reset deduplicator state."""
        self._exact_hashes.clear()
        self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._minhash_count = 0
        self.stats = DeduplicationStats()


def create_deduplicator(config: DataConfig) -> Deduplicator:
    """Create deduplicator from config.

    Args:
        config: Data configuration

    Returns:
        Configured Deduplicator
    """
    return Deduplicator(
        threshold=config.minhash_threshold,
        num_perm=config.minhash_num_perm,
        ngram=config.minhash_ngram,
    )


def deduplicate(
    data: Iterator[InstructionResponse],
    deduplicator: Deduplicator,
) -> Iterator[InstructionResponse]:
    """Remove duplicates from data stream.

    Args:
        data: Iterator of InstructionResponse
        deduplicator: Deduplicator instance

    Yields:
        Unique InstructionResponse items
    """
    for item in data:
        if not deduplicator.is_duplicate(item):
            yield item


def deduplicate_batch(
    items: List[InstructionResponse],
    config: DataConfig,
) -> List[InstructionResponse]:
    """Deduplicate a batch of items.

    Args:
        items: List of InstructionResponse
        config: Data configuration

    Returns:
        List of unique items
    """
    deduplicator = create_deduplicator(config)
    unique_items = list(deduplicate(iter(items), deduplicator))

    logger.info(f"Deduplication: {deduplicator.get_stats()}")
    return unique_items
