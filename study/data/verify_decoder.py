"""Round-trip verification for SONAR decoder.

Verifies that embeddings can be decoded to text and re-encoded with high fidelity.
This ensures the normalization pipeline preserves semantic content.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ecoflow.decoder import SonarDecoder
from study.data.normalize import denormalize, load_stats

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TEST_PATH = "study/datasets/splits/10k/test.pt"
DEFAULT_STATS_PATH = "study/datasets/normalization_stats.pt"
DEFAULT_FAILURES_PATH = "study/datasets/verification_failures.json"


class SonarEncoder:
    """SONAR encoder for text to embedding conversion."""

    def __init__(self, device: str = "cuda:0"):
        """
        Initialize the SONAR encoder.

        Args:
            device: Device to run encoder on
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        logger.info(f"Initializing SONAR encoder on device: {self.device}")

        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        self.encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
        )
        logger.info("SONAR encoder initialized successfully")

    def encode(self, texts: list[str]) -> Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: List of strings to encode

        Returns:
            Tensor of shape [N, 1024] with embeddings
        """
        embeddings = self.encoder.predict(texts, source_lang="eng_Latn")
        return embeddings

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> Tensor:
        """
        Encode texts in batches for memory efficiency.

        Args:
            texts: List of strings to encode
            batch_size: Number of texts to encode at once

        Returns:
            Tensor of shape [N, 1024] with embeddings
        """
        all_embeddings = []
        n_samples = len(texts)

        for i in range(0, n_samples, batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


def compute_cosine_similarity(
    original: Tensor,
    recovered: Tensor,
) -> Tensor:
    """
    Compute pairwise cosine similarity between original and recovered embeddings.

    Args:
        original: Original embeddings [N, D]
        recovered: Re-encoded embeddings [N, D]

    Returns:
        Tensor of shape [N] with cosine similarities
    """
    # Ensure same device
    if original.device != recovered.device:
        recovered = recovered.to(original.device)

    # Normalize and compute cosine similarity
    similarities = cosine_similarity(original, recovered, dim=1)
    return similarities


def verify_round_trip(
    embeddings: Tensor,
    texts: list[str],
    stats: Optional[dict] = None,
    threshold: float = 0.9,
    sample_size: int = 500,
    batch_size: int = 32,
    device: str = "cuda:0",
    normalized_input: bool = False,
) -> dict:
    """
    Verify SONAR decoder round-trip fidelity.

    Pipeline:
    1. Take original embedding
    2. Denormalize if needed
    3. Decode to text via SONAR decoder
    4. Re-encode decoded text via SONAR encoder
    5. Compute cosine similarity between original and re-encoded

    Args:
        embeddings: Original embeddings [N, 1024]
        texts: Original instruction texts (for logging failures)
        stats: Normalization stats (for denormalization if normalized_input=True)
        threshold: Cosine similarity threshold for pass/fail
        sample_size: Number of samples to test
        batch_size: Batch size for encoding/decoding
        device: Device for SONAR models
        normalized_input: Whether input embeddings are normalized

    Returns:
        Dictionary with:
            - mean_similarity: Mean cosine similarity
            - pass_rate: Fraction of samples passing threshold
            - threshold: Threshold used
            - n_tested: Number of samples tested
            - failures: List of failure cases with details
    """
    n_total = embeddings.shape[0]
    n_test = min(sample_size, n_total)

    logger.info(f"Testing {n_test} samples from {n_total} total")

    # Random sample indices (deterministic)
    torch.manual_seed(42)
    indices = torch.randperm(n_total)[:n_test].tolist()

    # Get test samples
    test_embeddings = embeddings[indices]
    test_texts = [texts[i] for i in indices]

    # Denormalize if needed
    if normalized_input and stats is not None:
        logger.info("Denormalizing embeddings before decoding...")
        test_embeddings_for_decode = denormalize(test_embeddings, stats)
    else:
        test_embeddings_for_decode = test_embeddings

    # Initialize SONAR models
    logger.info("Initializing SONAR decoder and encoder...")
    decoder = SonarDecoder(device=device, ngram_block_size=3)
    encoder = SonarEncoder(device=device)

    # Decode embeddings to text
    logger.info(f"Decoding {n_test} embeddings to text...")
    decoded_texts = decoder.decode_batch(
        test_embeddings_for_decode,
        batch_size=batch_size,
        max_seq_len=256,
        beam_size=5,
    )

    # Re-encode decoded texts
    logger.info(f"Re-encoding {n_test} decoded texts...")
    reencoded_embeddings = encoder.encode_batch(decoded_texts, batch_size=batch_size)

    # Compute cosine similarity
    # Compare original embeddings (before any normalization) with re-encoded
    similarities = compute_cosine_similarity(
        test_embeddings_for_decode.cpu(),
        reencoded_embeddings.cpu(),
    )

    # Calculate metrics
    mean_similarity = similarities.mean().item()
    pass_mask = similarities >= threshold
    pass_rate = pass_mask.float().mean().item()

    # Collect failures
    fail_indices = torch.where(~pass_mask)[0].tolist()
    failures = []
    for i in fail_indices:
        global_idx = indices[i]
        failures.append({
            "index": global_idx,
            "local_index": i,
            "similarity": similarities[i].item(),
            "original_text": test_texts[i],
            "decoded_text": decoded_texts[i],
        })

    logger.info(f"Mean cosine similarity: {mean_similarity:.4f}")
    logger.info(f"Pass rate (>= {threshold}): {pass_rate:.2%} ({int(pass_rate * n_test)}/{n_test})")
    logger.info(f"Failures: {len(failures)}")

    return {
        "mean_similarity": mean_similarity,
        "pass_rate": pass_rate,
        "threshold": threshold,
        "n_tested": n_test,
        "failures": failures,
    }


def save_failures(results: dict, path: str) -> None:
    """
    Save failure cases to JSON for manual review.

    Args:
        results: Results dictionary from verify_round_trip
        path: Output path for JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "mean_similarity": results["mean_similarity"],
        "pass_rate": results["pass_rate"],
        "threshold": results["threshold"],
        "n_tested": results["n_tested"],
        "n_failures": len(results["failures"]),
        "failures": results["failures"],
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved failure details to {path}")


def main():
    """Run round-trip verification."""
    parser = argparse.ArgumentParser(description="Verify SONAR decoder round-trip fidelity")
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH, help="Path to test split")
    parser.add_argument("--stats-path", default=DEFAULT_STATS_PATH, help="Path to normalization stats")
    parser.add_argument("--failures-path", default=DEFAULT_FAILURES_PATH, help="Output path for failures JSON")
    parser.add_argument("--threshold", type=float, default=0.9, help="Cosine similarity threshold")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of samples to test")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding/decoding")
    parser.add_argument("--device", default="cuda:0", help="Device for SONAR models")
    parser.add_argument("--normalized", action="store_true", help="Input embeddings are normalized")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load test data
    logger.info(f"Loading test data from {args.test_path}")
    test_data = torch.load(args.test_path, weights_only=False)
    embeddings = test_data["embeddings"]
    texts = test_data["instructions"]

    logger.info(f"Loaded {embeddings.shape[0]} test embeddings")

    # Load stats if needed
    stats = None
    if args.normalized:
        logger.info(f"Loading normalization stats from {args.stats_path}")
        stats = load_stats(args.stats_path)

    # Run verification
    results = verify_round_trip(
        embeddings=embeddings,
        texts=texts,
        stats=stats,
        threshold=args.threshold,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        device=args.device,
        normalized_input=args.normalized,
    )

    # Save failures if any
    if results["failures"]:
        save_failures(results, args.failures_path)
    else:
        logger.info("No failures to save")

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Samples tested: {results['n_tested']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Mean cosine similarity: {results['mean_similarity']:.4f}")
    print(f"Pass rate: {results['pass_rate']:.2%}")
    print(f"Failures: {len(results['failures'])}")

    # Check pass criteria
    if results["pass_rate"] >= 0.95:
        print("\n[PASS] Pipeline passes verification (>= 95% pass rate)")
        return 0
    else:
        print(f"\n[FAIL] Pipeline fails verification ({results['pass_rate']:.2%} < 95%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
