"""
Generation quality validation utilities for EcoFlow.

Provides functions to compute sample statistics, diversity metrics,
and text quality metrics for validating flow model outputs.
"""

import argparse
import logging
from typing import Dict, List, Optional, Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def compute_sample_statistics(samples: Tensor) -> Dict[str, float]:
    """
    Compute statistics of generated samples.

    Compares against target SONAR distribution:
    - Mean should be ~0.0
    - Std should be ~0.01
    - L2 norm should be ~0.32

    Args:
        samples: Tensor of shape [N, D] containing generated samples

    Returns:
        Dictionary with mean, std, l2_norm_mean, l2_norm_std
    """
    # Flatten to compute overall statistics
    sample_mean = samples.mean().item()
    sample_std = samples.std().item()

    # Compute L2 norms
    l2_norms = torch.norm(samples, dim=1)
    l2_norm_mean = l2_norms.mean().item()
    l2_norm_std = l2_norms.std().item()

    # Per-dimension statistics
    dim_means = samples.mean(dim=0)
    dim_stds = samples.std(dim=0)

    stats = {
        "mean": sample_mean,
        "std": sample_std,
        "l2_norm_mean": l2_norm_mean,
        "l2_norm_std": l2_norm_std,
        "dim_mean_mean": dim_means.mean().item(),
        "dim_mean_std": dim_means.std().item(),
        "dim_std_mean": dim_stds.mean().item(),
        "dim_std_std": dim_stds.std().item(),
    }

    return stats


def compute_diversity_metrics(samples: Tensor, n_pairs: int = 1000) -> Dict[str, float]:
    """
    Compute diversity metrics for generated samples.

    Measures pairwise cosine similarity to assess diversity.
    Target: mean cosine similarity < 0.5 indicates good diversity.

    Args:
        samples: Tensor of shape [N, D] containing generated samples
        n_pairs: Number of random pairs to sample for similarity (default: 1000)

    Returns:
        Dictionary with cosine_sim_mean, cosine_sim_std, cosine_sim_max
    """
    n_samples = samples.shape[0]

    if n_samples < 2:
        return {
            "cosine_sim_mean": 0.0,
            "cosine_sim_std": 0.0,
            "cosine_sim_max": 0.0,
        }

    # Normalize samples for cosine similarity
    normalized = samples / (torch.norm(samples, dim=1, keepdim=True) + 1e-8)

    # Sample random pairs
    if n_samples * (n_samples - 1) // 2 <= n_pairs:
        # Compute all pairwise similarities
        sim_matrix = torch.mm(normalized, normalized.t())
        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        similarities = sim_matrix[mask]
    else:
        # Sample random pairs
        idx1 = torch.randint(0, n_samples, (n_pairs,))
        idx2 = torch.randint(0, n_samples, (n_pairs,))
        # Ensure different indices
        same_idx = idx1 == idx2
        idx2[same_idx] = (idx2[same_idx] + 1) % n_samples

        similarities = (normalized[idx1] * normalized[idx2]).sum(dim=1)

    return {
        "cosine_sim_mean": similarities.mean().item(),
        "cosine_sim_std": similarities.std().item(),
        "cosine_sim_max": similarities.max().item(),
    }


def compute_text_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Compute quality metrics for decoded texts.

    Measures:
    - unique_token_ratio: Vocabulary diversity
    - avg_length: Mean character length
    - n_empty: Count of empty/short texts
    - n_coherent: Count of likely coherent texts

    Args:
        texts: List of decoded text strings

    Returns:
        Dictionary with text quality metrics
    """
    if not texts:
        return {
            "unique_token_ratio": 0.0,
            "avg_length": 0.0,
            "n_empty": 0,
            "n_coherent": 0,
            "coherent_ratio": 0.0,
        }

    # Token analysis
    all_tokens = []
    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)

    if all_tokens:
        unique_token_ratio = len(set(all_tokens)) / len(all_tokens)
    else:
        unique_token_ratio = 0.0

    # Length analysis
    lengths = [len(text) for text in texts]
    avg_length = sum(lengths) / len(lengths) if lengths else 0.0

    # Empty/short text count (less than 10 characters)
    n_empty = sum(1 for text in texts if len(text) < 10)

    # Coherent text estimate (>20 chars and multiple words)
    n_coherent = sum(
        1 for text in texts if len(text) > 20 and len(text.split()) > 2
    )

    coherent_ratio = n_coherent / len(texts) if texts else 0.0

    return {
        "unique_token_ratio": unique_token_ratio,
        "avg_length": avg_length,
        "n_empty": n_empty,
        "n_coherent": n_coherent,
        "coherent_ratio": coherent_ratio,
        "total_texts": len(texts),
    }


def validate_generation(
    model: Any,
    device: str,
    decoder: Optional[Any] = None,
    n_samples: int = 100,
    num_steps: int = 50,
    method: str = "euler",
) -> Dict[str, Any]:
    """
    Comprehensive validation of flow model generation quality.

    Generates samples, computes statistics, and optionally decodes to text.

    Args:
        model: FlowMatchingModel instance
        device: Device for computation
        decoder: Optional SonarDecoder instance for text decoding
        n_samples: Number of samples to generate (default: 100)
        num_steps: Number of ODE integration steps (default: 50)
        method: Integration method ('euler' or 'heun')

    Returns:
        Dictionary with all computed metrics
    """
    from src.ecoflow.flow_model import FlowMatchingModel

    # Ensure model is in eval mode
    if hasattr(model, "eval"):
        model.eval()

    logger.info(f"Generating {n_samples} samples with {num_steps} {method} steps...")

    # Generate samples
    with torch.no_grad():
        if method == "heun":
            samples = model.sample_heun(n_samples, device=device, num_steps=num_steps)
        else:
            samples = model.sample(n_samples, device=device, method=method, num_steps=num_steps)

    logger.info(f"Generated samples shape: {samples.shape}")

    # Compute sample statistics
    logger.info("Computing sample statistics...")
    stats = compute_sample_statistics(samples)
    logger.info(f"Sample mean: {stats['mean']:.6f} (target: ~0.0)")
    logger.info(f"Sample std: {stats['std']:.6f} (target: ~0.01)")
    logger.info(f"L2 norm mean: {stats['l2_norm_mean']:.4f} (target: ~0.32)")
    logger.info(f"L2 norm std: {stats['l2_norm_std']:.4f}")

    # Compute diversity metrics
    logger.info("Computing diversity metrics...")
    diversity = compute_diversity_metrics(samples)
    logger.info(f"Cosine sim mean: {diversity['cosine_sim_mean']:.4f} (target: <0.5)")
    logger.info(f"Cosine sim std: {diversity['cosine_sim_std']:.4f}")
    logger.info(f"Cosine sim max: {diversity['cosine_sim_max']:.4f}")

    # Compile results
    results = {
        "n_samples": n_samples,
        "num_steps": num_steps,
        "method": method,
        **stats,
        **diversity,
    }

    # Decode to text if decoder provided
    if decoder is not None:
        logger.info("Decoding samples to text...")
        texts = decoder.decode_batch(samples.to(device), batch_size=16)

        # Log ALL decoded texts (NEVER truncate per CLAUDE.md)
        logger.info("=" * 60)
        logger.info("DECODED TEXTS (FULL OUTPUT)")
        logger.info("=" * 60)
        for i, text in enumerate(texts):
            logger.info(f"[{i+1:3d}] {text}")
        logger.info("=" * 60)

        # Compute text metrics
        text_metrics = compute_text_metrics(texts)
        logger.info(f"Unique token ratio: {text_metrics['unique_token_ratio']:.4f} (target: >0.5)")
        logger.info(f"Average length: {text_metrics['avg_length']:.1f} chars (target: >20)")
        logger.info(f"Empty/short texts: {text_metrics['n_empty']}/{len(texts)}")
        logger.info(f"Coherent texts: {text_metrics['n_coherent']}/{len(texts)} ({text_metrics['coherent_ratio']*100:.1f}%)")

        results["texts"] = texts
        results.update(text_metrics)

    # Print summary
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Samples generated: {n_samples}")
    logger.info(f"Sample mean: {results['mean']:.6f}")
    logger.info(f"Sample std: {results['std']:.6f}")
    logger.info(f"L2 norm: {results['l2_norm_mean']:.4f} +/- {results['l2_norm_std']:.4f}")
    logger.info(f"Diversity (cosine sim): {results['cosine_sim_mean']:.4f}")
    if "coherent_ratio" in results:
        logger.info(f"Coherent text ratio: {results['coherent_ratio']*100:.1f}%")
    logger.info("=" * 60)

    return results


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda:0",
    use_ema: bool = True,
) -> Any:
    """
    Load FlowMatchingModel from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        use_ema: Whether to use EMA weights (default: True)

    Returns:
        FlowMatchingModel instance
    """
    from src.ecoflow.velocity_network import VelocityNetwork
    from src.ecoflow.flow_model import FlowMatchingModel

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    args = checkpoint.get("args", {})
    hidden_dim = args.get("hidden_dim", 512)
    num_layers = args.get("num_layers", 6)
    num_heads = args.get("num_heads", 8)

    logger.info(f"Model config: hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}")

    # Create model
    velocity_net = VelocityNetwork(
        input_dim=1024,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # Load weights
    if use_ema and "ema_shadow" in checkpoint:
        logger.info("Loading EMA weights")
        # EMA shadow is a dict mapping param names to values
        ema_shadow = checkpoint["ema_shadow"]
        state_dict = {}
        for name, param in velocity_net.named_parameters():
            if name in ema_shadow:
                state_dict[name] = ema_shadow[name]
            else:
                logger.warning(f"Parameter {name} not found in EMA shadow")
        velocity_net.load_state_dict(state_dict, strict=False)
    else:
        logger.info("Loading model state dict")
        velocity_net.load_state_dict(checkpoint["model_state_dict"])

    velocity_net = velocity_net.to(device)
    velocity_net.eval()

    # Wrap in FlowMatchingModel
    model = FlowMatchingModel(velocity_net)

    logger.info(f"Model loaded successfully. Epoch: {checkpoint.get('epoch', 'N/A')}, Best loss: {checkpoint.get('best_loss', 'N/A'):.6f}")

    return model


def main():
    """Main entry point for standalone validation."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Validate flow model generation quality"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of ODE integration steps (default: 50)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="euler",
        choices=["euler", "heun"],
        help="Integration method (default: euler)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        default=True,
        help="Decode samples to text (default: True)",
    )
    parser.add_argument(
        "--no-decode",
        action="store_false",
        dest="decode",
        help="Skip text decoding",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use EMA weights (default: True)",
    )

    args = parser.parse_args()

    # Load model
    model = load_model_from_checkpoint(
        args.checkpoint,
        device=args.device,
        use_ema=args.use_ema,
    )

    # Create decoder if needed
    decoder = None
    if args.decode:
        from src.ecoflow.decoder import SonarDecoder
        decoder = SonarDecoder(device=args.device)

    # Run validation
    results = validate_generation(
        model=model,
        device=args.device,
        decoder=decoder,
        n_samples=args.n_samples,
        num_steps=args.num_steps,
        method=args.method,
    )

    # Final report
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {results['n_samples']}")
    print(f"Steps: {results['num_steps']} ({results['method']})")
    print(f"Sample mean: {results['mean']:.6f}")
    print(f"Sample std: {results['std']:.6f}")
    print(f"L2 norm: {results['l2_norm_mean']:.4f}")
    print(f"Diversity: {results['cosine_sim_mean']:.4f}")
    if "coherent_ratio" in results:
        print(f"Coherent ratio: {results['coherent_ratio']*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
