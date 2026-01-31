"""Generation quality validation for EcoFlow flow models."""

import argparse
import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def compute_sample_statistics(samples: Tensor) -> dict[str, float]:
    """Compute statistics of generated samples (mean, std, L2 norms)."""
    l2_norms = torch.norm(samples, dim=1)
    dim_means = samples.mean(dim=0)
    dim_stds = samples.std(dim=0)

    return {
        "mean": samples.mean().item(),
        "std": samples.std().item(),
        "l2_norm_mean": l2_norms.mean().item(),
        "l2_norm_std": l2_norms.std().item(),
        "dim_mean_mean": dim_means.mean().item(),
        "dim_mean_std": dim_means.std().item(),
        "dim_std_mean": dim_stds.mean().item(),
        "dim_std_std": dim_stds.std().item(),
    }


def compute_diversity_metrics(samples: Tensor, n_pairs: int = 1000) -> dict[str, float]:
    """Compute pairwise cosine similarity to assess sample diversity."""
    n_samples = samples.shape[0]

    if n_samples < 2:
        return {"cosine_sim_mean": 0.0, "cosine_sim_std": 0.0, "cosine_sim_max": 0.0}

    normalized = samples / (torch.norm(samples, dim=1, keepdim=True) + 1e-8)

    if n_samples * (n_samples - 1) // 2 <= n_pairs:
        sim_matrix = torch.mm(normalized, normalized.t())
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        similarities = sim_matrix[mask]
    else:
        idx1 = torch.randint(0, n_samples, (n_pairs,))
        idx2 = torch.randint(0, n_samples, (n_pairs,))
        same_idx = idx1 == idx2
        idx2[same_idx] = (idx2[same_idx] + 1) % n_samples
        similarities = (normalized[idx1] * normalized[idx2]).sum(dim=1)

    return {
        "cosine_sim_mean": similarities.mean().item(),
        "cosine_sim_std": similarities.std().item(),
        "cosine_sim_max": similarities.max().item(),
    }


def compute_text_metrics(texts: list[str]) -> dict[str, float]:
    """Compute quality metrics for decoded texts (diversity, length, coherence)."""
    if not texts:
        return {
            "unique_token_ratio": 0.0,
            "avg_length": 0.0,
            "n_empty": 0,
            "n_coherent": 0,
            "coherent_ratio": 0.0,
            "total_texts": 0,
        }

    all_tokens = []
    for text in texts:
        all_tokens.extend(text.lower().split())

    unique_token_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    lengths = [len(text) for text in texts]
    avg_length = sum(lengths) / len(lengths)
    n_empty = sum(1 for text in texts if len(text) < 10)
    n_coherent = sum(1 for text in texts if len(text) > 20 and len(text.split()) > 2)

    return {
        "unique_token_ratio": unique_token_ratio,
        "avg_length": avg_length,
        "n_empty": n_empty,
        "n_coherent": n_coherent,
        "coherent_ratio": n_coherent / len(texts),
        "total_texts": len(texts),
    }


def validate_generation(
    model: Any,
    device: str,
    decoder: Any | None = None,
    n_samples: int = 100,
    num_steps: int = 50,
    method: str = "heun",
) -> dict[str, Any]:
    """Validate flow model by generating samples and computing quality metrics."""
    if hasattr(model, "eval"):
        model.eval()

    logger.info(f"Generating {n_samples} samples with {num_steps} {method} steps...")

    with torch.no_grad():
        samples = model.sample(n_samples, device=device, method=method, num_steps=num_steps)

    logger.info(f"Generated samples shape: {samples.shape}")

    stats = compute_sample_statistics(samples)
    logger.info(f"Sample mean: {stats['mean']:.6f}, std: {stats['std']:.6f}")
    logger.info(f"L2 norm: {stats['l2_norm_mean']:.4f} +/- {stats['l2_norm_std']:.4f}")

    diversity = compute_diversity_metrics(samples)
    logger.info(f"Cosine sim: {diversity['cosine_sim_mean']:.4f} (target: <0.5)")

    results = {
        "n_samples": n_samples,
        "num_steps": num_steps,
        "method": method,
        **stats,
        **diversity,
    }

    if decoder is not None:
        logger.info("Decoding samples to text...")
        texts = decoder.decode_batch(samples.to(device), batch_size=16)

        logger.info("=" * 60)
        logger.info("DECODED TEXTS")
        logger.info("=" * 60)
        for i, text in enumerate(texts):
            logger.info(f"[{i+1:3d}] {text}")
        logger.info("=" * 60)

        text_metrics = compute_text_metrics(texts)
        logger.info(f"Coherent: {text_metrics['n_coherent']}/{len(texts)} ({text_metrics['coherent_ratio']*100:.1f}%)")

        results["texts"] = texts
        results.update(text_metrics)

    return results


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda:0",
    use_ema: bool = True,
) -> Any:
    """Load FlowMatchingModel from checkpoint file."""
    from ecoflow.velocity_network import VelocityNetwork
    from ecoflow.flow_model import FlowMatchingModel

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = checkpoint.get("args", {})
    hidden_dim = args.get("hidden_dim", 512)
    num_layers = args.get("num_layers", 6)
    num_heads = args.get("num_heads", 8)

    logger.info(f"Model config: hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}")

    velocity_net = VelocityNetwork(
        input_dim=1024,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    if use_ema and "ema_shadow" in checkpoint:
        logger.info("Loading EMA weights")
        ema_shadow = checkpoint["ema_shadow"]
        state_dict = {}
        for name in velocity_net.state_dict():
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

    norm_stats = checkpoint.get("norm_stats")
    if norm_stats:
        logger.info("Loaded normalization stats")
    else:
        logger.warning("No normalization stats in checkpoint")

    model = FlowMatchingModel(velocity_net, norm_stats=norm_stats)

    best_loss = checkpoint.get("best_loss") or checkpoint.get("best_val_loss")
    loss_str = f"{best_loss:.6f}" if best_loss is not None else "N/A"
    logger.info(f"Loaded epoch {checkpoint.get('epoch', 'N/A')}, best loss: {loss_str}")

    return model


def main() -> None:
    """CLI entry point for validation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Validate flow model generation quality")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--num-steps", type=int, default=50, help="ODE integration steps")
    parser.add_argument("--method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--decode", action="store_true", default=True, help="Decode samples to text")
    parser.add_argument("--no-decode", action="store_false", dest="decode")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA weights")

    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint, device=args.device, use_ema=args.use_ema)

    decoder = None
    if args.decode:
        from ecoflow.decoder import SonarDecoder
        decoder = SonarDecoder(device=args.device)

    results = validate_generation(
        model=model,
        device=args.device,
        decoder=decoder,
        n_samples=args.n_samples,
        num_steps=args.num_steps,
        method=args.method,
    )

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {results['n_samples']}, Steps: {results['num_steps']} ({results['method']})")
    print(f"Sample mean: {results['mean']:.6f}, std: {results['std']:.6f}")
    print(f"L2 norm: {results['l2_norm_mean']:.4f}, Diversity: {results['cosine_sim_mean']:.4f}")
    if "coherent_ratio" in results:
        print(f"Coherent ratio: {results['coherent_ratio']*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
