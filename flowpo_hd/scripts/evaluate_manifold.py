#!/usr/bin/env python3
"""
Evaluate ManifoldKeeper quality.

Tests:
1. Sample generation: noise → ODE integrate → decode → valid English?
2. Reconstruction: instruction → SONAR → project to manifold → decode → similar?
3. Manifold velocity quality: does velocity at t=0.9 point towards valid text?

Usage:
    uv run python -m flowpo_hd.scripts.evaluate_manifold
    uv run python -m flowpo_hd.scripts.evaluate_manifold --checkpoint path/to/manifold_keeper.pt
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from flowpo_hd.config import FlowPOHDConfig, get_device
from flowpo_hd.manifold_keeper import ManifoldKeeperMLP, create_manifold_keeper
from flowpo_hd.utils import SONARHelper, set_seed, setup_logging, compute_embedding_stats

logger = logging.getLogger(__name__)


def is_valid_instruction(text: str) -> Tuple[bool, str]:
    """
    Check if text is a valid instruction.

    Criteria:
    - Not empty
    - Contains letters (not just symbols/numbers)
    - Reasonable length (10-500 chars)
    - No excessive repetition
    - Contains some English words

    Returns:
        (is_valid, reason)
    """
    # Empty check
    if not text or not text.strip():
        return False, "empty"

    text = text.strip()

    # Length check
    if len(text) < 10:
        return False, "too_short"
    if len(text) > 500:
        return False, "too_long"

    # Contains letters
    letters = sum(1 for c in text if c.isalpha())
    if letters < len(text) * 0.3:
        return False, "too_few_letters"

    # Repetition check (no word repeated >5 times)
    words = text.lower().split()
    if words:
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        max_count = max(word_counts.values())
        if max_count > 5 and max_count > len(words) * 0.3:
            return False, "repetitive"

    # Basic English words check
    common_words = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'and', 'or', 'of', 'in', 'on', 'with'}
    words_lower = set(w.lower().strip('.,!?') for w in words)
    if not words_lower.intersection(common_words):
        return False, "no_common_words"

    return True, "valid"


def evaluate_sample_quality(
    manifold_keeper: ManifoldKeeperMLP,
    sonar_helper: SONARHelper,
    n_samples: int = 50,
    num_ode_steps: int = 50,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate quality of generated samples.

    Process: noise → ODE integrate → SONAR decode → check validity

    Args:
        manifold_keeper: Trained model
        sonar_helper: SONAR helper
        n_samples: Number of samples to generate
        num_ode_steps: ODE integration steps
        device: Device

    Returns:
        Metrics dict
    """
    logger.info(f"Evaluating sample quality with {n_samples} samples...")
    manifold_keeper.eval()

    valid_count = 0
    invalid_reasons = {}
    samples_text = []
    embedding_norms = []

    with torch.no_grad():
        # Generate samples
        embeddings = manifold_keeper.sample(
            batch_size=n_samples,
            device=torch.device(device),
            num_steps=num_ode_steps,
        )

        # Compute norms
        norms = embeddings.norm(dim=-1)
        embedding_norms = norms.tolist()

        # Decode to text
        texts = sonar_helper.decode(embeddings)

        # Check validity
        for i, text in enumerate(texts):
            is_valid, reason = is_valid_instruction(text)
            if is_valid:
                valid_count += 1
            else:
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

            samples_text.append({
                "text": text[:200],
                "is_valid": is_valid,
                "reason": reason,
                "embedding_norm": norms[i].item(),
            })

    validity_rate = valid_count / n_samples

    logger.info(f"  Validity rate: {validity_rate:.2%} ({valid_count}/{n_samples})")
    logger.info(f"  Invalid reasons: {invalid_reasons}")
    logger.info(f"  Embedding norm: mean={sum(embedding_norms)/len(embedding_norms):.4f}")

    return {
        "validity_rate": validity_rate,
        "valid_count": valid_count,
        "total_samples": n_samples,
        "invalid_reasons": invalid_reasons,
        "mean_embedding_norm": sum(embedding_norms) / len(embedding_norms),
        "samples": samples_text[:10],  # Keep first 10 for inspection
    }


def evaluate_reconstruction_quality(
    manifold_keeper: ManifoldKeeperMLP,
    sonar_helper: SONARHelper,
    test_instructions: List[str],
    num_ode_steps: int = 50,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate reconstruction quality through manifold projection.

    Process: text → SONAR encode → project to manifold → decode → compare

    Args:
        manifold_keeper: Trained model
        sonar_helper: SONAR helper
        test_instructions: Instructions to test
        num_ode_steps: ODE steps
        device: Device

    Returns:
        Metrics dict
    """
    logger.info(f"Evaluating reconstruction quality with {len(test_instructions)} instructions...")
    manifold_keeper.eval()

    cosine_sims = []
    reconstructions = []

    with torch.no_grad():
        # Encode original
        original_embeddings = sonar_helper.encode(test_instructions)

        # Project to manifold (treat as t=0.5, integrate to t=1)
        projected = manifold_keeper.project_to_manifold(
            original_embeddings.to(device),
            num_steps=num_ode_steps,
            t_current=0.5,
        )

        # Compute cosine similarity before decode
        cos_sim_embed = F.cosine_similarity(
            original_embeddings.to(device),
            projected,
            dim=-1,
        )

        # Decode projected
        decoded = sonar_helper.decode(projected)

        # Re-encode decoded for text similarity
        decoded_embeddings = sonar_helper.encode(decoded)

        # Cosine similarity after roundtrip
        cos_sim_roundtrip = F.cosine_similarity(
            original_embeddings.to(device),
            decoded_embeddings.to(device),
            dim=-1,
        )

        for i, (orig, dec, sim_e, sim_r) in enumerate(
            zip(test_instructions, decoded, cos_sim_embed, cos_sim_roundtrip)
        ):
            cosine_sims.append({
                "embedding": sim_e.item(),
                "roundtrip": sim_r.item(),
            })
            reconstructions.append({
                "original": orig[:100],
                "decoded": dec[:100],
                "cos_sim_embedding": sim_e.item(),
                "cos_sim_roundtrip": sim_r.item(),
            })

    mean_embed_sim = sum(c["embedding"] for c in cosine_sims) / len(cosine_sims)
    mean_roundtrip_sim = sum(c["roundtrip"] for c in cosine_sims) / len(cosine_sims)

    logger.info(f"  Mean embedding cosine sim: {mean_embed_sim:.4f}")
    logger.info(f"  Mean roundtrip cosine sim: {mean_roundtrip_sim:.4f}")

    return {
        "mean_embedding_cosine_sim": mean_embed_sim,
        "mean_roundtrip_cosine_sim": mean_roundtrip_sim,
        "reconstructions": reconstructions[:5],  # Keep first 5
    }


def evaluate_manifold_velocity(
    manifold_keeper: ManifoldKeeperMLP,
    sonar_helper: SONARHelper,
    n_samples: int = 20,
    t_values: List[float] = [0.1, 0.5, 0.9],
    device: str = "cuda",
) -> Dict:
    """
    Evaluate manifold velocity quality at different timesteps.

    Tests whether velocity at different t values points towards valid text.

    Args:
        manifold_keeper: Trained model
        sonar_helper: SONAR helper
        n_samples: Samples per timestep
        t_values: Timesteps to test
        device: Device

    Returns:
        Metrics dict
    """
    logger.info(f"Evaluating manifold velocity at t={t_values}...")
    manifold_keeper.eval()

    results_by_t = {}

    with torch.no_grad():
        for t in t_values:
            # Start with random points (interpolated between noise and data)
            noise = torch.randn(n_samples, 1024, device=device) * 0.5
            data_approx = torch.randn(n_samples, 1024, device=device) * 0.2  # SONAR-like norm

            # Interpolate
            x_t = t * data_approx + (1 - t) * noise

            # Get velocity
            velocity = manifold_keeper.get_manifold_velocity(x_t, t=t)

            # Take one step with velocity
            step_size = 0.1
            x_stepped = x_t + step_size * velocity

            # Decode both
            texts_before = sonar_helper.decode(x_t)
            texts_after = sonar_helper.decode(x_stepped)

            # Check validity improvement
            valid_before = sum(1 for t in texts_before if is_valid_instruction(t)[0])
            valid_after = sum(1 for t in texts_after if is_valid_instruction(t)[0])

            results_by_t[str(t)] = {
                "valid_before": valid_before,
                "valid_after": valid_after,
                "improvement": valid_after - valid_before,
                "velocity_norm": velocity.norm(dim=-1).mean().item(),
            }

            logger.info(f"  t={t}: valid {valid_before} → {valid_after}, v_norm={velocity.norm(dim=-1).mean():.4f}")

    return {
        "timestep_results": results_by_t,
        "t_values": t_values,
        "n_samples": n_samples,
    }


def evaluate_ode_trajectory(
    manifold_keeper: ManifoldKeeperMLP,
    sonar_helper: SONARHelper,
    n_samples: int = 5,
    n_steps: int = 10,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate ODE trajectory from noise to data.

    Shows how validity evolves along the flow.

    Args:
        manifold_keeper: Trained model
        sonar_helper: SONAR helper
        n_samples: Number of trajectories
        n_steps: Checkpoints along trajectory
        device: Device

    Returns:
        Trajectory data
    """
    logger.info(f"Evaluating ODE trajectory with {n_samples} trajectories...")
    manifold_keeper.eval()

    trajectories = []
    t_checkpoints = torch.linspace(0, 1, n_steps)

    with torch.no_grad():
        # Start from noise
        x_0 = torch.randn(n_samples, 1024, device=device)

        for i, t_end in enumerate(t_checkpoints[1:], 1):
            t_start = t_checkpoints[i - 1].item()
            t_end_val = t_end.item()

            # Integrate from previous checkpoint
            if i == 1:
                x_current = x_0
            x_current = manifold_keeper.integrate(
                x_current,
                t_start=t_start,
                t_end=t_end_val,
                num_steps=5,
            )

            # Decode
            texts = sonar_helper.decode(x_current)

            # Check validity
            valid_count = sum(1 for t in texts if is_valid_instruction(t)[0])

            trajectories.append({
                "t": t_end_val,
                "valid_count": valid_count,
                "valid_rate": valid_count / n_samples,
                "example_text": texts[0][:100] if texts else "",
            })

            logger.info(f"  t={t_end_val:.2f}: {valid_count}/{n_samples} valid")

    return {
        "trajectories": trajectories,
        "n_samples": n_samples,
    }


def run_evaluation(
    config: FlowPOHDConfig,
    checkpoint_path: Optional[str] = None,
    n_samples: int = 50,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Run full ManifoldKeeper evaluation.

    Args:
        config: FlowPOHDConfig
        checkpoint_path: Path to ManifoldKeeper checkpoint
        n_samples: Samples per test
        output_path: Optional path to save results

    Returns:
        Full results dict
    """
    device = torch.device(config.device)
    set_seed(config.seed)

    # Initialize SONAR
    logger.info("Initializing SONAR...")
    sonar_helper = SONARHelper(
        device=config.device,
        normalize=config.sonar_normalize,
    )

    # Load ManifoldKeeper
    checkpoint_path = checkpoint_path or config.manifold_keeper_path
    logger.info(f"Loading ManifoldKeeper from {checkpoint_path}...")

    manifold_keeper = create_manifold_keeper(config).to(device)

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        manifold_keeper.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded trained ManifoldKeeper")
    else:
        logger.warning("No checkpoint found, using untrained model")

    manifold_keeper.eval()

    # Run evaluations
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "config": {
            "sonar_dim": config.sonar_dim,
            "mk_hidden_dim": config.mk_hidden_dim,
            "mk_num_blocks": config.mk_num_blocks,
        },
    }

    # 1. Sample quality
    logger.info("\n=== Sample Quality ===")
    results["sample_quality"] = evaluate_sample_quality(
        manifold_keeper, sonar_helper,
        n_samples=n_samples,
        device=config.device,
    )

    # 2. Reconstruction quality
    logger.info("\n=== Reconstruction Quality ===")
    test_instructions = [
        "Let's think step by step.",
        "Solve this problem carefully and show your work.",
        "Break down the problem into smaller steps.",
        "Think through this logically before answering.",
        "Consider all possibilities before giving your answer.",
    ]
    results["reconstruction_quality"] = evaluate_reconstruction_quality(
        manifold_keeper, sonar_helper,
        test_instructions=test_instructions,
        device=config.device,
    )

    # 3. Manifold velocity
    logger.info("\n=== Manifold Velocity ===")
    results["manifold_velocity"] = evaluate_manifold_velocity(
        manifold_keeper, sonar_helper,
        n_samples=20,
        device=config.device,
    )

    # 4. ODE trajectory
    logger.info("\n=== ODE Trajectory ===")
    results["ode_trajectory"] = evaluate_ode_trajectory(
        manifold_keeper, sonar_helper,
        n_samples=5,
        device=config.device,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Sample validity rate: {results['sample_quality']['validity_rate']:.2%}")
    logger.info(f"Reconstruction cosine sim: {results['reconstruction_quality']['mean_roundtrip_cosine_sim']:.4f}")
    logger.info("=" * 60)

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate ManifoldKeeper quality")

    parser.add_argument("--checkpoint", type=str,
                       default="flowpo_hd/checkpoints/manifold_keeper.pt",
                       help="Path to ManifoldKeeper checkpoint")
    parser.add_argument("--n-samples", type=int, default=50,
                       help="Number of samples per test")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Create config
    config = FlowPOHDConfig(
        device=get_device(args.device),
        seed=args.seed,
        manifold_keeper_path=args.checkpoint,
    )

    # Default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"flowpo_hd/results/manifold_eval_{timestamp}.json"

    # Run evaluation
    results = run_evaluation(
        config=config,
        checkpoint_path=args.checkpoint,
        n_samples=args.n_samples,
        output_path=args.output,
    )

    # Print key metrics
    print("\n" + "=" * 40)
    print("KEY METRICS")
    print("=" * 40)
    print(f"Sample validity: {results['sample_quality']['validity_rate']:.2%}")
    print(f"Reconstruction: {results['reconstruction_quality']['mean_roundtrip_cosine_sim']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
