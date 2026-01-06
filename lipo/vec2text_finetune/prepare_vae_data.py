"""Prepare training data from VAE decoder outputs.

The key insight: Vec2Text needs to learn to invert VAE decoder outputs,
not just raw GTR embeddings. VAE decoder outputs have a different distribution.

Pipeline:
    instruction → GTR(768D) → VAE_encode → z(32D) → VAE_decode → embedding(768D)

We create (embedding, instruction) pairs where embedding comes from VAE decode.

Usage:
    uv run python -m lipo.vec2text_finetune.prepare_vae_data
"""

import json
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_vae_and_encoder(vae_path: str, device: str = "cuda"):
    """Load trained VAE and GTR encoder."""
    from lipo.encoder import InstructionVAE, GTRInstructionEncoder

    logger.info(f"Loading VAE from {vae_path}...")
    checkpoint = torch.load(vae_path, map_location="cpu", weights_only=True)

    # Get dimensions from checkpoint
    latent_dim = checkpoint["config"]["latent_dim"]
    embedding_dim = checkpoint["config"]["embedding_dim"]

    vae = InstructionVAE(
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae = vae.to(device).eval()

    logger.info(f"  VAE loaded: {embedding_dim}D → {latent_dim}D → {embedding_dim}D")

    logger.info("Loading GTR encoder...")
    gtr = GTRInstructionEncoder(device=device)

    return vae, gtr


def encode_through_vae(
    instructions: List[str],
    vae,
    gtr,
    device: str = "cuda",
    batch_size: int = 64,
) -> List[Dict]:
    """Encode instructions through VAE pipeline.

    instruction → GTR → VAE_encode → z → VAE_decode → embedding

    Returns list of {text, embedding, original_embedding, cosine_sim}
    """
    results = []

    for i in tqdm(range(0, len(instructions), batch_size), desc="Processing"):
        batch = instructions[i:i + batch_size]

        # GTR encode
        with torch.no_grad():
            original_embeddings = gtr.encode(batch)  # [B, 768]
            original_embeddings = original_embeddings.to(device)

            # VAE encode → decode
            z, _, _ = vae.encode(original_embeddings)
            reconstructed = vae.decode(z)  # [B, 768]

            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                original_embeddings, reconstructed, dim=1
            )

        # Store results
        for j, text in enumerate(batch):
            results.append({
                "text": text,
                "embedding": reconstructed[j].cpu().tolist(),
                "original_embedding": original_embeddings[j].cpu().tolist(),
                "cosine_sim": cos_sim[j].item(),
            })

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", type=str, required=True,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--instructions-path", type=str,
                        default="lipo/data/ape_instructions.json",
                        help="Path to instructions JSON")
    parser.add_argument("--output-dir", type=str,
                        default="lipo/vec2text_finetune/data_vae",
                        help="Output directory")
    parser.add_argument("--eval-ratio", type=float, default=0.1,
                        help="Fraction for eval set")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load instructions
    logger.info(f"Loading instructions from {args.instructions_path}...")
    with open(args.instructions_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        instructions = data.get("instructions", [])
    else:
        instructions = data

    logger.info(f"  Loaded {len(instructions)} instructions")

    # Load VAE and encoder
    vae, gtr = load_vae_and_encoder(args.vae_path, args.device)

    # Process through VAE
    logger.info("Encoding instructions through VAE pipeline...")
    results = encode_through_vae(
        instructions, vae, gtr, args.device, args.batch_size
    )

    # Stats
    cosine_sims = [r["cosine_sim"] for r in results]
    mean_sim = sum(cosine_sims) / len(cosine_sims)
    min_sim = min(cosine_sims)
    max_sim = max(cosine_sims)

    logger.info(f"VAE reconstruction quality:")
    logger.info(f"  Mean cosine: {mean_sim:.4f}")
    logger.info(f"  Min: {min_sim:.4f}, Max: {max_sim:.4f}")

    # Split train/eval
    n_eval = int(len(results) * args.eval_ratio)

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(results)

    eval_data = results[:n_eval]
    train_data = results[n_eval:]

    # Save in format compatible with train_inverter.py
    train_output = [{"text": r["text"], "embedding": r["embedding"]} for r in train_data]
    eval_output = [{"text": r["text"], "embedding": r["embedding"]} for r in eval_data]

    train_path = output_dir / "vae_train.json"
    eval_path = output_dir / "vae_eval.json"

    with open(train_path, "w") as f:
        json.dump(train_output, f)

    with open(eval_path, "w") as f:
        json.dump(eval_output, f)

    logger.info(f"Saved {len(train_output)} train examples to {train_path}")
    logger.info(f"Saved {len(eval_output)} eval examples to {eval_path}")

    # Also save full data with diagnostics
    full_path = output_dir / "vae_full.json"
    with open(full_path, "w") as f:
        json.dump({
            "stats": {
                "mean_cosine": mean_sim,
                "min_cosine": min_sim,
                "max_cosine": max_sim,
                "total": len(results),
            },
            "data": results,
        }, f)
    logger.info(f"Saved full diagnostics to {full_path}")


if __name__ == "__main__":
    main()
