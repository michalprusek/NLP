#!/usr/bin/env python3
"""
Test TFA reconstruction quality with SONAR encode/decode.

Pipeline:
    Text → SONAR encode(1024D) → TFA encode(256D) → TFA decode(1024D) → SONAR decode → Text

Measures BLEU score between original and reconstructed text.
"""

import torch
import argparse
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test instructions (diverse examples)
TEST_INSTRUCTIONS = [
    "Think step by step and show your reasoning.",
    "Solve this math problem by breaking it down into smaller parts.",
    "Let's approach this systematically. First, identify the key variables.",
    "Please provide a detailed explanation with examples.",
    "Calculate the answer and verify your work.",
    "Use logical reasoning to solve this problem.",
    "Explain your thought process as you work through this.",
    "Break down the problem into manageable steps.",
    "Show all intermediate calculations clearly.",
    "Analyze the problem carefully before solving.",
]


def load_tfa_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load TFA model from checkpoint."""
    from lido_pp.backbone.cfm_encoder import TextFlowAutoencoder

    logger.info(f"Loading TFA from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    args = ckpt.get("args", {})
    input_dim = ckpt.get("input_dim", 1024)
    latent_dim = ckpt.get("latent_dim", args.get("latent_dim", 256))
    flow_dim = args.get("flow_dim", 512)

    # Get z-score stats
    zscore_mean = ckpt.get("zscore_mean", None)
    zscore_std = ckpt.get("zscore_std", None)

    logger.info(f"  Input dim: {input_dim}, Latent dim: {latent_dim}, Flow dim: {flow_dim}")
    logger.info(f"  Z-score stats: {'available' if zscore_mean is not None else 'not available'}")
    logger.info(f"  Val CosODE: {ckpt.get('val_cos_ode', 'N/A'):.4f}")

    # Create model
    model = TextFlowAutoencoder(
        input_dim=input_dim,
        flow_dim=flow_dim,
        latent_dim=latent_dim,
        num_ode_steps=20,
        num_train_ode_steps=20,
        num_velocity_layers=args.get("velocity_layers", 6),
        normalize_output=False,  # No L2 norm for SONAR decoder
        zscore_mean=zscore_mean,
        zscore_std=zscore_std,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


def load_sonar(device: str = "cuda:0"):
    """Load SONAR encoder and decoder."""
    from sonar.inference_pipelines.text import (
        TextToEmbeddingModelPipeline,
        EmbeddingToTextModelPipeline,
    )

    logger.info("Loading SONAR encoder...")
    torch_device = torch.device(device)
    encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch_device,
    )

    logger.info("Loading SONAR decoder...")
    decoder = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_decoder",
        device=torch_device,
    )

    return encoder, decoder


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU score."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        score = bleu.sentence_score(hypothesis, [reference])
        return score.score / 100.0  # Normalize to 0-1
    except ImportError:
        # Fallback: simple word overlap
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        if len(ref_words) == 0:
            return 0.0
        overlap = len(ref_words & hyp_words)
        return overlap / len(ref_words)


def test_reconstruction(
    tfa_model,
    sonar_encoder,
    sonar_decoder,
    instructions: List[str],
    device: str = "cuda:0",
) -> Tuple[List[dict], float]:
    """
    Test reconstruction pipeline.

    Returns:
        List of results for each instruction and average BLEU score.
    """
    results = []

    with torch.no_grad():
        for i, text in enumerate(instructions):
            logger.info(f"\n[{i+1}/{len(instructions)}] Original: {text}")

            # Step 1: SONAR encode (text → 1024D)
            sonar_emb = sonar_encoder.predict([text], source_lang="eng_Latn")
            sonar_emb = sonar_emb.to(device)  # (1, 1024)

            # Step 2: Z-score normalize (if TFA was trained with z-score data)
            if tfa_model.has_zscore_stats():
                sonar_emb_norm = tfa_model.normalize_zscore(sonar_emb)
            else:
                sonar_emb_norm = sonar_emb

            # Step 3: TFA encode (1024D → 256D)
            latent = tfa_model.encode(sonar_emb_norm)  # (1, 256)

            # Step 4: TFA decode (256D → 1024D) with z-score denormalization
            recon_emb = tfa_model.decode(latent, denormalize_zscore=True)  # (1, 1024)

            # Step 5: Compute cosine similarity (embedding quality)
            cos_sim = torch.nn.functional.cosine_similarity(
                sonar_emb, recon_emb, dim=-1
            ).item()

            # Step 6: SONAR decode (1024D → text)
            recon_text = sonar_decoder.predict(
                recon_emb.cpu(),
                target_lang="eng_Latn",
                max_seq_len=128,
            )[0]

            # Step 7: Compute BLEU
            bleu = compute_bleu(text, recon_text)

            logger.info(f"  Reconstructed: {recon_text}")
            logger.info(f"  Cosine sim: {cos_sim:.4f}, BLEU: {bleu:.4f}")

            results.append({
                "original": text,
                "reconstructed": recon_text,
                "cosine_sim": cos_sim,
                "bleu": bleu,
                "latent_norm": latent.norm().item(),
            })

    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    avg_cos = sum(r["cosine_sim"] for r in results) / len(results)

    return results, avg_bleu, avg_cos


def main():
    parser = argparse.ArgumentParser(description="Test TFA reconstruction quality")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="lido_pp/checkpoints_zscore/tfa_best.pt",
        help="Path to TFA checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for computation",
    )
    parser.add_argument(
        "--custom-text",
        type=str,
        nargs="+",
        help="Custom text(s) to test instead of defaults",
    )
    args = parser.parse_args()

    # Load models
    tfa_model = load_tfa_model(args.checkpoint, args.device)
    sonar_encoder, sonar_decoder = load_sonar(args.device)

    # Get test instructions
    instructions = args.custom_text if args.custom_text else TEST_INSTRUCTIONS

    # Run reconstruction test
    print("\n" + "=" * 70)
    print("TFA Reconstruction Test")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {len(instructions)}")
    print("=" * 70)

    results, avg_bleu, avg_cos = test_reconstruction(
        tfa_model, sonar_encoder, sonar_decoder, instructions, args.device
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average Cosine Similarity: {avg_cos:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print("-" * 70)

    # Detailed results table
    print("\nDetailed Results:")
    print("-" * 70)
    for i, r in enumerate(results):
        print(f"[{i+1}] CosSim: {r['cosine_sim']:.4f} | BLEU: {r['bleu']:.4f}")
        print(f"    Original:      {r['original'][:60]}...")
        print(f"    Reconstructed: {r['reconstructed'][:60]}...")
        print()


if __name__ == "__main__":
    main()
