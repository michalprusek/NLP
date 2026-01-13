#!/usr/bin/env python3
"""
Comprehensive LID-O++ Training Script.

This script runs the full training pipeline:
1. Load APE instructions
2. Train Latent Injection projector
3. Train FlowDiT with CFM loss
4. Evaluate round-trip quality
5. Save checkpoints

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m lido_pp.train_full \
        --projector-epochs 50 \
        --flow-epochs 1000 \
        --device cuda:0
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_ape_instructions(path: str, max_instructions: int = None) -> list:
    """Load APE instructions from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    instructions = data.get('instructions', data)
    if isinstance(instructions, dict):
        instructions = list(instructions.values())

    # Filter out non-English or weird instructions
    filtered = []
    for inst in instructions:
        if isinstance(inst, str) and len(inst) > 10 and len(inst) < 500:
            # Basic English check
            if inst[0].isascii() and '步' not in inst and '电' not in inst:
                filtered.append(inst)

    if max_instructions:
        filtered = filtered[:max_instructions]

    logger.info(f"Loaded {len(filtered)} instructions from {path}")
    return filtered


def train_projector(
    encoder,
    decoder,
    instructions: list,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    checkpoint_dir: str = "lido_pp/checkpoints",
):
    """Train the Latent Injection projector."""
    from lido_pp.backbone import ProjectorTrainer, RoundTripEvaluator

    logger.info(f"Training projector for {epochs} epochs on {len(instructions)} instructions")

    trainer = ProjectorTrainer(decoder, encoder, lr=lr)

    best_loss = float('inf')
    best_similarity = 0.0

    for epoch in range(epochs):
        metrics = trainer.train_epoch(instructions, batch_size=batch_size)

        loss = metrics['epoch_loss']
        acc = metrics['epoch_accuracy']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={loss:.4f}, acc={acc:.2%}")

        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(
                decoder.projector.state_dict(),
                os.path.join(checkpoint_dir, "projector_best.pt")
            )

        # Periodic evaluation
        if (epoch + 1) % 10 == 0:
            evaluator = RoundTripEvaluator(encoder, decoder)
            test_texts = random.sample(instructions, min(10, len(instructions)))
            results = evaluator.evaluate_batch(test_texts)
            sim = results['mean_cosine_similarity']
            logger.info(f"  Round-trip cosine similarity: {sim:.4f}")

            if sim > best_similarity:
                best_similarity = sim

    # Save final model
    torch.save(
        decoder.projector.state_dict(),
        os.path.join(checkpoint_dir, "projector_final.pt")
    )

    logger.info(f"Projector training complete. Best loss: {best_loss:.4f}, Best similarity: {best_similarity:.4f}")
    return best_loss, best_similarity


def train_flow_dit(
    flow_model,
    encoder,
    instructions: list,
    epochs: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-4,
    checkpoint_dir: str = "lido_pp/checkpoints",
    device: str = "cuda:0",
):
    """Train FlowDiT with Conditional Flow Matching."""
    from lido_pp.flow import conditional_flow_matching_loss

    logger.info(f"Training FlowDiT for {epochs} epochs")

    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Pre-encode all instructions
    logger.info("Pre-encoding instructions...")
    with torch.no_grad():
        all_embeddings = encoder.encode_embedding_batch(instructions, normalize=True)
    logger.info(f"Encoded {len(instructions)} instructions to shape {all_embeddings.shape}")

    best_loss = float('inf')
    losses = []

    for epoch in range(epochs):
        # Sample batch
        indices = torch.randperm(len(instructions))[:batch_size]
        context = all_embeddings[indices].to(device)

        # Expand context to 4 tokens (FlowDiT expects context_tokens dimension)
        context = context.unsqueeze(1).expand(-1, 4, -1)  # (B, 4, 768)

        # Sample noise and target
        x_0 = torch.randn(batch_size, 32, device=device)  # Noise
        x_1 = torch.randn(batch_size, 32, device=device)  # Target (random for now)

        # Forward
        optimizer.zero_grad()
        loss, metrics = conditional_flow_matching_loss(flow_model, x_0, x_1, context)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(
                flow_model.state_dict(),
                os.path.join(checkpoint_dir, "flow_dit_best.pt")
            )

        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = np.mean(losses[-100:])
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    # Save final model
    torch.save(
        flow_model.state_dict(),
        os.path.join(checkpoint_dir, "flow_dit_final.pt")
    )

    logger.info(f"FlowDiT training complete. Best loss: {best_loss:.4f}")
    return best_loss


def final_evaluation(encoder, decoder, flow_model, instructions: list, device: str):
    """Run final evaluation of the full pipeline."""
    from lido_pp.backbone import RoundTripEvaluator
    from lido_pp.flow import euler_integrate

    logger.info("Running final evaluation...")

    # Round-trip evaluation
    evaluator = RoundTripEvaluator(encoder, decoder)
    test_texts = random.sample(instructions, min(50, len(instructions)))
    results = evaluator.evaluate_batch(test_texts)

    logger.info(f"Final Results:")
    logger.info(f"  Mean cosine similarity: {results['mean_cosine_similarity']:.4f}")
    logger.info(f"  Std cosine similarity: {results['std_cosine_similarity']:.4f}")
    logger.info(f"  Exact match rate: {results['exact_match_rate']:.2%}")
    logger.info(f"  Mean length ratio: {results['mean_length_ratio']:.2f}")

    # Test FlowDiT integration
    logger.info("\nFlow integration test:")
    with torch.no_grad():
        embeddings = encoder.encode_embedding_batch(test_texts[:8], normalize=True)
        context = embeddings.unsqueeze(1).expand(-1, 4, -1).to(device)
        x_0 = torch.randn(8, 32, device=device)

        result = euler_integrate(flow_model, x_0, context, num_steps=20)
        logger.info(f"  Final x shape: {result.x_final.shape}")
        logger.info(f"  Mean curvature: {result.curvature.mean():.6f}")

    # Show example reconstructions
    logger.info("\nExample reconstructions:")
    for i, text in enumerate(test_texts[:5]):
        with torch.no_grad():
            latent = encoder.encode_tensor(text)
            decoded = decoder.decode_single(latent, max_new_tokens=80, temperature=0.7)
        logger.info(f"  Original:  {repr(text[:60])}...")
        logger.info(f"  Decoded:   {repr(decoded[:60])}...")
        logger.info("")

    return results


def main():
    parser = argparse.ArgumentParser(description="LID-O++ Comprehensive Training")
    parser.add_argument("--projector-epochs", type=int, default=50)
    parser.add_argument("--flow-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-instructions", type=int, default=500)
    parser.add_argument("--ape-path", type=str, default="lipo/data/ape_instructions.json")
    parser.add_argument("--checkpoint-dir", type=str, default="lido_pp/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("lido_pp/results", exist_ok=True)

    logger.info("=" * 60)
    logger.info("  LID-O++ Comprehensive Training")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Projector epochs: {args.projector_epochs}")
    logger.info(f"Flow epochs: {args.flow_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max instructions: {args.max_instructions}")

    # Load instructions
    instructions = load_ape_instructions(args.ape_path, args.max_instructions)

    # Initialize models
    logger.info("\n[1/5] Loading GritLM encoder...")
    from lido_pp.backbone import GritLMUnifiedEncoder, create_latent_injection_decoder

    encoder = GritLMUnifiedEncoder(
        model_name="GritLM/GritLM-7B",
        device=args.device,
        dtype="float16",
    )

    logger.info("\n[2/5] Creating Latent Injection decoder...")
    decoder = create_latent_injection_decoder(encoder, num_prefix_tokens=4)

    # Train projector
    logger.info("\n[3/5] Training projector...")
    start_time = time.time()
    proj_loss, proj_sim = train_projector(
        encoder, decoder, instructions,
        epochs=args.projector_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
    proj_time = time.time() - start_time
    logger.info(f"Projector training took {proj_time/60:.1f} minutes")

    # Train FlowDiT
    logger.info("\n[4/5] Training FlowDiT...")
    from lido_pp.flow import FlowDiT

    flow_model = FlowDiT(
        latent_dim=32,
        context_dim=encoder.output_dim,
        hidden_dim=512,
        num_layers=6,
    ).to(args.device)

    start_time = time.time()
    flow_loss = train_flow_dit(
        flow_model, encoder, instructions,
        epochs=args.flow_epochs,
        batch_size=32,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    flow_time = time.time() - start_time
    logger.info(f"FlowDiT training took {flow_time/60:.1f} minutes")

    # Final evaluation
    logger.info("\n[5/5] Final evaluation...")
    results = final_evaluation(encoder, decoder, flow_model, instructions, args.device)

    # Save results
    results_path = os.path.join(
        "lido_pp/results",
        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_path, 'w') as f:
        json.dump({
            "projector_loss": proj_loss,
            "projector_similarity": proj_sim,
            "flow_loss": flow_loss,
            "final_similarity": results['mean_cosine_similarity'],
            "projector_time_min": proj_time / 60,
            "flow_time_min": flow_time / 60,
            "config": vars(args),
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("=" * 60)
    logger.info("  Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
