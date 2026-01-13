#!/usr/bin/env python
"""
Full Round-Trip Test for LID-O++ Encoder/Decoder.

This script tests the complete encode-decode pipeline:
    Text → GritLM Encoder → Latent (768D) → Latent Injection → Text'

It measures:
1. Semantic similarity (cosine of embeddings)
2. Reconstruction quality (before and after projector training)
3. Memory usage

Usage:
    # Quick test (untrained projector)
    uv run python lido_pp/test_roundtrip.py

    # With projector training
    uv run python lido_pp/test_roundtrip.py --train-epochs 10

    # Custom model
    uv run python lido_pp/test_roundtrip.py --model GritLM/GritLM-7B
"""

import argparse
import torch
import time
import gc
from typing import List, Dict


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def test_roundtrip(
    model_name: str = "GritLM/GritLM-7B",
    device: str = "cuda:0",
    train_epochs: int = 0,
    num_prefix_tokens: int = 4,
):
    """Run full round-trip test."""

    print_section("Loading GritLM Encoder")

    from lido_pp.backbone import (
        GritLMUnifiedEncoder,
        create_latent_injection_decoder,
        ProjectorTrainer,
        RoundTripEvaluator,
    )

    # Load encoder
    start_time = time.time()
    encoder = GritLMUnifiedEncoder(
        model_name=model_name,
        output_dim=768,
        use_latent_attention=True,
        device=device,
        dtype="float16",
    )
    load_time = time.time() - start_time
    print(f"  Encoder loaded in {load_time:.1f}s")
    print_memory()

    print_section("Creating Latent Injection Decoder")

    decoder = create_latent_injection_decoder(
        encoder,
        num_prefix_tokens=num_prefix_tokens,
    )
    print(f"  Projector parameters: {sum(p.numel() for p in decoder.projector.parameters()):,}")
    print_memory()

    # Test instructions
    test_instructions = [
        "Let's think step by step.",
        "Solve this math problem carefully and show your work.",
        "Break down the problem into smaller parts.",
        "Calculate the answer and verify it.",
        "Think about each step before proceeding.",
    ]

    print_section("Testing Encoding")

    # Test encoding
    for text in test_instructions[:3]:
        latent = encoder.encode_tensor(text)
        print(f"  '{text[:40]}...'")
        print(f"    → Latent shape: {latent.shape}, norm: {latent.norm().item():.4f}")

    print_section("Testing Decoding (Untrained Projector)")

    # Test decoding with untrained projector
    print("  Note: Without training, reconstruction will be poor")
    for text in test_instructions[:3]:
        latent = encoder.encode_tensor(text)
        try:
            reconstructed = decoder.decode_single(
                latent,
                max_new_tokens=64,
                temperature=0.7,
                do_sample=True,
            )
            print(f"  Original:     '{text}'")
            print(f"  Reconstructed: '{reconstructed[:80]}'")
            print()
        except Exception as e:
            print(f"  Decoding error: {e}")
            print()

    # Train projector if requested
    if train_epochs > 0:
        print_section(f"Training Projector ({train_epochs} epochs)")

        # Use more instructions for training
        train_instructions = test_instructions * 20  # Repeat for more data

        trainer = ProjectorTrainer(
            decoder=decoder,
            encoder=encoder,
            lr=1e-4,
            max_length=64,
        )

        for epoch in range(train_epochs):
            metrics = trainer.train_epoch(train_instructions, batch_size=4)
            print(f"  Epoch {epoch+1}: loss={metrics['epoch_loss']:.4f}, "
                  f"acc={metrics['epoch_accuracy']:.4f}, "
                  f"ppl={metrics['epoch_perplexity']:.2f}")

        print_section("Testing Decoding (Trained Projector)")

        for text in test_instructions[:3]:
            latent = encoder.encode_tensor(text)
            reconstructed = decoder.decode_single(
                latent,
                max_new_tokens=64,
                temperature=0.7,
                do_sample=True,
            )
            print(f"  Original:     '{text}'")
            print(f"  Reconstructed: '{reconstructed[:80]}'")
            print()

    print_section("Round-Trip Evaluation")

    evaluator = RoundTripEvaluator(encoder, decoder)

    print("  Evaluating semantic similarity...")
    results = evaluator.evaluate_batch(test_instructions)

    print(f"  Mean cosine similarity: {results['mean_cosine_similarity']:.4f}")
    print(f"  Std cosine similarity:  {results['std_cosine_similarity']:.4f}")
    print(f"  Exact match rate:       {results['exact_match_rate']:.4f}")
    print(f"  Mean length ratio:      {results['mean_length_ratio']:.2f}")

    print_section("Final Memory Usage")
    print_memory()

    # Cleanup
    del encoder, decoder
    gc.collect()
    torch.cuda.empty_cache()

    print("\n  [OK] Round-trip test completed!")


def main():
    parser = argparse.ArgumentParser(description="Test LID-O++ round-trip encoding/decoding")
    parser.add_argument("--model", default="GritLM/GritLM-7B", help="GritLM model name")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--train-epochs", type=int, default=0, help="Projector training epochs (0=skip)")
    parser.add_argument("--num-prefix-tokens", type=int, default=4, help="Number of prefix tokens")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  LID-O++ Round-Trip Test")
    print("="*60)
    print(f"\n  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Prefix tokens: {args.num_prefix_tokens}")
    print(f"  Training epochs: {args.train_epochs}")

    if not torch.cuda.is_available() and "cuda" in args.device:
        print("\n  [ERROR] CUDA not available!")
        return

    test_roundtrip(
        model_name=args.model,
        device=args.device,
        train_epochs=args.train_epochs,
        num_prefix_tokens=args.num_prefix_tokens,
    )


if __name__ == "__main__":
    main()
