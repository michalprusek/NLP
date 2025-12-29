"""CLI entry point for InvBO decoder inversion.

Usage:
    # Train and run full pipeline
    uv run python -m generation.invbo_decoder.run

    # With InvBO inversion loop (recommended)
    uv run python -m generation.invbo_decoder.run --use-inversion --method trust_region

    # With custom parameters
    uv run python -m generation.invbo_decoder.run --gp-epochs 500 --method random

    # Load pre-trained models
    uv run python -m generation.invbo_decoder.run --load generation/invbo_decoder/results/

    # Validate inversion gap
    uv run python -m generation.invbo_decoder.run --validate-gap
"""

import argparse
import torch
from pathlib import Path

from generation.invbo_decoder.training import InvBOTrainer, TrainingConfig
from generation.invbo_decoder.inference import InvBOInference


def main():
    parser = argparse.ArgumentParser(
        description="InvBO Decoder Inversion for Instruction Optimization"
    )

    # Data paths
    parser.add_argument(
        "--instructions",
        type=str,
        default="datasets/inversion/instructions_100.txt",
        help="Path to instructions file",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="datasets/inversion/grid_100_qend.jsonl",
        help="Path to grid JSONL file",
    )

    # Training parameters
    parser.add_argument(
        "--gp-epochs",
        type=int,
        default=3000,
        help="GP training epochs",
    )
    parser.add_argument(
        "--decoder-epochs",
        type=int,
        default=1000,
        help="Decoder training epochs",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=10,
        help="Latent space dimension",
    )

    # Loss parameters
    parser.add_argument(
        "--lambda-cycle",
        type=float,
        default=1.0,
        help="Cyclic loss weight",
    )
    parser.add_argument(
        "--lambda-embedding",
        type=float,
        default=0.5,
        help="Embedding cosine loss weight",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Soft tolerance for cyclic loss",
    )

    # VAE mode parameters
    parser.add_argument(
        "--use-vae",
        action="store_true",
        help="Use VAE mode (recommended for smooth latent space)",
    )
    parser.add_argument(
        "--vae-beta",
        type=float,
        default=0.1,
        help="VAE KL regularization weight",
    )
    parser.add_argument(
        "--vae-epochs",
        type=int,
        default=1000,
        help="VAE training epochs",
    )
    parser.add_argument(
        "--vae-annealing",
        type=int,
        default=500,
        help="KL annealing epochs (0 → beta)",
    )

    # Optimization parameters
    parser.add_argument(
        "--method",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "random", "trust_region"],
        help="Latent optimization method (trust_region recommended for stability)",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=10,
        help="L-BFGS restarts",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=1000,
        help="Random/trust-region sampling candidates",
    )
    parser.add_argument(
        "--trust-radius",
        type=float,
        default=0.3,
        help="Trust region perturbation radius (fraction of latent std)",
    )

    # InvBO inversion parameters
    parser.add_argument(
        "--use-inversion",
        action="store_true",
        help="Use InvBO-style inversion loop (recommended)",
    )
    parser.add_argument(
        "--max-inversion-iters",
        type=int,
        default=3,
        help="Maximum inversion iterations",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.5,
        help="Threshold for triggering re-inversion",
    )

    # Vec2Text parameters
    parser.add_argument(
        "--vec2text-steps",
        type=int,
        default=50,
        help="Vec2Text correction steps",
    )
    parser.add_argument(
        "--vec2text-beam",
        type=int,
        default=4,
        help="Vec2Text beam width",
    )

    # Save/load
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save trained models to directory",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load pre-trained models from directory",
    )

    # Validation
    parser.add_argument(
        "--validate-gap",
        action="store_true",
        help="Validate inversion gap on random samples",
    )
    parser.add_argument(
        "--gap-samples",
        type=int,
        default=10,
        help="Number of samples for gap validation",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("InvBO Decoder Inversion for Instruction Optimization")
    print("=" * 70)

    # Create config
    config = TrainingConfig(
        instructions_path=args.instructions,
        grid_path=args.grid,
        latent_dim=args.latent_dim,
        gp_epochs=args.gp_epochs,
        decoder_epochs=args.decoder_epochs,
        lambda_cycle=args.lambda_cycle,
        lambda_cosine=args.lambda_embedding,
        cycle_tolerance=args.tolerance,
        use_vae=args.use_vae,
        vae_beta=args.vae_beta,
        vae_epochs=args.vae_epochs,
        vae_annealing_epochs=args.vae_annealing,
        device=args.device,
    )

    # Initialize trainer
    trainer = InvBOTrainer(config)

    if args.load:
        # Load pre-trained models
        print(f"\nLoading models from {args.load}...")
        trainer.load(args.load)
    else:
        # Train from scratch
        print("\nStarting training...")
        gp, decoder = trainer.train(verbose=True)

        if args.save:
            trainer.save(args.save)

    # Create inference pipeline
    inference = InvBOInference(
        gp=trainer.gp,
        decoder=trainer.decoder,
        gtr=trainer.gtr,
        vec2text_steps=args.vec2text_steps,
        vec2text_beam=args.vec2text_beam,
    )

    # Validate inversion gap if requested
    if args.validate_gap:
        inference.validate_inversion_gap(n_samples=args.gap_samples, verbose=True)

    # Run optimization
    if args.use_inversion:
        print("\n[Using InvBO-style inversion loop]")
        result = inference.optimize_with_inversion(
            method=args.method,
            n_restarts=args.n_restarts,
            n_candidates=args.n_candidates,
            trust_radius=args.trust_radius,
            max_inversion_iters=args.max_inversion_iters,
            gap_threshold=args.gap_threshold,
            verbose=True,
        )
    else:
        result = inference.run_optimization(
            method=args.method,
            n_restarts=args.n_restarts,
            n_candidates=args.n_candidates,
            trust_radius=args.trust_radius,
            verbose=True,
        )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nNovel Instruction (generated by InvBO):")
    print(f"  {result.instruction_text}")
    print(f"\nMetrics:")
    print(f"  Predicted error rate: {result.predicted_error:.4f}")
    print(f"  Expected improvement: {result.ei_value:.6f}")
    print(f"  Vec2Text cosine similarity: {result.cosine_similarity:.4f}")
    print(f"  Best observed error: {trainer.gp.y_best:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
