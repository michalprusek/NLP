#!/usr/bin/env python
"""
LID-O++ Command Line Interface.

Entry point for training and inference with LID-O++.

Usage:
    # Full training pipeline
    uv run python -m lido_pp.run train --epochs 1000

    # Train projector only
    uv run python -m lido_pp.run train-projector --epochs 100

    # Run Reflow on trained model
    uv run python -m lido_pp.run reflow --load-flow checkpoints/flow_best.pt

    # Test round-trip
    uv run python -m lido_pp.run test-roundtrip --train-epochs 10

    # Analyze FCU distribution
    uv run python -m lido_pp.run analyze-fcu --load-flow checkpoints/flow_best.pt
"""

import argparse
import logging
import sys
import torch
from pathlib import Path

from lido_pp.config import LIDOPPConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_train(args):
    """Train FlowDiT model."""
    from lido_pp.backbone import GritLMUnifiedEncoder
    from lido_pp.flow import FlowDiT
    from lido_pp.training import (
        LIDOPPTrainer,
        InstructionDataset,
        FlowMatchingDataLoader,
        load_ape_instructions,
    )

    config = LIDOPPConfig(
        device=args.device,
        flow_epochs=args.epochs,
        flow_batch_size=args.batch_size,
        flow_lr=args.lr,
        oat_weight=args.oat_weight,
    )

    logging.info("Loading GritLM encoder...")
    encoder = GritLMUnifiedEncoder(
        model_name=config.gritlm_model,
        device=config.device,
        dtype="float16",
    )

    # Load instructions
    logging.info("Loading instructions...")
    instructions = load_ape_instructions()
    if not instructions:
        logging.warning("No APE instructions found, using synthetic data")
        instructions = [f"Instruction {i}" for i in range(100)]

    dataset = InstructionDataset(instructions)
    dataloader = FlowMatchingDataLoader(
        dataset=dataset,
        encoder=encoder,
        batch_size=config.flow_batch_size,
        device=config.device,
    )

    # Create trainer
    trainer = LIDOPPTrainer(config, encoder=encoder)

    # Train
    logging.info(f"Starting training for {args.epochs} epochs...")
    results = trainer.train_flow(
        train_dataloader=dataloader,
        epochs=args.epochs,
        use_oat=args.oat_weight > 0,
        eval_interval=args.eval_interval,
    )

    logging.info(f"Training complete. Final loss: {results['final_loss']:.6f}")

    # Run Reflow if requested
    if args.reflow:
        logging.info("Running Reflow...")
        context_source = encoder.encode_embedding_batch(instructions[:1000])
        reflow_results = trainer.run_reflow(context_source=context_source)
        logging.info(f"Reflow complete. 1-step error: {reflow_results['one_step_quality']['l2_error']:.6f}")


def cmd_train_projector(args):
    """Train latent injection projector."""
    from lido_pp.backbone import GritLMUnifiedEncoder, create_latent_injection_decoder, ProjectorTrainer
    from lido_pp.training import load_ape_instructions

    logging.info("Loading GritLM encoder...")
    encoder = GritLMUnifiedEncoder(
        model_name=args.model,
        device=args.device,
        dtype="float16",
    )

    logging.info("Creating decoder...")
    decoder = create_latent_injection_decoder(
        encoder,
        num_prefix_tokens=args.num_prefix_tokens,
    )

    # Load instructions
    instructions = load_ape_instructions()
    if not instructions:
        instructions = [
            "Let's think step by step.",
            "Solve this problem carefully.",
            "Show your work.",
        ] * 100

    logging.info(f"Training projector on {len(instructions)} instructions for {args.epochs} epochs...")
    trainer = ProjectorTrainer(decoder, encoder, lr=args.lr)

    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(instructions, batch_size=args.batch_size)
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}: loss={metrics['epoch_loss']:.4f}, acc={metrics['epoch_accuracy']:.4f}")

    # Save projector
    output_path = Path(args.output_dir) / "projector.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.projector.state_dict(), output_path)
    logging.info(f"Saved projector to {output_path}")


def cmd_test_roundtrip(args):
    """Test encode-decode round-trip."""
    from lido_pp.backbone import GritLMUnifiedEncoder, create_latent_injection_decoder, ProjectorTrainer, RoundTripEvaluator

    logging.info("Loading GritLM encoder...")
    encoder = GritLMUnifiedEncoder(
        model_name=args.model,
        device=args.device,
        dtype="float16",
    )

    logging.info("Creating decoder...")
    decoder = create_latent_injection_decoder(encoder, num_prefix_tokens=4)

    # Test instructions
    test_instructions = [
        "Let's think step by step.",
        "Solve this math problem carefully and show your work.",
        "Break down the problem into smaller parts.",
        "Calculate the answer and verify it.",
        "Think about each step before proceeding.",
    ]

    # Train projector if requested
    if args.train_epochs > 0:
        logging.info(f"Training projector for {args.train_epochs} epochs...")
        trainer = ProjectorTrainer(decoder, encoder, lr=1e-4)
        train_instructions = test_instructions * 50

        for epoch in range(args.train_epochs):
            metrics = trainer.train_epoch(train_instructions, batch_size=4)
            if (epoch + 1) % 5 == 0:
                logging.info(f"Epoch {epoch+1}: loss={metrics['epoch_loss']:.4f}")

    # Evaluate
    logging.info("Evaluating round-trip quality...")
    evaluator = RoundTripEvaluator(encoder, decoder)
    results = evaluator.evaluate_batch(test_instructions)

    logging.info(f"\nResults:")
    logging.info(f"  Mean cosine similarity: {results['mean_cosine_similarity']:.4f}")
    logging.info(f"  Exact match rate: {results['exact_match_rate']:.4f}")
    logging.info(f"  Mean length ratio: {results['mean_length_ratio']:.2f}")

    # Show examples
    logging.info("\nExamples:")
    for text in test_instructions[:3]:
        latent = encoder.encode_tensor(text)
        reconstructed = decoder.decode_single(latent, max_new_tokens=64, temperature=0.7)
        logging.info(f"  Original: '{text}'")
        logging.info(f"  Decoded:  '{reconstructed[:80]}'")
        logging.info("")


def cmd_analyze_fcu(args):
    """Analyze FCU distribution."""
    from lido_pp.flow import FlowDiT
    from lido_pp.active_learning import batch_fcu_analysis

    logging.info("Loading FlowDiT model...")

    if args.load_flow:
        state = torch.load(args.load_flow, map_location=args.device)
        # Infer dimensions from state dict
        latent_dim = state["model_state_dict"]["input_proj.weight"].shape[1]
        hidden_dim = state["model_state_dict"]["input_proj.weight"].shape[0]

        model = FlowDiT(latent_dim=latent_dim, hidden_dim=hidden_dim)
        model.load_state_dict(state["model_state_dict"])
    else:
        model = FlowDiT(latent_dim=768, hidden_dim=512, num_layers=6)

    model = model.to(args.device)

    # Generate test data
    logging.info("Analyzing FCU on random samples...")
    z = torch.randn(args.num_samples, model.latent_dim, device=args.device)
    context = torch.randn(args.num_samples, 4, 768, device=args.device)

    stats = batch_fcu_analysis(model, z, context, num_steps=20)

    logging.info("\nFCU Statistics:")
    for key, value in stats.items():
        logging.info(f"  {key}: {value:.6f}")


def cmd_reflow(args):
    """Run Reflow on trained model."""
    from lido_pp.flow import FlowDiT, ReflowTrainer, ReflowConfig, verify_one_step_inference

    logging.info(f"Loading FlowDiT from {args.load_flow}...")
    state = torch.load(args.load_flow, map_location=args.device)

    # Infer dimensions
    latent_dim = state["model_state_dict"]["input_proj.weight"].shape[1]
    hidden_dim = state["model_state_dict"]["input_proj.weight"].shape[0]

    model = FlowDiT(latent_dim=latent_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(args.device)

    # Reflow config
    config = ReflowConfig(
        epochs_per_round=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_pairs=args.num_pairs,
    )

    logging.info(f"Running Reflow for {args.epochs} epochs...")
    trainer = ReflowTrainer(model, config, args.device)

    context_source = torch.randn(args.num_pairs, 4, 768)
    result = trainer.train(context_source=context_source, latent_dim=latent_dim)

    # Verify
    x_0 = torch.randn(10, latent_dim, device=args.device)
    quality = verify_one_step_inference(model, x_0, reference_steps=50)

    logging.info(f"\nReflow complete:")
    logging.info(f"  Final straightness: {result.final_straightness}")
    logging.info(f"  1-step L2 error: {quality['l2_error']:.6f}")
    logging.info(f"  1-step cosine sim: {quality['cosine_similarity']:.4f}")

    # Save
    output_path = Path(args.output_dir) / "flow_reflowed.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, output_path)
    logging.info(f"Saved reflowed model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LID-O++ Training and Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train FlowDiT model")
    train_parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--oat-weight", type=float, default=0.1, help="OAT regularization weight")
    train_parser.add_argument("--device", default="cuda:0", help="Device")
    train_parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation interval")
    train_parser.add_argument("--reflow", action="store_true", help="Run Reflow after training")

    # Train projector command
    proj_parser = subparsers.add_parser("train-projector", help="Train latent injection projector")
    proj_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    proj_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    proj_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    proj_parser.add_argument("--model", default="GritLM/GritLM-7B", help="GritLM model")
    proj_parser.add_argument("--device", default="cuda:0", help="Device")
    proj_parser.add_argument("--num-prefix-tokens", type=int, default=4, help="Prefix tokens")
    proj_parser.add_argument("--output-dir", default="lido_pp/checkpoints", help="Output directory")

    # Test roundtrip command
    rt_parser = subparsers.add_parser("test-roundtrip", help="Test encode-decode round-trip")
    rt_parser.add_argument("--model", default="GritLM/GritLM-7B", help="GritLM model")
    rt_parser.add_argument("--device", default="cuda:0", help="Device")
    rt_parser.add_argument("--train-epochs", type=int, default=0, help="Projector training epochs")

    # Analyze FCU command
    fcu_parser = subparsers.add_parser("analyze-fcu", help="Analyze FCU distribution")
    fcu_parser.add_argument("--load-flow", help="Path to FlowDiT checkpoint")
    fcu_parser.add_argument("--device", default="cuda:0", help="Device")
    fcu_parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")

    # Reflow command
    reflow_parser = subparsers.add_parser("reflow", help="Run Reflow on trained model")
    reflow_parser.add_argument("--load-flow", required=True, help="Path to FlowDiT checkpoint")
    reflow_parser.add_argument("--epochs", type=int, default=2000, help="Reflow epochs")
    reflow_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    reflow_parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    reflow_parser.add_argument("--num-pairs", type=int, default=10000, help="Trajectory pairs")
    reflow_parser.add_argument("--device", default="cuda:0", help="Device")
    reflow_parser.add_argument("--output-dir", default="lido_pp/checkpoints", help="Output directory")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch command
    if args.command == "train":
        cmd_train(args)
    elif args.command == "train-projector":
        cmd_train_projector(args)
    elif args.command == "test-roundtrip":
        cmd_test_roundtrip(args)
    elif args.command == "analyze-fcu":
        cmd_analyze_fcu(args)
    elif args.command == "reflow":
        cmd_reflow(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
