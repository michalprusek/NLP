#!/usr/bin/env python3
"""
Test ManifoldKeeper generation: noise → ODE flow → scale → SONAR decode.
"""

import argparse
import torch
from flowpo_hd.manifold_keeper import ManifoldKeeperMLP
from flowpo_hd.utils import SONARHelper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="/home/prusek/NLP/flowpo_hd/checkpoints/latest.pt")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-norm", type=float, default=0.28,
                       help="Target norm for SONAR decoder (training data mean)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Auto-detect num_blocks from checkpoint
    num_blocks = len([k for k in checkpoint["model_state_dict"].keys() if k.startswith("blocks.") and k.endswith(".adaln.proj.weight")])
    print(f"Detected {num_blocks} blocks from checkpoint")

    model = ManifoldKeeperMLP(
        dim=1024,
        hidden_dim=2048,
        time_dim=256,
        num_blocks=num_blocks,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model at step {checkpoint.get('global_step', 'unknown')}")

    # Initialize SONAR decoder
    print("Initializing SONAR decoder...")
    sonar = SONARHelper(device=args.device, normalize=False)

    # Generate samples
    print(f"\n{'='*60}")
    print(f"Generating {args.n_samples} samples with {args.steps} ODE steps")
    print(f"{'='*60}\n")

    with torch.no_grad():
        # Sample from prior - scale noise to match training data distribution
        # Training data: mean_norm=0.28, so noise should be scaled accordingly
        z_noise = torch.randn(args.n_samples, 1024, device=device)
        # Scale noise to have similar norm to training data
        z_noise = z_noise * (args.target_norm / z_noise.norm(dim=-1, keepdim=True).mean())
        print(f"Scaled noise: shape={z_noise.shape}, norm={z_noise.norm(dim=-1).mean():.4f}")

        # Flow: t=0 (noise) → t=1 (data manifold)
        z_data = model.integrate(z_noise, t_start=0.0, t_end=1.0, num_steps=args.steps)
        raw_norm = z_data.norm(dim=-1).mean()
        print(f"After flow (raw): norm={raw_norm:.4f}")

        # Scale to target norm for SONAR decoder
        z_scaled = z_data * (args.target_norm / z_data.norm(dim=-1, keepdim=True))
        print(f"After scaling: norm={z_scaled.norm(dim=-1).mean():.4f}")

        # Decode with SONAR
        texts = sonar.decode(z_scaled)

    print(f"\n{'='*60}")
    print("GENERATED INSTRUCTIONS:")
    print(f"{'='*60}\n")

    for i, text in enumerate(texts):
        print(f"[{i+1}] {text}")
        print()

    # Also test reconstruction of a real instruction
    print(f"\n{'='*60}")
    print("RECONSTRUCTION TEST:")
    print(f"{'='*60}\n")

    test_instructions = [
        "Let's think step by step and solve this problem carefully.",
        "Break down the problem into smaller steps before solving.",
    ]

    with torch.no_grad():
        # Encode
        z_orig = sonar.encode(test_instructions)
        print(f"Original embeddings: norm={z_orig.norm(dim=-1).mean():.4f}")

        # Project to manifold (assume we're at t=0.5, integrate to t=1)
        z_projected = model.integrate(z_orig.to(device), t_start=0.5, t_end=1.0, num_steps=25)
        proj_norm = z_projected.norm(dim=-1).mean()
        print(f"After projection: norm={proj_norm:.4f}")

        # Scale back to original norm
        z_proj_scaled = z_projected * (z_orig.norm(dim=-1, keepdim=True).to(device) / z_projected.norm(dim=-1, keepdim=True))

        # Decode
        decoded = sonar.decode(z_proj_scaled)

    for i, (orig, dec) in enumerate(zip(test_instructions, decoded)):
        print(f"[{i+1}] Original:    {orig}")
        print(f"    Decoded:     {dec}")
        print()


if __name__ == "__main__":
    main()
