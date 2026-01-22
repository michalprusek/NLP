#!/usr/bin/env python3
"""
Test ManifoldKeeper projection: real embeddings + perturbation → ODE flow → decode.

This tests the more realistic use case: project perturbed/optimized embeddings
back to the instruction manifold.
"""

import argparse
import torch
import torch.nn.functional as F
from flowpo_hd.manifold_keeper import ManifoldKeeperMLP
from flowpo_hd.utils import SONARHelper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="/home/prusek/NLP/flowpo_hd/checkpoints/latest.pt")
    parser.add_argument("--data-path", type=str,
                       default="flowpo_hd/data/mega_raw_encoded.pt")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--noise-scale", type=float, default=0.1,
                       help="Scale of perturbation relative to embedding norm")
    parser.add_argument("--t-start", type=float, default=0.5,
                       help="Starting t for ODE (0=pure noise, 1=data)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Auto-detect num_blocks from checkpoint
    num_blocks = len([k for k in checkpoint["model_state_dict"].keys()
                      if k.startswith("blocks.") and k.endswith(".adaln.proj.weight")])
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

    # Initialize SONAR
    print("Initializing SONAR...")
    sonar = SONARHelper(device=args.device, normalize=False)

    # Load training data
    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path, map_location="cpu")
    embeddings = data["embeddings"]
    instructions = data.get("instructions", None)
    print(f"Loaded {len(embeddings):,} embeddings")

    # Sample random indices
    indices = torch.randperm(len(embeddings))[:args.n_samples]
    x_orig = embeddings[indices].to(device)
    orig_instructions = [instructions[i] for i in indices] if instructions else None

    print(f"\n{'='*60}")
    print(f"Testing projection with noise_scale={args.noise_scale}, t_start={args.t_start}")
    print(f"{'='*60}\n")

    with torch.no_grad():
        # Original norms and stats
        orig_norms = x_orig.norm(dim=-1)
        print(f"Original embeddings: norm={orig_norms.mean():.4f} ± {orig_norms.std():.4f}")

        # Add perturbation
        noise = torch.randn_like(x_orig)
        noise = noise / noise.norm(dim=-1, keepdim=True) * orig_norms.unsqueeze(-1) * args.noise_scale
        x_perturbed = x_orig + noise
        pert_norms = x_perturbed.norm(dim=-1)
        print(f"Perturbed embeddings: norm={pert_norms.mean():.4f} ± {pert_norms.std():.4f}")

        # Cosine similarity before projection
        cos_sim_before = F.cosine_similarity(x_orig, x_perturbed, dim=-1)
        print(f"Cosine similarity (orig vs perturbed): {cos_sim_before.mean():.4f}")

        # Project using ODE flow (t_start → 1.0)
        x_projected = model.integrate(x_perturbed, t_start=args.t_start, t_end=1.0,
                                       num_steps=args.steps)
        proj_norms = x_projected.norm(dim=-1)
        print(f"Projected embeddings: norm={proj_norms.mean():.4f} ± {proj_norms.std():.4f}")

        # Cosine similarity after projection
        cos_sim_after = F.cosine_similarity(x_orig, x_projected, dim=-1)
        print(f"Cosine similarity (orig vs projected): {cos_sim_after.mean():.4f}")
        print(f"Improvement: {(cos_sim_after - cos_sim_before).mean():.4f}")

        # Scale to original norms for decoding
        x_proj_scaled = x_projected * (orig_norms.unsqueeze(-1) / x_projected.norm(dim=-1, keepdim=True))

        # Decode
        decoded_orig = sonar.decode(x_orig)
        decoded_perturbed = sonar.decode(x_perturbed)
        decoded_projected = sonar.decode(x_proj_scaled)

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}\n")

    for i in range(args.n_samples):
        print(f"[{i+1}] Ground truth:  {orig_instructions[i][:100] if orig_instructions else 'N/A'}...")
        print(f"    Decoded orig:   {decoded_orig[i][:100]}...")
        print(f"    Decoded pert:   {decoded_perturbed[i][:100]}...")
        print(f"    Decoded proj:   {decoded_projected[i][:100]}...")
        print(f"    cos_sim: before={cos_sim_before[i]:.4f}, after={cos_sim_after[i]:.4f}")
        print()

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    print(f"Mean cosine similarity before projection: {cos_sim_before.mean():.4f}")
    print(f"Mean cosine similarity after projection:  {cos_sim_after.mean():.4f}")
    print(f"Improvement:                              {(cos_sim_after - cos_sim_before).mean():.4f}")


if __name__ == "__main__":
    main()
