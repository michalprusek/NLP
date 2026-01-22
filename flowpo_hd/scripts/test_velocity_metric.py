#!/usr/bin/env python3
"""
Test using velocity magnitude at t=1 as manifold distance metric.

Hypothesis: Points on the manifold should have low velocity at t=1 (nowhere to go).
Off-manifold points should have high velocity (need to move to reach manifold).
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
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    num_blocks = len([k for k in checkpoint["model_state_dict"].keys()
                      if k.startswith("blocks.") and k.endswith(".adaln.proj.weight")])

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

    # Load training data
    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path, map_location="cpu")
    embeddings = data["embeddings"]
    print(f"Loaded {len(embeddings):,} embeddings")

    # Sample random indices
    indices = torch.randperm(len(embeddings))[:args.n_samples]
    x_real = embeddings[indices].to(device)
    print(f"x_real shape: {x_real.shape}")  # Should be (n_samples, 1024)

    # Create perturbed versions with different noise levels
    noise_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    print(f"\n{'='*70}")
    print(f"Testing velocity magnitude at different t values")
    print(f"{'='*70}\n")

    t_values = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]

    with torch.no_grad():
        for noise_scale in noise_scales:
            # Add noise
            if noise_scale > 0:
                noise = torch.randn_like(x_real) * noise_scale * x_real.norm(dim=-1, keepdim=True).mean()
                x_test = x_real + noise
            else:
                x_test = x_real.clone()

            print(f"\nNoise scale: {noise_scale}")
            print(f"  Input norm: {x_test.norm(dim=-1).mean():.4f}")

            for t in t_values:
                t_tensor = torch.full((x_test.size(0),), t, device=device)
                # Note: model forward is (t, x), not (x, t)!
                velocity = model(t_tensor, x_test)
                v_norm = velocity.norm(dim=-1).mean()
                print(f"  t={t:.2f}: velocity_norm={v_norm:.4f}")

    # Compare real data vs random noise
    print(f"\n{'='*70}")
    print(f"Comparing real data vs pure noise")
    print(f"{'='*70}\n")

    with torch.no_grad():
        # Pure noise (scaled to match data norm)
        noise_pure = torch.randn(args.n_samples, 1024, device=device)
        noise_pure = noise_pure * (x_real.norm(dim=-1).mean() / noise_pure.norm(dim=-1, keepdim=True).mean())

        for t in t_values:
            t_tensor = torch.full((args.n_samples,), t, device=device)

            # Note: model forward is (t, x), not (x, t)!
            v_real = model(t_tensor, x_real)
            v_noise = model(t_tensor, noise_pure)

            print(f"t={t:.2f}: real_data_v_norm={v_real.norm(dim=-1).mean():.4f}, "
                  f"noise_v_norm={v_noise.norm(dim=-1).mean():.4f}, "
                  f"ratio={v_noise.norm(dim=-1).mean() / v_real.norm(dim=-1).mean():.2f}")


if __name__ == "__main__":
    main()
