#!/usr/bin/env python3
"""Test reconstruction quality of trained Cascading Matryoshka Flow."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import torch.nn.functional as F

from vec2text_vae.matryoshka_funnel import CascadingMatryoshkaGTRFunnelFlow


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    ckpt_path = Path("vec2text_vae/checkpoints/cascading_matryoshka_funnel_best.pt")
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    matryoshka_dims = ckpt['matryoshka_dims']
    latent_dim = ckpt['latent_dim']

    flow = CascadingMatryoshkaGTRFunnelFlow(
        input_dim=768,
        latent_dim=latent_dim,
        matryoshka_dims=matryoshka_dims,
    ).to(device)
    flow.load_state_dict(ckpt['model_state_dict'])
    flow.eval()
    print(f"Model loaded: {sum(p.numel() for p in flow.parameters()):,} parameters")

    # Load some embeddings
    emb_path = Path("vec2text_vae/cache/gtr_embeddings_full.pt")
    if not emb_path.exists():
        emb_path = Path("vec2text_vae/cache/gtr_embeddings.pt")

    print(f"Loading embeddings from {emb_path}...")
    embeddings = torch.load(emb_path, map_location=device)
    print(f"Embeddings shape: {embeddings.shape}")

    # Test on random samples
    n_test = 100
    indices = torch.randperm(len(embeddings))[:n_test]
    test_emb = embeddings[indices].to(device)

    print(f"\n{'='*60}")
    print(f"Testing reconstruction on {n_test} random samples")
    print(f"{'='*60}")

    with torch.no_grad():
        # Full latent reconstruction
        z = flow.encode(test_emb)
        x_recon_full = flow.decode(z)
        cos_sim_full = F.cosine_similarity(test_emb, x_recon_full, dim=-1)

        print(f"\nFull latent ({latent_dim}D):")
        print(f"  Cosine similarity: {cos_sim_full.mean():.4f} ± {cos_sim_full.std():.4f}")
        print(f"  Min: {cos_sim_full.min():.4f}, Max: {cos_sim_full.max():.4f}")

        # Per-level reconstruction
        print(f"\nPer-level reconstruction:")
        for level in matryoshka_dims:
            # Zero out dimensions beyond level
            z_partial = torch.cat([
                z[:, :level],
                torch.zeros(z.size(0), latent_dim - level, device=device)
            ], dim=-1) if level < latent_dim else z

            x_recon = flow.decode(z_partial, active_dim=level)
            cos_sim = F.cosine_similarity(test_emb, x_recon, dim=-1)
            print(f"  {level:3d}D: {cos_sim.mean():.4f} ± {cos_sim.std():.4f}")

        # Test cascade prediction quality
        print(f"\nCascade prediction accuracy:")
        for level_in, level_out in zip(matryoshka_dims[:-1], matryoshka_dims[1:]):
            z_prefix = z[:, :level_in]
            z_pred = flow.decoder.predict_next_level(z_prefix, level_in, deterministic=True)
            z_true = z[:, level_in:level_out]
            mse = F.mse_loss(z_pred, z_true)
            cos_sim = F.cosine_similarity(z_pred, z_true, dim=-1).mean()
            print(f"  {level_in:3d}D → {level_out:3d}D: MSE={mse:.6f}, cos_sim={cos_sim:.4f}")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
