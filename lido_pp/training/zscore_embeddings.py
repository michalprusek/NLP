"""
Z-Score Standardization for SONAR Embeddings.

Transforms embeddings to zero mean and unit variance per dimension.
Saves mean/std for denormalization during inference.

Usage:
    uv run python -m lido_pp.training.zscore_embeddings \
        --input lido_pp/data/sonar_unified_unnorm.pt \
        --output lido_pp/data/sonar_unified_zscore.pt
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Z-Score standardize SONAR embeddings")
    parser.add_argument(
        "--input",
        type=str,
        default="lido_pp/data/sonar_unified_unnorm.pt",
        help="Input embeddings file (unnormalized)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lido_pp/data/sonar_unified_zscore.pt",
        help="Output embeddings file (z-score normalized)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for numerical stability in std division",
    )
    args = parser.parse_args()

    print(f"Loading embeddings from {args.input}...")
    data = torch.load(args.input, weights_only=False)

    if isinstance(data, dict):
        embeddings = data["embeddings"]
        metadata = data.get("metadata", {})
    else:
        embeddings = data
        metadata = {}

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Original metadata: {metadata}")

    # Compute statistics per dimension
    print("\nComputing z-score statistics...")
    mean = embeddings.mean(dim=0)  # (embedding_dim,)
    std = embeddings.std(dim=0)    # (embedding_dim,)

    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")

    # Check for zero/near-zero std dimensions
    low_std_dims = (std < args.eps).sum().item()
    if low_std_dims > 0:
        print(f"WARNING: {low_std_dims} dimensions have std < {args.eps}")

    # Apply z-score standardization
    print("\nApplying z-score standardization...")
    embeddings_zscore = (embeddings - mean) / (std + args.eps)

    # Verify standardization
    new_mean = embeddings_zscore.mean(dim=0)
    new_std = embeddings_zscore.std(dim=0)
    print(f"After z-score:")
    print(f"  Mean range: [{new_mean.min():.6f}, {new_mean.max():.6f}] (should be ~0)")
    print(f"  Std range: [{new_std.min():.6f}, {new_std.max():.6f}] (should be ~1)")
    print(f"  Mean of means: {new_mean.mean():.8f}")
    print(f"  Mean of stds: {new_std.mean():.6f}")

    # Compare norms before/after
    orig_norms = embeddings.norm(dim=-1)
    zscore_norms = embeddings_zscore.norm(dim=-1)
    print(f"\nNorm comparison:")
    print(f"  Original: mean={orig_norms.mean():.2f}, std={orig_norms.std():.2f}")
    print(f"  Z-score:  mean={zscore_norms.mean():.2f}, std={zscore_norms.std():.2f}")

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata.update({
        "normalized": "zscore",
        "zscore_eps": args.eps,
        "zscore_timestamp": datetime.now().isoformat(),
        "original_file": args.input,
    })

    # Save
    output_data = {
        "embeddings": embeddings_zscore,
        "mean": mean,
        "std": std,
        "metadata": new_metadata,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {args.output}...")
    torch.save(output_data, args.output)

    # Verify save
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved successfully ({file_size_mb:.1f} MB)")

    # Verification: load and check denormalization works
    print("\nVerifying denormalization...")
    loaded = torch.load(args.output, weights_only=False)
    denorm = loaded["embeddings"] * loaded["std"] + loaded["mean"]
    reconstruction_error = (denorm - embeddings).abs().max().item()
    print(f"Max reconstruction error: {reconstruction_error:.2e} (should be ~0)")

    if reconstruction_error < 1e-5:
        print("✓ Denormalization verified successfully!")
    else:
        print("⚠ WARNING: Denormalization error is high!")

    print("\n" + "=" * 60)
    print("Z-Score Standardization Complete")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Mean/Std saved for denormalization during inference")
    print("=" * 60)


if __name__ == "__main__":
    main()
