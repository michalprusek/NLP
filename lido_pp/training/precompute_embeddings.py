"""
Pre-compute GritLM embeddings for Alpaca dataset.

This script pre-computes embeddings once so they can be reused
for VAE, Projector, and FlowDiT training without re-encoding.

Output format:
{
    "embeddings": Tensor (N, 768),
    "instructions": List[str],
    "metadata": {
        "model": "GritLM/GritLM-7B",
        "dataset": "alpaca",
        "n_samples": N,
        "timestamp": "...",
    }
}
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Pre-compute GritLM embeddings")
    parser.add_argument(
        "--dataset",
        type=str,
        default="alpaca",
        choices=["alpaca", "ultrachat", "combined", "custom"],
        help="Dataset to encode",
    )
    parser.add_argument(
        "--ultrachat-samples",
        type=int,
        default=50000,
        help="Number of UltraChat samples for combined dataset",
    )
    parser.add_argument(
        "--custom-path",
        type=str,
        default=None,
        help="Path to custom JSON file with instructions (for --dataset custom)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lido_pp/data/alpaca_embeddings.pt",
        help="Output path for embeddings",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to encode (None = all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for encoding",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GritLM/GritLM-7B",
        help="GritLM model name",
    )
    parser.add_argument(
        "--use-gtr",
        action="store_true",
        help="Use GTR encoder instead of GritLM (faster, smaller)",
    )
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load instructions based on dataset
    print(f"Loading {args.dataset} dataset...")

    if args.dataset == "alpaca":
        from lido_pp.training.alpaca_dataset import load_alpaca_dataset
        instructions = load_alpaca_dataset(max_samples=args.max_samples)

    elif args.dataset == "ultrachat":
        from lido_pp.training.alpaca_dataset import load_ultrachat_dataset
        instructions = load_ultrachat_dataset(max_samples=args.max_samples)

    elif args.dataset == "combined":
        from lido_pp.training.alpaca_dataset import load_combined_dataset
        # 0 means all samples (None)
        uc_samples = None if args.ultrachat_samples == 0 else args.ultrachat_samples
        instructions = load_combined_dataset(
            alpaca_samples=args.max_samples,
            ultrachat_samples=uc_samples,
        )

    elif args.dataset == "custom":
        if not args.custom_path:
            raise ValueError("--custom-path required for custom dataset")
        with open(args.custom_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            if isinstance(data[0], str):
                instructions = data
            else:
                instructions = [d.get("instruction", d.get("text", "")) for d in data]
        else:
            raise ValueError("Custom file must be a JSON list")

        if args.max_samples:
            instructions = instructions[:args.max_samples]

    print(f"Loaded {len(instructions)} instructions")

    # Initialize encoder
    print(f"\nInitializing encoder...")

    if args.use_gtr:
        from lipo.encoder import GTRInstructionEncoder
        encoder = GTRInstructionEncoder(device=args.device)
        model_name = "sentence-transformers/gtr-t5-base"
    else:
        from lido_pp.backbone.gritlm_encoder import GritLMUnifiedEncoder
        encoder = GritLMUnifiedEncoder(
            model_name=args.model,
            device=args.device,
            dtype="float16",
        )
        model_name = args.model

    print(f"Encoder ready: {model_name}")

    # Encode all instructions
    print(f"\nEncoding {len(instructions)} instructions with batch_size={args.batch_size}...")

    embeddings = encoder.encode_batch(
        instructions,
        batch_size=args.batch_size,
        show_progress=True,
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # Convert to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Prepare output data
    data = {
        "embeddings": embeddings_tensor,
        "instructions": instructions,
        "metadata": {
            "model": model_name,
            "dataset": args.dataset,
            "n_samples": len(instructions),
            "embedding_dim": embeddings.shape[1],
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Save
    print(f"\nSaving to {args.output}...")
    torch.save(data, args.output)

    # Verify
    loaded = torch.load(args.output, weights_only=False)
    print(f"\nVerification:")
    print(f"  Embeddings shape: {loaded['embeddings'].shape}")
    print(f"  Instructions: {len(loaded['instructions'])}")
    print(f"  Model: {loaded['metadata']['model']}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Also save instructions separately for convenience
    instructions_path = output_path.with_suffix(".json")
    with open(instructions_path, "w") as f:
        json.dump(instructions, f, indent=2)
    print(f"  Instructions saved to: {instructions_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
