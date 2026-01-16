"""
Pre-compute SONAR embeddings for FlowPO training.

This script pre-computes SONAR embeddings (1024D) once so they can be reused
for TFA (Text Flow Autoencoder), Flow-DiT, and Decoder training.

SONAR (Meta's reconstruction-optimized encoder) is the recommended encoder
because it's trained for translation, preserving reconstruction information.

Output format:
{
    "embeddings": Tensor (N, 1024),
    "instructions": List[str],
    "metadata": {
        "model": "sonar",
        "dataset": "alpaca",
        "n_samples": N,
        "embedding_dim": 1024,
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
    parser = argparse.ArgumentParser(description="Pre-compute SONAR embeddings for FlowPO")
    parser.add_argument(
        "--dataset",
        type=str,
        default="alpaca",
        choices=["alpaca", "ultrachat", "combined", "universal", "custom"],
        help="Dataset to encode (universal = 1.5M+ from OpenOrca, UltraChat, Code, ShareGPT, Alpaca)",
    )
    parser.add_argument(
        "--ultrachat-samples",
        type=int,
        default=50000,
        help="Number of UltraChat samples for combined dataset",
    )
    # Universal dataset sample counts
    parser.add_argument(
        "--openorca-samples",
        type=int,
        default=500000,
        help="Number of OpenOrca samples for universal dataset",
    )
    parser.add_argument(
        "--code-samples",
        type=int,
        default=200000,
        help="Number of code samples (CodeAlpaca + Glaive) for universal dataset",
    )
    parser.add_argument(
        "--sharegpt-samples",
        type=int,
        default=100000,
        help="Number of ShareGPT samples for universal dataset",
    )
    parser.add_argument(
        "--existing-embeddings",
        type=str,
        default=None,
        help="Path to existing embeddings to merge with (for universal dataset)",
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
        default="lido_pp/data/sonar_embeddings.pt",
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
        default=1024,
        help="Batch size for encoding (optimized for L40S 48GB VRAM)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for encoding",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="sonar",
        choices=["sonar", "gtr"],
        help="Encoder to use: sonar (recommended, 1024D), gtr (lightweight, 768D)",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="eng_Latn",
        help="Source language for SONAR encoder",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="L2 normalize embeddings (default: False for SONAR decoder compatibility)",
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

    elif args.dataset == "universal":
        from lido_pp.training.alpaca_dataset import load_universal_dataset

        print("\n" + "=" * 70)
        print("Loading UNIVERSAL dataset for production training")
        print("Sources: OpenOrca, UltraChat, CodeAlpaca, ShareGPT, Alpaca")
        print(f"Target samples: {args.openorca_samples + args.ultrachat_samples + args.code_samples + args.sharegpt_samples + 52000}")
        print("=" * 70 + "\n")

        instructions = load_universal_dataset(
            openorca_samples=args.openorca_samples,
            ultrachat_samples=args.ultrachat_samples,
            codealpaca_samples=args.code_samples,
            sharegpt_samples=args.sharegpt_samples,
            alpaca_samples=52000,  # All Alpaca
            existing_path=args.existing_embeddings,
            deduplicate=True,
        )

        if args.max_samples and len(instructions) > args.max_samples:
            instructions = instructions[:args.max_samples]

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
    print(f"\nInitializing {args.encoder} encoder...")

    if args.encoder == "sonar":
        from lido_pp.backbone.sonar_encoder import SONAREncoder

        print("\n" + "=" * 70)
        print("Using SONAR encoder (recommended for FlowPO)")
        print("SONAR is reconstruction-optimized (DAE + translation loss)")
        print("Output dimension: 1024D")
        print("=" * 70 + "\n")

        encoder = SONAREncoder(
            device=args.device,
            source_lang=args.source_lang,
            normalize=args.normalize,
        )
        model_name = "sonar-text-encoder"
        embedding_dim = 1024

    elif args.encoder == "gtr":
        from lipo.encoder import GTRInstructionEncoder
        encoder = GTRInstructionEncoder(device=args.device)
        model_name = "sentence-transformers/gtr-t5-base"
        embedding_dim = 768

    print(f"Encoder ready: {model_name}")

    # Encode all instructions
    print(f"\nEncoding {len(instructions)} instructions with batch_size={args.batch_size}...")

    all_embeddings = []

    for i in tqdm(range(0, len(instructions), args.batch_size), desc="Encoding"):
        batch = instructions[i:i + args.batch_size]

        with torch.no_grad():
            if args.encoder == "sonar":
                batch_embeddings = encoder.encode(batch)
            else:  # gtr
                batch_embeddings = torch.tensor(encoder.encode(batch))

        all_embeddings.append(batch_embeddings.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    print(f"Embeddings shape: {embeddings_tensor.shape}")

    # Prepare output data
    data = {
        "embeddings": embeddings_tensor,
        "instructions": instructions,
        "metadata": {
            "model": model_name,
            "encoder_type": args.encoder,
            "dataset": args.dataset,
            "n_samples": len(instructions),
            "embedding_dim": embeddings_tensor.shape[1],
            "normalized": args.normalize,
            "source_lang": args.source_lang if args.encoder == "sonar" else None,
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
