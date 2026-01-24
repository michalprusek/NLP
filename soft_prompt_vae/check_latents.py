"""Quick check if latent space encodes different inputs differently.

Verifies that the VAE encoder is differentiating inputs (i.e., producing
different latent codes for different instructions). If latent distances are
near zero, the problem is in the encoder. If distances are significant but
generations are identical, the problem is in the decoder.
"""

import argparse
import torch
from soft_prompt_vae.model import LlamaSoftPromptVAE
from soft_prompt_vae.config import VAEConfig
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Check latent space differentiation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., soft_prompt_vae/checkpoints/checkpoint-1400)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda:0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (uses default if not specified)",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        from pathlib import Path
        config = VAEConfig.load(Path(args.config))
    else:
        config = VAEConfig()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = LlamaSoftPromptVAE(config.model, use_ddp=False)
    ckpt = torch.load(
        f"{args.checkpoint}/checkpoint.pt",
        map_location=args.device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device).eval()
    print(f"Loaded checkpoint from step {ckpt['global_step']}, epoch {ckpt['epoch']}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Test different instructions
    instructions = [
        "Explain machine learning.",
        "Write Python code for factorial.",
        "What are benefits of exercise?",
    ]

    print("\nChecking if latent space encodes different inputs differently...")
    print("=" * 60)

    latents = []
    for instr in instructions:
        tokens = tokenizer(
            instr,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokens["input_ids"].to(args.device)
        attention_mask = tokens["attention_mask"].to(args.device)

        with torch.no_grad():
            z, mu, logvar = model.encode(input_ids, attention_mask)

        latents.append(mu.cpu())
        print(f"Instruction: {instr[:40]}...")
        print(f"  mu mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")
        print(f"  mu[:5]: {[round(x, 3) for x in mu[0, :5].tolist()]}")
        print()

    # Compute pairwise distances
    print("Pairwise L2 distances between latent means:")
    for i in range(len(latents)):
        for j in range(i + 1, len(latents)):
            dist = torch.norm(latents[i] - latents[j]).item()
            print(f"  [{i+1}] vs [{j+1}]: {dist:.4f}")

    print()
    print("If distances > 0, the encoder IS differentiating inputs.")
    print("The problem is in generation, not encoding.")


if __name__ == "__main__":
    main()
