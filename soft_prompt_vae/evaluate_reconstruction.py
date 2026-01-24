"""Quick evaluation script for VAE reconstruction quality."""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from soft_prompt_vae.model import LlamaSoftPromptVAE
from soft_prompt_vae.config import VAEConfig

def load_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load model from checkpoint."""
    config = VAEConfig()
    model = LlamaSoftPromptVAE(config.model, use_ddp=False)

    # Load checkpoint
    ckpt_dir = Path(checkpoint_path)
    checkpoint = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint['global_step']}, epoch {checkpoint['epoch']}")
    return model, config

def encode_decode(model, tokenizer, instruction: str, device: str = "cuda:1", max_length: int = 128):
    """Encode instruction to latent and decode back."""
    # Tokenize
    tokens = tokenizer(
        instruction,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        # Encode to latent
        z, mu, logvar = model.encode(input_ids, attention_mask)

        # Decode back
        # Use greedy decoding for reconstruction
        generated = model.generate(
            z,
            max_length=max_length,
            temperature=1.0,
            do_sample=False,  # greedy
        )

    # Decode tokens to text
    reconstructed = tokenizer.decode(generated[0], skip_special_tokens=True)

    return reconstructed, mu.cpu(), logvar.cpu()

def compute_metrics(original: str, reconstructed: str):
    """Compute reconstruction metrics."""
    # Exact match
    exact_match = original.strip() == reconstructed.strip()

    # Word-level metrics
    orig_words = original.lower().split()
    recon_words = reconstructed.lower().split()

    # Word overlap (Jaccard)
    orig_set = set(orig_words)
    recon_set = set(recon_words)
    if len(orig_set | recon_set) > 0:
        jaccard = len(orig_set & recon_set) / len(orig_set | recon_set)
    else:
        jaccard = 0.0

    # Character-level edit distance ratio
    from difflib import SequenceMatcher
    char_similarity = SequenceMatcher(None, original.lower(), reconstructed.lower()).ratio()

    # Prefix match (how much of beginning matches)
    prefix_match = 0
    for i, (c1, c2) in enumerate(zip(original, reconstructed)):
        if c1 == c2:
            prefix_match = i + 1
        else:
            break
    prefix_ratio = prefix_match / max(len(original), 1)

    return {
        "exact_match": exact_match,
        "jaccard": jaccard,
        "char_similarity": char_similarity,
        "prefix_ratio": prefix_ratio,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VAE reconstruction quality")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., soft_prompt_vae/checkpoints/checkpoint-600)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda:0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (uses default if not specified)",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    # Load config from file if specified, otherwise use default
    if args.config:
        config = VAEConfig.load(Path(args.config))
        model = LlamaSoftPromptVAE(config.model, use_ddp=False)
        checkpoint = torch.load(
            Path(args.checkpoint) / "checkpoint.pt",
            map_location=args.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(args.device)
        model.eval()
        print(f"Loaded checkpoint from step {checkpoint['global_step']}, epoch {checkpoint['epoch']}")
    else:
        model, config = load_model(args.checkpoint, args.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sample instructions to test
    test_instructions = [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function that calculates factorial.",
        "What are the benefits of regular exercise?",
        "Describe the process of photosynthesis.",
        "How do I make a good cup of coffee?",
        "Summarize the main points of climate change.",
        "Write a haiku about autumn leaves.",
        "Explain how a neural network learns.",
        "What is the difference between SQL and NoSQL?",
        "Give me tips for better time management.",
        "How does encryption protect data?",
        "Explain the theory of relativity simply.",
    ][:args.num_samples]

    print(f"\nEvaluating {len(test_instructions)} instructions...\n")
    print("=" * 80)

    results = []
    for i, instruction in enumerate(test_instructions):
        reconstructed, mu, logvar = encode_decode(model, tokenizer, instruction, args.device)
        metrics = compute_metrics(instruction, reconstructed)

        results.append({
            "id": i + 1,
            "original": instruction,
            "reconstructed": reconstructed,
            **metrics,
        })

        print(f"[{i+1}] Original:      {instruction[:60]}...")
        print(f"    Reconstructed: {reconstructed[:60]}...")
        print(f"    Char sim: {metrics['char_similarity']:.2%} | Jaccard: {metrics['jaccard']:.2%} | Exact: {metrics['exact_match']}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    avg_char_sim = sum(r["char_similarity"] for r in results) / len(results)
    avg_jaccard = sum(r["jaccard"] for r in results) / len(results)
    exact_matches = sum(r["exact_match"] for r in results)

    print(f"Average char similarity: {avg_char_sim:.2%}")
    print(f"Average Jaccard:         {avg_jaccard:.2%}")
    print(f"Exact matches:           {exact_matches}/{len(results)}")

    # Save results
    output_path = Path("results") / "vae_reconstruction_eval.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
