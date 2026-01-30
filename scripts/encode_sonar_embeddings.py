#!/usr/bin/env python3
"""
Encode entire dataset with SONAR text encoder.

Creates datasets/sonar_embeddings.pt with shape [N, 1024].
SONAR embeddings are NOT L2-normalized (unlike GTR).
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline


def main():
    device = "cuda:0"
    texts_path = "datasets/combined_texts.json"
    output_path = "datasets/sonar_embeddings.pt"
    batch_size = 256  # Optimal for SONAR (memory-independent, ~5h total)

    print("=" * 60)
    print("SONAR Dataset Encoding")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input: {texts_path}")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size}")
    print()

    # Load texts
    print("Loading texts...")
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    print(f"Loaded {len(texts):,} texts")

    # Load SONAR encoder
    print("\nLoading SONAR text encoder...")
    t2vec = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(device),
    )

    # Encode in batches
    print(f"\nEncoding {len(texts):,} texts...")
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i : i + batch_size]

        with torch.no_grad():
            embeddings = t2vec.predict(batch_texts, source_lang="eng_Latn")
            all_embeddings.append(embeddings.cpu())

        # Memory management
        if i > 0 and i % (batch_size * 100) == 0:
            torch.cuda.empty_cache()

    # Concatenate all embeddings
    print("\nConcatenating embeddings...")
    all_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"\nFinal embeddings shape: {all_embeddings.shape}")
    print(f"  Mean: {all_embeddings.mean().item():.6f}")
    print(f"  Std: {all_embeddings.std().item():.6f}")
    print(f"  L2 norm (mean): {all_embeddings.norm(dim=-1).mean().item():.4f}")

    # Save
    print(f"\nSaving to {output_path}...")
    torch.save(all_embeddings, output_path)

    file_size = Path(output_path).stat().st_size / (1024 ** 3)
    print(f"Saved! File size: {file_size:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
