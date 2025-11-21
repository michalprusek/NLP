#!/usr/bin/env python3
"""Generate embeddings for prompts using sentence-transformers."""

import csv
import json
from sentence_transformers import SentenceTransformer

# Model selection - all-mpnet-base-v2 is excellent for semantic similarity
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INPUT_FILE = "prompt_dataset.csv"

# Create output filename with model name
model_short_name = MODEL_NAME.split('/')[-1]
OUTPUT_FILE = f"prompt_embeddings_{model_short_name}.csv"

print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Read prompts from CSV
prompts = []
print(f"\nReading prompts from {INPUT_FILE}...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompts.append(row['prompt'])

print(f"Loaded {len(prompts)} prompts")

# Generate embeddings (batch processing for efficiency)
print("\nGenerating embeddings...")
embeddings = model.encode(prompts, show_progress_bar=True, batch_size=32)
print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

# Write to CSV
print(f"\nWriting embeddings to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Write header
    writer.writerow(['prompt', 'embedding'])

    # Write data rows (embedding as JSON array for easy parsing)
    for prompt, embedding in zip(prompts, embeddings):
        embedding_json = json.dumps(embedding.tolist())
        writer.writerow([prompt, embedding_json])

print(f"✓ Successfully created {OUTPUT_FILE}")
print(f"  - Prompts: {len(prompts)}")
print(f"  - Embedding dimension: {embeddings.shape[1]}")
print(f"  - Model: {MODEL_NAME}")
