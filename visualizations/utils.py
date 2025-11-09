#!/usr/bin/env python3
"""
Shared utility functions for visualization scripts.
"""

import json
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


def load_optimization_results(json_path: str) -> Tuple[List[str], List[float], List[int]]:
    """
    Load prompts, scores, and iterations from optimization JSON.

    Args:
        json_path: Path to optimization results JSON

    Returns:
        Tuple of (prompts, scores, iterations)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    prompts = []
    scores = []
    iterations = []

    for entry in data['history']:
        prompts.append(entry['prompt'])
        scores.append(entry['score'])
        iterations.append(entry['iteration'])

    print(f"Loaded {len(prompts)} prompts from {data['method'].upper()} optimization")

    # Handle different JSON formats
    if 'model' in data:
        print(f"Model: {data['model']}")
    elif 'task_model' in data:
        print(f"Task model: {data['task_model']}")
        if 'meta_model' in data and data['meta_model'] != data['task_model']:
            print(f"Meta model: {data['meta_model']}")

    print(f"Iterations: 0-{max(iterations)}")
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")

    return prompts, scores, iterations


def embed_prompts(
    prompts: List[str],
    model_name: str = 'all-mpnet-base-v2',
    device: str = 'cpu'
) -> np.ndarray:
    """
    Create embeddings for prompts using sentence transformers.

    Args:
        prompts: List of prompt strings
        model_name: HuggingFace model name for embeddings
            - 'all-MiniLM-L6-v2': Fast, 384-dim
            - 'all-mpnet-base-v2': Better quality, 768-dim (default)
            - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual
        device: Device to use ('cpu', 'cuda', 'mps')

    Returns:
        Embeddings array of shape (n_prompts, embedding_dim)
    """
    print(f"\nGenerating embeddings with {model_name}...")
    model = SentenceTransformer(model_name, device=device)

    embeddings = model.encode(
        prompts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings