#!/usr/bin/env python3
"""Encode full combined dataset to GTR embeddings."""

import json
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling with attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def get_gtr_embeddings(texts: list, device: str = "cuda", batch_size: int = 2048) -> torch.Tensor:
    """Get GTR embeddings (vec2text compatible: mean_pool + L2 normalize)."""
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading GTR-T5-Base encoder...")
    model = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base")
    encoder = model.encoder.to(device)
    encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", total=n_batches):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            model_output = encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            embeddings = mean_pool(model_output.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    output_path = Path("vec2text_vae/cache/gtr_embeddings_full.pt")

    # Check if already exists
    if output_path.exists():
        emb = torch.load(output_path, map_location='cpu')
        logger.info(f"Full embeddings already cached: {emb.shape}")
        return

    # Load texts
    texts_path = Path("vec2text_vae/cache/combined_texts.json")
    logger.info(f"Loading texts from {texts_path}...")
    with open(texts_path) as f:
        texts = json.load(f)
    logger.info(f"Loaded {len(texts):,} texts")

    # Encode with large batch size for A100
    embeddings = get_gtr_embeddings(texts, device, batch_size=4096)
    logger.info(f"Encoded embeddings shape: {embeddings.shape}")

    # Save
    logger.info(f"Saving to {output_path}...")
    torch.save(embeddings, output_path)
    logger.info("Done!")


if __name__ == "__main__":
    main()
