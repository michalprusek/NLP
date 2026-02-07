"""Estimate intrinsic dimensionality for all GuacaMol tasks using TwoNN + MLE.

Scores all ZINC molecules on each task, takes top-K, encodes to latent space,
normalizes to unit sphere, and runs TwoNN + MLE estimators.

No capping applied — reports raw estimates.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/home/prusek/NLP")

from shared.guacamol.codec import SELFIESVAECodec
from shared.guacamol.oracle import GuacaMolOracle

TASKS = ["adip", "med2", "pdop", "rano", "osmb", "siga", "zale", "valt", "dhop", "shop", "fexo", "med1"]
ZINC_PATH = "/home/prusek/NLP/datasets/zinc/zinc_all.txt"
TOP_KS = [100, 250, 500]


def main():
    print("Loading ZINC molecules...")
    with open(ZINC_PATH) as f:
        all_smiles = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(all_smiles)} molecules")

    print("Loading VAE codec...")
    codec = SELFIESVAECodec.from_pretrained(device="cuda")

    import skdim

    results = {}

    for task_id in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"{'='*60}")

        oracle = GuacaMolOracle(task_id=task_id)

        print(f"Scoring {len(all_smiles)} molecules...")
        scores = oracle.score_batch(all_smiles)
        scores_np = np.array(scores)

        # Sort by score descending
        sorted_idx = np.argsort(scores_np)[::-1]

        task_results = {}
        for top_k in TOP_KS:
            top_idx = sorted_idx[:top_k]
            top_smiles = [all_smiles[i] for i in top_idx]
            top_scores = scores_np[top_idx]

            print(f"\n  Top {top_k}: score range [{top_scores[-1]:.4f}, {top_scores[0]:.4f}]")

            # Encode to latent space
            with torch.no_grad():
                embeddings = codec.encode(top_smiles)  # [K, 256]

            # Normalize to unit sphere
            norms = torch.norm(embeddings, dim=1, keepdim=True)
            directions = (embeddings / norms).cpu().numpy()

            # TwoNN
            twonn = skdim.id.TwoNN(discard_fraction=0.1)
            twonn.fit(directions)
            d_twonn = float(twonn.dimension_)

            # MLE
            mle = skdim.id.MLE(K=20)
            mle.fit(directions)
            d_mle = float(mle.dimension_)

            avg = (d_twonn + d_mle) / 2.0

            print(f"  Top {top_k}: TwoNN={d_twonn:.1f}  MLE={d_mle:.1f}  avg={avg:.1f}")
            task_results[top_k] = {"twonn": d_twonn, "mle": d_mle, "avg": avg,
                                    "score_min": float(top_scores[-1]), "score_max": float(top_scores[0])}

        results[task_id] = task_results

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY: Raw ID Estimates (no capping)")
    print(f"{'='*80}")
    print(f"{'Task':<8} | {'Top-100':<25} | {'Top-250':<25} | {'Top-500':<25}")
    print(f"{'':8} | {'TwoNN  MLE    avg':<25} | {'TwoNN  MLE    avg':<25} | {'TwoNN  MLE    avg':<25}")
    print("-" * 90)

    for task_id in TASKS:
        tr = results[task_id]
        parts = []
        for k in TOP_KS:
            r = tr[k]
            parts.append(f"{r['twonn']:5.1f}  {r['mle']:5.1f}  {r['avg']:5.1f}")
        print(f"{task_id:<8} | {parts[0]:<25} | {parts[1]:<25} | {parts[2]:<25}")

    # Score ranges
    print(f"\n{'Task':<8} | {'Score range (top-100)':<25} | {'Score range (top-500)':<25}")
    print("-" * 65)
    for task_id in TASKS:
        tr = results[task_id]
        r100 = tr[100]
        r500 = tr[500]
        print(f"{task_id:<8} | [{r100['score_min']:.4f}, {r100['score_max']:.4f}]{'':>8} | [{r500['score_min']:.4f}, {r500['score_max']:.4f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
