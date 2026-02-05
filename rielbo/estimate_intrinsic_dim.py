"""Intrinsic Dimensionality Estimation for SELFIES VAE latent space.

Estimates the true ID of the 256D latent manifold to set optimal
subspace_dim for SphericalSubspaceBO.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.estimate_intrinsic_dim \
        --n-samples 5000 --device cuda
"""

import argparse
import logging
import time

import numpy as np
import torch
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Estimate intrinsic dimensionality of SELFIES VAE latent space")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of molecules to encode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--zinc", action="store_true", help="Use ZINC dataset (249K) instead of GuacaMol (20K)")
    args = parser.parse_args()

    # ============================================================
    # Step 1: Encode molecules
    # ============================================================
    logger.info(f"Loading codec on {args.device}...")
    from shared.guacamol.codec import SELFIESVAECodec
    codec = SELFIESVAECodec.from_pretrained(device=args.device)

    if args.zinc:
        logger.info("Loading ZINC dataset...")
        with open("datasets/zinc/zinc_all.txt") as f:
            all_smiles = [line.strip() for line in f if line.strip()]
    else:
        logger.info("Loading GuacaMol dataset...")
        import pandas as pd
        df = pd.read_csv("datasets/guacamol/guacamol_train_data_first_20k.csv")
        all_smiles = df["smile"].tolist()

    # Sample
    n = min(args.n_samples, len(all_smiles))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_smiles), size=n, replace=False)
    smiles_list = [all_smiles[i] for i in indices]
    logger.info(f"Encoding {n} molecules...")

    t0 = time.time()
    embeddings_torch = codec.encode_batch(smiles_list, batch_size=128)
    logger.info(f"Encoding took {time.time() - t0:.1f}s, shape: {embeddings_torch.shape}")

    # Remove zero vectors (invalid molecules)
    norms = embeddings_torch.norm(dim=-1)
    valid_mask = norms > 1e-6
    embeddings_torch = embeddings_torch[valid_mask]
    logger.info(f"Valid embeddings: {embeddings_torch.shape[0]}/{n}")

    # Raw embeddings (for variance analysis)
    embeddings = embeddings_torch.cpu().numpy()

    # Normalized to unit sphere (matching BO setup)
    norms_np = np.linalg.norm(embeddings, axis=1, keepdims=True)
    directions = embeddings / norms_np

    print("\n" + "=" * 60)
    print("INTRINSIC DIMENSIONALITY ESTIMATION")
    print(f"Dataset: {'ZINC' if args.zinc else 'GuacaMol'}, N={embeddings.shape[0]}, D={embeddings.shape[1]}")
    print("=" * 60)

    # ============================================================
    # Step 2: Linear methods (PCA, Participation Ratio, ARD-style)
    # ============================================================
    print("\n--- Linear Methods ---")

    # PCA on raw embeddings
    pca = PCA().fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eigenvalues = pca.explained_variance_

    d_80 = int(np.searchsorted(cumvar, 0.80) + 1)
    d_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    d_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    d_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    print(f"PCA cumulative variance: d_80={d_80}, d_90={d_90}, d_95={d_95}, d_99={d_99}")

    # Participation Ratio
    ev = eigenvalues[eigenvalues > 0]
    PR = float((ev.sum()) ** 2 / (ev ** 2).sum())
    print(f"Participation Ratio: {PR:.1f}")

    # Per-dimension variance (ARD-style)
    per_dim_var = embeddings.var(axis=0)
    sorted_var = np.sort(per_dim_var)[::-1]
    active_1pct = int((sorted_var > 0.01 * sorted_var[0]).sum())
    active_5pct = int((sorted_var > 0.05 * sorted_var[0]).sum())
    active_10pct = int((sorted_var > 0.10 * sorted_var[0]).sum())
    print(f"Active dims (>1% max var): {active_1pct}")
    print(f"Active dims (>5% max var): {active_5pct}")
    print(f"Active dims (>10% max var): {active_10pct}")

    # PCA on directions (unit sphere)
    pca_dir = PCA().fit(directions)
    cumvar_dir = np.cumsum(pca_dir.explained_variance_ratio_)
    d_95_dir = int(np.searchsorted(cumvar_dir, 0.95) + 1)
    ev_dir = pca_dir.explained_variance_
    ev_dir = ev_dir[ev_dir > 0]
    PR_dir = float((ev_dir.sum()) ** 2 / (ev_dir ** 2).sum())
    print(f"\nPCA on unit sphere directions: d_95={d_95_dir}, PR={PR_dir:.1f}")

    # ============================================================
    # Step 3: Nonlinear methods (scikit-dimension)
    # ============================================================
    print("\n--- Nonlinear Methods (scikit-dimension) ---")

    import skdim

    # TwoNN
    t0 = time.time()
    twonn = skdim.id.TwoNN(discard_fraction=0.1)
    twonn.fit(embeddings)
    print(f"TwoNN (raw):       {twonn.dimension_:.1f}  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    twonn_dir = skdim.id.TwoNN(discard_fraction=0.1)
    twonn_dir.fit(directions)
    print(f"TwoNN (S^255):     {twonn_dir.dimension_:.1f}  ({time.time() - t0:.1f}s)")

    # MLE
    t0 = time.time()
    mle = skdim.id.MLE(K=20)
    mle.fit(embeddings)
    print(f"MLE K=20 (raw):    {mle.dimension_:.1f}  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    mle_dir = skdim.id.MLE(K=20)
    mle_dir.fit(directions)
    print(f"MLE K=20 (S^255):  {mle_dir.dimension_:.1f}  ({time.time() - t0:.1f}s)")

    # DANCo
    t0 = time.time()
    try:
        danco = skdim.id.DANCo()
        danco.fit(embeddings)
        danco_val = danco.dimension_
        print(f"DANCo (raw):       {danco_val:.1f}  ({time.time() - t0:.1f}s)")
    except Exception as e:
        danco_val = None
        print(f"DANCo (raw):       FAILED - {e}")

    # ESS
    t0 = time.time()
    try:
        ess = skdim.id.ESS()
        ess.fit(embeddings[:2000])  # ESS can be slow on large datasets
        ess_val = ess.dimension_
        print(f"ESS (raw, N=2k):   {ess_val:.1f}  ({time.time() - t0:.1f}s)")
    except Exception as e:
        ess_val = None
        print(f"ESS (raw):         FAILED - {e}")

    # FisherS
    t0 = time.time()
    try:
        fishers = skdim.id.FisherS()
        fishers.fit(embeddings[:2000])
        fishers_val = fishers.dimension_
        print(f"FisherS (N=2k):    {fishers_val:.1f}  ({time.time() - t0:.1f}s)")
    except Exception as e:
        fishers_val = None
        print(f"FisherS:           FAILED - {e}")

    # ============================================================
    # Step 4: GRIDE multiscale (DADApy)
    # ============================================================
    print("\n--- Multiscale Analysis (DADApy GRIDE) ---")

    from dadapy import Data

    # On raw embeddings
    t0 = time.time()
    data_raw = Data(coordinates=embeddings)
    data_raw.compute_distances(maxk=100)
    id_2nn_raw, id_err_raw, _ = data_raw.compute_id_2NN()
    print(f"DADApy TwoNN (raw):  {id_2nn_raw:.1f} ± {id_err_raw:.1f}  ({time.time() - t0:.1f}s)")

    # On unit sphere directions
    t0 = time.time()
    data_dir = Data(coordinates=directions)
    data_dir.compute_distances(maxk=100)
    id_2nn_dir, id_err_dir, _ = data_dir.compute_id_2NN()
    print(f"DADApy TwoNN (S^255): {id_2nn_dir:.1f} ± {id_err_dir:.1f}  ({time.time() - t0:.1f}s)")

    # GRIDE multiscale on raw embeddings
    t0 = time.time()
    ids_gride, errs_gride, scales_gride = data_raw.return_id_scaling_gride(range_max=64)
    print(f"\nGRIDE Multiscale (raw embeddings):  ({time.time() - t0:.1f}s)")
    for s, i, e in zip(scales_gride, ids_gride, errs_gride):
        print(f"  neighbors={s:5.1f}: ID={i:.1f} ± {e:.1f}")

    # GRIDE multiscale on directions
    t0 = time.time()
    ids_gride_dir, errs_gride_dir, scales_gride_dir = data_dir.return_id_scaling_gride(range_max=64)
    print(f"\nGRIDE Multiscale (unit sphere S^255):  ({time.time() - t0:.1f}s)")
    for s, i, e in zip(scales_gride_dir, ids_gride_dir, errs_gride_dir):
        print(f"  neighbors={s:5.1f}: ID={i:.1f} ± {e:.1f}")

    # ============================================================
    # Step 5: Local ID distribution
    # ============================================================
    print("\n--- Local ID Distribution (lPCA) ---")

    t0 = time.time()
    lpca = skdim.id.lPCA()
    lpca.fit_pw(embeddings, n_neighbors=100, n_jobs=8)
    local_ids = lpca.dimension_pw_
    print(f"Local PCA ID (raw):  ({time.time() - t0:.1f}s)")
    print(f"  mean={np.mean(local_ids):.1f}, median={np.median(local_ids):.1f}, std={np.std(local_ids):.1f}")
    print(f"  10th pctl={np.percentile(local_ids, 10):.1f}, "
          f"25th={np.percentile(local_ids, 25):.1f}, "
          f"75th={np.percentile(local_ids, 75):.1f}, "
          f"90th={np.percentile(local_ids, 90):.1f}")

    # ============================================================
    # Step 6: Norm distribution analysis
    # ============================================================
    print("\n--- Norm Distribution ---")
    norms_flat = norms_np.flatten()
    print(f"  mean={norms_flat.mean():.2f}, std={norms_flat.std():.2f}")
    print(f"  min={norms_flat.min():.2f}, max={norms_flat.max():.2f}")
    print(f"  10th={np.percentile(norms_flat, 10):.2f}, "
          f"90th={np.percentile(norms_flat, 90):.2f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY OF ID ESTIMATES")
    print("=" * 60)

    estimates = {
        "PCA d_95 (raw)": d_95,
        "PCA d_95 (S^255)": d_95_dir,
        "Participation Ratio (raw)": PR,
        "Participation Ratio (S^255)": PR_dir,
        "Active dims >5% var": active_5pct,
        "TwoNN (raw)": twonn.dimension_,
        "TwoNN (S^255)": twonn_dir.dimension_,
        "MLE K=20 (raw)": mle.dimension_,
        "MLE K=20 (S^255)": mle_dir.dimension_,
        "DADApy TwoNN (raw)": id_2nn_raw,
        "DADApy TwoNN (S^255)": id_2nn_dir,
        "lPCA median (raw)": np.median(local_ids),
    }
    if danco_val is not None:
        estimates["DANCo (raw)"] = danco_val
    if ess_val is not None:
        estimates["ESS (raw, N=2k)"] = ess_val
    if fishers_val is not None:
        estimates["FisherS (N=2k)"] = fishers_val

    for name, val in estimates.items():
        print(f"  {name:30s}: {val:.1f}")

    # Recommendation
    nonlinear = [
        twonn.dimension_, twonn_dir.dimension_,
        mle.dimension_, mle_dir.dimension_,
        id_2nn_raw, id_2nn_dir,
        np.median(local_ids),
    ]
    if danco_val is not None:
        nonlinear.append(danco_val)
    if ess_val is not None:
        nonlinear.append(ess_val)

    recommended = int(np.median(nonlinear))
    print(f"\n  >>> RECOMMENDED subspace_dim: {recommended}")
    print(f"  >>> (median of {len(nonlinear)} nonlinear estimates)")
    print(f"  >>> Test range: [{max(4, recommended // 2)}, {recommended * 2}]")

    # Points per dimension analysis
    total_pts = 600  # 100 cold-start + 500 iterations
    for d in [8, 12, 16, recommended, 24, 32, 48, 64]:
        ratio = total_pts / d
        quality = "excellent" if ratio > 20 else "good" if ratio > 10 else "marginal" if ratio > 5 else "poor"
        marker = " <<<" if d == recommended else ""
        print(f"    d={d:3d}: {ratio:.1f} pts/dim ({quality}){marker}")


if __name__ == "__main__":
    main()
