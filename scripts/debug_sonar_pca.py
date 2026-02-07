"""Test PCA projection for SONAR embeddings.

Random QR captures 1.7% variance in 1024D SONAR space.
PCA should capture much more by aligning with the principal directions
of variation among seed prompts.

Also test higher subspace dimensions.
"""

import torch
import torch.nn.functional as F
import numpy as np

print("Loading SONAR codec...")
from rielbo_gsm8k.sonar_codec import SonarCodec
codec = SonarCodec(device="cpu")

# All seed prompts
from rielbo_gsm8k.seed_prompts import get_seed_prompts
prompts = get_seed_prompts()
print(f"Loaded {len(prompts)} seed prompts")

with torch.no_grad():
    embeddings = codec.encode(prompts)  # [26, 1024]

norms = embeddings.norm(dim=-1)
mean_norm = norms.mean().item()
unit_emb = F.normalize(embeddings, p=2, dim=-1)

print(f"Embedding norms: mean={mean_norm:.4f}, range=[{norms.min():.4f}, {norms.max():.4f}]")

# --- PCA on unit embeddings ---
print("\n" + "=" * 60)
print("PCA analysis of seed embeddings on S^1023")
print("=" * 60)

centered = unit_emb - unit_emb.mean(dim=0, keepdim=True)
U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

print(f"Singular values (top 25): {S.tolist()}")
cumvar = (S**2).cumsum(0) / (S**2).sum()
print(f"Cumulative variance explained:")
for d in [4, 8, 16, 25]:
    if d <= len(S):
        print(f"  d={d}: {cumvar[d-1].item():.4f}")

# --- Test PCA projection at different d ---
print("\n" + "=" * 60)
print("PCA projection round-trip at different d")
print("=" * 60)

test_idx = [0, 2, 4]  # 3 diverse prompts
test_prompts = [prompts[i] for i in test_idx]

for d in [4, 8, 16, 25]:
    if d > len(S):
        continue
    A_pca = Vt[:d].T  # [1024, d] — top d principal components

    # Project and lift
    v = unit_emb[test_idx] @ A_pca  # [3, d] (NOT normalized — keep PCA coords)
    u_hat = v @ A_pca.T  # [3, 1024] — back in ambient space
    u_hat = u_hat + unit_emb.mean(dim=0, keepdim=True)  # add back mean
    # Don't normalize — PCA preserves structure better without re-normalization

    # Rescale to original SONAR scale
    x_hat = u_hat * mean_norm

    decoded = codec.decode(x_hat)
    print(f"\n--- d={d}, cumvar={cumvar[d-1].item():.3f} ---")
    for j, (idx, orig, dec) in enumerate(zip(test_idx, test_prompts, decoded)):
        re_emb = codec.encode([dec])
        cos = F.cosine_similarity(embeddings[idx:idx+1], re_emb).item()
        print(f"  [{idx}] cos={cos:.4f}")
        print(f"    orig: {orig[:80]}")
        print(f"    dec:  {dec[:80]}")

# --- Test: PCA projection with unit sphere normalization ---
print("\n" + "=" * 60)
print("PCA projection WITH unit sphere normalization (for SphericalSubspaceBO compatibility)")
print("=" * 60)

for d in [8, 16, 25]:
    if d > len(S):
        continue
    A_pca = Vt[:d].T  # [1024, d]

    # Project and normalize (like SphericalSubspaceBO does)
    v = F.normalize(unit_emb[test_idx] @ A_pca, p=2, dim=-1)  # [3, d] on S^(d-1)
    u_hat = F.normalize(v @ A_pca.T, p=2, dim=-1)  # lift back to S^1023
    x_hat = u_hat * mean_norm

    decoded = codec.decode(x_hat)
    print(f"\n--- d={d} (sphere-normalized) ---")
    for j, (idx, orig, dec) in enumerate(zip(test_idx, test_prompts, decoded)):
        re_emb = codec.encode([dec])
        cos = F.cosine_similarity(embeddings[idx:idx+1], re_emb).item()
        cos_unit = F.cosine_similarity(unit_emb[idx:idx+1], u_hat[j:j+1]).item()
        print(f"  [{idx}] cos_emb={cos:.4f} cos_unit={cos_unit:.4f}")
        print(f"    orig: {orig[:80]}")
        print(f"    dec:  {dec[:80]}")

# --- Test: Random QR vs PCA at d=16 ---
print("\n" + "=" * 60)
print("Comparison: Random QR vs PCA at d=16 (sphere-normalized)")
print("=" * 60)

torch.manual_seed(42)
A_rand = torch.linalg.qr(torch.randn(1024, 16))[0]
A_pca16 = Vt[:16].T

for name, A in [("Random QR", A_rand), ("PCA-16", A_pca16)]:
    cos_vals = []
    for i in range(len(prompts)):
        v = F.normalize(unit_emb[i:i+1] @ A, p=2, dim=-1)
        u_hat = F.normalize(v @ A.T, p=2, dim=-1)
        cos = F.cosine_similarity(unit_emb[i:i+1], u_hat).item()
        cos_vals.append(cos)
    print(f"{name}: mean_cos={np.mean(cos_vals):.4f}, min={np.min(cos_vals):.4f}, max={np.max(cos_vals):.4f}")

# --- Test: higher d with random QR ---
print("\n" + "=" * 60)
print("Random QR at higher d (sphere-normalized)")
print("=" * 60)

for d in [16, 32, 64, 128, 256]:
    torch.manual_seed(42)
    A_rand = torch.linalg.qr(torch.randn(1024, d))[0]
    cos_vals = []
    for i in range(len(prompts)):
        v = F.normalize(unit_emb[i:i+1] @ A_rand, p=2, dim=-1)
        u_hat = F.normalize(v @ A_rand.T, p=2, dim=-1)
        cos = F.cosine_similarity(unit_emb[i:i+1], u_hat).item()
        cos_vals.append(cos)

    # Decode a sample
    v_test = F.normalize(unit_emb[0:1] @ A_rand, p=2, dim=-1)
    u_test = F.normalize(v_test @ A_rand.T, p=2, dim=-1)
    x_test = u_test * mean_norm
    dec = codec.decode(x_test)[0]

    print(f"d={d:3d}: mean_cos={np.mean(cos_vals):.4f} | decode[0]: {dec[:70]}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
