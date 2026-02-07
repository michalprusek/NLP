"""Debug SONAR round-trip fidelity through subspace projection pipeline.

Tests where the signal is lost:
1. SONAR encode→decode (no projection)
2. SONAR encode→normalize→denormalize→decode (sphere mapping)
3. SONAR encode→normalize→project→lift→denormalize→decode (full pipeline)
4. GP candidate generation→decode (what BO actually does)
"""

import torch
import torch.nn.functional as F
import numpy as np

# --- 1. Load SONAR ---
print("=" * 60)
print("Loading SONAR codec on CPU...")
from rielbo_gsm8k.sonar_codec import SonarCodec
codec = SonarCodec(device="cpu")

# --- 2. Test prompts ---
prompts = [
    "Let's think step by step.",
    "Show your work and give the final answer.",
    "Read the problem carefully. Set up an equation and solve it step by step. State your final answer as a number.",
    "Solve this math word problem. Be precise with your arithmetic.",
    "Break this problem into clear steps:\n1. Identify the given information\n2. Determine what needs to be found\n3. Set up the calculation\n4. Solve step by step\n5. Verify your answer",
]

# --- 3. Encode ---
print("\n" + "=" * 60)
print("STEP 1: SONAR encode/decode round-trip (no projection)")
print("=" * 60)
with torch.no_grad():
    embeddings = codec.encode(prompts)  # [5, 1024]

print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding norms: {embeddings.norm(dim=-1).tolist()}")
print(f"Embedding min/max: {embeddings.min().item():.4f} / {embeddings.max().item():.4f}")
print(f"Pairwise cosine similarities:")
for i in range(len(prompts)):
    for j in range(i+1, len(prompts)):
        cos = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
        print(f"  [{i}] vs [{j}]: {cos:.4f}")

# Direct round-trip
decoded = codec.decode(embeddings)
print(f"\nDirect round-trip:")
for i, (orig, dec) in enumerate(zip(prompts, decoded)):
    re_emb = codec.encode([dec])
    cos = F.cosine_similarity(embeddings[i:i+1], re_emb).item()
    print(f"  [{i}] cos={cos:.4f}")
    print(f"    orig: {orig[:80]}")
    print(f"    dec:  {dec[:80]}")

# --- 4. Normalize to unit sphere ---
print("\n" + "=" * 60)
print("STEP 2: Unit sphere normalization → decode")
print("=" * 60)
norms = embeddings.norm(dim=-1)
mean_norm = norms.mean().item()
print(f"Mean norm: {mean_norm:.4f}")
print(f"Norm range: [{norms.min().item():.4f}, {norms.max().item():.4f}]")

unit_emb = F.normalize(embeddings, p=2, dim=-1)  # project to S^1023
rescaled = unit_emb * mean_norm  # rescale back with mean norm

decoded_rescaled = codec.decode(rescaled)
print(f"\nUnit sphere → rescale(mean_norm={mean_norm:.4f}) → decode:")
for i, (orig, dec) in enumerate(zip(prompts, decoded_rescaled)):
    re_emb = codec.encode([dec])
    cos = F.cosine_similarity(embeddings[i:i+1], re_emb).item()
    print(f"  [{i}] cos={cos:.4f}")
    print(f"    orig: {orig[:80]}")
    print(f"    dec:  {dec[:80]}")

# --- 5. Subspace projection round-trip ---
print("\n" + "=" * 60)
print("STEP 3: Subspace projection (1024D → 16D → 1024D) → decode")
print("=" * 60)

D, d = 1024, 16
torch.manual_seed(42)
A_raw = torch.randn(D, d)
A, _ = torch.linalg.qr(A_raw)  # orthonormal [D, d]

# Project: u → v = normalize(u @ A)
v = F.normalize(unit_emb @ A, p=2, dim=-1)  # [5, 16] on S^15
print(f"Subspace embedding shape: {v.shape}")
print(f"Subspace norms: {v.norm(dim=-1).tolist()}")  # should be 1.0

# Lift: v → u_hat = normalize(v @ A.T)
u_hat = F.normalize(v @ A.T, p=2, dim=-1)  # [5, 1024] back on S^1023

# Check how much info we lost in projection
for i in range(len(prompts)):
    cos = F.cosine_similarity(unit_emb[i:i+1], u_hat[i:i+1]).item()
    print(f"  [{i}] projection cos(original, lifted): {cos:.4f}")

# Rescale and decode
x_hat = u_hat * mean_norm
decoded_proj = codec.decode(x_hat)
print(f"\nSubspace projection → rescale → decode:")
for i, (orig, dec) in enumerate(zip(prompts, decoded_proj)):
    re_emb = codec.encode([dec])
    cos_orig = F.cosine_similarity(embeddings[i:i+1], re_emb).item()
    print(f"  [{i}] cos_to_orig={cos_orig:.4f}")
    print(f"    orig: {orig[:80]}")
    print(f"    dec:  {dec[:80]}")

# --- 6. Test with individual norms instead of mean_norm ---
print("\n" + "=" * 60)
print("STEP 4: Subspace projection with INDIVIDUAL norms → decode")
print("=" * 60)
x_hat_ind = u_hat * norms.unsqueeze(-1)  # per-sample norm
decoded_ind = codec.decode(x_hat_ind)
print(f"Individual norms → decode:")
for i, (orig, dec) in enumerate(zip(prompts, decoded_ind)):
    re_emb = codec.encode([dec])
    cos_orig = F.cosine_similarity(embeddings[i:i+1], re_emb).item()
    print(f"  [{i}] cos_to_orig={cos_orig:.4f} (norm={norms[i].item():.4f})")
    print(f"    orig: {orig[:80]}")
    print(f"    dec:  {dec[:80]}")

# --- 7. Test what random points on the subspace decode to ---
print("\n" + "=" * 60)
print("STEP 5: Random subspace points → decode (what BO candidates look like)")
print("=" * 60)
torch.manual_seed(123)
n_random = 10
v_random = F.normalize(torch.randn(n_random, d), p=2, dim=-1)  # random S^15
u_random = F.normalize(v_random @ A.T, p=2, dim=-1)  # lift to S^1023
x_random = u_random * mean_norm  # rescale

decoded_random = codec.decode(x_random)
print(f"Random subspace points (scaled by mean_norm={mean_norm:.4f}):")
for i, dec in enumerate(decoded_random):
    print(f"  [{i}] {dec[:100]}")

# --- 8. Perturbation test: small steps from a good prompt ---
print("\n" + "=" * 60)
print("STEP 6: Small perturbations from 'Let's think step by step.'")
print("=" * 60)
base_v = v[0:1]  # "Let's think step by step." in subspace
for eps in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
    noise = torch.randn(1, d) * eps
    v_pert = F.normalize(base_v + noise, p=2, dim=-1)
    u_pert = F.normalize(v_pert @ A.T, p=2, dim=-1)
    x_pert = u_pert * mean_norm
    dec = codec.decode(x_pert)[0]
    cos_sub = F.cosine_similarity(base_v, v_pert).item()
    cos_full = F.cosine_similarity(unit_emb[0:1], u_pert).item()
    print(f"  eps={eps:.2f} cos_sub={cos_sub:.4f} cos_full={cos_full:.4f} → {dec[:80]}")

# --- 9. Check: what fraction of info is in the 16D subspace? ---
print("\n" + "=" * 60)
print("STEP 7: Variance explained by subspace")
print("=" * 60)
# Project all seed embeddings and check reconstruction error
proj_component = unit_emb @ A @ A.T  # projection onto subspace
residual = unit_emb - proj_component
var_explained = 1 - (residual.norm(dim=-1)**2 / unit_emb.norm(dim=-1)**2)
print(f"Variance explained per sample: {var_explained.tolist()}")
print(f"Mean variance explained: {var_explained.mean().item():.4f}")
print(f"Expected for random 16/1024: {16/1024:.4f}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
