"""Test full BO pipeline with SONAR + PCA projection (no LLM eval).

Simulates what SphericalSubspaceBOv2 does:
1. Encode seed prompts with SONAR
2. Cold start with PCA projection
3. Run GP + acquisition → get candidate embedding
4. Decode with SONAR → check if math-related prompt
"""

import torch
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("Loading SONAR codec...")
from rielbo_gsm8k.sonar_codec import SonarCodec
codec = SonarCodec(device="cpu")

# Seed prompts + scores from previous run
import json
with open("rielbo_gsm8k/results/seed_cache_s42.json") as f:
    cached = json.load(f)

seed_prompts = [e["prompt"] for e in cached]
seed_scores = torch.tensor([e["score"] for e in cached], dtype=torch.float32)

print(f"Loaded {len(seed_prompts)} seeds, best: {seed_scores.max():.4f}")

# --- Test with PCA projection ---
print("\n" + "=" * 60)
print("Testing SubspaceBOv2 with PCA projection (no oracle calls)")
print("=" * 60)

from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config

# Dummy oracle that returns 0
class DummyOracle:
    def score(self, text):
        return 0.0

config = V2Config.from_preset("explore")
config.projection_type = "pca"
config.ur_std_low = 0.05

optimizer = SphericalSubspaceBOv2(
    codec=codec,
    oracle=DummyOracle(),
    input_dim=1024,
    subspace_dim=16,
    config=config,
    device="cpu",
    n_candidates=2000,
    acqf="ts",
    trust_region=0.8,
    seed=42,
)

optimizer.cold_start(seed_prompts, seed_scores)

# Now simulate BO steps — only test GP + decode, skip oracle
print("\n--- Simulating 20 BO steps (dummy scores) ---")
for i in range(20):
    # Get acquisition candidate
    u_opt, acq_diag = optimizer._optimize_acquisition()
    x_opt, norm_diag = optimizer._reconstruct_embedding(u_opt)

    # Decode
    decoded = codec.decode(x_opt)
    prompt = decoded[0] if decoded else ""

    # Round-trip check
    rt_cos = 0.0
    if prompt:
        re_emb = codec.encode([prompt])
        rt_cos = F.cosine_similarity(x_opt, re_emb.to(x_opt.device)).item()

    # Check if duplicate
    is_dup = prompt in optimizer.smiles_observed
    is_math = any(w in prompt.lower() for w in
                  ["math", "solve", "step", "answer", "number", "calculate",
                   "equation", "problem", "work", "computation", "arithmetic"])

    print(
        f"[{i+1:2d}] gp_mean={acq_diag.get('gp_mean',0):.3f} "
        f"gp_std={acq_diag.get('gp_std',0):.3f} "
        f"rt_cos={rt_cos:.3f} dup={is_dup} math={is_math}"
    )
    print(f"     → {prompt[:100]}")

    if is_dup or not prompt:
        # Skip — don't add to training data
        optimizer._update_ur_tr(gp_std=acq_diag.get("gp_std", 0))
        continue

    # Add to training data with dummy score (use GP prediction as proxy)
    score = acq_diag.get("gp_mean", 0.5)  # use GP mean as fake score
    optimizer.train_X = torch.cat([optimizer.train_X, x_opt], dim=0)
    optimizer.train_U = torch.cat([optimizer.train_U, u_opt], dim=0)
    optimizer.train_Y = torch.cat([
        optimizer.train_Y,
        torch.tensor([score], device=optimizer.device, dtype=torch.float32)
    ])
    optimizer.smiles_observed.append(prompt)

    if i % 10 == 9:
        optimizer._fit_gp()

    optimizer._update_ur_tr(gp_std=acq_diag.get("gp_std", 0))

print("\n" + "=" * 60)
print("Comparison: Random QR projection")
print("=" * 60)

config_rand = V2Config.from_preset("explore")
config_rand.projection_type = "random"
config_rand.ur_std_low = 0.05

optimizer_rand = SphericalSubspaceBOv2(
    codec=codec,
    oracle=DummyOracle(),
    input_dim=1024,
    subspace_dim=16,
    config=config_rand,
    device="cpu",
    n_candidates=2000,
    acqf="ts",
    trust_region=0.8,
    seed=42,
)

optimizer_rand.cold_start(seed_prompts, seed_scores)

print("\n--- Simulating 10 BO steps with Random QR ---")
for i in range(10):
    u_opt, acq_diag = optimizer_rand._optimize_acquisition()
    x_opt, norm_diag = optimizer_rand._reconstruct_embedding(u_opt)
    decoded = codec.decode(x_opt)
    prompt = decoded[0] if decoded else ""
    rt_cos = 0.0
    if prompt:
        re_emb = codec.encode([prompt])
        rt_cos = F.cosine_similarity(x_opt, re_emb.to(x_opt.device)).item()
    print(f"[{i+1:2d}] rt_cos={rt_cos:.3f} → {prompt[:100]}")
    optimizer_rand._update_ur_tr(gp_std=acq_diag.get("gp_std", 0))

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
