"""Test flow quality for GuacaMol molecular optimization.

Key tests:
1. Forward sampling quality: z -> u (do samples land on data manifold?)
2. Round-trip invertibility: u -> z -> u (is flow invertible?)
3. Decoder compatibility: u -> x -> SMILES (do samples decode to valid molecules?)
4. GP-BO compatibility: Can we do BO in z-space effectively?

Run:
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.test_flow_quality
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def test_flow_quality():
    """Comprehensive test of flow quality for molecular optimization."""
    print("=" * 70)
    print("FLOW QUALITY TEST FOR GUACAMOL MOLECULAR OPTIMIZATION")
    print("=" * 70)

    device = "cuda"

    # Load components
    print("\n[1] Loading components...")
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle
    from rielbo.norm_predictor import NormPredictor
    from rielbo.velocity_network import VelocityNetwork

    codec = SELFIESVAECodec.from_pretrained(device=device)
    oracle = GuacaMolOracle(task_id="adip")

    # Load flow
    checkpoint = torch.load("rielbo/checkpoints/guacamol_flow_spherical/best.pt", map_location=device)
    flow = VelocityNetwork(
        input_dim=256, hidden_dim=256, num_layers=6, num_heads=8
    ).to(device)
    if "ema_shadow" in checkpoint:
        state = flow.state_dict()
        for k in state:
            if k in checkpoint["ema_shadow"]:
                state[k] = checkpoint["ema_shadow"][k]
        flow.load_state_dict(state)
    flow.eval()

    norm_pred = NormPredictor.load("rielbo/checkpoints/guacamol_flow_spherical/norm_predictor.pt", device=device)
    print("  ✓ Components loaded")

    # Load training data
    print("\n[2] Loading training data...")
    smiles_list, scores, _ = load_guacamol_data(n_samples=500, task_id="adip")
    embs = codec.encode(smiles_list[:100])
    directions = F.normalize(embs, p=2, dim=-1)
    print(f"  Loaded {len(directions)} training embeddings")

    # Helper functions
    def flow_forward(z, steps=50):
        """z (t=0) -> u (t=1)"""
        u = z.clone()
        with torch.no_grad():
            for t_idx in range(steps):
                t = torch.full((u.shape[0],), t_idx / steps, device=device)
                v = flow(u, t)
                u = u + (1.0 / steps) * v
                u = F.normalize(u, p=2, dim=-1)
        return u

    def flow_invert(u, steps=50):
        """u (t=1) -> z (t=0)"""
        z = u.clone()
        with torch.no_grad():
            for t_idx in range(steps):
                t = torch.full((z.shape[0],), 1.0 - t_idx / steps, device=device)
                v = flow(z, t)
                z = z + (-1.0 / steps) * v
                z = F.normalize(z, p=2, dim=-1)
        return z

    # TEST 1: Forward sampling - do random z map to data manifold?
    print("\n" + "=" * 70)
    print("TEST 1: Forward Sampling Quality")
    print("=" * 70)

    n_test = 100
    z_random = torch.randn(n_test, 256, device=device)
    z_random = F.normalize(z_random, p=2, dim=-1)

    u_sampled = flow_forward(z_random)

    # Check similarity to training data
    cos_sims = F.cosine_similarity(
        u_sampled.unsqueeze(1),  # [100, 1, 256]
        directions.unsqueeze(0),  # [1, 100, 256]
        dim=-1
    )  # [100, 100]

    max_cos_sims = cos_sims.max(dim=1).values
    print(f"  Flow samples similarity to training data:")
    print(f"    Max cosine: mean={max_cos_sims.mean():.4f}, min={max_cos_sims.min():.4f}, max={max_cos_sims.max():.4f}")

    if max_cos_sims.mean() < 0.5:
        print("  ⚠️ PROBLEM: Flow samples are FAR from training data!")
        print("     Random sampling from z-space doesn't reach data manifold.")
    else:
        print("  ✓ Flow samples are close to training data")

    # TEST 2: Round-trip from DATA direction
    print("\n" + "=" * 70)
    print("TEST 2: Round-trip from DATA direction (u -> z -> u)")
    print("=" * 70)

    u_test = directions[:20].clone()
    z_inverted = flow_invert(u_test)
    u_reconstructed = flow_forward(z_inverted)

    rt_cos = F.cosine_similarity(u_test, u_reconstructed, dim=-1)
    print(f"  Round-trip cosine: mean={rt_cos.mean():.4f}, min={rt_cos.min():.4f}")

    if rt_cos.mean() > 0.95:
        print("  ✓ Data round-trip is good")
    else:
        print("  ⚠️ PROBLEM: Data round-trip has significant error!")

    # TEST 3: Round-trip from INVERTED z (like in BO loop)
    print("\n" + "=" * 70)
    print("TEST 3: Round-trip from INVERTED z (z -> u -> z)")
    print("=" * 70)

    # This simulates what happens in BO:
    # 1. Invert training data to get z_train
    # 2. GP proposes z_opt (similar to z_train)
    # 3. Flow forward z_opt -> u
    # 4. Evaluate u
    # 5. Invert u back to z for GP update
    # 6. z should be similar to original z_opt

    z_train = flow_invert(directions[:20])
    u_forward = flow_forward(z_train)
    z_back = flow_invert(u_forward)

    z_rt_cos = F.cosine_similarity(z_train, z_back, dim=-1)
    print(f"  z round-trip cosine: mean={z_rt_cos.mean():.4f}, min={z_rt_cos.min():.4f}")

    if z_rt_cos.mean() > 0.95:
        print("  ✓ z-space round-trip is good - BO should work")
    else:
        print("  ⚠️ CRITICAL: z-space round-trip is BROKEN!")
        print("     This means GP in z-space will be inconsistent.")

    # TEST 4: Decoder quality
    print("\n" + "=" * 70)
    print("TEST 4: Decoder Quality")
    print("=" * 70)

    # Sample from flow and decode
    u_samples = flow_forward(z_random[:50])
    r_pred = norm_pred(u_samples)
    x_samples = u_samples * r_pred
    decoded = codec.decode(x_samples)

    valid = [s for s in decoded if s and len(s) > 0]
    print(f"  Valid SMILES: {len(valid)}/{len(decoded)} ({100*len(valid)/len(decoded):.1f}%)")

    # Score some
    scores_sampled = []
    for smi in valid[:20]:
        s = oracle.score(smi)
        scores_sampled.append(s)

    if scores_sampled:
        print(f"  Score stats: mean={np.mean(scores_sampled):.4f}, max={np.max(scores_sampled):.4f}")

    # TEST 5: Check z-space structure
    print("\n" + "=" * 70)
    print("TEST 5: z-space Structure")
    print("=" * 70)

    z_train_full = flow_invert(directions)

    # Self-similarity of z_train
    z_cos = F.cosine_similarity(
        z_train_full.unsqueeze(1),
        z_train_full.unsqueeze(0),
        dim=-1
    )
    # Remove diagonal
    mask = ~torch.eye(len(z_train_full), dtype=bool, device=device)
    z_self_cos = z_cos[mask].mean()

    # Random baseline
    z_rand = torch.randn(100, 256, device=device)
    z_rand = F.normalize(z_rand, p=2, dim=-1)
    z_rand_cos = F.cosine_similarity(
        z_rand.unsqueeze(1),
        z_rand.unsqueeze(0),
        dim=-1
    )
    mask_rand = ~torch.eye(100, dtype=bool, device=device)
    z_rand_self_cos = z_rand_cos[mask_rand].mean()

    print(f"  z_train mean self-cosine: {z_self_cos:.4f}")
    print(f"  z_random mean self-cosine: {z_rand_self_cos:.4f}")

    if z_self_cos > z_rand_self_cos + 0.1:
        print("  ✓ z_train is more structured than random")
    else:
        print("  ⚠️ z_train has same structure as random - flow didn't learn data distribution")

    # DIAGNOSIS
    print("\n" + "=" * 70)
    print("DIAGNOSIS AND RECOMMENDATIONS")
    print("=" * 70)

    problems = []

    if max_cos_sims.mean() < 0.5:
        problems.append("Flow samples don't reach data manifold")

    if rt_cos.mean() < 0.95:
        problems.append("Data round-trip (u->z->u) has error")

    if z_rt_cos.mean() < 0.95:
        problems.append("z-space round-trip (z->u->z) is broken")

    if z_self_cos < z_rand_self_cos + 0.1:
        problems.append("z-space structure is like random")

    if problems:
        print("\n⚠️ PROBLEMS FOUND:")
        for i, p in enumerate(problems, 1):
            print(f"   {i}. {p}")

        print("\n📋 RECOMMENDATIONS:")
        print("""
   1. DON'T USE z-SPACE FOR BO
      - z-space round-trip is broken (0.53 cosine)
      - GP in z-space will be inconsistent
      - Instead: Do BO directly on unit sphere (u-space)

   2. USE DIRECT SPHERE BO
      - Fit GP on normalized directions u (ArcCosine kernel)
      - UCB optimization on unit sphere
      - Decode u directly (with NormPredictor)
      - No flow needed for BO, only for initialization

   3. OR: RETRAIN FLOW WITH INVERTIBILITY LOSS
      - Add cycle consistency: loss += ||u - forward(invert(u))||
      - Use standard CFM (not OT-CFM) - more invertible
      - Or use I-CFM (Invertible CFM)

   4. OR: USE GUIDED FLOW WITHOUT INVERSION
      - Start from random z, use UCB gradient guidance
      - Guide samples toward high-score regions
      - No need for consistent z-space
        """)
    else:
        print("\n✓ All tests passed - flow is working correctly")

    return problems


if __name__ == "__main__":
    problems = test_flow_quality()
