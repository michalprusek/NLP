"""Debug script for GuacaMol RieLBO optimization."""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def debug_pipeline():
    """Debug the full RieLBO pipeline."""

    print("=" * 60)
    print("DEBUGGING RieLBO GuacaMol Pipeline")
    print("=" * 60)

    device = "cuda"

    # 1. Load components
    print("\n[1] Loading components...")

    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle
    from rielbo.norm_predictor import NormPredictor
    from rielbo.velocity_network import VelocityNetwork

    codec = SELFIESVAECodec.from_pretrained(device=device)
    oracle = GuacaMolOracle(task_id="adip")

    # Load flow
    checkpoint = torch.load("rielbo/checkpoints/guacamol_flow_spherical/best.pt", map_location=device, weights_only=False)
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

    print("  ✓ All components loaded")

    # 2. Test encoding/decoding round-trip
    print("\n[2] Testing codec round-trip...")
    test_smiles = ["CCO", "CCCC", "c1ccccc1", "CC(=O)O"]
    embs = codec.encode(test_smiles)
    decoded = codec.decode(embs)
    print(f"  Original: {test_smiles}")
    print(f"  Decoded:  {decoded}")
    print(f"  Embedding norms: {embs.norm(dim=-1).tolist()}")

    # 3. Test NormPredictor accuracy
    print("\n[3] Testing NormPredictor accuracy...")
    smiles_list, scores, _ = load_guacamol_data(n_samples=100, task_id="adip")
    embs = codec.encode(smiles_list[:20])
    true_norms = embs.norm(dim=-1)
    directions = F.normalize(embs, p=2, dim=-1)
    pred_norms = norm_pred(directions).squeeze()

    print(f"  True norms:  min={true_norms.min():.2f}, max={true_norms.max():.2f}, mean={true_norms.mean():.2f}")
    print(f"  Pred norms:  min={pred_norms.min():.2f}, max={pred_norms.max():.2f}, mean={pred_norms.mean():.2f}")
    print(f"  MAE: {(pred_norms - true_norms).abs().mean():.4f}")
    print(f"  MAPE: {((pred_norms - true_norms).abs() / true_norms).mean() * 100:.2f}%")

    # 4. Test flow forward/backward
    print("\n[4] Testing flow forward/backward consistency...")

    # Take a real embedding, invert, forward, check reconstruction
    test_emb = embs[0:1]  # [1, 256]
    test_dir = F.normalize(test_emb, p=2, dim=-1)

    # Invert: direction at t=1 -> z at t=0
    z = test_dir.clone()
    with torch.no_grad():
        for t_idx in range(50):
            t = torch.full((1,), 1.0 - t_idx / 50, device=device)
            v = flow(z, t)
            z = z + (-1.0 / 50) * v
            z = F.normalize(z, p=2, dim=-1)

    print(f"  After inversion: z_norm = {z.norm().item():.4f}")

    # Forward: z at t=0 -> direction at t=1
    x_rec = z.clone()
    with torch.no_grad():
        for t_idx in range(50):
            t = torch.full((1,), t_idx / 50, device=device)
            v = flow(x_rec, t)
            x_rec = x_rec + (1.0 / 50) * v
            x_rec = F.normalize(x_rec, p=2, dim=-1)

    # Check cosine similarity
    cos_sim = F.cosine_similarity(test_dir, x_rec, dim=-1).item()
    l2_error = (test_dir - x_rec).norm().item()
    print(f"  Round-trip cosine similarity: {cos_sim:.4f}")
    print(f"  Round-trip L2 error: {l2_error:.4f}")

    # 5. Test flow sampling quality
    print("\n[5] Testing flow sample quality...")

    # Sample from flow
    n_samples = 100
    z_samples = torch.randn(n_samples, 256, device=device)
    z_samples = F.normalize(z_samples, p=2, dim=-1)

    with torch.no_grad():
        for t_idx in range(50):
            t = torch.full((n_samples,), t_idx / 50, device=device)
            v = flow(z_samples, t)
            z_samples = z_samples + (1.0 / 50) * v
            z_samples = F.normalize(z_samples, p=2, dim=-1)

    # Predict norms and reconstruct
    pred_norms_samples = norm_pred(z_samples)
    full_embs = z_samples * pred_norms_samples

    print(f"  Sampled directions norm: {z_samples.norm(dim=-1).mean():.4f} (should be 1.0)")
    print(f"  Predicted norms: mean={pred_norms_samples.mean():.2f}, std={pred_norms_samples.std():.2f}")
    print(f"  Full embedding norms: mean={full_embs.norm(dim=-1).mean():.2f}")

    # Decode and check validity
    decoded_samples = codec.decode(full_embs)
    valid_count = sum(1 for s in decoded_samples if s and len(s) > 0)
    print(f"  Valid decoded SMILES: {valid_count}/{n_samples}")

    # Score some samples
    print("\n[6] Scoring flow samples...")
    scores_samples = []
    for i, smi in enumerate(decoded_samples[:20]):
        if smi:
            score = oracle.score(smi)
            scores_samples.append(score)
            if i < 5:
                print(f"  {smi[:50]}: {score:.4f}")

    if scores_samples:
        print(f"  Score stats: mean={np.mean(scores_samples):.4f}, max={np.max(scores_samples):.4f}")

    # 7. Check GP acquisition landscape
    print("\n[7] Analyzing acquisition optimization...")

    # Load some training data and fit GP
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition import UpperConfidenceBound

    # Get top molecules
    smiles_list, scores_tensor, _ = load_guacamol_data(n_samples=100, task_id="adip")
    embs = codec.encode(smiles_list)
    directions = F.normalize(embs, p=2, dim=-1)

    # Fit GP
    train_X = directions.double().to(device)
    train_Y = scores_tensor.double().unsqueeze(-1).to(device)

    gp = SingleTaskGP(train_X, train_Y).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()

    # Check GP predictions on training data
    with torch.no_grad():
        pred = gp.posterior(train_X)
        pred_mean = pred.mean.squeeze()
        pred_std = pred.variance.sqrt().squeeze()

    print(f"  GP train predictions: mean_error={((pred_mean - train_Y.squeeze()).abs()).mean():.4f}")
    print(f"  GP uncertainty: mean_std={pred_std.mean():.4f}")

    # Check UCB on flow samples
    ucb = UpperConfidenceBound(gp, beta=4.0)
    with torch.no_grad():
        flow_dirs = z_samples.double()
        ucb_values = ucb(flow_dirs.unsqueeze(1))

    print(f"  UCB on flow samples: mean={ucb_values.mean():.4f}, max={ucb_values.max():.4f}")

    # Compare UCB of best training point vs flow samples
    best_train_idx = train_Y.argmax()
    best_train_ucb = ucb(train_X[best_train_idx:best_train_idx+1].unsqueeze(1))
    print(f"  UCB of best training point: {best_train_ucb.item():.4f}")

    # 8. Key diagnostic: Are flow samples near training data?
    print("\n[8] Checking flow sample proximity to training data...")

    # Cosine similarity between flow samples and training data
    cos_sims = F.cosine_similarity(
        flow_dirs.float().unsqueeze(1),  # [100, 1, 256]
        directions.unsqueeze(0),  # [1, 100, 256]
        dim=-1
    )  # [100, 100]

    max_cos_sims = cos_sims.max(dim=1).values
    print(f"  Flow samples max cosine to training: mean={max_cos_sims.mean():.4f}, min={max_cos_sims.min():.4f}")

    # This tells us if flow is generating samples in the right region
    if max_cos_sims.mean() < 0.5:
        print("  ⚠️ WARNING: Flow samples are FAR from training data!")
        print("     This suggests flow may not be generating on-manifold samples.")
    else:
        print("  ✓ Flow samples are reasonably close to training data.")

    # 9. Check z-space distribution (inverted training data vs random)
    print("\n[9] Checking z-space distribution...")

    # Invert training directions to z-space
    print("  Inverting training directions to z-space...")
    z_train = []
    for i in tqdm(range(len(directions)), desc="  Inverting"):
        z = directions[i:i+1].clone()
        with torch.no_grad():
            for t_idx in range(50):
                t = torch.full((1,), 1.0 - t_idx / 50, device=device)
                v = flow(z, t)
                z = z + (-1.0 / 50) * v
                z = F.normalize(z, p=2, dim=-1)
        z_train.append(z.squeeze(0))
    z_train = torch.stack(z_train)

    # Compare z_train distribution to random z
    z_random = torch.randn(100, 256, device=device)
    z_random = F.normalize(z_random, p=2, dim=-1)

    # Self-similarity of z_train
    z_train_cos = F.cosine_similarity(
        z_train.unsqueeze(1),
        z_train.unsqueeze(0),
        dim=-1
    )
    z_train_mean_cos = (z_train_cos.sum() - z_train_cos.diag().sum()) / (len(z_train) * (len(z_train) - 1))

    # Similarity between z_train and z_random
    z_cross_cos = F.cosine_similarity(
        z_train.unsqueeze(1),
        z_random.unsqueeze(0),
        dim=-1
    )
    z_cross_mean = z_cross_cos.mean()

    print(f"  z_train mean self-cosine: {z_train_mean_cos:.4f}")
    print(f"  z_train-z_random mean cosine: {z_cross_mean:.4f}")

    # Check if z_train is concentrated (high self-similarity) or spread like z_random
    if z_train_mean_cos > 0.3:
        print("  ✓ z_train is concentrated (good - training data forms a cluster in z-space)")
    else:
        print("  ⚠️ z_train is spread like random (flow may not have learned data structure)")

    # 10. Test if flow preserves GP-relevant structure
    print("\n[10] Checking flow round-trip for GP consistency...")

    # Take some z_train points, flow forward, then invert back
    test_z = z_train[:10]
    z_roundtrip = []
    for z_i in test_z:
        # Forward: z → u
        u = z_i.unsqueeze(0).clone()
        with torch.no_grad():
            for t_idx in range(50):
                t = torch.full((1,), t_idx / 50, device=device)
                v = flow(u, t)
                u = u + (1.0 / 50) * v
                u = F.normalize(u, p=2, dim=-1)

        # Invert: u → z
        z_back = u.clone()
        with torch.no_grad():
            for t_idx in range(50):
                t = torch.full((1,), 1.0 - t_idx / 50, device=device)
                v = flow(z_back, t)
                z_back = z_back + (-1.0 / 50) * v
                z_back = F.normalize(z_back, p=2, dim=-1)

        z_roundtrip.append(z_back.squeeze(0))

    z_roundtrip = torch.stack(z_roundtrip)
    roundtrip_cos = F.cosine_similarity(test_z, z_roundtrip, dim=-1)
    print(f"  z → u → z round-trip cosine: mean={roundtrip_cos.mean():.4f}, min={roundtrip_cos.min():.4f}")

    if roundtrip_cos.mean() > 0.99:
        print("  ✓ Flow is invertible - GP will be consistent")
    else:
        print("  ⚠️ Flow is NOT well invertible - GP may have inconsistencies")

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    debug_pipeline()
