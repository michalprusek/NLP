"""GP LOO Cross-Validation for subspace dimension selection.

Uses Geodesic Matérn kernel with ARD lengthscales. Per-dimension
lengthscales rescale inputs before re-normalizing to S^(d-1) and
computing geodesic distance. Dimensions the GP doesn't need get
large lengthscales → suppressed → LOO penalizes unnecessary dims.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/loocv_dim_selection.py
"""

import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/prusek/NLP")

import gpytorch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from shared.guacamol.codec import SELFIESVAECodec
from shared.guacamol.oracle import GuacaMolOracle

TASK = "adip"
DIMS = [8, 12, 16, 20, 24]
N_PROJECTIONS = 10
N_COLD_START = 100
SEED = 42


class GeodesicMaternARDKernel(gpytorch.kernels.Kernel):
    """Geodesic Matérn with per-dimension ARD lengthscales.

    1. Rescale: x_i' = x_i / ℓ_i
    2. Re-normalize: u = x' / ||x'||
    3. Geodesic distance: θ = arccos(u · v)
    4. Matérn(θ)

    Dimensions with large ℓ_i get suppressed after re-normalization,
    making the geodesic angle insensitive to them. This is the natural
    ARD extension of the isotropic geodesic Matérn.
    """

    has_lengthscale = True

    def __init__(self, nu: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"nu must be 0.5, 1.5, or 2.5, got {nu}")
        self.nu = nu

    def _rescale_and_normalize(self, x):
        """Apply ARD lengthscales and re-normalize to unit sphere."""
        x_scaled = x / self.lengthscale  # [N, d] / [1, d]
        return F.normalize(x_scaled, p=2, dim=-1)

    def forward(self, x1, x2, diag=False, **params):
        x1_n = self._rescale_and_normalize(x1)
        x2_n = self._rescale_and_normalize(x2)

        if diag:
            cos_sim = (x1_n * x2_n).sum(dim=-1)
        else:
            cos_sim = x1_n @ x2_n.transpose(-2, -1)

        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        dist = torch.arccos(cos_sim)

        if self.nu == 0.5:
            return torch.exp(-dist)
        elif self.nu == 1.5:
            s = math.sqrt(3) * dist
            return (1.0 + s) * torch.exp(-s)
        else:  # 2.5
            s = math.sqrt(5) * dist
            return (1.0 + s + s.pow(2) / 3.0) * torch.exp(-s)


def analytical_loo(model, X, Y):
    """Analytical LOO log-likelihood (Rasmussen & Williams eq 5.12)."""
    model.eval()
    with torch.no_grad():
        K = model.covar_module(X).evaluate()
        noise = model.likelihood.noise.item()
        K_noisy = K + noise * torch.eye(len(X), dtype=X.dtype, device=X.device)

        K_inv = torch.linalg.inv(K_noisy)
        alpha = K_inv @ Y

        diag = K_inv.diag()
        loo_mean = Y - alpha / diag
        loo_var = 1.0 / diag

        residual = Y - loo_mean
        ll = -0.5 * torch.log(2 * torch.pi * loo_var) - 0.5 * residual**2 / loo_var

    return ll.sum().item(), ll.std().item(), ll.cpu().numpy()


def fit_gp_geodesic_ard(V, Y, nu=1.5):
    """Fit GP with Geodesic Matérn ARD kernel."""
    d = V.shape[-1]
    base = GeodesicMaternARDKernel(nu=nu, ard_num_dims=d)
    covar = gpytorch.kernels.ScaleKernel(base)
    model = SingleTaskGP(V.double(), Y.double().unsqueeze(-1), covar_module=covar)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading ZINC molecules...")
    with open("/home/prusek/NLP/datasets/zinc/zinc_all.txt") as f:
        all_smiles = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(all_smiles)} molecules")

    print(f"Scoring on {TASK}...")
    oracle = GuacaMolOracle(task_id=TASK)
    scores = oracle.score_batch(all_smiles)
    scores_t = torch.tensor(scores, dtype=torch.float32)

    # Top-100 (score-conditioned)
    sorted_idx = scores_t.argsort(descending=True)
    top_idx = sorted_idx[:N_COLD_START]
    top_smiles = [all_smiles[i] for i in top_idx]
    top_scores = scores_t[top_idx]
    print(f"Top-{N_COLD_START} score range: [{top_scores[-1]:.4f}, {top_scores[0]:.4f}]")

    # Encode
    print("Loading VAE codec...")
    codec = SELFIESVAECodec.from_pretrained(device=device)

    print("Encoding top molecules...")
    with torch.no_grad():
        top_emb = codec.encode(top_smiles).to(device)
    top_U = F.normalize(top_emb, p=2, dim=-1)
    top_Y = top_scores.to(device)

    # Standardize Y
    top_Y_std = (top_Y - top_Y.mean()) / (top_Y.std() + 1e-8)

    U, Y = top_U, top_Y_std

    print(f"\n{'='*70}")
    print(f"LOO CV: Geodesic Matérn ARD — TOP-100 on {TASK}")
    print(f"{'='*70}")
    print(f"{'d':>4} | {'LOO LL (mean±std)':>25} | {'per-proj LOO LLs'}")
    print("-" * 90)

    results = {}
    for d in DIMS:
        proj_lls = []
        t0 = time.time()
        for k in range(N_PROJECTIONS):
            torch.manual_seed(SEED + k * 1000)
            A, _ = torch.linalg.qr(torch.randn(U.shape[1], d, device=device))
            V = F.normalize(U @ A, p=2, dim=-1)

            try:
                model = fit_gp_geodesic_ard(V, Y, nu=1.5)
                ll_sum, ll_std, _ = analytical_loo(model, V.double(), Y.double())
                proj_lls.append(ll_sum)

                # Extract ARD lengthscales for the last projection
                if k == 0:
                    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().squeeze()
                    ls_str = ", ".join(f"{x:.2f}" for x in sorted(ls.tolist()))
            except Exception as e:
                print(f"  d={d}, proj {k}: GP fit failed: {e}")
                proj_lls.append(float("nan"))

        elapsed = time.time() - t0
        valid = [x for x in proj_lls if not np.isnan(x)]
        if valid:
            mean_ll = np.mean(valid)
            std_ll = np.std(valid)
            ll_str = ", ".join(f"{x:.1f}" for x in proj_lls[:6])
            if len(proj_lls) > 6:
                ll_str += "..."
            print(f"{d:4d} | {mean_ll:10.1f} ± {std_ll:5.1f}      | {ll_str}  ({elapsed:.1f}s)")
            print(f"       lengthscales (proj 0, sorted): [{ls_str}]")
            results[d] = (mean_ll, std_ll)
        else:
            print(f"{d:4d} | FAILED")

    if results:
        best_d = max(results, key=lambda d: results[d][0])
        print(f"\n>>> Best d = {best_d}  (LOO LL = {results[best_d][0]:.1f} ± {results[best_d][1]:.1f})")
        print(f">>> All: {', '.join(f'd={d}: {results[d][0]:.1f}' for d in DIMS if d in results)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
