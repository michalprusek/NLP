"""Run RieLBO on GuacaMol with Direct Sphere BO (no flow).

Direct approach without flow model:
- GP operates directly on unit sphere (directions)
- NormPredictor recovers magnitude
- No z-space inversion needed

This avoids the broken z-space round-trip problem.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_direct \
        --norm-predictor rielbo/checkpoints/guacamol_flow_spherical/norm_predictor.pt \
        --task-id adip --n-cold-start 100 --iterations 500
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DirectSphereBO:
    """Direct Bayesian Optimization on unit sphere.

    No flow model - GP operates directly on normalized embeddings.
    """

    def __init__(
        self,
        norm_predictor_path: str,
        codec,
        oracle,
        device: str = "cuda",
        n_candidates: int = 512,
        ucb_beta: float = 2.0,
    ):
        self.device = device
        self.codec = codec
        self.oracle = oracle
        self.n_candidates = n_candidates
        self.ucb_beta = ucb_beta
        self.input_dim = 256  # SELFIES VAE

        # Load norm predictor
        from rielbo.norm_predictor import NormPredictor
        logger.info(f"Loading NormPredictor from {norm_predictor_path}")
        self.norm_predictor = NormPredictor.load(norm_predictor_path, device=device)

        # GP on unit sphere
        self.gp = None
        self.likelihood = None

        # State
        self.train_U = None  # directions [N, D]
        self.train_Y = None  # scores [N]
        self.smiles_observed = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.iteration = 0

        self.history = {
            "iteration": [],
            "best_score": [],
            "current_score": [],
            "n_evaluated": [],
        }

    def _fit_gp(self):
        """Fit GP on current training data using ArcCosine kernel for sphere."""
        import gpytorch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        X = self.train_U.double()
        Y = self.train_Y.double().unsqueeze(-1)

        # ArcCosine kernel for unit sphere data
        # k(x, y) = 1 - arccos(x·y) / π
        class ArcCosineKernel(gpytorch.kernels.Kernel):
            has_lengthscale = False

            def forward(self, x1, x2, diag=False, **params):
                # Normalize inputs to unit sphere
                x1_norm = x1 / (x1.norm(dim=-1, keepdim=True) + 1e-8)
                x2_norm = x2 / (x2.norm(dim=-1, keepdim=True) + 1e-8)

                if diag:
                    # Diagonal: x1[i] · x2[i]
                    cos_sim = (x1_norm * x2_norm).sum(dim=-1)
                else:
                    # Full matrix: x1 @ x2.T
                    cos_sim = x1_norm @ x2_norm.transpose(-2, -1)

                # Clamp to valid range for arccos
                cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)

                # ArcCosine kernel: 1 - arccos(cos_sim) / π
                return 1.0 - torch.arccos(cos_sim) / torch.pi

        try:
            # Create GP with ArcCosine kernel
            covar_module = gpytorch.kernels.ScaleKernel(ArcCosineKernel())

            self.gp = SingleTaskGP(
                X, Y,
                covar_module=covar_module,
            ).to(self.device)
            self.likelihood = self.gp.likelihood
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()
        except Exception as e:
            logger.warning(f"GP fitting with ArcCosine failed: {e}. Using fallback.")
            self.gp = SingleTaskGP(
                X, Y,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.likelihood = self.gp.likelihood
            self.likelihood.noise = 0.1
            self.gp.eval()

    def _optimize_acquisition(self) -> torch.Tensor:
        """Find u* = argmax UCB(u) on unit sphere."""
        from botorch.acquisition import UpperConfidenceBound
        from botorch.optim import optimize_acqf

        acq = UpperConfidenceBound(self.gp, beta=self.ucb_beta**2)

        # Optimize in box, then project to sphere
        bounds = torch.tensor(
            [[-1.0] * self.input_dim, [1.0] * self.input_dim],
            device=self.device,
            dtype=torch.double,
        )

        try:
            u_opt, _ = optimize_acqf(
                acq,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=self.n_candidates,
            )
            # Project to unit sphere
            u_opt = F.normalize(u_opt.float(), p=2, dim=-1)
            return u_opt
        except Exception as e:
            logger.warning(f"Acquisition optimization failed: {e}. Using random.")
            z = torch.randn(1, self.input_dim, device=self.device)
            return F.normalize(z, p=2, dim=-1)

    def _reconstruct_embedding(self, directions: torch.Tensor) -> torch.Tensor:
        """Reconstruct full embedding from direction using NormPredictor."""
        with torch.no_grad():
            directions = F.normalize(directions, p=2, dim=-1)
            magnitudes = self.norm_predictor(directions)
            embeddings = directions * magnitudes
        return embeddings

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules."""
        logger.info(f"Cold start with {len(smiles_list)} molecules")

        # Encode molecules
        logger.info("Encoding molecules...")
        embeddings = []
        batch_size = 64
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Encoding"):
            batch = smiles_list[i:i + batch_size]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0).to(self.device)

        # Extract directions (for GP) - DIRECT, no flow inversion!
        directions = F.normalize(embeddings, p=2, dim=-1)

        # Use ALL points for GP
        self.train_U = directions.to(self.device)
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        # Track best
        best_idx = scores.argmax().item()
        self.best_score = scores[best_idx].item()
        self.best_smiles = smiles_list[best_idx]

        # Fit GP on directions
        self._fit_gp()

        logger.info(f"Cold start complete. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")
        logger.info(f"GP trained on {len(self.train_U)} directions")

    def step(self) -> dict:
        """Run one BO iteration.

        1. Optimize UCB on sphere → u* (direction)
        2. NormPredictor: u* → r (magnitude)
        3. Reconstruct: x = u* * r
        4. Decode x → SMILES
        5. Evaluate with oracle
        6. Update GP on direction u*
        """
        self.iteration += 1

        # 1. Optimize acquisition on sphere
        u_opt = self._optimize_acquisition()

        # 2. Predict magnitude
        with torch.no_grad():
            r = self.norm_predictor(u_opt)

        # 3. Reconstruct full embedding
        x = u_opt * r

        # 4. Decode to SMILES
        smiles_list = self.codec.decode(x)
        smiles = smiles_list[0] if smiles_list else ""

        if not smiles:
            logger.warning("Failed to decode valid SMILES")
            return {
                "score": 0.0,
                "best_score": self.best_score,
                "smiles": "",
                "is_duplicate": True,
                "pred_norm": r.item(),
            }

        # Check duplicate
        is_duplicate = smiles in self.smiles_observed
        if is_duplicate:
            return {
                "score": 0.0,
                "best_score": self.best_score,
                "smiles": smiles,
                "is_duplicate": True,
                "pred_norm": r.item(),
            }

        # 5. Evaluate with oracle
        score = self.oracle.score(smiles)

        # 6. Update GP on direction (DIRECT - no inversion needed!)
        self.train_U = torch.cat([self.train_U, u_opt], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device)])
        self.smiles_observed.append(smiles)

        # Refit GP periodically (every 10 iterations for efficiency)
        if self.iteration % 10 == 0:
            self._fit_gp()

        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles
            logger.info(f"New best! Score: {score:.4f}, SMILES: {smiles}")

        return {
            "score": score,
            "best_score": self.best_score,
            "smiles": smiles,
            "is_duplicate": False,
            "pred_norm": r.item(),
        }

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        logger.info(f"Starting Direct Sphere BO for {n_iterations} iterations")
        logger.info("No flow model - GP operates directly on unit sphere")

        pbar = tqdm(range(n_iterations), desc="Optimizing")
        n_duplicates = 0

        for i in pbar:
            result = self.step()

            # Track history
            self.history["iteration"].append(i)
            self.history["best_score"].append(self.best_score)
            self.history["current_score"].append(result["score"])
            self.history["n_evaluated"].append(len(self.smiles_observed))

            if result["is_duplicate"]:
                n_duplicates += 1

            pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "norm": f"{result['pred_norm']:.2f}",
                "dup": n_duplicates,
            })

            if (i + 1) % log_interval == 0:
                logger.info(
                    f"Iter {i+1}/{n_iterations} | "
                    f"Best: {self.best_score:.4f} | "
                    f"Current: {result['score']:.4f} | "
                    f"pred_norm: {result['pred_norm']:.2f} | "
                    f"Evaluated: {len(self.smiles_observed)}"
                )

        logger.info(f"Optimization complete. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")


def main():
    parser = argparse.ArgumentParser(description="Run Direct Sphere BO on GuacaMol")

    parser.add_argument(
        "--norm-predictor", type=str,
        default="rielbo/checkpoints/guacamol_flow_spherical/norm_predictor.pt",
        help="Path to NormPredictor checkpoint"
    )
    parser.add_argument(
        "--task-id", type=str, default="adip",
        help="GuacaMol task ID"
    )
    parser.add_argument(
        "--n-cold-start", type=int, default=100,
        help="Number of molecules for cold start"
    )
    parser.add_argument(
        "--iterations", type=int, default=500,
        help="Number of BO iterations"
    )
    parser.add_argument(
        "--ucb-beta", type=float, default=2.0,
        help="UCB exploration parameter"
    )
    parser.add_argument(
        "--n-candidates", type=int, default=512,
        help="Number of acquisition candidates"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load components
    logger.info("Loading codec and oracle...")
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    codec = SELFIESVAECodec.from_pretrained(device=args.device)
    oracle = GuacaMolOracle(task_id=args.task_id)

    # Load cold start data
    logger.info("Loading cold start data...")
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start,
        task_id=args.task_id,
    )

    # Create optimizer
    optimizer = DirectSphereBO(
        norm_predictor_path=args.norm_predictor,
        codec=codec,
        oracle=oracle,
        device=args.device,
        n_candidates=args.n_candidates,
        ucb_beta=args.ucb_beta,
    )

    # Cold start
    optimizer.cold_start(smiles_list, scores)

    # Run optimization
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    # Save results
    results_dir = "rielbo/results/guacamol"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"direct_sphere_{args.task_id}_{timestamp}.json")

    results = {
        "task_id": args.task_id,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "history": optimizer.history,
        "args": vars(args),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
