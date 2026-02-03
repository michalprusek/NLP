"""Run RieLBO on GuacaMol with Spherical Flow BO.

Supports two modes (auto-detected from checkpoint):

1. Stereographic (preferred): R^D ↔ S^D bijection
   - No NormPredictor needed
   - Exact magnitude recovery
   - Flow operates on D+1 dimensional sphere

2. Legacy (NormPredictor): S^{D-1} × R^+ product manifold
   - Requires trained NormPredictor
   - Approximate magnitude recovery
   - Flow operates on D dimensional sphere

Usage (stereographic):
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol \
        --flow-checkpoint rielbo/checkpoints/guacamol_stereo/best.pt \
        --iterations 500

Usage (legacy with NormPredictor):
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol \
        --flow-checkpoint rielbo/checkpoints/guacamol_flow_spherical/best.pt \
        --norm-predictor rielbo/checkpoints/guacamol_flow_spherical/norm_predictor.pt \
        --iterations 500
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
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


class SphericalGP:
    """GP surrogate on unit sphere using ArcCosine-like kernel.

    For unit vectors, we use a kernel based on angular distance.
    k(u, v) = (1 - arccos(u·v)/π) works well for spherical data.
    """

    def __init__(self, input_dim: int = 256, device: str = "cuda"):
        self.input_dim = input_dim
        self.device = device
        self.model = None
        self.train_X = None
        self.train_Y = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Fit GP on (direction, score) pairs."""
        import gpytorch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        # Ensure unit vectors
        X = F.normalize(X, p=2, dim=-1)

        self.train_X = X.to(self.device).double()
        self.train_Y = Y.to(self.device).double().unsqueeze(-1)

        # Handle NaN/Inf
        if torch.isnan(self.train_X).any() or torch.isinf(self.train_X).any():
            logger.warning("Training X contains NaN/Inf! Replacing.")
            nan_mask = torch.isnan(self.train_X) | torch.isinf(self.train_X)
            self.train_X[nan_mask] = 0.0
            self.train_X = F.normalize(self.train_X, p=2, dim=-1)

        if torch.isnan(self.train_Y).any() or torch.isinf(self.train_Y).any():
            valid_y = self.train_Y[~torch.isnan(self.train_Y) & ~torch.isinf(self.train_Y)]
            fill_val = valid_y.mean() if len(valid_y) > 0 else 0.5
            nan_mask = torch.isnan(self.train_Y) | torch.isinf(self.train_Y)
            self.train_Y[nan_mask] = fill_val

        try:
            self.model = SingleTaskGP(self.train_X, self.train_Y).to(self.device)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)
            self.model.eval()
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            # Numerical issues (Cholesky, singular matrix) - use fallback
            logger.warning(f"GP fitting failed (numerical): {e}. Using fallback.")
            self.model = SingleTaskGP(
                self.train_X, self.train_Y,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.model.likelihood.noise = 0.1
            self.model.eval()
        except (torch.cuda.OutOfMemoryError, MemoryError):
            # Critical memory errors - re-raise
            raise

    def update(self, X_new: torch.Tensor, Y_new: torch.Tensor) -> None:
        """Update GP with new observations."""
        X_new = F.normalize(X_new, p=2, dim=-1)
        X_new = X_new.to(self.device).double()
        Y_new = Y_new.to(self.device).double()

        if self.train_X is None:
            self.fit(X_new, Y_new)
        else:
            new_X = torch.cat([self.train_X, X_new], dim=0)
            new_Y = torch.cat([self.train_Y.squeeze(-1), Y_new], dim=0)
            self.fit(new_X, new_Y)

    def optimize_acquisition(
        self,
        n_candidates: int = 512,
        n_restarts: int = 10,
        alpha: float = 2.0,
    ) -> torch.Tensor:
        """Find u* = argmax UCB(u) on unit sphere."""
        from botorch.acquisition import UpperConfidenceBound
        from botorch.optim import optimize_acqf

        if self.model is None:
            # Random direction on sphere
            z = torch.randn(1, self.input_dim, device=self.device)
            return F.normalize(z, p=2, dim=-1)

        acq = UpperConfidenceBound(self.model, beta=alpha**2)

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
                num_restarts=n_restarts,
                raw_samples=n_candidates,
            )
            # Project to unit sphere
            u_opt = F.normalize(u_opt.float(), p=2, dim=-1)
            return u_opt
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            # Numerical issues - use random fallback
            logger.warning(f"Acquisition optimization failed (numerical): {e}. Using random.")
            z = torch.randn(1, self.input_dim, device=self.device)
            return F.normalize(z, p=2, dim=-1)
        except (torch.cuda.OutOfMemoryError, MemoryError):
            # Critical memory errors - re-raise
            raise


class RieLBOSpherical:
    """RieLBO with Decoupled Direction & Magnitude.

    Supports two modes:
    1. Stereographic (preferred): Deterministic bijection R^D ↔ S^D
       - No NormPredictor needed
       - Exact magnitude recovery
       - Uses D+1 dimensional sphere

    2. Legacy (NormPredictor): Learned magnitude prediction
       - Requires separate trained NormPredictor
       - Approximate magnitude recovery
       - Uses D dimensional sphere

    The mode is auto-detected from the checkpoint.
    """

    def __init__(
        self,
        flow_checkpoint: str,
        norm_predictor_path: str | None = None,
        codec=None,
        oracle=None,
        device: str = "cuda",
        top_k: int = 100,
        n_candidates: int = 512,
        ucb_beta: float = 2.0,
    ):
        self.device = device
        self.codec = codec
        self.oracle = oracle
        self.top_k = top_k
        self.n_candidates = n_candidates
        self.ucb_beta = ucb_beta

        # Load flow model (auto-detects stereographic)
        self._load_flow_model(flow_checkpoint)

        # Load magnitude recovery (stereographic or NormPredictor)
        if self.is_stereographic:
            logger.info("Using stereographic projection for magnitude recovery")
            self.norm_predictor = None
        else:
            if norm_predictor_path is None:
                raise ValueError(
                    "norm_predictor_path required for non-stereographic checkpoints"
                )
            self._load_norm_predictor(norm_predictor_path)

        # GP on unit sphere (uses flow's input_dim which is D+1 for stereographic)
        self.gp = SphericalGP(input_dim=self.input_dim, device=device)

        # State
        self.train_U = None  # directions [N, D] or [N, D+1] for stereographic
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

    def _load_flow_model(self, checkpoint_path: str):
        """Load trained spherical flow model.

        Auto-detects stereographic projection from checkpoint metadata.
        """
        from rielbo.velocity_network import VelocityNetwork

        logger.info(f"Loading spherical flow from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Check for stereographic projection
        self.is_stereographic = checkpoint.get("is_stereographic", False)

        if self.is_stereographic:
            # Stereographic: model input_dim is D+1, original data is D
            from rielbo.stereographic import StereographicTransform

            original_input_dim = checkpoint.get("original_input_dim")
            radius_scaling = checkpoint.get("radius_scaling")

            if original_input_dim is None or radius_scaling is None:
                raise ValueError(
                    "Stereographic checkpoint missing original_input_dim or radius_scaling"
                )

            self.stereo = StereographicTransform(original_input_dim, radius_scaling)
            self.input_dim = self.stereo.output_dim  # D+1
            self.original_input_dim = original_input_dim

            logger.info(
                f"Stereographic projection: {original_input_dim}D -> {self.input_dim}D, "
                f"R={radius_scaling:.4f}"
            )
        else:
            # Legacy: flow operates on normalized D-dimensional embeddings
            args = checkpoint.get("args", {})
            self.input_dim = args.get("input_dim", 256)
            self.original_input_dim = self.input_dim
            self.stereo = None

        args = checkpoint.get("args", {})
        hidden_dim = args.get("hidden_dim", 256)
        num_layers = args.get("num_layers", 6)
        num_heads = args.get("num_heads", 8)

        self.flow_model = VelocityNetwork(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        ).to(self.device)

        # Load EMA weights
        if "ema_shadow" in checkpoint:
            ema_state = checkpoint["ema_shadow"]
            model_state = self.flow_model.state_dict()
            for name in model_state:
                if name in ema_state:
                    model_state[name] = ema_state[name]
            self.flow_model.load_state_dict(model_state)
            logger.info("Loaded EMA weights")
        else:
            self.flow_model.load_state_dict(checkpoint["model_state_dict"])

        self.flow_model.eval()
        self.is_spherical = checkpoint.get("is_spherical", True)

        if not self.is_spherical and not self.is_stereographic:
            logger.warning("Flow checkpoint is NOT spherical! Results may be suboptimal.")

        logger.info(
            f"Flow model: input_dim={self.input_dim}, spherical={self.is_spherical}, "
            f"stereographic={self.is_stereographic}"
        )

    def _load_norm_predictor(self, path: str):
        """Load trained norm predictor."""
        from rielbo.norm_predictor import NormPredictor

        logger.info(f"Loading NormPredictor from {path}")
        self.norm_predictor = NormPredictor.load(path, device=self.device)

    def _flow_sample_directions(self, n_samples: int, num_steps: int = 50) -> torch.Tensor:
        """Sample directions from spherical flow."""
        with torch.no_grad():
            # Start from random directions on sphere
            u = torch.randn(n_samples, self.input_dim, device=self.device)
            u = F.normalize(u, p=2, dim=-1)

            # Integrate flow (stays on sphere)
            dt = 1.0 / num_steps
            for t_idx in range(num_steps):
                t = torch.full((n_samples,), t_idx * dt, device=self.device)
                v = self.flow_model(u, t)
                u = u + dt * v
                u = F.normalize(u, p=2, dim=-1)  # Project back to sphere

        return u

    def _lift_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Lift raw embedding to the sphere used by the flow.

        For stereographic: use stereographic lift (D -> D+1)
        For legacy: normalize to unit sphere (D -> D)
        """
        if self.is_stereographic:
            return self.stereo.lift(x)
        else:
            return F.normalize(x, p=2, dim=-1)

    def _invert_to_direction(self, u: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """Invert flow output to latent direction via reverse ODE.

        Args:
            u: Points on sphere (D+1 for stereographic, D for legacy)
            num_steps: Integration steps

        Returns:
            Latent directions in noise space
        """
        with torch.no_grad():
            # Ensure on unit sphere
            u = F.normalize(u, p=2, dim=-1)

            # Reverse integration
            dt = -1.0 / num_steps
            for t_idx in range(num_steps):
                t = torch.full((u.shape[0],), 1.0 - t_idx / num_steps, device=self.device)
                v = self.flow_model(u, t)
                u = u + dt * v
                u = F.normalize(u, p=2, dim=-1)

        return u

    def _reconstruct_embedding(self, u: torch.Tensor) -> torch.Tensor:
        """Reconstruct full embedding from flow output.

        For stereographic: exact magnitude recovery via inverse projection
        For legacy: approximate magnitude via NormPredictor
        """
        with torch.no_grad():
            if self.is_stereographic:
                # Stereographic: exact magnitude recovery
                # u is on S^D (D+1 dimensional sphere), project to R^D
                return self.stereo.project(u)
            else:
                # Legacy: use NormPredictor
                directions = F.normalize(u, p=2, dim=-1)
                magnitudes = self.norm_predictor(directions)
                return directions * magnitudes

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules.

        Uses ALL evaluated points for GP training (not just top-k).

        For stereographic: embeddings are lifted to S^D (D+1 dims)
        For legacy: embeddings are normalized to S^{D-1} (D dims)
        """
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

        # Lift to sphere (stereographic: D -> D+1, legacy: D -> D normalized)
        logger.info("Lifting embeddings to sphere...")
        sphere_points = self._lift_embedding(embeddings)

        # Invert to z-space directions
        logger.info("Inverting to latent directions...")
        z_directions = []
        for i in tqdm(range(len(sphere_points)), desc="Inverting"):
            z = self._invert_to_direction(sphere_points[i:i+1])
            z_directions.append(z.squeeze(0))
        z_directions = torch.stack(z_directions)

        # Use ALL points for GP (not just top-k)
        self.train_U = z_directions.to(self.device)
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        # Track best
        best_idx = scores.argmax().item()
        self.best_score = scores[best_idx].item()
        self.best_smiles = smiles_list[best_idx]

        # Fit GP on ALL directions
        self.gp.fit(self.train_U, self.train_Y)

        logger.info(f"Cold start complete. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")
        logger.info(f"GP trained on {len(self.train_U)} points (dim={self.input_dim})")

    def step(self) -> dict:
        """Run one BO iteration.

        For stereographic:
        1. Optimize UCB on S^D → z* (D+1 dims)
        2. Flow forward: z* → u (point on S^D)
        3. Stereographic project: u → x (D dims with exact magnitude)
        4. Decode x → SMILES
        5. Evaluate with oracle
        6. Update GP

        For legacy (NormPredictor):
        1. Optimize UCB on S^{D-1} → z*
        2. Flow forward: z* → u (direction on S^{D-1})
        3. NormPredictor: u → r, x = u * r
        4-6. Same as above
        """
        self.iteration += 1

        # 1. Optimize acquisition on sphere
        z_opt = self.gp.optimize_acquisition(
            n_candidates=self.n_candidates,
            n_restarts=10,
            alpha=self.ucb_beta,
        )

        # 2. Flow forward: z → u (on data manifold)
        with torch.no_grad():
            u = z_opt.clone()
            u = F.normalize(u, p=2, dim=-1)

            dt = 1.0 / 50
            for t_idx in range(50):
                t = torch.full((u.shape[0],), t_idx * dt, device=self.device)
                v = self.flow_model(u, t)
                u = u + dt * v
                u = F.normalize(u, p=2, dim=-1)

        # 3. Reconstruct full embedding
        x = self._reconstruct_embedding(u)
        embedding_norm = x.norm(dim=-1).item()

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
                "embedding_norm": embedding_norm,
            }

        # Check duplicate
        is_duplicate = smiles in self.smiles_observed
        if is_duplicate:
            return {
                "score": 0.0,
                "best_score": self.best_score,
                "smiles": smiles,
                "is_duplicate": True,
                "embedding_norm": embedding_norm,
            }

        # 5. Evaluate with oracle
        score = self.oracle.score(smiles)

        # 6. Invert new point to z-space (CRITICAL: GP operates in z-space!)
        z_new = self._invert_to_direction(u)

        # Update GP on z-space direction (same space as cold start!)
        self.gp.update(z_new, torch.tensor([score], device=self.device))

        # Update state - store z-space directions
        self.train_U = torch.cat([self.train_U, z_new], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device)])
        self.smiles_observed.append(smiles)

        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles

        return {
            "score": score,
            "best_score": self.best_score,
            "smiles": smiles,
            "is_duplicate": False,
            "embedding_norm": embedding_norm,
        }

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        logger.info(f"Starting optimization for {n_iterations} iterations")
        logger.info("Using Decoupled Direction & Magnitude (S^{d-1} × R^+)")

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
                "norm": f"{result['embedding_norm']:.2f}",
                "dup": n_duplicates,
            })

            if (i + 1) % log_interval == 0:
                logger.info(
                    f"Iter {i+1}/{n_iterations} | "
                    f"Best: {self.best_score:.4f} | "
                    f"Current: {result['score']:.4f} | "
                    f"||x||: {result['embedding_norm']:.2f} | "
                    f"Evaluated: {len(self.smiles_observed)}"
                )

        logger.info(f"Optimization complete. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")

    def plot_progress(self, save_path=None):
        """Plot optimization progress."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax1 = axes[0]
        ax1.plot(self.history["iteration"], self.history["best_score"], "b-", lw=2, label="Best")
        ax1.scatter(self.history["iteration"], self.history["current_score"],
                   c="gray", s=10, alpha=0.5, label="Current")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Score")
        ax1.set_title("Optimization Progress")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(self.history["n_evaluated"], self.history["best_score"], "r-", lw=2)
        ax2.set_xlabel("Oracle Calls")
        ax2.set_ylabel("Best Score")
        ax2.set_title("Sample Efficiency")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        return fig


def main():
    parser = argparse.ArgumentParser(description="Run RieLBO with Decoupled Direction & Magnitude")

    parser.add_argument("--flow-checkpoint", type=str,
                       default="rielbo/checkpoints/guacamol_flow_spherical/best.pt")
    parser.add_argument("--norm-predictor", type=str, default=None,
                       help="Path to NormPredictor (only needed for non-stereographic checkpoints)")
    parser.add_argument("--task-id", type=str, default="pdop")
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--n-candidates", type=int, default=512)
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="rielbo/results/guacamol")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load codec and oracle
    logger.info("Loading codec and oracle...")
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    codec = SELFIESVAECodec.from_pretrained(device="cuda")
    oracle = GuacaMolOracle(task_id=args.task_id)

    # Load cold start data
    logger.info("Loading cold start data...")
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start,
        task_id=args.task_id,
    )

    # Create optimizer
    optimizer = RieLBOSpherical(
        flow_checkpoint=args.flow_checkpoint,
        norm_predictor_path=args.norm_predictor,
        codec=codec,
        oracle=oracle,
        device="cuda",
        top_k=args.top_k,
        n_candidates=args.n_candidates,
        ucb_beta=args.ucb_beta,
    )

    # Cold start
    optimizer.cold_start(smiles_list, scores)

    # Run optimization
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    # Save results
    results = {
        "task_id": args.task_id,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_cold_start": args.n_cold_start,
        "n_oracle_calls": len(optimizer.smiles_observed),
        "history": optimizer.history,
        "args": vars(args),
    }

    results_path = os.path.join(args.output_dir, f"results_spherical_{args.task_id}_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Plot
    plot_path = os.path.join(args.output_dir, f"progress_spherical_{args.task_id}_{timestamp}.png")
    optimizer.plot_progress(save_path=plot_path)


if __name__ == "__main__":
    main()
