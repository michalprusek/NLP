"""Spherical Subspace BO v3: Practical Improvements for Sample Efficiency.

Three targeted improvements over v1:
1. Windowed/Local GP: Fit on 50 nearest + 30 random points (prevents posterior collapse)
2. Multi-Projection Ensemble: K=3 QR matrices, round-robin selection
3. Every-step refit with Y-normalization (cheap with 80-point window)

Pipeline:
    x [N, D] -> normalize -> u [N, D] on S^(D-1) (directions)
    u [N, D] -> project_k -> v [N, d] on S^(d-1)  (d=16, using A[k])
    Window: select 50 nearest + 30 random from training data
    Y-normalize: Y_norm = (Y - mean) / std
    GP operates on S^15 with ArcCosine kernel (80 points)
    v* -> lift_k -> u* on S^(D-1) -> x* = u* * mean_norm -> decode

Usage:
    from rielbo.subspace_bo_v3 import SphericalSubspaceBOv3

    optimizer = SphericalSubspaceBOv3(
        codec=codec,
        oracle=oracle,
        n_projections=3,
        window_local=50,
        window_random=30,
    )
"""

import logging

import gpytorch
import torch
import torch.nn.functional as F
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from rielbo.gp_diagnostics import GPDiagnostics
from rielbo.kernels import create_kernel

logger = logging.getLogger(__name__)


class SphericalSubspaceBOv3:
    """Spherical Subspace BO v3 with windowed GP and multi-projection ensemble.

    Key improvements over v1:
    1. Windowed GP: Only uses 80 nearest+random points to prevent posterior collapse
       (GP std stays > 0.005 vs 0.00006 in v1 at late iterations)
    2. Multi-projection: K=3 QR matrices rotated each step for coverage
    3. Every-step refit: Cheap with small window (~0.02s per fit)
    4. Y-normalization: Keeps GP well-conditioned

    Why this works:
    - 80 points in 16D: 5 pts/dim -> GP generalizes
    - 100+ points in 16D: GP memorizes -> posterior std -> 0
    - Different projections capture different VAE latent directions
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        subspace_dim: int = 16,
        device: str = "cuda",
        n_candidates: int = 2000,
        ucb_beta: float = 2.0,
        acqf: str = "ts",
        trust_region: float = 0.8,
        seed: int = 42,
        verbose: bool = True,
        kernel: str = "arccosine",
        # V3 improvements
        n_projections: int = 3,
        window_local: int = 50,
        window_random: int = 30,
    ):
        if subspace_dim >= input_dim:
            raise ValueError(
                f"subspace_dim ({subspace_dim}) must be < input_dim ({input_dim})"
            )
        if subspace_dim < 2:
            raise ValueError(f"subspace_dim must be >= 2, got {subspace_dim}")
        if not (0 < trust_region <= 2.0):
            raise ValueError(f"trust_region must be in (0, 2], got {trust_region}")
        if n_projections < 1:
            raise ValueError(f"n_projections must be >= 1, got {n_projections}")
        if window_local < 1:
            raise ValueError(f"window_local must be >= 1, got {window_local}")
        if window_random < 0:
            raise ValueError(f"window_random must be >= 0, got {window_random}")

        self.device = device
        self.codec = codec
        self.oracle = oracle
        self.input_dim = input_dim
        self.subspace_dim = subspace_dim
        self.n_candidates = n_candidates
        self.ucb_beta = ucb_beta
        self.acqf = acqf
        self.trust_region = trust_region
        self.verbose = verbose
        self.seed = seed
        self.kernel = kernel
        self.fallback_count = 0

        # V3 parameters
        self.n_projections = n_projections
        self.window_local = window_local
        self.window_random = window_random
        self.window_size = window_local + window_random

        # Gradient optimization parameters
        self.grad_enabled = acqf in ("grad_ucb", "grad_ei", "grad_ts")
        self.n_grad_starts = 10
        self.n_grad_steps = 50
        self.grad_lr = 0.01

        # Initialize K orthonormal projection matrices
        torch.manual_seed(seed)
        self.projections = []
        for k in range(n_projections):
            A_raw = torch.randn(input_dim, subspace_dim, device=device)
            A, _ = torch.linalg.qr(A_raw)
            self.projections.append(A)

        logger.info(
            f"SubspaceBOv3: S^{input_dim-1} -> S^{subspace_dim-1}, "
            f"kernel={kernel}, n_proj={n_projections}, "
            f"window={window_local}+{window_random}"
        )

        # GP
        self.gp = None
        self.likelihood = None

        # Training data
        self.train_X = None  # Original embeddings [N, D]
        self.train_U = None  # Directions [N, D] on S^(D-1)
        self.train_Y = None  # Scores [N]
        self.mean_norm = None
        self.smiles_observed = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.iteration = 0

        # Windowed GP data (recomputed each step)
        self._window_V = None  # Windowed subspace points [W, d]
        self._window_Y = None  # Windowed scores [W]
        self._window_indices = None  # Indices into train_* arrays
        self._y_mean = None
        self._y_std = None

        self.history = {
            "iteration": [],
            "best_score": [],
            "current_score": [],
            "n_evaluated": [],
            "gp_mean": [],
            "gp_std": [],
            "nearest_train_cos": [],
            "embedding_norm": [],
            "projection_idx": [],
            "window_size": [],
        }

        # GP diagnostics
        self.gp_diagnostics = GPDiagnostics(verbose=True)
        self.diagnostic_history = []

    def _current_projection_idx(self) -> int:
        """Get current projection index (round-robin)."""
        return self.iteration % self.n_projections

    def _current_A(self) -> torch.Tensor:
        """Get current projection matrix."""
        return self.projections[self._current_projection_idx()]

    def project_to_subspace(self, u: torch.Tensor, A: torch.Tensor | None = None) -> torch.Tensor:
        """Project from S^(D-1) to S^(d-1) using given projection."""
        if A is None:
            A = self._current_A()
        v = u @ A
        return F.normalize(v, p=2, dim=-1)

    def lift_to_original(self, v: torch.Tensor, A: torch.Tensor | None = None) -> torch.Tensor:
        """Lift from S^(d-1) to S^(D-1) using given projection."""
        if A is None:
            A = self._current_A()
        u = v @ A.T
        return F.normalize(u, p=2, dim=-1)

    def _create_kernel(self):
        """Create covariance kernel."""
        if self.kernel == "arccosine":
            return create_kernel(kernel_type="arccosine", kernel_order=0, use_scale=True)
        elif self.kernel == "matern":
            return create_kernel(kernel_type="matern", use_scale=True)
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'. Valid: arccosine, matern")

    def _select_window(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select windowed training subset: K_local nearest + K_random random.

        Returns (V_window, Y_window) projected through A.
        """
        N = len(self.train_Y)
        total_window = self.window_local + self.window_random

        if N <= total_window:
            # Use all data if we have fewer points than window size
            V = self.project_to_subspace(self.train_U, A)
            self._window_indices = torch.arange(N, device=self.device)
            return V, self.train_Y

        # Find best point
        best_idx = self.train_Y.argmax()
        u_best = self.train_U[best_idx:best_idx + 1]  # [1, D]

        # Compute cosine similarity to best in original space
        cos_sims = (self.train_U @ u_best.T).squeeze()  # [N]

        # Select top-K_local nearest (by cosine similarity)
        _, local_indices = cos_sims.topk(self.window_local)

        # Select K_random from remaining
        mask = torch.ones(N, dtype=torch.bool, device=self.device)
        mask[local_indices] = False
        remaining_indices = torch.where(mask)[0]

        n_random = min(self.window_random, len(remaining_indices))
        if n_random > 0:
            perm = torch.randperm(len(remaining_indices), device=self.device)[:n_random]
            random_indices = remaining_indices[perm]
            window_indices = torch.cat([local_indices, random_indices])
        else:
            window_indices = local_indices

        self._window_indices = window_indices

        # Project windowed data
        V = self.project_to_subspace(self.train_U[window_indices], A)
        Y = self.train_Y[window_indices]

        return V, Y

    def _fit_gp(self, A: torch.Tensor | None = None):
        """Fit GP on windowed subspace data with Y-normalization."""
        if A is None:
            A = self._current_A()

        # Select window
        V_window, Y_window = self._select_window(A)
        self._window_V = V_window
        self._window_Y = Y_window

        X = V_window.double()

        # Y-normalization: center and scale
        y_mean = Y_window.mean()
        y_std = Y_window.std()
        if y_std < 1e-8:
            y_std = torch.tensor(1.0, device=self.device)
        Y_norm = ((Y_window - y_mean) / y_std).double().unsqueeze(-1)

        # Store normalization params for acquisition
        self._y_mean = y_mean
        self._y_std = y_std

        try:
            covar_module = self._create_kernel()
            self.gp = SingleTaskGP(X, Y_norm, covar_module=covar_module).to(self.device)
            self.likelihood = self.gp.likelihood
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()

            # Diagnostics every 10 iterations
            if self.verbose and self.iteration % 10 == 0:
                metrics = self.gp_diagnostics.analyze(
                    self.gp, X.float(), Y_norm.squeeze(-1).float()
                )
                self.gp_diagnostics.log_summary(metrics, prefix=f"[Iter {self.iteration}]")
                self.diagnostic_history.append(
                    self.gp_diagnostics.get_summary_dict(metrics)
                )
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            self.fallback_count += 1
            logger.error(f"GP fit failed (fallback #{self.fallback_count}): {e}")
            self.gp = SingleTaskGP(
                X, Y_norm,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.likelihood = self.gp.likelihood
            self.likelihood.noise = 0.1
            self.gp.eval()

    def _generate_sobol_candidates(self, n_candidates: int, A: torch.Tensor) -> torch.Tensor:
        """Generate candidates in trust region around best point (in windowed data)."""
        best_idx = self._window_Y.argmax()
        v_best = self._window_V[best_idx:best_idx + 1]  # [1, d]

        half_length = self.trust_region / 2
        tr_lb = v_best - half_length
        tr_ub = v_best + half_length

        sobol = SobolEngine(self.subspace_dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=torch.float32, device=self.device)

        v_cand = tr_lb + (tr_ub - tr_lb) * pert
        v_cand = F.normalize(v_cand, p=2, dim=-1)

        return v_cand

    def _riemannian_gradient_ascent(
        self, v_starts: torch.Tensor, acqf_type: str = "ucb"
    ) -> torch.Tensor:
        """Riemannian gradient ascent on S^(d-1) to optimize acquisition function.

        Uses tangent space projection + retraction (normalization) to stay on sphere.
        Works with UCB, EI, or TS (via posterior mean of a drawn sample).

        Args:
            v_starts: [K, d] starting points on S^(d-1)
            acqf_type: "ucb", "ei", or "ts" (base type, without "grad_" prefix)

        Returns:
            [1, d] best optimized point on S^(d-1)
        """
        K = v_starts.shape[0]
        best_v = None
        best_acqf_val = float("-inf")

        # For TS: draw one posterior sample and optimize its mean
        ts_sample = None
        if acqf_type == "ts":
            with torch.no_grad():
                post = self.gp.posterior(self._window_V.double())
                ts_sample = post.rsample()  # [1, W, 1] — one function draw

        # Best-f for EI (in normalized space)
        best_f_norm = None
        if acqf_type == "ei":
            best_f_norm = ((self.train_Y.max() - self._y_mean) / self._y_std).double()

        for k in range(K):
            v = v_starts[k:k+1].clone().double().detach().requires_grad_(True)  # [1, d]

            for step in range(self.n_grad_steps):
                if v.grad is not None:
                    v.grad.zero_()

                post = self.gp.posterior(v)

                if acqf_type == "ucb":
                    acqf_val = post.mean.squeeze() + self.ucb_beta * post.variance.sqrt().squeeze()
                elif acqf_type == "ei":
                    # Analytical EI: E[max(f - f*, 0)]
                    mu = post.mean.squeeze()
                    sigma = post.variance.sqrt().squeeze().clamp(min=1e-8)
                    z = (mu - best_f_norm) / sigma
                    normal = torch.distributions.Normal(0, 1)
                    acqf_val = sigma * (z * normal.cdf(z) + normal.log_prob(z).exp())
                elif acqf_type == "ts":
                    # Optimize the drawn function's prediction at v
                    acqf_val = post.mean.squeeze()
                else:
                    acqf_val = post.mean.squeeze()

                acqf_val.backward()

                with torch.no_grad():
                    g = v.grad  # [1, d]
                    # Tangent space projection: remove normal component
                    g_tan = g - (g * v).sum(dim=-1, keepdim=True) * v
                    # Gradient ascent + sphere retraction
                    v_new = v + self.grad_lr * g_tan
                    v.data = F.normalize(v_new, p=2, dim=-1)

            # Evaluate final point
            with torch.no_grad():
                post = self.gp.posterior(v)
                if acqf_type == "ucb":
                    val = (post.mean.squeeze() + self.ucb_beta * post.variance.sqrt().squeeze()).item()
                elif acqf_type == "ei":
                    mu = post.mean.squeeze()
                    sigma = post.variance.sqrt().squeeze().clamp(min=1e-8)
                    z = (mu - best_f_norm) / sigma
                    normal = torch.distributions.Normal(0, 1)
                    val = (sigma * (z * normal.cdf(z) + normal.log_prob(z).exp())).item()
                else:
                    val = post.mean.squeeze().item()

                if val > best_acqf_val:
                    best_acqf_val = val
                    best_v = v.detach().clone()

        return best_v.float()

    def _optimize_acquisition(self, A: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Find optimal v* using acquisition function on windowed GP."""
        diag = {}

        try:
            v_cand = self._generate_sobol_candidates(self.n_candidates, A)

            if self.grad_enabled:
                # Hybrid: Sobol warm-start → gradient refinement on sphere
                base_acqf = self.acqf.replace("grad_", "")

                # Score all Sobol candidates with UCB to find top-k starts
                with torch.no_grad():
                    post = self.gp.posterior(v_cand.double())
                    ucb_vals = post.mean.squeeze() + self.ucb_beta * post.variance.sqrt().squeeze()
                _, top_indices = ucb_vals.topk(self.n_grad_starts)
                v_starts = v_cand[top_indices]  # [K, d]

                # Riemannian gradient ascent on S^(d-1)
                v_opt = self._riemannian_gradient_ascent(v_starts, acqf_type=base_acqf)

            elif self.acqf == "ts":
                thompson = MaxPosteriorSampling(model=self.gp, replacement=False)
                v_opt = thompson(v_cand.double().unsqueeze(0), num_samples=1)
                v_opt = v_opt.squeeze(0).float()
                v_opt = F.normalize(v_opt, p=2, dim=-1)

            elif self.acqf == "ei":
                # best_f in normalized space
                best_f_norm = ((self.train_Y.max() - self._y_mean) / self._y_std).double()
                ei = qExpectedImprovement(self.gp, best_f=best_f_norm)
                with torch.no_grad():
                    ei_vals = ei(v_cand.double().unsqueeze(-2))
                best_idx = ei_vals.argmax()
                v_opt = v_cand[best_idx:best_idx + 1]

            elif self.acqf == "ucb":
                with torch.no_grad():
                    post = self.gp.posterior(v_cand.double())
                    ucb_vals = post.mean.squeeze() + self.ucb_beta * post.variance.sqrt().squeeze()
                best_idx = ucb_vals.argmax()
                v_opt = v_cand[best_idx:best_idx + 1]

            else:
                raise ValueError(f"Unknown acquisition function: {self.acqf}")

            # Diagnostics (in normalized Y space, then convert back)
            with torch.no_grad():
                post = self.gp.posterior(v_opt.double())
                gp_mean_norm = post.mean.item()
                gp_std_norm = post.variance.sqrt().item()
                # Convert back to original Y scale
                diag["gp_mean"] = gp_mean_norm * self._y_std.item() + self._y_mean.item()
                diag["gp_std"] = gp_std_norm * self._y_std.item()
                cos_sims = (v_opt @ self._window_V.T).squeeze()
                diag["nearest_train_cos"] = cos_sims.max().item()

            u_opt = self.lift_to_original(v_opt, A)
            return u_opt, diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            logger.error(f"Acquisition failed: {e}")
            u_opt = F.normalize(torch.randn(1, self.input_dim, device=self.device), dim=-1)
            return u_opt, {"gp_mean": 0, "gp_std": 1, "nearest_train_cos": 0, "is_fallback": True}

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules."""
        logger.info(f"Cold start: {len(smiles_list)} molecules")

        from tqdm import tqdm
        embeddings = []
        for i in tqdm(range(0, len(smiles_list), 64), desc="Encoding"):
            batch = smiles_list[i:i + 64]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0).to(self.device)

        # Compute mean norm
        self.mean_norm = embeddings.norm(dim=-1).mean().item()
        logger.info(f"Mean embedding norm: {self.mean_norm:.2f}")

        # Store data
        self.train_X = embeddings
        self.train_U = F.normalize(embeddings, p=2, dim=-1)
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        # Track best
        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]

        # Initial GP fit with first projection
        self._fit_gp(self.projections[0])

        logger.info(f"Cold start done. Best: {self.best_score:.4f} (n={len(self.train_Y)})")
        logger.info(f"Best SMILES: {self.best_smiles}")

    def step(self) -> dict:
        """One BO iteration with windowed GP and round-robin projection."""
        self.iteration += 1

        # Round-robin projection selection
        proj_idx = self._current_projection_idx()
        A = self.projections[proj_idx]

        # Refit GP every step (cheap with 80-point window)
        self._fit_gp(A)

        # Optimize on subspace, lift to original sphere
        u_opt, diag = self._optimize_acquisition(A)
        diag["projection_idx"] = proj_idx
        diag["window_size"] = len(self._window_Y)

        # Reconstruct embedding with mean norm
        x_opt = u_opt * self.mean_norm
        diag["embedding_norm"] = self.mean_norm

        # Decode
        smiles_list = self.codec.decode(x_opt)
        smiles = smiles_list[0] if smiles_list else ""

        if not smiles:
            logger.debug(f"Decode failed at iter {self.iteration}")
            return {"score": 0.0, "best_score": self.best_score, "smiles": "",
                    "is_duplicate": True, "is_decode_failure": True, **diag}

        if smiles in self.smiles_observed:
            return {"score": 0.0, "best_score": self.best_score, "smiles": smiles,
                    "is_duplicate": True, **diag}

        # Evaluate
        score = self.oracle.score(smiles)

        # Update training data
        self.train_X = torch.cat([self.train_X, x_opt], dim=0)
        self.train_U = torch.cat([self.train_U, u_opt], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device, dtype=torch.float32)])
        self.smiles_observed.append(smiles)

        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles
            logger.info(f"New best! {score:.4f}: {smiles}")

        return {"score": score, "best_score": self.best_score, "smiles": smiles,
                "is_duplicate": False, **diag}

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        from tqdm import tqdm

        logger.info(f"SubspaceBOv3: {n_iterations} iterations")
        logger.info(
            f"S^{self.input_dim-1} -> S^{self.subspace_dim-1}, "
            f"{self.n_projections} projections, "
            f"window={self.window_local}+{self.window_random}"
        )

        pbar = tqdm(range(n_iterations), desc="Optimizing")
        n_dup = 0

        for i in pbar:
            result = self.step()

            self.history["iteration"].append(i)
            self.history["best_score"].append(self.best_score)
            self.history["current_score"].append(result["score"])
            self.history["n_evaluated"].append(len(self.smiles_observed))
            self.history["gp_mean"].append(result.get("gp_mean", 0))
            self.history["gp_std"].append(result.get("gp_std", 0))
            self.history["nearest_train_cos"].append(result.get("nearest_train_cos", 0))
            self.history["embedding_norm"].append(result.get("embedding_norm", 0))
            self.history["projection_idx"].append(result.get("projection_idx", 0))
            self.history["window_size"].append(result.get("window_size", 0))

            if result["is_duplicate"]:
                n_dup += 1

            pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "proj": result.get("projection_idx", 0),
                "gp_s": f"{result.get('gp_std', 0):.4f}",
                "dup": n_dup,
            })

            if (i + 1) % log_interval == 0 and self.verbose:
                logger.info(
                    f"Iter {i+1}/{n_iterations} | Best: {self.best_score:.4f} | "
                    f"Curr: {result['score']:.4f} | "
                    f"GP: {result.get('gp_mean', 0):.2f}+/-{result.get('gp_std', 0):.4f} | "
                    f"proj: A[{result.get('projection_idx', 0)}] | "
                    f"win: {result.get('window_size', 0)}"
                )

        logger.info(f"Done. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")
