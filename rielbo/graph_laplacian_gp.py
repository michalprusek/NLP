"""Graph Laplacian GP for manifold-aware Bayesian Optimization.

Uses Graph Laplacian eigenvectors (spectral embedding) to define a kernel
that respects the topology of the data manifold. This kernel "sees" holes
in the latent space - if two points are close in Euclidean distance but
have no valid path between them, the graph distance will be large.

Key insight: Euclidean distance in VAE latent space doesn't reflect
semantic distance. Graph Laplacian kernel measures distance along the
data manifold, avoiding "forbidden zones" (invalid molecule regions).

Algorithm:
1. Build k-NN graph from labeled points + unlabeled anchors
2. Compute normalized Graph Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
3. Compute top-k eigenvectors (smallest eigenvalues, excluding first)
4. Spectral embedding: φ(x) = [v_1(x), v_2(x), ..., v_k(x)]
5. Heat kernel: k(x,y) = exp(-||φ(x) - φ(y)||^2 / (2σ^2))

This is equivalent to diffusion distance on the graph.

References:
- Belkin & Niyogi (2003) "Laplacian Eigenmaps for Dimensionality Reduction"
- Coifman & Lafon (2006) "Diffusion Maps"
- Zhu et al. (2003) "Semi-Supervised Learning Using Gaussian Fields"
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import numpy as np

import gpytorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

logger = logging.getLogger(__name__)


class SpectralEmbedding:
    """Spectral embedding via Graph Laplacian eigenvectors.

    Computes a low-dimensional embedding that preserves graph structure.
    Points connected by edges are close in the embedding space.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        n_components: int = 32,
        sigma: float = "auto",  # Auto-compute from data
        device: str = "cuda",
    ):
        """
        Args:
            n_neighbors: Number of neighbors for k-NN graph
            n_components: Number of eigenvectors to use (embedding dimension)
            sigma: Bandwidth for Gaussian edge weights ("auto" to compute from data)
            device: Torch device
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self._sigma_input = sigma
        self.sigma = None  # Computed at fit time
        self.device = device

        # Computed at fit time
        self.anchor_points = None  # [n_anchors, D]
        self.eigenvectors = None   # [n_anchors, n_components]
        self.eigenvalues = None    # [n_components]

    def _compute_adaptive_sigma(self, dists_sq: torch.Tensor) -> float:
        """Compute adaptive sigma based on local distances.

        Uses median of k-th nearest neighbor distances (robust to outliers).
        """
        # Get k-th nearest neighbor distance for each point (excluding self)
        k = min(self.n_neighbors, dists_sq.shape[0] - 1)
        knn_dists_sq, _ = torch.topk(dists_sq, k=k+1, largest=False, dim=1)
        knn_dists_sq = knn_dists_sq[:, 1:]  # Exclude self

        # Use median of average k-NN distance
        median_dist = torch.sqrt(knn_dists_sq.mean(dim=1)).median().item()

        # Sigma = median distance (heuristic that works well in practice)
        return max(median_dist, 0.01)  # Floor to avoid numerical issues

    def _build_knn_graph(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build k-NN graph adjacency matrix with Gaussian weights.

        Args:
            X: Points [N, D]

        Returns:
            (Adjacency matrix [N, N], squared distances [N, N])
        """
        N = X.shape[0]

        # Compute pairwise distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
        dists_sq = X_norm_sq + X_norm_sq.T - 2 * X @ X.T
        dists_sq = dists_sq.clamp(min=0)  # Numerical stability

        # Auto-compute sigma if needed
        if self._sigma_input == "auto":
            self.sigma = self._compute_adaptive_sigma(dists_sq)
            logger.info(f"Auto-computed sigma: {self.sigma:.4f}")
        else:
            self.sigma = self._sigma_input

        # Find k nearest neighbors for each point
        _, indices = torch.topk(dists_sq, k=self.n_neighbors + 1, largest=False, dim=1)

        # Build adjacency matrix with Gaussian weights
        # A[i,j] = exp(-||x_i - x_j||^2 / (2σ^2)) if j is neighbor of i
        A = torch.zeros(N, N, device=self.device)

        for i in range(N):
            neighbors = indices[i, 1:]  # Exclude self (index 0)
            weights = torch.exp(-dists_sq[i, neighbors] / (2 * self.sigma ** 2))
            A[i, neighbors] = weights

        # Symmetrize: A = (A + A^T) / 2
        A = (A + A.T) / 2

        return A, dists_sq

    def _compute_laplacian_eigenvectors(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvectors of normalized Graph Laplacian.

        L_sym = I - D^(-1/2) A D^(-1/2)

        Returns eigenvectors corresponding to smallest eigenvalues
        (excluding the trivial zero eigenvalue).
        """
        N = A.shape[0]

        # Degree matrix
        D = A.sum(dim=1)
        D_inv_sqrt = torch.zeros_like(D)
        nonzero = D > 1e-10
        D_inv_sqrt[nonzero] = 1.0 / torch.sqrt(D[nonzero])

        # Normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
        # Equivalent to: L_sym = D^(-1/2) (D - A) D^(-1/2)
        D_inv_sqrt_mat = torch.diag(D_inv_sqrt)
        L_sym = torch.eye(N, device=self.device) - D_inv_sqrt_mat @ A @ D_inv_sqrt_mat

        # Convert to numpy for eigsh (sparse eigensolver)
        L_np = L_sym.cpu().numpy()

        # Compute smallest k+1 eigenvectors (k+1 because we skip the first one)
        n_compute = min(self.n_components + 1, N - 1)

        try:
            eigenvalues, eigenvectors = eigsh(
                csr_matrix(L_np),
                k=n_compute,
                which='SM',  # Smallest magnitude
                return_eigenvectors=True,
            )
        except Exception as e:
            logger.warning(f"eigsh failed: {e}, falling back to dense eigenvector computation")
            eigenvalues, eigenvectors = np.linalg.eigh(L_np)
            eigenvalues = eigenvalues[:n_compute]
            eigenvectors = eigenvectors[:, :n_compute]

        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Skip first eigenvector (constant, eigenvalue ≈ 0)
        eigenvalues = eigenvalues[1:self.n_components + 1]
        eigenvectors = eigenvectors[:, 1:self.n_components + 1]

        return (
            torch.tensor(eigenvalues, device=self.device, dtype=torch.float32),
            torch.tensor(eigenvectors, device=self.device, dtype=torch.float32),
        )

    def fit(self, anchor_points: torch.Tensor) -> "SpectralEmbedding":
        """Fit spectral embedding on anchor points.

        Args:
            anchor_points: Points to build graph from [N, D]
        """
        self.anchor_points = anchor_points.to(self.device)

        logger.info(f"Building k-NN graph (k={self.n_neighbors}) on {len(anchor_points)} points...")
        A, _ = self._build_knn_graph(self.anchor_points)

        logger.info(f"Computing {self.n_components} Laplacian eigenvectors...")
        self.eigenvalues, self.eigenvectors = self._compute_laplacian_eigenvectors(A)

        logger.info(f"Spectral embedding ready. Eigenvalue range: [{self.eigenvalues.min():.4f}, {self.eigenvalues.max():.4f}]")

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform new points to spectral embedding space.

        Uses Nyström extension: interpolate eigenvectors based on
        similarity to anchor points.

        Args:
            X: New points [..., M, D] - supports batch dimensions

        Returns:
            Spectral coordinates [..., M, n_components]
        """
        if self.anchor_points is None:
            raise RuntimeError("Must call fit() before transform()")

        X = X.to(self.device)
        input_dtype = X.dtype

        # Handle batch dimensions: reshape to [batch_size, D] for computation
        original_shape = X.shape[:-1]  # All dimensions except last (D)
        D = X.shape[-1]
        X_flat = X.reshape(-1, D)  # [total_points, D]

        M = X_flat.shape[0]
        N = self.anchor_points.shape[0]

        # Convert anchor points to same dtype as input (GPyTorch uses double)
        anchor_pts = self.anchor_points.to(input_dtype)

        # Compute distances to anchor points
        # ||x - a||^2 for each x in X and a in anchor_points
        X_norm_sq = (X_flat ** 2).sum(dim=1, keepdim=True)  # [M, 1]
        A_norm_sq = (anchor_pts ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        dists_sq = X_norm_sq + A_norm_sq.T - 2 * X_flat @ anchor_pts.T  # [M, N]
        dists_sq = dists_sq.clamp(min=0)

        # Gaussian weights to anchor points
        weights = torch.exp(-dists_sq / (2 * self.sigma ** 2))  # [M, N]

        # Normalize weights
        weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-10)
        weights = weights / weights_sum  # [M, N]

        # Interpolate eigenvectors: φ(x) = Σ_a w(x,a) * v(a)
        eigenvecs = self.eigenvectors.to(input_dtype)
        embedding_flat = weights @ eigenvecs  # [M, n_components]

        # Reshape back to original batch shape
        output_shape = original_shape + (self.n_components,)
        embedding = embedding_flat.reshape(output_shape)

        return embedding

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        self.fit(X)
        return self.eigenvectors  # For anchor points, just return eigenvectors


class GraphLaplacianKernel(gpytorch.kernels.Kernel):
    """Kernel based on distance in spectral embedding space.

    k(x, y) = exp(-||φ(x) - φ(y)||^2 / (2l^2))

    where φ is the spectral embedding from Graph Laplacian eigenvectors.
    """

    has_lengthscale = True

    def __init__(self, spectral_embedding: SpectralEmbedding, **kwargs):
        super().__init__(**kwargs)
        self.spectral_embedding = spectral_embedding

    def forward(self, x1, x2, diag=False, **params):
        # Transform to spectral space
        # x1: [..., N1, D] -> phi1: [..., N1, n_components]
        # x2: [..., N2, D] -> phi2: [..., N2, n_components]
        phi1 = self.spectral_embedding.transform(x1)
        phi2 = self.spectral_embedding.transform(x2)

        # RBF kernel in spectral space
        if diag:
            # phi1, phi2 have same shape [..., N, n_components]
            diff = phi1 - phi2
            dist_sq = (diff ** 2).sum(dim=-1)  # [..., N]
        else:
            # Compute pairwise squared distances
            # phi1: [..., N1, K], phi2: [..., N2, K]
            # Result: [..., N1, N2]
            phi1_sq = (phi1 ** 2).sum(dim=-1, keepdim=True)  # [..., N1, 1]
            phi2_sq = (phi2 ** 2).sum(dim=-1, keepdim=True)  # [..., N2, 1]

            # For broadcasting: phi2_sq needs to be [..., 1, N2]
            phi2_sq = phi2_sq.transpose(-1, -2)  # [..., 1, N2]

            # Matrix multiply: phi1 @ phi2.T -> [..., N1, N2]
            cross_term = torch.matmul(phi1, phi2.transpose(-1, -2))

            dist_sq = phi1_sq + phi2_sq - 2 * cross_term  # [..., N1, N2]

        dist_sq = dist_sq.clamp(min=0)

        # RBF kernel
        return torch.exp(-dist_sq / (2 * self.lengthscale ** 2))


class GraphLaplacianBO:
    """Bayesian Optimization with Graph Laplacian kernel.

    Uses spectral embedding to define a kernel that respects the topology
    of the data manifold. This helps avoid sampling in "forbidden zones"
    (regions with no valid molecules).

    The key idea:
    1. Build graph from labeled + unlabeled anchor points
    2. Compute spectral embedding via Laplacian eigenvectors
    3. Use RBF kernel in spectral space for GP
    4. This kernel "sees" holes in the data

    Improvements over basic implementation:
    - Adaptive sigma based on local distances
    - Dynamic trust region that expands when stuck
    - Periodic graph rebuilding to incorporate new points
    - Perturbation on duplicate detection
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        device: str = "cuda",
        # Graph parameters
        n_neighbors: int = 15,
        n_components: int = 32,
        n_anchors: int = 2000,
        graph_sigma: str = "auto",  # Auto-compute from data
        # BO parameters
        n_candidates: int = 2000,
        acqf: str = "ts",
        trust_region: float = 0.5,  # Start smaller for more focused search
        trust_region_max: float = 1.5,  # Can expand when stuck
        # Adaptive parameters
        rebuild_interval: int = 50,  # Rebuild graph every N iterations
        stuck_threshold: int = 20,  # Expand trust region after N iterations without improvement
        perturbation_scale: float = 0.1,  # Scale for perturbation on duplicate
        seed: int = 42,
        verbose: bool = True,
    ):
        self.device = device
        self.codec = codec
        self.oracle = oracle
        self.input_dim = input_dim
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_anchors = n_anchors
        self.graph_sigma = graph_sigma
        self.n_candidates = n_candidates
        self.acqf = acqf
        self.trust_region_init = trust_region
        self.trust_region = trust_region
        self.trust_region_max = trust_region_max
        self.rebuild_interval = rebuild_interval
        self.stuck_threshold = stuck_threshold
        self.perturbation_scale = perturbation_scale
        self.verbose = verbose
        self.seed = seed

        torch.manual_seed(seed)

        # Spectral embedding (computed at cold start)
        self.spectral_embedding = None
        self.anchor_smiles = []  # Track anchor molecules

        # GP
        self.gp = None

        # Training data
        self.train_X = None
        self.train_Y = None
        self.mean_norm = None
        self.smiles_observed = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.iteration = 0

        # Adaptive tracking
        self.iterations_since_improvement = 0
        self.last_best_score = float("-inf")

        self.history = {
            "iteration": [],
            "best_score": [],
            "current_score": [],
            "n_evaluated": [],
            "trust_region": [],
        }

        logger.info(
            f"Graph Laplacian BO: {input_dim}D, k={n_neighbors}, "
            f"components={n_components}, anchors={n_anchors}, sigma={graph_sigma}"
        )

    def _load_anchor_points(self, n_anchors: int) -> tuple[torch.Tensor, list[str]]:
        """Load unlabeled anchor points from dataset."""
        from shared.guacamol.data import load_guacamol_data

        logger.info(f"Loading {n_anchors} anchor points...")
        smiles_list, _, _ = load_guacamol_data(n_samples=n_anchors)

        # Encode to embeddings
        from tqdm import tqdm
        embeddings = []
        for i in tqdm(range(0, len(smiles_list), 64), desc="Encoding anchors", disable=not self.verbose):
            batch = smiles_list[i:i+64]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0).to(self.device), smiles_list

    def _build_spectral_embedding(self, train_X: torch.Tensor) -> None:
        """Build spectral embedding from training + anchor points."""
        # Load anchor points
        anchor_embeddings, self.anchor_smiles = self._load_anchor_points(self.n_anchors)

        # Combine with training points
        all_points = torch.cat([train_X, anchor_embeddings], dim=0)

        # Normalize to unit sphere
        all_points = F.normalize(all_points, p=2, dim=-1)

        # Compute spectral embedding with adaptive sigma
        self.spectral_embedding = SpectralEmbedding(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            sigma=self.graph_sigma,
            device=self.device,
        )
        self.spectral_embedding.fit(all_points)

    def _rebuild_spectral_embedding(self) -> None:
        """Rebuild spectral embedding with current training data."""
        logger.info(f"Rebuilding spectral embedding with {len(self.train_X)} training points...")

        # Use current training points + fresh anchors
        anchor_embeddings, self.anchor_smiles = self._load_anchor_points(self.n_anchors)

        # Combine
        all_points = torch.cat([self.train_X, anchor_embeddings], dim=0)
        all_points = F.normalize(all_points, p=2, dim=-1)

        # Recompute
        self.spectral_embedding = SpectralEmbedding(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            sigma=self.graph_sigma,
            device=self.device,
        )
        self.spectral_embedding.fit(all_points)

    def _fit_gp(self):
        """Fit GP with Graph Laplacian kernel."""
        # Normalize training points
        X_norm = F.normalize(self.train_X, p=2, dim=-1)

        X = X_norm.double()
        Y = self.train_Y.double().unsqueeze(-1)

        # Create kernel
        covar_module = gpytorch.kernels.ScaleKernel(
            GraphLaplacianKernel(self.spectral_embedding)
        ).to(self.device)

        self.gp = SingleTaskGP(X, Y, covar_module=covar_module).to(self.device)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            logger.warning(f"GP fit failed: {e}")

        self.gp.eval()

    def _generate_candidates(self, n_candidates: int) -> torch.Tensor:
        """Generate candidates in trust region around best point.

        Uses a combination of:
        1. Trust region sampling around best point
        2. Some global exploration candidates
        """
        best_idx = self.train_Y.argmax()
        x_best = F.normalize(self.train_X[best_idx:best_idx+1], p=2, dim=-1)

        # 80% local, 20% global exploration
        n_local = int(n_candidates * 0.8)
        n_global = n_candidates - n_local

        # Local candidates: tangent space sampling around best
        tangent = torch.randn(n_local, self.input_dim, device=self.device)
        proj = (tangent * x_best).sum(dim=-1, keepdim=True) * x_best
        tangent = tangent - proj
        tangent = F.normalize(tangent, p=2, dim=-1)

        # Sample angles within trust region
        max_angle = self.trust_region * torch.pi / 4
        angles = torch.rand(n_local, 1, device=self.device) * max_angle

        local_candidates = torch.cos(angles) * x_best + torch.sin(angles) * tangent
        local_candidates = F.normalize(local_candidates, p=2, dim=-1)

        # Global candidates: random points on sphere
        global_candidates = torch.randn(n_global, self.input_dim, device=self.device)
        global_candidates = F.normalize(global_candidates, p=2, dim=-1)

        candidates = torch.cat([local_candidates, global_candidates], dim=0)
        return candidates

    def _perturb_point(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Add random perturbation to escape local region."""
        noise = torch.randn_like(x) * scale
        perturbed = x + noise
        return F.normalize(perturbed, p=2, dim=-1)

    def _optimize_acquisition(self) -> torch.Tensor:
        """Find optimal point using Thompson Sampling or EI."""
        candidates = self._generate_candidates(self.n_candidates)

        if self.acqf == "ts":
            thompson = MaxPosteriorSampling(model=self.gp, replacement=False)
            x_opt = thompson(candidates.double().unsqueeze(0), num_samples=1)
            x_opt = x_opt.squeeze(0).float()
        elif self.acqf == "ei":
            ei = qExpectedImprovement(self.gp, best_f=self.train_Y.max().double())
            with torch.no_grad():
                ei_vals = ei(candidates.double().unsqueeze(-2))
            best_idx = ei_vals.argmax()
            x_opt = candidates[best_idx:best_idx+1]
        else:
            raise ValueError(f"Unknown acqf: {self.acqf}")

        return F.normalize(x_opt, p=2, dim=-1)

    def _update_trust_region(self, improved: bool):
        """Adaptive trust region based on progress."""
        if improved:
            # Shrink trust region when improving (focus exploitation)
            self.trust_region = max(self.trust_region * 0.9, self.trust_region_init * 0.5)
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

            # Expand trust region when stuck
            if self.iterations_since_improvement >= self.stuck_threshold:
                self.trust_region = min(self.trust_region * 1.2, self.trust_region_max)
                logger.info(f"Expanding trust region to {self.trust_region:.2f} (stuck for {self.iterations_since_improvement} iters)")

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules."""
        logger.info(f"Cold start: {len(smiles_list)} molecules")

        from tqdm import tqdm
        embeddings = []
        for i in tqdm(range(0, len(smiles_list), 64), desc="Encoding", disable=not self.verbose):
            batch = smiles_list[i:i+64]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0).to(self.device)

        self.mean_norm = embeddings.norm(dim=-1).mean().item()
        logger.info(f"Mean embedding norm: {self.mean_norm:.2f}")

        # Store data
        self.train_X = embeddings
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        # Track best
        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]
        self.last_best_score = self.best_score

        # Build spectral embedding (this is the expensive part)
        self._build_spectral_embedding(embeddings)

        # Fit GP
        self._fit_gp()

        logger.info(f"Cold start done. Best: {self.best_score:.4f}")

    def step(self) -> dict:
        """One BO iteration."""
        self.iteration += 1

        x_opt = self._optimize_acquisition()
        x_opt_scaled = x_opt * self.mean_norm

        smiles_list = self.codec.decode(x_opt_scaled)
        smiles = smiles_list[0] if smiles_list else ""

        # Handle duplicates with perturbation
        attempts = 0
        max_attempts = 5
        while (not smiles or smiles in self.smiles_observed) and attempts < max_attempts:
            attempts += 1
            # Perturb and retry
            x_opt = self._perturb_point(x_opt, self.perturbation_scale * attempts)
            x_opt_scaled = x_opt * self.mean_norm
            smiles_list = self.codec.decode(x_opt_scaled)
            smiles = smiles_list[0] if smiles_list else ""

        if not smiles or smiles in self.smiles_observed:
            return {"score": 0.0, "best_score": self.best_score,
                    "smiles": smiles, "is_duplicate": True}

        score = self.oracle.score(smiles)

        # Update training data
        self.train_X = torch.cat([self.train_X, x_opt_scaled], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device, dtype=torch.float32)])
        self.smiles_observed.append(smiles)

        # Check for improvement
        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_smiles = smiles
            logger.info(f"New best! {score:.4f}: {smiles}")

        # Update trust region
        self._update_trust_region(improved)

        # Periodic operations
        # Refit GP every 10 iterations
        if self.iteration % 10 == 0:
            self._fit_gp()

        # Rebuild graph periodically to incorporate new points
        if self.iteration % self.rebuild_interval == 0:
            self._rebuild_spectral_embedding()
            self._fit_gp()

        return {"score": score, "best_score": self.best_score,
                "smiles": smiles, "is_duplicate": False}

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        from tqdm import tqdm

        logger.info(f"Graph Laplacian BO: {n_iterations} iterations")

        pbar = tqdm(range(n_iterations), desc="Optimizing")
        n_dup = 0

        for i in pbar:
            result = self.step()

            self.history["iteration"].append(i)
            self.history["best_score"].append(self.best_score)
            self.history["current_score"].append(result["score"])
            self.history["n_evaluated"].append(len(self.smiles_observed))
            self.history["trust_region"].append(self.trust_region)

            if result["is_duplicate"]:
                n_dup += 1

            pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "dup": n_dup,
                "tr": f"{self.trust_region:.2f}",
            })

            if (i + 1) % log_interval == 0 and self.verbose:
                logger.info(
                    f"Iter {i+1}/{n_iterations} | Best: {self.best_score:.4f} | "
                    f"Curr: {result['score']:.4f} | TR: {self.trust_region:.2f}"
                )

        logger.info(f"Done. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")
