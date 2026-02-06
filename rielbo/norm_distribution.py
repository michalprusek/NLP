"""Probabilistic norm reconstruction for spherical embeddings.

Instead of using a fixed mean norm for reconstruction, this models the
distribution of norms and samples multiple candidates for evaluation.

This provides diversity in reconstruction and can be combined with GP
uncertainty to select the best candidate.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NormDistribution:
    """Model the distribution of embedding norms.

    Supports:
    - Gaussian: Simple mean/std model
    - Histogram: Non-parametric empirical distribution
    - GMM: Gaussian Mixture Model for multimodal distributions
    """

    def __init__(
        self,
        method: str = "gaussian",
        n_bins: int = 50,
        n_components: int = 3,
        device: str = "cuda",
    ):
        """
        Args:
            method: "gaussian", "histogram", or "gmm"
            n_bins: Number of bins for histogram method
            n_components: Number of components for GMM
            device: Torch device
        """
        self.method = method
        self.n_bins = n_bins
        self.n_components = n_components
        self.device = device

        # Parameters (fitted)
        self.mean = None
        self.std = None
        self.min_norm = None
        self.max_norm = None

        # For histogram
        self.bin_edges = None
        self.bin_probs = None

        # For GMM
        self.gmm_means = None
        self.gmm_stds = None
        self.gmm_weights = None

    def fit(self, norms: torch.Tensor) -> "NormDistribution":
        """Fit distribution from observed norms.

        Args:
            norms: Tensor of shape [N] containing embedding norms

        Returns:
            self for chaining
        """
        norms = norms.float().to(self.device)

        # Basic statistics
        self.mean = norms.mean().item()
        self.std = norms.std().item()
        self.min_norm = norms.min().item()
        self.max_norm = norms.max().item()

        if self.method == "gaussian":
            # Already computed
            pass

        elif self.method == "histogram":
            # Build histogram
            self.bin_edges = torch.linspace(
                self.min_norm, self.max_norm, self.n_bins + 1,
                device=self.device
            )
            counts = torch.histc(norms, bins=self.n_bins, min=self.min_norm, max=self.max_norm)
            self.bin_probs = counts / counts.sum()

        elif self.method == "gmm":
            # Simple EM for GMM
            self._fit_gmm(norms)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        logger.info(
            f"NormDistribution fitted: mean={self.mean:.2f}, std={self.std:.2f}, "
            f"range=[{self.min_norm:.2f}, {self.max_norm:.2f}]"
        )

        return self

    def _fit_gmm(self, norms: torch.Tensor, n_iter: int = 50):
        """Fit GMM using simple EM algorithm."""
        n = len(norms)
        k = self.n_components

        # Initialize with k-means++
        self.gmm_means = torch.zeros(k, device=self.device)
        self.gmm_stds = torch.ones(k, device=self.device) * self.std / k
        self.gmm_weights = torch.ones(k, device=self.device) / k

        # Random initialization
        indices = torch.randperm(n)[:k]
        self.gmm_means = norms[indices].clone()

        for _ in range(n_iter):
            # E-step: compute responsibilities
            log_probs = torch.zeros(n, k, device=self.device)
            for j in range(k):
                diff = norms - self.gmm_means[j]
                log_probs[:, j] = (
                    torch.log(self.gmm_weights[j] + 1e-10)
                    - 0.5 * torch.log(2 * torch.pi * self.gmm_stds[j]**2 + 1e-10)
                    - 0.5 * diff**2 / (self.gmm_stds[j]**2 + 1e-10)
                )

            # Normalize responsibilities
            log_sum = torch.logsumexp(log_probs, dim=1, keepdim=True)
            resp = torch.exp(log_probs - log_sum)  # [n, k]

            # M-step: update parameters
            Nk = resp.sum(dim=0) + 1e-10  # [k]
            self.gmm_weights = Nk / n
            self.gmm_means = (resp.T @ norms) / Nk  # [k]

            for j in range(k):
                diff = norms - self.gmm_means[j]
                self.gmm_stds[j] = torch.sqrt((resp[:, j] * diff**2).sum() / Nk[j] + 1e-6)

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample norms from the fitted distribution.

        Args:
            n_samples: Number of samples

        Returns:
            Tensor of shape [n_samples] containing sampled norms
        """
        if self.mean is None:
            raise RuntimeError("Must call fit() before sample()")

        if self.method == "gaussian":
            samples = torch.randn(n_samples, device=self.device) * self.std + self.mean

        elif self.method == "histogram":
            # Sample bin indices according to probabilities
            bin_indices = torch.multinomial(self.bin_probs, n_samples, replacement=True)

            # Sample uniformly within each bin
            bin_width = (self.max_norm - self.min_norm) / self.n_bins
            bin_starts = self.bin_edges[bin_indices]
            samples = bin_starts + torch.rand(n_samples, device=self.device) * bin_width

        elif self.method == "gmm":
            # Sample component assignments
            comp_indices = torch.multinomial(self.gmm_weights, n_samples, replacement=True)

            # Sample from each component
            samples = torch.randn(n_samples, device=self.device)
            samples = samples * self.gmm_stds[comp_indices] + self.gmm_means[comp_indices]

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Clamp to observed range
        samples = samples.clamp(self.min_norm * 0.9, self.max_norm * 1.1)

        return samples


class ProbabilisticReconstructor:
    """Reconstruct embeddings with probabilistic norm sampling.

    Instead of using a single mean norm, samples multiple norms and
    selects the best candidate based on GP prediction or decoding success.
    """

    def __init__(
        self,
        norm_dist: NormDistribution,
        n_candidates: int = 5,
        selection: str = "gp_mean",  # "gp_mean", "gp_ucb", "random"
        device: str = "cuda",
    ):
        """
        Args:
            norm_dist: Fitted NormDistribution
            n_candidates: Number of norm candidates to try
            selection: How to select best candidate
            device: Torch device
        """
        self.norm_dist = norm_dist
        self.n_candidates = n_candidates
        self.selection = selection
        self.device = device

    def reconstruct(
        self,
        direction: torch.Tensor,
        gp=None,
        project_fn=None,
    ) -> tuple[torch.Tensor, dict]:
        """Reconstruct embedding from direction with probabilistic norm.

        Args:
            direction: Unit direction [1, D] or [D]
            gp: Optional GP model for selection
            project_fn: Optional function to project to GP space

        Returns:
            (best_embedding, diagnostics_dict)
        """
        direction = F.normalize(direction.reshape(1, -1), p=2, dim=-1)
        d = direction.shape[-1]

        # Sample candidate norms
        norms = self.norm_dist.sample(self.n_candidates)  # [n_candidates]

        # Create candidate embeddings
        candidates = direction * norms.unsqueeze(-1)  # [n_candidates, D]

        diag = {
            "sampled_norms": norms.tolist(),
            "n_candidates": self.n_candidates,
        }

        if self.selection == "random" or gp is None:
            # Random selection
            idx = torch.randint(0, self.n_candidates, (1,)).item()
            best = candidates[idx:idx+1]
            diag["selected_norm"] = norms[idx].item()
            diag["selection_method"] = "random"

        elif self.selection == "gp_mean":
            # Select by highest GP mean
            if project_fn is not None:
                cand_proj = project_fn(F.normalize(candidates, p=2, dim=-1))
            else:
                cand_proj = F.normalize(candidates, p=2, dim=-1)

            with torch.no_grad():
                posterior = gp.posterior(cand_proj.double())
                means = posterior.mean.squeeze()

            idx = means.argmax().item()
            best = candidates[idx:idx+1]
            diag["selected_norm"] = norms[idx].item()
            diag["gp_means"] = means.tolist()
            diag["selection_method"] = "gp_mean"

        elif self.selection == "gp_ucb":
            # Select by highest GP UCB
            if project_fn is not None:
                cand_proj = project_fn(F.normalize(candidates, p=2, dim=-1))
            else:
                cand_proj = F.normalize(candidates, p=2, dim=-1)

            with torch.no_grad():
                posterior = gp.posterior(cand_proj.double())
                means = posterior.mean.squeeze()
                stds = posterior.variance.sqrt().squeeze()
                ucb = means + 2.0 * stds

            idx = ucb.argmax().item()
            best = candidates[idx:idx+1]
            diag["selected_norm"] = norms[idx].item()
            diag["gp_ucb"] = ucb.tolist()
            diag["selection_method"] = "gp_ucb"

        else:
            raise ValueError(f"Unknown selection method: {self.selection}")

        return best, diag

    def reconstruct_batch(
        self,
        directions: torch.Tensor,
        gp=None,
        project_fn=None,
    ) -> tuple[torch.Tensor, dict]:
        """Reconstruct multiple directions.

        For simplicity, uses mean norm for batch reconstruction.
        Use single reconstruct() for probabilistic selection.

        Args:
            directions: Unit directions [N, D]

        Returns:
            (embeddings, diagnostics)
        """
        directions = F.normalize(directions, p=2, dim=-1)
        embeddings = directions * self.norm_dist.mean
        return embeddings, {"method": "batch_mean", "norm": self.norm_dist.mean}
