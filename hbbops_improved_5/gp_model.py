"""
Multi-Fidelity Deep Kernel Gaussian Process for HbBoPs Improved 5

This module implements a GP model that:
1. Uses a structural-aware deep kernel on prompt embeddings
2. Learns correlation between fidelity levels via IndexKernel
3. Combines both through a product kernel: K = K_deep × K_fidelity

NO LOGIT TRANSFORM - GP models accuracy directly in [0, 1] space.
"""

import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel, IndexKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from typing import Dict, Optional
import numpy as np


class FeatureExtractor(nn.Module):
    """
    Structural-aware feature extractor for deep kernel GP.

    Architecture from HbBoPs paper (Section 3.2):
    - Separate encoders for instruction and exemplar embeddings
    - Joint encoder combines both into latent space

    The separate pathways allow the model to learn distinct representations
    for instructions vs exemplars, recognizing their different roles in prompts.

    Input: (instruction_emb[768], exemplar_emb[768]) -> concatenated 1536D
    Output: 10D latent features for GP kernel
    """

    def __init__(self, input_dim: int = 768, latent_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Separate encoder for instructions
        self.instruction_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Separate encoder for exemplars
        self.exemplar_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Joint encoder combining both
        self.joint_encoder = nn.Sequential(
            nn.Linear(2 * 32, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, instruction_emb: torch.Tensor, exemplar_emb: torch.Tensor) -> torch.Tensor:
        """
        Extract latent features from instruction and exemplar embeddings.

        Args:
            instruction_emb: (N, 768) instruction embeddings
            exemplar_emb: (N, 768) exemplar embeddings

        Returns:
            (N, latent_dim) latent features
        """
        inst_features = self.instruction_encoder(instruction_emb)
        ex_features = self.exemplar_encoder(exemplar_emb)
        combined = torch.cat([inst_features, ex_features], dim=1)
        return self.joint_encoder(combined)


class MultiFidelityDeepKernelGP(ExactGP):
    """
    Multi-Fidelity Gaussian Process with structural-aware deep kernel.

    Kernel structure: K = K_deep(z, z') × K_fidelity(f, f')

    where:
    - z = feature_extractor(instruction_emb, exemplar_emb) is 10D latent
    - K_deep is Matern 5/2 ARD kernel on latent space
    - K_fidelity is IndexKernel learning correlation between fidelity levels

    NO LOGIT TRANSFORM - GP models accuracy directly.

    Args:
        train_x: (N, 2*input_dim + 1) embeddings with fidelity idx in last column
        train_y: (N,) standardized accuracy values
        likelihood: FixedNoiseGaussianLikelihood with per-point noise
        feature_extractor: FeatureExtractor instance
        num_fidelities: Number of discrete fidelity levels
        input_dim: Dimension of embeddings (768 for BERT)
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: FixedNoiseGaussianLikelihood,
        feature_extractor: FeatureExtractor,
        num_fidelities: int,
        input_dim: int = 768
    ):
        # train_x has shape (N, 2*input_dim + 1), last column is fidelity_idx
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()

        # Deep kernel on latent space (latent_dim dimensions)
        self.deep_kernel = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=feature_extractor.latent_dim,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
        )

        # Fidelity correlation kernel
        # IndexKernel learns a num_fidelities × num_fidelities covariance matrix
        # rank determines the low-rank approximation (full rank = num_fidelities)
        self.fidelity_kernel = IndexKernel(
            num_tasks=num_fidelities,
            rank=min(num_fidelities, 3)  # Low-rank for efficiency
        )

        self.feature_extractor = feature_extractor
        self.input_dim = input_dim
        self.num_fidelities = num_fidelities

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass computing GP prior distribution.

        Args:
            x: (N, 2*input_dim + 1) embeddings with fidelity in last column

        Returns:
            MultivariateNormal distribution over function values
        """
        # Split input: embeddings and fidelity index
        embeddings = x[:, :-1]  # (N, 2*input_dim)
        fidelity_idx = x[:, -1].long()  # (N,)

        # Split embeddings into instruction and exemplar
        instruction_emb = embeddings[:, :self.input_dim]
        exemplar_emb = embeddings[:, self.input_dim:]

        # Extract latent features via deep kernel network
        latent = self.feature_extractor(instruction_emb, exemplar_emb)

        # Compute deep kernel on latent space
        covar_deep = self.deep_kernel(latent)

        # Compute fidelity kernel
        # IndexKernel expects indices as (N, 1) tensor
        fidelity_idx_expanded = fidelity_idx.unsqueeze(-1)
        covar_fidelity = self.fidelity_kernel(fidelity_idx_expanded)

        # Product kernel: element-wise (Hadamard) product of covariance matrices
        covar = covar_deep.mul(covar_fidelity)

        # Mean function (zero mean)
        mean = self.mean_module(latent)

        return MultivariateNormal(mean, covar)


def prepare_gp_input(X_embeddings: torch.Tensor, fidelity_idx: torch.Tensor) -> torch.Tensor:
    """
    Prepare input for MultiFidelityDeepKernelGP by concatenating embeddings with fidelity.

    Args:
        X_embeddings: (N, 2*input_dim) concatenated instruction+exemplar embeddings
        fidelity_idx: (N,) fidelity indices

    Returns:
        (N, 2*input_dim + 1) tensor ready for GP
    """
    fidelity_col = fidelity_idx.float().unsqueeze(-1)
    return torch.cat([X_embeddings, fidelity_col], dim=1)


class GPNormalizationParams:
    """
    Container for normalization parameters needed for inference.

    These parameters are saved with the model checkpoint and restored
    when loading a pre-trained GP for inference on new prompts.

    NO LOGIT TRANSFORM - uses accuracy directly.
    """

    def __init__(
        self,
        X_min: torch.Tensor,
        X_max: torch.Tensor,
        y_mean: float,
        y_std: float,
        fidelity_to_idx: Dict[int, int],
        max_fidelity_idx: int,
        epsilon: float = 0.001
    ):
        self.X_min = X_min
        self.X_max = X_max
        self.y_mean = y_mean
        self.y_std = y_std
        self.fidelity_to_idx = fidelity_to_idx
        self.max_fidelity_idx = max_fidelity_idx
        self.epsilon = epsilon

    def normalize_x(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize input embeddings to unit cube [0, 1]."""
        denominator = self.X_max - self.X_min
        denominator[denominator == 0] = 1.0
        return (X - self.X_min) / denominator

    def standardize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Standardize y values (zero mean, unit variance)."""
        return (y - self.y_mean) / self.y_std

    def destandardize_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Convert standardized values back to raw accuracy."""
        return y_norm * self.y_std + self.y_mean

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'X_min': self.X_min.cpu() if isinstance(self.X_min, torch.Tensor) else self.X_min,
            'X_max': self.X_max.cpu() if isinstance(self.X_max, torch.Tensor) else self.X_max,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'fidelity_to_idx': self.fidelity_to_idx,
            'max_fidelity_idx': self.max_fidelity_idx,
            'epsilon': self.epsilon
        }

    @classmethod
    def from_dict(cls, d: dict, device: torch.device = None) -> 'GPNormalizationParams':
        """Create from dictionary."""
        device = device or torch.device('cpu')
        X_min = d['X_min']
        X_max = d['X_max']
        if isinstance(X_min, torch.Tensor):
            X_min = X_min.to(device)
            X_max = X_max.to(device)
        return cls(
            X_min=X_min,
            X_max=X_max,
            y_mean=d['y_mean'],
            y_std=d['y_std'],
            fidelity_to_idx=d['fidelity_to_idx'],
            max_fidelity_idx=d['max_fidelity_idx'],
            epsilon=d.get('epsilon', 0.001)
        )
