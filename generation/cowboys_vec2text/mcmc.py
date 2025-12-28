"""Preconditioned Crank-Nicolson (pCN) MCMC Sampler.

Implements the core COWBOYS algorithm for latent space optimization.
Replaces gradient-based optimization with probabilistic sampling that
respects the VAE's Gaussian prior while exploring high-EI regions.

This is the instruction-only version (no exemplars).

Key insight: pCN proposals preserve N(0,I) as the stationary distribution,
making them ideal for VAE latent spaces where the prior is Gaussian.

Reference: COWBOYS paper - "Return of the Latent Space COWBOYS"
"""

import torch
import gpytorch
from botorch.acquisition.analytic import LogExpectedImprovement
from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .vae import InstructionVAE
    from robust_vec2text.exemplar_selector import InstructionSelector
    from .trust_region import TrustRegionManager


@dataclass
class MCMCConfig:
    """Configuration for pCN MCMC sampling.

    Attributes:
        n_steps: Total MCMC steps per chain
        beta: pCN step size - controls exploration vs exploitation
              Higher beta = larger steps = more exploration
              Optimal range: 0.05-0.3 for 32D latent space
        n_chains: Number of parallel MCMC chains
        warmup_steps: Burn-in steps before collecting samples
        thinning: Keep every N-th sample to reduce autocorrelation
        target_accept_rate: Optimal acceptance rate for MH (0.234 is theoretical optimum)
        adapt_beta: Whether to adapt beta during warmup
        min_beta: Minimum beta value during adaptation
        max_beta: Maximum beta value during adaptation
    """

    n_steps: int = 500
    beta: float = 0.1
    n_chains: int = 5
    warmup_steps: int = 50
    thinning: int = 10
    target_accept_rate: float = 0.234
    adapt_beta: bool = True
    min_beta: float = 0.01
    max_beta: float = 0.5


@dataclass
class MCMCResult:
    """Result of MCMC sampling."""

    candidates: List[torch.Tensor]  # All collected samples
    best_latent: torch.Tensor  # Best sample by log EI
    best_log_ei: float  # Log EI of best sample
    accept_rate: float  # Overall acceptance rate
    final_beta: float  # Final adapted beta value


class NegatedGPModelWrapper(torch.nn.Module):
    """Wrapper that negates GP predictions for BoTorch maximization.

    BoTorch assumes higher predictions = better. Our GP predicts error rates
    where lower = better. This wrapper outputs -predictions so BoTorch
    correctly maximizes (which corresponds to minimizing error).
    """

    def __init__(self, gp_model, likelihood):
        super().__init__()
        self.gp_model = gp_model
        self.likelihood = likelihood
        self.train_inputs = gp_model.train_inputs
        self.train_targets = -gp_model.train_targets
        self.num_outputs = 1

    def forward(self, X):
        self.gp_model.eval()
        self.likelihood.eval()
        pred = self.gp_model(X)
        return gpytorch.distributions.MultivariateNormal(-pred.mean, pred.covariance_matrix)

    def posterior(self, X, observation_noise=False, **kwargs):
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        self.gp_model.eval()
        with gpytorch.settings.fast_pred_var():
            pred = self.gp_model(X.squeeze(-2) if X.dim() > 2 else X)
            negated_dist = gpytorch.distributions.MultivariateNormal(
                -pred.mean, pred.covariance_matrix
            )
        return GPyTorchPosterior(negated_dist)


class pCNSampler:
    """Preconditioned Crank-Nicolson MCMC for latent space optimization.

    This is the instruction-only version (no exemplars).

    Implements the pCN proposal:
        z_new = sqrt(1 - beta^2) * z_old + beta * epsilon,  epsilon ~ N(0, I)

    This proposal preserves the N(0, I) prior as the stationary distribution,
    making it ideal for VAE latent spaces.

    Uses Metropolis-Hastings acceptance based on log Expected Improvement.
    Combined with Trust Regions for constrained exploration.

    Key advantages over gradient ascent:
    1. Naturally respects the VAE prior (avoids "holes" in latent space)
    2. Explores multiple modes instead of converging to one
    3. Produces diverse set of candidates for evaluation
    4. No gradient computation required (works with non-differentiable objectives)
    """

    def __init__(
        self,
        vae: "InstructionVAE",
        instruction_selector: "InstructionSelector",
        device: str = "cuda",
    ):
        """Initialize pCN sampler.

        Args:
            vae: Trained VAE for decoding latents to embeddings
            instruction_selector: InstructionSelector GP for error prediction
            device: Computation device
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vae = vae.to(self.device)
        self.instruction_selector = instruction_selector

        # Cache for normalization parameters
        self._log_ei_acqf = None
        self._best_y_cached = None

    def compute_log_ei(
        self,
        z: torch.Tensor,
        best_y: float,
    ) -> torch.Tensor:
        """Compute log Expected Improvement for latent point.

        Pipeline:
            z (32D) -> VAE decode -> inst_emb (768D) -> GP -> predictions -> LogEI

        Uses BoTorch LogExpectedImprovement for numerical stability.

        Args:
            z: Latent point (32,) or (batch, 32)
            best_y: Best observed error rate

        Returns:
            Log EI value(s)
        """
        # Ensure correct shape
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device)

        # 1. Decode VAE latent to instruction embedding
        with torch.no_grad():
            inst_emb = self.vae.decode(z)  # (batch, 768)

        # 2. Prepare GP input - instruction only (768D, not 1536D)
        X = inst_emb  # (batch, 768)

        # Normalize using InstructionSelector's stored params
        selector = self.instruction_selector
        denominator = selector.X_max - selector.X_min
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        X_norm = (X - selector.X_min) / denominator

        # 3. BoTorch LogEI computation
        X_botorch = X_norm.unsqueeze(-2)  # (batch, 1, 768)

        selector.gp_model.eval()
        selector.likelihood.eval()

        # Create or update negated wrapper
        wrapped_model = NegatedGPModelWrapper(selector.gp_model, selector.likelihood)

        # Normalize best_y
        best_y_norm = (best_y - selector.y_mean.item()) / selector.y_std.item()

        log_ei_acqf = LogExpectedImprovement(
            model=wrapped_model,
            best_f=-best_y_norm,  # Negated for minimization
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            log_ei = log_ei_acqf(X_botorch)

        return log_ei.squeeze()

    def pcn_proposal(
        self,
        z_current: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """Generate pCN proposal.

        z_new = sqrt(1 - beta^2) * z_current + beta * epsilon

        This preserves N(0, I) as the stationary distribution because:
        - If z ~ N(0, I), then z_new ~ N(0, I) as well
        - The proposal is reversible with same probability

        Args:
            z_current: Current latent (32,)
            beta: Step size parameter

        Returns:
            Proposal latent (32,)
        """
        epsilon = torch.randn_like(z_current)
        sqrt_term = torch.sqrt(torch.tensor(1.0 - beta**2, device=z_current.device))
        return sqrt_term * z_current + beta * epsilon

    def metropolis_hastings_step(
        self,
        z_current: torch.Tensor,
        log_ei_current: torch.Tensor,
        best_y: float,
        beta: float,
        trust_region: Optional["TrustRegionManager"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Perform single MH step with pCN proposal.

        The acceptance probability is based on the log EI ratio.
        For pCN, the proposal is symmetric so no correction is needed.

        Args:
            z_current: Current latent (32,)
            log_ei_current: Current log EI
            best_y: Best observed error rate
            beta: pCN step size
            trust_region: Optional trust region for constraint checking

        Returns:
            Tuple of (new_z, new_log_ei, accepted)
        """
        # Generate proposal
        z_proposal = self.pcn_proposal(z_current, beta)

        # Trust region check (hard constraint)
        if trust_region is not None:
            if not trust_region.is_within_region(z_proposal):
                return z_current, log_ei_current, False

        # Compute acceptance probability
        log_ei_proposal = self.compute_log_ei(z_proposal, best_y)

        # MH acceptance: log_alpha = log_ei_proposal - log_ei_current
        # pCN proposal is symmetric, so no Hastings correction needed
        log_alpha = log_ei_proposal - log_ei_current

        # Accept with probability min(1, alpha)
        if torch.log(torch.rand(1, device=z_current.device)) < log_alpha:
            return z_proposal, log_ei_proposal, True
        else:
            return z_current, log_ei_current, False

    def sample(
        self,
        initial_latent: torch.Tensor,
        best_y: float,
        config: MCMCConfig,
        trust_region: Optional["TrustRegionManager"] = None,
        verbose: bool = True,
    ) -> MCMCResult:
        """Run pCN MCMC sampling.

        Collects samples from the posterior distribution:
            p(z | best_y) propto p(z) * exp(LogEI(z))

        where p(z) = N(0, I) is the VAE prior.

        Args:
            initial_latent: Starting point (32,)
            best_y: Best observed error rate
            config: MCMC configuration
            trust_region: Optional trust region constraints
            verbose: Print progress

        Returns:
            MCMCResult with collected samples and statistics
        """
        candidates = []
        z_current = initial_latent.clone().to(self.device)
        log_ei_current = self.compute_log_ei(z_current, best_y)

        beta = config.beta
        accept_count = 0
        total_steps = 0

        # Track best sample
        best_z = z_current.clone()
        best_log_ei = log_ei_current.item() if isinstance(log_ei_current, torch.Tensor) else log_ei_current

        for step in range(config.n_steps):
            z_current, log_ei_current, accepted = self.metropolis_hastings_step(
                z_current, log_ei_current, best_y, beta, trust_region
            )

            total_steps += 1
            if accepted:
                accept_count += 1

                # Update best
                current_log_ei = log_ei_current.item() if isinstance(log_ei_current, torch.Tensor) else log_ei_current
                if current_log_ei > best_log_ei:
                    best_log_ei = current_log_ei
                    best_z = z_current.clone()

            # Adaptive beta during warmup
            if config.adapt_beta and step < config.warmup_steps:
                accept_rate = accept_count / (step + 1)
                if accept_rate < config.target_accept_rate:
                    beta = max(config.min_beta, beta * 0.95)
                else:
                    beta = min(config.max_beta, beta * 1.05)

            # Collect samples after warmup with thinning
            if step >= config.warmup_steps and (step - config.warmup_steps) % config.thinning == 0:
                candidates.append(z_current.clone())

        final_accept_rate = accept_count / total_steps if total_steps > 0 else 0.0

        if verbose:
            print(f"    Chain: {len(candidates)} samples, accept_rate={final_accept_rate:.3f}, beta={beta:.4f}")

        return MCMCResult(
            candidates=candidates,
            best_latent=best_z,
            best_log_ei=best_log_ei,
            accept_rate=final_accept_rate,
            final_beta=beta,
        )

    def sample_multiple_chains(
        self,
        initial_latents: List[torch.Tensor],
        best_y: float,
        config: MCMCConfig,
        trust_region: Optional["TrustRegionManager"] = None,
        verbose: bool = True,
    ) -> MCMCResult:
        """Run multiple MCMC chains from different starting points.

        Multiple chains help:
        1. Explore different regions of latent space
        2. Assess convergence (similar results = good mixing)
        3. Increase sample diversity

        Args:
            initial_latents: Starting points for each chain
            best_y: Best observed error rate
            config: MCMC configuration
            trust_region: Optional trust region constraints
            verbose: Print progress

        Returns:
            Combined MCMCResult from all chains
        """
        all_samples = []
        total_accept = 0
        total_steps = 0
        best_z = initial_latents[0].clone()
        best_log_ei = float("-inf")
        final_beta = config.beta

        for i, init_z in enumerate(initial_latents):
            if verbose:
                print(f"  Chain {i + 1}/{len(initial_latents)}")

            result = self.sample(init_z, best_y, config, trust_region, verbose=verbose)

            all_samples.extend(result.candidates)
            total_accept += result.accept_rate * config.n_steps
            total_steps += config.n_steps
            final_beta = result.final_beta  # Use last adapted beta

            if result.best_log_ei > best_log_ei:
                best_log_ei = result.best_log_ei
                best_z = result.best_latent.clone()

        overall_accept_rate = total_accept / total_steps if total_steps > 0 else 0.0

        if verbose:
            print(f"  Total: {len(all_samples)} samples, overall_accept_rate={overall_accept_rate:.3f}")

        return MCMCResult(
            candidates=all_samples,
            best_latent=best_z,
            best_log_ei=best_log_ei,
            accept_rate=overall_accept_rate,
            final_beta=final_beta,
        )
