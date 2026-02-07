"""Acquisition function strategies: TS, EI, UCB, and AcquisitionSchedule.

Extracted from V2's _optimize_acquisition() and _get_effective_acqf().
Consolidates the duplicated TS/EI/UCB switch across V1, V2, VanillaBO.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from botorch.acquisition import qExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP

from rielbo.core.config import AcquisitionConfig, TrustRegionConfig

logger = logging.getLogger(__name__)


class AcquisitionSelector:
    """Selects best candidate using TS, EI, or UCB.

    Stateless — all state comes from arguments.
    """

    def __init__(self, config: AcquisitionConfig):
        self.config = config

    def select(
        self,
        gp: SingleTaskGP,
        candidates: torch.Tensor,
        train_Y: torch.Tensor,
        acqf: str | None = None,
        ucb_beta: float | None = None,
        pca_mode: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Select the best candidate point.

        Args:
            gp: Fitted GP model
            candidates: [N, d] candidate points
            train_Y: Training targets for EI best_f
            acqf: Override acquisition function name
            ucb_beta: Override UCB beta
            pca_mode: If True, don't normalize output to sphere

        Returns:
            (selected_point [1, d], diagnostics_dict)
        """
        acqf = acqf or self.config.acqf
        ucb_beta = ucb_beta or self.config.ucb_beta
        diag: dict = {"acqf_used": acqf}

        try:
            if acqf == "ts":
                thompson = MaxPosteriorSampling(model=gp, replacement=False)
                v_opt = thompson(candidates.double().unsqueeze(0), num_samples=1)
                v_opt = v_opt.squeeze(0).float()
                if not pca_mode:
                    v_opt = F.normalize(v_opt, p=2, dim=-1)

            elif acqf == "ei":
                ei = qExpectedImprovement(gp, best_f=train_Y.max().double())
                with torch.no_grad():
                    ei_vals = ei(candidates.double().unsqueeze(-2))
                best_idx = ei_vals.argmax()
                v_opt = candidates[best_idx : best_idx + 1]

            elif acqf == "ucb":
                with torch.no_grad():
                    post = gp.posterior(candidates.double())
                    ucb_vals = (
                        post.mean.squeeze()
                        + ucb_beta * post.variance.sqrt().squeeze()
                    )
                best_idx = ucb_vals.argmax()
                v_opt = candidates[best_idx : best_idx + 1]

            else:
                raise ValueError(f"Unknown acquisition function: {acqf}")

            # Diagnostics
            with torch.no_grad():
                post = gp.posterior(v_opt.double())
                diag["gp_mean"] = post.mean.item()
                diag["gp_std"] = post.variance.sqrt().item()

            return v_opt, diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            logger.error(f"Acquisition failed: {e}")
            # Random fallback
            v_opt = candidates[0:1]
            return v_opt, {
                "gp_mean": 0,
                "gp_std": 1,
                "nearest_train_cos": 0,
                "is_fallback": True,
                "acqf_used": acqf,
            }


class AcquisitionSchedule:
    """Switches acquisition function based on GP posterior uncertainty.

    When GP std is low (collapsing), switches to UCB with high beta
    for exploration boost. When GP std is high, uses UCB with low beta.
    Otherwise uses the default acquisition function.
    """

    def __init__(
        self,
        acqf_config: AcquisitionConfig,
        tr_config: TrustRegionConfig,
    ):
        self.acqf_config = acqf_config
        self.tr_config = tr_config

    def get_effective_acqf(
        self, gp_std: float, gp=None,
    ) -> tuple[str, float]:
        """Return (acqf_name, ucb_beta) based on GP state.

        Args:
            gp_std: Current GP posterior standard deviation
            gp: GP model (for noise-relative thresholds)
        """
        if not self.acqf_config.schedule:
            return self.acqf_config.acqf, self.acqf_config.ucb_beta

        cfg = self.tr_config

        noise_std = 1.0
        if cfg.ur_relative and gp is not None:
            try:
                noise_std = gp.likelihood.noise.item() ** 0.5
            except Exception:
                pass

        eff_low = cfg.ur_std_low * noise_std
        eff_high = cfg.ur_std_high * noise_std

        if gp_std < eff_low:
            return "ucb", self.acqf_config.acqf_ucb_beta_high
        elif gp_std > eff_high:
            return "ucb", self.acqf_config.acqf_ucb_beta_low
        else:
            return self.acqf_config.acqf, self.acqf_config.ucb_beta


def create_acquisition(
    config: AcquisitionConfig,
    tr_config: TrustRegionConfig | None = None,
) -> tuple[AcquisitionSelector, AcquisitionSchedule | None]:
    """Factory for acquisition components.

    Returns:
        (selector, schedule_or_None)
    """
    selector = AcquisitionSelector(config)
    schedule = None
    if config.schedule and tr_config is not None:
        schedule = AcquisitionSchedule(config, tr_config)
    return selector, schedule
