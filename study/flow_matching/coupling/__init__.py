"""Coupling methods for flow matching.

Provides coupling methods for flow matching training:
- I-CFM (independent coupling): Random pairing of noise and data
- OT-CFM (optimal transport coupling): Sinkhorn-based optimal pairing
- Reflow: Pre-generated pairs from teacher ODE for straighter paths
- SI (stochastic interpolant): Non-linear interpolation schedules

Use create_coupling() factory to instantiate.

Example:
    >>> from study.flow_matching.coupling import create_coupling
    >>> icfm = create_coupling('icfm')
    >>> t, x_t, u_t = icfm.sample(x0, x1)
    >>>
    >>> otcfm = create_coupling('otcfm', reg=0.5)
    >>> t, x_t, u_t = otcfm.sample(x0, x1)
    >>>
    >>> # Reflow requires pre-generated pairs
    >>> reflow = create_coupling('reflow', pair_tensors=(x0_pairs, x1_pairs))
    >>> t, x_t, u_t = reflow.sample(None, None)  # Ignores inputs
    >>>
    >>> # Stochastic Interpolant with GVP schedule
    >>> si = create_coupling('si')  # or 'si-gvp'
    >>> t, x_t, u_t = si.sample(x0, x1)
"""

from study.flow_matching.coupling.icfm import ICFMCoupling
from study.flow_matching.coupling.otcfm import OTCFMCoupling
from study.flow_matching.coupling.reflow import ReflowCoupling
from study.flow_matching.coupling.stochastic import StochasticInterpolantCoupling


def create_coupling(method: str, **kwargs):
    """Create coupling method by name.

    Args:
        method: Coupling method name:
            - 'icfm': Independent CFM
            - 'otcfm': Optimal Transport CFM
            - 'reflow': Rectified Flow
            - 'si' or 'si-gvp': Stochastic Interpolant with GVP schedule
            - 'si-linear': Stochastic Interpolant with linear schedule
        **kwargs: Passed to coupling constructor
            - For icfm: sigma (float)
            - For otcfm: sigma (float), reg (float), normalize_cost (bool)
            - For reflow: pair_tensors (tuple of x0, x1), batch_size (int)
            - For si: schedule (str, default 'gvp')

    Returns:
        Coupling instance with sample() method

    Raises:
        ValueError: If method is unknown or required kwargs missing

    Example:
        >>> icfm = create_coupling('icfm')
        >>> otcfm = create_coupling('otcfm', reg=0.5, normalize_cost=True)
        >>> reflow = create_coupling('reflow', pair_tensors=(x0, x1))
        >>> si = create_coupling('si')  # GVP schedule (default)
        >>> si_linear = create_coupling('si-linear')  # Linear schedule
    """
    if method == "icfm":
        return ICFMCoupling(**kwargs)
    elif method == "otcfm":
        return OTCFMCoupling(**kwargs)
    elif method == "reflow":
        if "pair_tensors" not in kwargs:
            raise ValueError("ReflowCoupling requires pair_tensors=(x0, x1) argument")
        return ReflowCoupling(**kwargs)
    elif method in ("si", "si-gvp"):
        # Stochastic Interpolant with GVP schedule (default)
        schedule = kwargs.pop("schedule", "gvp")
        return StochasticInterpolantCoupling(schedule=schedule, **kwargs)
    elif method == "si-linear":
        # Stochastic Interpolant with linear schedule (for ablation)
        return StochasticInterpolantCoupling(schedule="linear", **kwargs)
    else:
        raise ValueError(
            f"Unknown coupling method: {method}. "
            "Choose 'icfm', 'otcfm', 'reflow', 'si', 'si-gvp', or 'si-linear'"
        )


__all__ = [
    "create_coupling",
    "ICFMCoupling",
    "OTCFMCoupling",
    "ReflowCoupling",
    "StochasticInterpolantCoupling",
]
