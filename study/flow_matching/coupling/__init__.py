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
    >>> reflow = create_coupling('reflow', pair_tensors=(x0_pairs, x1_pairs))
    >>> t, x_t, u_t = reflow.sample(None, None)  # Ignores inputs
    >>>
    >>> si = create_coupling('si')  # or 'si-gvp'
    >>> t, x_t, u_t = si.sample(x0, x1)
"""

from study.flow_matching.coupling.icfm import ICFMCoupling, linear_interpolate
from study.flow_matching.coupling.otcfm import OTCFMCoupling
from study.flow_matching.coupling.reflow import ReflowCoupling
from study.flow_matching.coupling.spherical import (
    SphericalCoupling,
    SphericalOTCoupling,
    slerp_interpolate,
)
from study.flow_matching.coupling.stochastic import StochasticInterpolantCoupling

_COUPLING_CLASSES = {
    "icfm": ICFMCoupling,
    "otcfm": OTCFMCoupling,
    "reflow": ReflowCoupling,
    "si": StochasticInterpolantCoupling,
    "si-gvp": StochasticInterpolantCoupling,
    "si-linear": StochasticInterpolantCoupling,
    "spherical": SphericalCoupling,
    "spherical-ot": SphericalOTCoupling,
}


def create_coupling(method: str, **kwargs):
    """Create coupling method by name.

    Args:
        method: Coupling method name:
            - 'icfm': Independent CFM
            - 'otcfm': Optimal Transport CFM
            - 'reflow': Rectified Flow
            - 'si' or 'si-gvp': Stochastic Interpolant with GVP schedule
            - 'si-linear': Stochastic Interpolant with linear schedule
            - 'spherical': Spherical (SLERP) interpolation for normalized embeddings
            - 'spherical-ot': Spherical OT coupling (Sinkhorn + SLERP)
        **kwargs: Passed to coupling constructor.
            - For otcfm: reg (float), normalize_cost (bool)
            - For reflow: pair_tensors (tuple of x0, x1), batch_size (int)
            - For si: schedule (str), normalize_loss (bool)
            - For spherical/spherical-ot: normalize_inputs (bool)

    Returns:
        Coupling instance with sample() method.

    Raises:
        ValueError: If method is unknown or required kwargs missing.
    """
    if method not in _COUPLING_CLASSES:
        available = ", ".join(sorted(_COUPLING_CLASSES.keys()))
        raise ValueError(f"Unknown coupling method: {method}. Available: {available}")

    if method == "reflow" and "pair_tensors" not in kwargs:
        raise ValueError("ReflowCoupling requires pair_tensors=(x0, x1) argument")

    # Handle schedule extraction for SI variants
    if method == "si-gvp":
        kwargs.setdefault("schedule", "gvp")
    elif method == "si-linear":
        kwargs["schedule"] = "linear"
    elif method == "si":
        kwargs.setdefault("schedule", "gvp")

    return _COUPLING_CLASSES[method](**kwargs)


__all__ = [
    "create_coupling",
    "linear_interpolate",
    "slerp_interpolate",
    "ICFMCoupling",
    "OTCFMCoupling",
    "ReflowCoupling",
    "StochasticInterpolantCoupling",
    "SphericalCoupling",
    "SphericalOTCoupling",
]
