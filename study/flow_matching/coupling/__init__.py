"""Coupling methods for flow matching.

Provides I-CFM (independent coupling) and OT-CFM (optimal transport coupling)
for flow matching training. Use create_coupling() factory to instantiate.

Example:
    >>> from study.flow_matching.coupling import create_coupling
    >>> icfm = create_coupling('icfm')
    >>> t, x_t, u_t = icfm.sample(x0, x1)
    >>>
    >>> otcfm = create_coupling('otcfm', reg=0.5)
    >>> t, x_t, u_t = otcfm.sample(x0, x1)
"""

from study.flow_matching.coupling.icfm import ICFMCoupling
from study.flow_matching.coupling.otcfm import OTCFMCoupling


def create_coupling(method: str, **kwargs):
    """Create coupling method by name.

    Args:
        method: 'icfm' or 'otcfm'
        **kwargs: Passed to coupling constructor
            - For icfm: sigma (float)
            - For otcfm: sigma (float), reg (float), normalize_cost (bool)

    Returns:
        Coupling instance with sample() method

    Raises:
        ValueError: If method is not 'icfm' or 'otcfm'

    Example:
        >>> icfm = create_coupling('icfm')
        >>> otcfm = create_coupling('otcfm', reg=0.5, normalize_cost=True)
    """
    if method == "icfm":
        return ICFMCoupling(**kwargs)
    elif method == "otcfm":
        return OTCFMCoupling(**kwargs)
    else:
        raise ValueError(f"Unknown coupling method: {method}. Choose 'icfm' or 'otcfm'")


__all__ = ["create_coupling", "ICFMCoupling", "OTCFMCoupling"]
