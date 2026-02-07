"""Composable BO components for RieLBO."""

from rielbo.components.surrogate import SphericalGPSurrogate, EuclideanGPSurrogate
from rielbo.components.trust_region import (
    AdaptiveTR,
    URTR,
    StaticTR,
    TuRBOTR,
    create_trust_region,
)
from rielbo.components.projection import (
    QRProjection,
    PCAProjection,
    IdentityProjection,
    LASSSelector,
    create_projection,
)
from rielbo.components.acquisition import (
    AcquisitionSelector,
    AcquisitionSchedule,
    create_acquisition,
)
from rielbo.components.candidate_gen import (
    GeodesicGenerator,
    SobolBoxGenerator,
    create_candidate_generator,
)
from rielbo.components.norm_reconstruction import (
    MeanNormReconstructor,
    ProbabilisticNormReconstructor,
    create_norm_reconstructor,
)

__all__ = [
    "SphericalGPSurrogate",
    "EuclideanGPSurrogate",
    "AdaptiveTR",
    "URTR",
    "StaticTR",
    "TuRBOTR",
    "create_trust_region",
    "QRProjection",
    "PCAProjection",
    "IdentityProjection",
    "LASSSelector",
    "create_projection",
    "AcquisitionSelector",
    "AcquisitionSchedule",
    "create_acquisition",
    "GeodesicGenerator",
    "SobolBoxGenerator",
    "create_candidate_generator",
    "MeanNormReconstructor",
    "ProbabilisticNormReconstructor",
    "create_norm_reconstructor",
]
