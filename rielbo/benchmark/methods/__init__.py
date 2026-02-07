"""Method wrappers for benchmark framework."""

from rielbo.benchmark.methods.subspace import SubspaceBOBenchmark
from rielbo.benchmark.methods.turbo import TuRBOBenchmark
from rielbo.benchmark.methods.lolbo import LOLBOBenchmark
from rielbo.benchmark.methods.vanilla import VanillaBOBenchmark
from rielbo.benchmark.methods.baxus import BAxUSBenchmark
from rielbo.benchmark.methods.cmaes import CMAESBenchmark
from rielbo.benchmark.methods.invbo import InvBOBenchmark
from rielbo.benchmark.methods.adas import AdaSBOTRBenchmark, AdaSBOStagBenchmark
from rielbo.benchmark.methods.random_sampling import RandomSamplingBenchmark

# Registry of available methods
METHODS = {
    "subspace": SubspaceBOBenchmark,
    "turbo": TuRBOBenchmark,
    "lolbo": LOLBOBenchmark,
    "vanilla": VanillaBOBenchmark,
    "baxus": BAxUSBenchmark,
    "cmaes": CMAESBenchmark,
    "invbo": InvBOBenchmark,
    "adas_tr": AdaSBOTRBenchmark,
    "adas_stag": AdaSBOStagBenchmark,
    "random": RandomSamplingBenchmark,
}


def get_method(name: str):
    """Get method class by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHODS.keys())}")
    return METHODS[name]


__all__ = [
    "SubspaceBOBenchmark", "TuRBOBenchmark", "LOLBOBenchmark", "VanillaBOBenchmark",
    "BAxUSBenchmark", "CMAESBenchmark", "InvBOBenchmark",
    "AdaSBOTRBenchmark", "AdaSBOStagBenchmark",
    "RandomSamplingBenchmark",
    "METHODS", "get_method",
]
