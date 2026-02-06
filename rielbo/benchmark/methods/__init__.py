"""Method wrappers for benchmark framework."""

from rielbo.benchmark.methods.subspace import SubspaceBOBenchmark
from rielbo.benchmark.methods.turbo import TuRBOBenchmark
from rielbo.benchmark.methods.lolbo import LOLBOBenchmark
from rielbo.benchmark.methods.vanilla import VanillaBOBenchmark
from rielbo.benchmark.methods.baxus import BAxUSBenchmark
from rielbo.benchmark.methods.cmaes import CMAESBenchmark
from rielbo.benchmark.methods.invbo import InvBOBenchmark

# Registry of available methods
METHODS = {
    "subspace": SubspaceBOBenchmark,
    "turbo": TuRBOBenchmark,
    "lolbo": LOLBOBenchmark,
    "vanilla": VanillaBOBenchmark,
    "baxus": BAxUSBenchmark,
    "cmaes": CMAESBenchmark,
    "invbo": InvBOBenchmark,
}


def get_method(name: str):
    """Get method class by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHODS.keys())}")
    return METHODS[name]


__all__ = [
    "SubspaceBOBenchmark", "TuRBOBenchmark", "LOLBOBenchmark", "VanillaBOBenchmark",
    "BAxUSBenchmark", "CMAESBenchmark", "InvBOBenchmark",
    "METHODS", "get_method",
]
