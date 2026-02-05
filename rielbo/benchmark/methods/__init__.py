"""Method wrappers for benchmark framework."""

from rielbo.benchmark.methods.subspace import SubspaceBOBenchmark
from rielbo.benchmark.methods.subspace_v3 import SubspaceBOv3Benchmark
from rielbo.benchmark.methods.subspace_v4 import SubspaceBOv4Benchmark
from rielbo.benchmark.methods.subspace_v5 import SubspaceBOv5Benchmark
from rielbo.benchmark.methods.turbo import TuRBOBenchmark
from rielbo.benchmark.methods.lolbo import LOLBOBenchmark
from rielbo.benchmark.methods.vanilla import VanillaBOBenchmark

# Registry of available methods
METHODS = {
    "subspace": SubspaceBOBenchmark,
    "subspace_v3": SubspaceBOv3Benchmark,
    "subspace_v4": SubspaceBOv4Benchmark,
    "subspace_v5": SubspaceBOv5Benchmark,
    "turbo": TuRBOBenchmark,
    "lolbo": LOLBOBenchmark,
    "vanilla": VanillaBOBenchmark,
}


def get_method(name: str):
    """Get method class by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHODS.keys())}")
    return METHODS[name]


__all__ = ["SubspaceBOBenchmark", "SubspaceBOv3Benchmark", "SubspaceBOv4Benchmark", "SubspaceBOv5Benchmark", "TuRBOBenchmark", "LOLBOBenchmark", "VanillaBOBenchmark", "METHODS", "get_method"]
