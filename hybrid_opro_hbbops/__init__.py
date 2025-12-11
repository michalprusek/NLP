"""
Hybrid OPRO + HbBoPs: Combining instruction generation with multi-fidelity GP screening.

Algorithm:
    1. Phase 1: Full Hyperband on initial instruction × exemplar grid
    2. Phase 2: Extract top instructions → OPRO generates new candidates
    3. Phase 3: GP screening (8 instructions × 25 dynamic exemplars = 200 candidates)
    4. Phase 4: Full-fidelity evaluation of top 10 by GP prediction
    5. Phase 5: Retrain GP on accumulated high-fidelity data
    6. Repeat 2-5 until budget exhausted
"""

from .config import HybridConfig, ScoredInstruction, DesignPoint, PromptCandidate
from .exemplar_sampler import ExemplarSampler
from .opro_adapter import OPROInstructionGenerator
from .hybrid_optimizer import HybridOPROHbBoPs

__all__ = [
    "HybridConfig",
    "ScoredInstruction",
    "DesignPoint",
    "PromptCandidate",
    "ExemplarSampler",
    "OPROInstructionGenerator",
    "HybridOPROHbBoPs",
]
