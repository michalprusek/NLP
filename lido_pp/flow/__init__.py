"""
FlowPO: Flow Matching modules.

Components:
- Flow-DiT: Transformer-based velocity field
- ODE Solvers: Euler, midpoint, RK4 integration
- Reflow: Trajectory straightening for 1-step inference
- GP-Guided Flow: Acquisition gradient injection (Novel Contribution #2)
"""

from lido_pp.flow.timestep_embed import TimestepEmbedding
from lido_pp.flow.flow_dit import FlowDiT, AdaLayerNorm, FlowTransformerBlock
from lido_pp.flow.losses import (
    conditional_flow_matching_loss,
    oat_regularization,
    oat_flow_matching_loss,
    measure_trajectory_straightness,
)
from lido_pp.flow.ode_solver import (
    euler_integrate,
    midpoint_integrate,
    rk4_integrate,
    one_step_integrate,
    integrate,
    compute_curvature_only,
    IntegrationResult,
)
from lido_pp.flow.reflow import (
    ReflowTrainer,
    ReflowConfig,
    ReflowResult,
    verify_one_step_inference,
)
from lido_pp.flow.gp_guided_flow import (
    GPGuidedFlowGenerator,
    GuidedGenerationResult,
    AcquisitionGradientGuide,
    create_guided_generator,
)

__all__ = [
    # Embedding
    "TimestepEmbedding",
    # Model
    "FlowDiT",
    "AdaLayerNorm",
    "FlowTransformerBlock",
    # Losses
    "conditional_flow_matching_loss",
    "oat_regularization",
    "oat_flow_matching_loss",
    "measure_trajectory_straightness",
    # ODE Solvers
    "euler_integrate",
    "midpoint_integrate",
    "rk4_integrate",
    "one_step_integrate",
    "integrate",
    "compute_curvature_only",
    "IntegrationResult",
    # Reflow
    "ReflowTrainer",
    "ReflowConfig",
    "ReflowResult",
    "verify_one_step_inference",
    # GP-Guided Flow (Novel Contribution #2)
    "GPGuidedFlowGenerator",
    "GuidedGenerationResult",
    "AcquisitionGradientGuide",
    "create_guided_generator",
]
