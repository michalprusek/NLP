"""Model factory for flow matching velocity networks.

This module provides a factory function for creating velocity networks
by architecture name, enabling CLI-based model selection.

Usage:
    from study.flow_matching.models import create_model, SimpleMLP, DiTVelocityNetwork, UNetMLP

    model = create_model("mlp")  # Creates SimpleMLP with defaults
    model = create_model("dit")  # Creates DiTVelocityNetwork with defaults
    model = create_model("unet")  # Creates UNetMLP with defaults
    model = create_model("mamba")  # Creates MambaVelocityNetwork (if mamba-ssm available)
    model = create_model("mlp", hidden_dim=512)  # Custom config

    # Scaling variants (Tiny/Small/Base)
    model = create_model("mlp", "tiny")  # ~300K params
    model = create_model("dit", "base")  # ~20M params
    model = create_model("unet", "small", hidden_dims=(384, 192))  # Small config with override
"""

import logging
from typing import Optional

from study.flow_matching.models.mlp import SimpleMLP, timestep_embedding
from study.flow_matching.models.dit import DiTVelocityNetwork, AdaLNBlock
from study.flow_matching.models.unet_mlp import UNetMLP, FiLMLayer
from study.flow_matching.models.mamba_velocity import MambaVelocityNetwork, MAMBA_AVAILABLE
from study.flow_matching.models.scaling import SCALING_CONFIGS, get_scaled_config, list_available_scales

logger = logging.getLogger(__name__)

__all__ = [
    "create_model",
    "SimpleMLP",
    "DiTVelocityNetwork",
    "AdaLNBlock",
    "UNetMLP",
    "FiLMLayer",
    "MambaVelocityNetwork",
    "MAMBA_AVAILABLE",
    "timestep_embedding",
    "SCALING_CONFIGS",
    "get_scaled_config",
    "list_available_scales",
]

# Default configurations for each architecture (used when scale=None)
_DEFAULTS = {
    "mlp": {"input_dim": 1024, "hidden_dim": 256, "num_layers": 5, "time_embed_dim": 256},
    "dit": {"input_dim": 1024, "hidden_dim": 384, "num_layers": 3, "num_heads": 6, "time_embed_dim": 256},
    "unet": {"input_dim": 1024, "hidden_dims": (512, 256), "time_embed_dim": 256},
    "mamba": {"input_dim": 1024, "hidden_dim": 256, "num_layers": 4, "d_state": 16, "chunk_size": 64, "time_embed_dim": 256},
}


def create_model(arch: str, scale: Optional[str] = None, **kwargs) -> "torch.nn.Module":
    """Create velocity network by architecture name with optional scaling.

    Args:
        arch: Architecture name. Supported values:
            - "mlp": SimpleMLP (~920K params with defaults)
            - "dit": DiTVelocityNetwork (~9.3M params with defaults)
            - "unet": UNetMLP (~6.9M params with defaults)
            - "mamba": MambaVelocityNetwork (~2M params, EXPERIMENTAL, requires mamba-ssm)
        scale: Optional scale level ('tiny', 'small', 'base').
            If provided, uses predefined scaling configuration.
            If None, uses architecture defaults (backward compatible).
        **kwargs: Architecture-specific arguments passed to constructor.
            These override values from scaling config if both are specified.
            For "mlp": input_dim, hidden_dim, num_layers, time_embed_dim
            For "dit": input_dim, hidden_dim, num_layers, num_heads, time_embed_dim
            For "unet": input_dim, hidden_dims (tuple), time_embed_dim
            For "mamba": input_dim, hidden_dim, num_layers, d_state, chunk_size, time_embed_dim

    Returns:
        Velocity network module with forward(x, t) -> v signature.

    Raises:
        ValueError: If architecture name is not recognized or scale is invalid.
        ImportError: If "mamba" is requested but mamba-ssm is not installed.

    Examples:
        >>> model = create_model("mlp")  # Default config
        >>> model = create_model("mlp", "tiny")  # Tiny config (~300K params)
        >>> model = create_model("mlp", "base", hidden_dim=512)  # Base with override
        >>> model = create_model("dit", "small")  # Current default equivalent
        >>> model = create_model("unet", "base")  # Large U-Net (~15M params)
    """
    import torch  # Deferred import for type annotation

    # Validate architecture
    if arch not in ["mlp", "dit", "unet", "mamba"]:
        available = ["mlp", "dit", "unet"]
        if MAMBA_AVAILABLE:
            available.append("mamba")
        else:
            available.append("mamba (requires mamba-ssm)")
        raise ValueError(
            f"Unknown architecture: '{arch}'. "
            f"Available architectures: {available}"
        )

    # Build configuration: start with defaults, overlay scale config, then kwargs
    if scale is not None:
        # Get scale-specific config
        config = get_scaled_config(arch, scale)
        # Add defaults for keys not in scale config (like input_dim, time_embed_dim)
        full_config = dict(_DEFAULTS.get(arch, {}))
        full_config.update(config)
    else:
        # Use defaults directly
        full_config = dict(_DEFAULTS.get(arch, {}))

    # Check for conflicting overrides and warn
    if scale is not None and kwargs:
        scale_config = get_scaled_config(arch, scale)
        conflicts = [k for k in kwargs if k in scale_config and kwargs[k] != scale_config[k]]
        if conflicts:
            logger.warning(
                f"Overriding scale='{scale}' config with explicit kwargs: {conflicts}. "
                f"Scale values: {[scale_config[k] for k in conflicts]}, "
                f"Override values: {[kwargs[k] for k in conflicts]}"
            )

    # Apply kwargs overrides
    full_config.update(kwargs)

    # Create model
    if arch == "mlp":
        return SimpleMLP(
            input_dim=full_config.get("input_dim", 1024),
            hidden_dim=full_config.get("hidden_dim", 256),
            num_layers=full_config.get("num_layers", 5),
            time_embed_dim=full_config.get("time_embed_dim", 256),
        )
    elif arch == "dit":
        return DiTVelocityNetwork(
            input_dim=full_config.get("input_dim", 1024),
            hidden_dim=full_config.get("hidden_dim", 384),
            num_layers=full_config.get("num_layers", 3),
            num_heads=full_config.get("num_heads", 6),
            time_embed_dim=full_config.get("time_embed_dim", 256),
        )
    elif arch == "unet":
        return UNetMLP(
            input_dim=full_config.get("input_dim", 1024),
            hidden_dims=full_config.get("hidden_dims", (512, 256)),
            time_embed_dim=full_config.get("time_embed_dim", 256),
        )
    elif arch == "mamba":
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is not installed. Install with: pip install mamba-ssm\n"
                "Note: mamba-ssm requires Linux, NVIDIA GPU, and CUDA 11.6+."
            )
        return MambaVelocityNetwork(
            input_dim=full_config.get("input_dim", 1024),
            hidden_dim=full_config.get("hidden_dim", 256),
            num_layers=full_config.get("num_layers", 4),
            d_state=full_config.get("d_state", 16),
            chunk_size=full_config.get("chunk_size", 64),
            time_embed_dim=full_config.get("time_embed_dim", 256),
        )
