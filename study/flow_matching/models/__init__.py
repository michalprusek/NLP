"""Model factory for flow matching velocity networks.

This module provides a factory function for creating velocity networks
by architecture name, enabling CLI-based model selection.

Usage:
    from study.flow_matching.models import create_model, SimpleMLP, DiTVelocityNetwork

    model = create_model("mlp")  # Creates SimpleMLP with defaults
    model = create_model("dit")  # Creates DiTVelocityNetwork with defaults
    model = create_model("mlp", hidden_dim=512)  # Custom config
"""

from study.flow_matching.models.mlp import SimpleMLP, timestep_embedding
from study.flow_matching.models.dit import DiTVelocityNetwork, AdaLNBlock

__all__ = ["create_model", "SimpleMLP", "DiTVelocityNetwork", "AdaLNBlock", "timestep_embedding"]


def create_model(arch: str, **kwargs) -> "torch.nn.Module":
    """Create velocity network by architecture name.

    Args:
        arch: Architecture name. Supported values:
            - "mlp": SimpleMLP (~920K params with defaults)
            - "dit": DiTVelocityNetwork (~9.3M params with defaults)
        **kwargs: Architecture-specific arguments passed to constructor.
            For "mlp": input_dim, hidden_dim, num_layers, time_embed_dim
            For "dit": input_dim, hidden_dim, num_layers, num_heads, time_embed_dim

    Returns:
        Velocity network module with forward(x, t) -> v signature.

    Raises:
        ValueError: If architecture name is not recognized.

    Examples:
        >>> model = create_model("mlp")
        >>> model = create_model("dit")
        >>> model = create_model("dit", hidden_dim=512, num_layers=6)
    """
    import torch  # Deferred import for type annotation

    if arch == "mlp":
        return SimpleMLP(
            input_dim=kwargs.get("input_dim", 1024),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 5),
            time_embed_dim=kwargs.get("time_embed_dim", 256),
        )
    elif arch == "dit":
        return DiTVelocityNetwork(
            input_dim=kwargs.get("input_dim", 1024),
            hidden_dim=kwargs.get("hidden_dim", 384),
            num_layers=kwargs.get("num_layers", 3),
            num_heads=kwargs.get("num_heads", 6),
            time_embed_dim=kwargs.get("time_embed_dim", 256),
        )
    else:
        available = ["mlp", "dit"]
        raise ValueError(
            f"Unknown architecture: '{arch}'. "
            f"Available architectures: {available}"
        )
