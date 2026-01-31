import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """Simple MLP for coupling layer scale/shift prediction."""
    def __init__(self, in_dim, out_dim, hidden_dim=256, n_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        # Initialize last layer with zeros for identity flow at start
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CouplingLayer(nn.Module):
    """Affine coupling layer for RealNVP."""
    def __init__(self, dim, hidden_dim=256, mask_type="even"):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type
        
        # Create mask
        # 1 means input is passed through unchanged (and used to condition)
        # 0 means input is transformed
        mask = torch.zeros(dim)
        if mask_type == "even":
            mask[0::2] = 1
        else: # odd
            mask[1::2] = 1
        self.register_buffer('mask', mask)
        
        # Network predicts scale and shift for transformed transform parameters
        # Input to net is masked part (size dim), but effective input is only where mask=1
        self.net = MLP(dim, dim * 2, hidden_dim=hidden_dim)

    def forward(self, x):
        # x: [B, D]
        # x_id = x * mask (part that stays same)
        # x_tr = x * (1-mask) (part that gets transformed)
        
        x_masked = x * self.mask
        
        # Predict s and t
        out = self.net(x_masked)
        s, t = out.chunk(2, dim=-1)
        
        # Apply tanh for stability in scale
        s = torch.tanh(s) * self.mask.logical_not().float()
        t = t * self.mask.logical_not().float()
        
        # Affine transform: y = x .* exp(s) + t
        # For masked part: s=0, t=0 -> y = x
        y = x * torch.exp(s) + t
        
        # Log determinant: sum(s)
        log_det = s.sum(dim=-1)
        
        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask
        
        out = self.net(y_masked)
        s, t = out.chunk(2, dim=-1)
        
        s = torch.tanh(s) * self.mask.logical_not().float()
        t = t * self.mask.logical_not().float()
        
        # Inverse: x = (y - t) .* exp(-s)
        x = (y - t) * torch.exp(-s)
        
        return x

class RealNVP(nn.Module):
    """RealNVP Normalizing Flow model."""
    def __init__(self, dim=1024, n_layers=6, hidden_dim=512):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        
        # Create alternating coupling layers
        for i in range(n_layers):
            mask_type = "even" if i % 2 == 0 else "odd"
            self.layers.append(
                CouplingLayer(dim, hidden_dim=hidden_dim, mask_type=mask_type)
            )
            
        # Base distribution: Standard Normal
        self.register_buffer('base_loc', torch.zeros(dim))
        self.register_buffer('base_scale', torch.ones(dim))

    def forward(self, x):
        """Transform data x to latent z. Returns z and log_det_J."""
        log_det_sum = 0
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum

    def inverse(self, z):
        """Transform latent z to data x (generate samples)."""
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        """Compute log probability of data x."""
        z, log_det = self.forward(x)
        
        # Log prob of base dist (Standard Normal)
        # log p(z) = -0.5 * (z^2 + log(2pi))
        log_p_z = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        
        # log p(x) = log p(z) + log |det J|
        return log_p_z + log_det

    def sample(self, n_samples, device="cuda"):
        """Generate n_samples from the flow."""
        z = torch.randn(n_samples, self.dim, device=device)
        return self.inverse(z)
