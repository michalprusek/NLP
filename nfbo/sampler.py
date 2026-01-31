import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

from nfbo.model import RealNVP
from ecoflow.gp_surrogate import SonarGPSurrogate

logger = logging.getLogger(__name__)

class NFBoSampler:
    """
    Latent BO Sampler using RealNVP.
    
    Paper: "Latent Bayesian Optimization via Autoregressive Normalizing Flows" (Lee et al., 2025)
    
    Strategy:
    1. Train RealNVP on ALL available data to learn the manifold p(x).
       (Or on top-k if we want to focus on high-value regions, but paper suggests bijection)
    2. Map observed data X to latent Z using flow: z = f(x).
    3. Fit GP surrogate in latent space: y = g(z).
    4. Optimize acquisition function in Z-space: z* = argmax acq(z).
    5. Decode optimal z* to candidate x*: x* = f^-1(z*).
    """
    def __init__(
        self,
        dim=1024,
        device="cuda",
        flow_lr=1e-3,
        flow_epochs=50,
        n_flow_layers=6,
        hidden_dim=512,
        gp_alpha=1.0,
    ):
        self.dim = dim
        self.device = device
        self.flow_lr = flow_lr
        self.flow_epochs = flow_epochs
        self.gp_alpha = gp_alpha
        
        # RealNVP model (the bijection)
        self.flow = RealNVP(dim=dim, n_layers=n_flow_layers, hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.Adam(self.flow.parameters(), lr=flow_lr)
        
        # GP model for latent space
        # We reuse SonarGPSurrogate logic but applied to Z
        self.gp = SonarGPSurrogate(D=dim, device=device) # will re-fit on Z
        
    def fit_flow(self, train_X):
        """
        Fit the flow model to the observed data to learn the manifold.
        """
        if len(train_X) < 10:
            return  # Not enough data
            
        dataset = TensorDataset(train_X)
        dataloader = DataLoader(dataset, batch_size=min(32, len(train_X)), shuffle=True)
        
        self.flow.train()
        for epoch in range(self.flow_epochs):
            epoch_loss = 0
            for batch in dataloader:
                x = batch[0]
                self.optimizer.zero_grad()
                
                # Maximize log likelihood
                log_prob = self.flow.log_prob(x)
                loss = -log_prob.mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

    def generate_candidates(self, train_X, train_Y, n_candidates=1):
        """
        Generate candidates by optimizing Acq(z) in latent space.
        
        Steps:
        1. Map train_X -> train_Z
        2. Fit GP on (train_Z, train_Y)
        3. Optimize UCB(z) using gradient ascent in Z-space
        4. Map resulting z* -> x*
        """
        # 1. Map to Z
        self.flow.eval()
        with torch.no_grad():
            # forward returns (z, log_det), we only need z
            train_Z, _ = self.flow(train_X)
            
        # 2. Fit GP on Z
        # Note: normalizer in GP handles Z scaling
        self.gp.fit(train_Z, train_Y)
        
        # 3. Optimize Acquisition in Z
        # We start optimization from the best Z found so far (or random points)
        best_idx = train_Y.argmax()
        start_z = train_Z[best_idx].clone().unsqueeze(0).to(self.device)  # [1, D]

        # Or batch of random starts for robustness
        # z ~ N(0, I) is the prior base, so valid Zs are normally distributed
        random_starts = torch.randn(16, self.dim, device=self.device)
        candidates_z = torch.cat([start_z, random_starts], dim=0)
        candidates_z.requires_grad_(True)
        
        # Optimize Z using Gradient Ascent
        # We use the GP's ucb_gradient method which handles differentiability via autograd internally
        lr = 0.1
        
        for _ in range(50): # 50 steps of optimization
            # Get gradient of UCB w.r.t Z
            grad = self.gp.ucb_gradient(candidates_z, alpha=self.gp_alpha)
            
            # Gradient ascent step
            # Note: ucb_gradient returns d(UCB)/dZ
            candidates_z.data += lr * grad
            
        # Select best Z from the optimized batch
        with torch.no_grad():
            self.gp.model.eval()
            mean, std = self.gp.predict(candidates_z)
            final_ucb = mean + self.gp_alpha * std
            best_z_idx = final_ucb.argmax()
            best_z = candidates_z[best_z_idx:best_z_idx+1]
            
        # 4. Decode Z -> X
        with torch.no_grad():
            best_x = self.flow.inverse(best_z)
            
        # Return candidates (here we just return one best, but could return top-k)
        if n_candidates == 1:
            return best_x
        else:
            # If multiple requested, sample from top optimized Zs
            # For strict single-point BO logic, best is fine.
            # But let's return top k distinct if possible
            # Just repeat best for now if > 1 is requested trivially
            return best_x.repeat(n_candidates, 1)

