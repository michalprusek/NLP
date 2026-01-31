import torch
import logging
from ecoflow.optimization_loop import BOOptimizationLoop
from nfbo.sampler import NFBoSampler
import random

logger = logging.getLogger(__name__)

class NFBoLoop(BOOptimizationLoop):
    """
    NF-BO optimization loop.
    Replaces Flow Matching + Guided ODE with Latent BO (RealNVP bijection + Z-space GP).
    """
    def __init__(
        self,
        nfbo_sampler: NFBoSampler,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nfbo_sampler = nfbo_sampler
        
    def initialize(self):
        """Initialize by fitting flow on initial random samples."""
        result = super().initialize()
        
        logger.info(f"Fitting NF-BO flow on {len(self.train_X)} initial samples...")
        self.nfbo_sampler.fit_flow(self.train_X)
        
        return result

    def step(self, n_candidates: int = 64, **kwargs) -> dict:
        """
        Execute one NF-BO iteration (Latent BO cycle).
        """
        self.iteration += 1
        logger.info(f"\n=== NF-BO Iteration {self.iteration} ===")

        # 1. Generate candidate(s) by optimizing in Z-space
        logger.info(f"Generating optimized candidate via Latent BO (Z-space optimization)...")
        # We only need 1 best candidate for sequential BO
        candidates = self.nfbo_sampler.generate_candidates(
            self.train_X, 
            self.train_Y, 
            n_candidates=1
        )
        
        selected_embedding = candidates[0:1] # [1, D]
        
        # 1.5 L2-r filtering (optional for NF-BO, but good safety check)
        if self.l2r_filter_enabled and self.encoder is not None:
             l2 = self._compute_round_trip_fidelity(selected_embedding)
             logger.info(f"  Candidate L2-r: {l2.item():.4f}")
             if l2.item() > self.l2r_threshold:
                 logger.warning("  Candidate OFF-MANIFOLD (L2-r high). decoding might fail.")
                 
        logger.info("Decoding selected embedding...")
        prompts = self._decode_safe(selected_embedding)
        prompt = prompts[0]
        
        # 2. Evaluate
        if self._is_valid_prompt(prompt):
             scores = self._evaluate_prompts([prompt])
             score = scores[0]
        else:
             logger.info("  Invalid prompt generated.")
             score = 0.0
             
        # 3. Update Data
        new_X = selected_embedding.to(self.device)
        new_Y = torch.tensor([score], device=self.device, dtype=torch.float32)
        
        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)
        
        # Track best
        if score > self.best_score:
            self.best_score = score
            self.best_prompt = prompt
            logger.info(f"NEW BEST: {self.best_score:.4f}")
        self.best_so_far_list.append(self.best_score)
        
        # 4. Update Flow Model (re-learn manifold with new point)
        # Note: In Latent BO, we usually update the flow periodically or every step
        logger.info("Refining flow model with new data...")
        self.nfbo_sampler.fit_flow(self.train_X)
        
        result = {
            "iteration": self.iteration,
            "score": score,
            "best_so_far": self.best_score,
            "prompt": prompt,
        }
        return result
