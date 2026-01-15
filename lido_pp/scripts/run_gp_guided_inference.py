#!/usr/bin/env python3
"""
FlowPO GP-Guided Inference.

Loads TFA checkpoint, trains GP on existing evaluations, and generates
optimized latents via GP-guided flow matching.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m lido_pp.scripts.run_gp_guided_inference \
        --tfa-checkpoint lido_pp/checkpoints/tfa_best.pt \
        --hbbops-path lipo/data/hbbops_results_20260102.json \
        --batch-size 16 \
        --num-steps 20
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_tfa_checkpoint(checkpoint_path: str, device: str = "cuda:0") -> nn.Module:
    """Load TFA model from checkpoint."""
    from lido_pp.backbone.cfm_encoder import TextFlowAutoencoder

    logger.info(f"Loading TFA checkpoint from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    input_dim = ckpt.get("input_dim", 1024)
    latent_dim = ckpt.get("latent_dim", 256)

    # Get flow_dim from args (training arguments)
    args = ckpt.get("args", {})
    flow_dim = args.get("flow_dim", 512) if isinstance(args, dict) else 512
    velocity_layers = args.get("velocity_layers", 6) if isinstance(args, dict) else 6
    dropout = args.get("dropout", 0.0) if isinstance(args, dict) else 0.0

    logger.info(f"TFA config: input_dim={input_dim}, latent_dim={latent_dim}, flow_dim={flow_dim}")

    # Create model
    tfa = TextFlowAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        flow_dim=flow_dim,
        num_velocity_layers=velocity_layers,
        dropout=dropout,
    ).to(device)

    # Load state dict
    tfa.load_state_dict(ckpt["model_state_dict"])
    tfa.eval()

    logger.info(f"TFA loaded successfully. Val CosODE: {ckpt.get('val_cos_ode', 'N/A')}")

    return tfa


def load_hbbops_evaluations(
    hbbops_path: str,
    min_fidelity: int = 600,
) -> List[Dict]:
    """Load HbBoPs evaluations with fidelity threshold."""
    logger.info(f"Loading HbBoPs evaluations from {hbbops_path}")

    with open(hbbops_path) as f:
        data = json.load(f)

    metadata = data["metadata"]
    results = data["results"]
    max_fid = metadata["max_fidelity"]

    logger.info(f"Total evaluations: {len(results)}, max fidelity: {max_fid}")

    evaluations = []
    for idx, item in results.items():
        if item["fidelity"] >= min_fidelity:
            evaluations.append({
                "id": idx,
                "instruction": item["instruction"],
                "accuracy": item["accuracy"],
                "error_rate": item["error_rate"],
                "fidelity": item["fidelity"],
            })

    logger.info(f"Loaded {len(evaluations)} evaluations with fidelity >= {min_fidelity}")

    # Sort by error rate for logging
    evaluations.sort(key=lambda x: x["error_rate"])
    if evaluations:
        best = evaluations[0]
        worst = evaluations[-1]
        logger.info(f"  Best: error={best['error_rate']:.3f} (fid={best['fidelity']})")
        logger.info(f"  Worst: error={worst['error_rate']:.3f} (fid={worst['fidelity']})")

    return evaluations


def encode_instructions_to_latent(
    instructions: List[str],
    tfa: nn.Module,
    device: str = "cuda:0",
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode instructions through SONAR -> TFA to get latent vectors."""
    from lido_pp.backbone.sonar_encoder import SONAREncoder

    logger.info(f"Encoding {len(instructions)} instructions to latent space")

    # Initialize SONAR
    sonar = SONAREncoder(device=device)

    # Encode in batches
    all_latents = []

    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]

        # SONAR encode
        embeddings = sonar.encode(batch)  # (B, 1024)
        embeddings = embeddings.to(device)

        # TFA encode to latent
        with torch.no_grad():
            latents = tfa.encode(embeddings)  # (B, latent_dim)

        all_latents.append(latents)
        logger.debug(f"Encoded batch {i // batch_size + 1}")

    latents = torch.cat(all_latents, dim=0)
    logger.info(f"Encoded to latents: {latents.shape}")

    return latents


class TFAVelocityWrapper(nn.Module):
    """Adapts TFA velocity (t, x) -> v to GPGuidedFlowGenerator interface (x, t, ctx) -> v."""

    def __init__(self, tfa: nn.Module):
        super().__init__()
        self.tfa = tfa
        self.velocity = tfa.velocity
        self.to_latent = tfa.to_latent
        self.from_latent = tfa.from_latent

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity in flow space."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.velocity(t, x)


class LatentGPWrapper(nn.Module):
    """Maps GP predictions from latent space to flow space for gradient computation."""

    def __init__(self, gp, to_latent: nn.Module, from_latent: nn.Module):
        super().__init__()
        self.gp = gp
        self.to_latent = to_latent
        self.from_latent = from_latent
        self.best_error_rate = getattr(gp, 'best_error_rate', 1.0)
        self.X_train = getattr(gp, 'X_train', None)
        self.y_train = getattr(gp, 'y_train', None)

    def predict(self, x_flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict on flow-space vectors by projecting to latent first."""
        z = self.to_latent(x_flow)
        return self.gp.predict(z)

    def compute_guidance_gradient(
        self,
        x_flow: torch.Tensor,
        ucb_beta: float = 4.0,
    ) -> torch.Tensor:
        """Compute guidance gradient in flow space."""
        z = self.to_latent(x_flow)

        if hasattr(self.gp, 'compute_guidance_gradient'):
            grad_latent = self.gp.compute_guidance_gradient(z, ucb_beta)
        else:
            z_grad = z.detach().requires_grad_(True)
            mean, std = self.gp.predict(z_grad)
            reward = -mean + ucb_beta * std
            reward.sum().backward()
            grad_latent = z_grad.grad if z_grad.grad is not None else torch.zeros_like(z)

        # Project gradient back: grad_flow = grad_latent @ W
        W = self.to_latent.weight
        return torch.matmul(grad_latent, W)


def run_gp_guided_inference(
    tfa: nn.Module,
    gp,  # HighDimGP
    train_latents: torch.Tensor,
    num_samples: int = 16,
    num_steps: int = 20,
    guidance_scale: float = 1.0,
    ucb_beta: float = 4.0,
    exploration_noise: float = 0.5,
    device: str = "cuda:0",
) -> Dict:
    """
    Run GP-guided flow generation with proper initialization.

    Initializes from training latent distribution (not N(0,1)) since TFA
    was trained on from_latent(z) where z comes from encoded instructions.
    """
    from lido_pp.flow.gp_guided_flow import GPGuidedFlowGenerator

    logger.info(f"Running GP-guided inference: samples={num_samples}, steps={num_steps}")

    flow_dim = tfa.flow_dim
    latent_dim = tfa.latent_dim
    logger.info(f"Dimensions: flow_dim={flow_dim}, latent_dim={latent_dim}")

    # Compute training latent statistics for proper initialization
    train_mean = train_latents.mean(dim=0)  # (latent_dim,)
    train_std = train_latents.std(dim=0).clamp(min=1e-6)  # (latent_dim,)
    logger.info(f"Training latent stats: mean norm={train_mean.norm():.4f}, mean std={train_std.mean():.6f}")

    # Sample initial latents from training distribution with exploration noise
    # z_init ~ N(train_mean, (exploration_noise * train_std)^2)
    noise = torch.randn(num_samples, latent_dim, device=device)
    z_init = train_mean.unsqueeze(0) + exploration_noise * train_std.unsqueeze(0) * noise
    logger.info(f"Initial latents: mean norm={z_init.mean(dim=0).norm():.4f}, std={z_init.std(dim=0).mean():.6f}")

    # Project initial latents to flow space via from_latent
    # This gives proper initialization at t=0 that the TFA velocity field expects
    with torch.no_grad():
        x0_flow = tfa.from_latent(z_init)  # (B, flow_dim)
    logger.info(f"Initial flow vectors: norm mean={x0_flow.norm(dim=-1).mean():.4f}")

    # Create wrappers
    velocity_wrapper = TFAVelocityWrapper(tfa).to(device)
    gp_wrapper = LatentGPWrapper(gp, tfa.to_latent, tfa.from_latent).to(device)

    # Create GP-guided generator operating in flow space
    generator = GPGuidedFlowGenerator(
        flowdit=velocity_wrapper,
        latent_dim=flow_dim,  # Generate in flow space!
        guidance_scale=guidance_scale,
        schedule="linear",
        ucb_beta=ucb_beta,
    )
    generator.set_gp_model(gp_wrapper)
    generator = generator.to(device)

    # Generate in flow space with proper initialization!
    result = generator.generate(
        batch_size=num_samples,
        num_steps=num_steps,
        acquisition="ucb",
        return_trajectory=False,
        initial_noise=x0_flow,  # Use proper initialization, not randn!
    )

    # Project to latent space
    with torch.no_grad():
        latents = tfa.to_latent(result.latents)  # (B, latent_dim)

    # Decode latents to SONAR embeddings using FULL TFA decode path
    # This ensures proper ODE integration: latent → from_latent → ODE(0→1) → dec_proj
    with torch.no_grad():
        embeddings = tfa.decode(latents, normalize=True)  # (B, input_dim=1024)

    # Get GP predictions on true latents
    mean, std = gp.predict(latents)

    logger.info(f"Generated {num_samples} samples")
    logger.info(f"  Predicted error rates: {mean.min():.3f} - {mean.max():.3f}")
    logger.info(f"  Prediction std: {std.min():.3f} - {std.max():.3f}")

    return {
        "latents": latents,
        "flow_vectors": result.latents,  # In flow space
        "embeddings": embeddings,
        "predicted_error_rates": mean,
        "predicted_std": std,
        "guidance_norms": result.guidance_norms,
    }


def main():
    parser = argparse.ArgumentParser(description="FlowPO GP-Guided Inference")

    # Paths
    parser.add_argument(
        "--tfa-checkpoint",
        type=str,
        default="lido_pp/checkpoints/tfa_best.pt",
        help="Path to TFA checkpoint",
    )
    parser.add_argument(
        "--hbbops-path",
        type=str,
        default="lipo/data/hbbops_results_20260102.json",
        help="Path to HbBoPs evaluations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="lido_pp/results",
        help="Output directory",
    )

    # Generation parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="ODE integration steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="GP guidance strength",
    )
    parser.add_argument(
        "--ucb-beta",
        type=float,
        default=4.0,
        help="UCB exploration parameter (higher = more exploration)",
    )

    # GP parameters
    parser.add_argument(
        "--gp-type",
        type=str,
        choices=["isotropic", "saas", "adaptive"],
        default="isotropic",
        help="GP type to use",
    )
    parser.add_argument(
        "--min-fidelity",
        type=int,
        default=600,
        help="Minimum fidelity threshold for evaluations (default: 600)",
    )
    parser.add_argument(
        "--exploration-noise",
        type=float,
        default=0.5,
        help="Noise scale for initialization (relative to training std, default: 0.5)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log configuration
    logger.info("=" * 60)
    logger.info("FlowPO GP-Guided Inference")
    logger.info("=" * 60)
    logger.info(f"TFA checkpoint: {args.tfa_checkpoint}")
    logger.info(f"HbBoPs path: {args.hbbops_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"GP type: {args.gp_type}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Num steps: {args.num_steps}")
    logger.info(f"Guidance scale: {args.guidance_scale}")
    logger.info(f"UCB beta: {args.ucb_beta}")
    logger.info(f"Min fidelity: {args.min_fidelity}")
    logger.info(f"Exploration noise: {args.exploration_noise}")

    # Check device
    if "cuda" in args.device and not torch.cuda.is_available():
        logger.error("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # 1. Load TFA checkpoint
    logger.info("\n[1/4] Loading TFA checkpoint...")
    tfa = load_tfa_checkpoint(args.tfa_checkpoint, args.device)
    latent_dim = tfa.latent_dim
    logger.info(f"TFA latent dim: {latent_dim}")

    # 2. Load HbBoPs evaluations
    logger.info("\n[2/4] Loading HbBoPs evaluations...")
    evaluations = load_hbbops_evaluations(args.hbbops_path, min_fidelity=args.min_fidelity)

    if len(evaluations) == 0:
        logger.error(f"No evaluations found with fidelity >= {args.min_fidelity}!")
        sys.exit(1)

    instructions = [e["instruction"] for e in evaluations]
    error_rates = torch.tensor([e["error_rate"] for e in evaluations], device=args.device)

    logger.info(f"Loaded {len(evaluations)} evaluations (fidelity >= {args.min_fidelity})")
    logger.info(f"Error rate range: [{error_rates.min():.3f}, {error_rates.max():.3f}]")

    # 3. Encode instructions to latent space
    logger.info("\n[3/4] Encoding instructions to latent space...")
    latents = encode_instructions_to_latent(instructions, tfa, args.device)

    # 4. Train GP
    logger.info("\n[4/4] Training GP...")
    from lido_pp.gp.high_dim_gp import (
        IsotropicHighDimGP,
        SaasHighDimGP,
        AdaptiveHighDimGP,
    )

    if args.gp_type == "isotropic":
        gp = IsotropicHighDimGP(
            latent_dim=latent_dim,
            device=args.device,
            ucb_beta=args.ucb_beta,
        )
    elif args.gp_type == "saas":
        gp = SaasHighDimGP(latent_dim=latent_dim, device=args.device)
    else:  # adaptive
        gp = AdaptiveHighDimGP(
            latent_dim=latent_dim,
            device=args.device,
            ucb_beta=args.ucb_beta,
        )

    gp.fit(latents, error_rates)
    logger.info(f"GP trained on {len(latents)} points")

    # 5. Run GP-guided inference
    logger.info("\n[5/5] Running GP-guided inference...")
    results = run_gp_guided_inference(
        tfa=tfa,
        gp=gp,
        train_latents=latents,  # Pass training latents for proper initialization!
        num_samples=args.batch_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        ucb_beta=args.ucb_beta,
        exploration_noise=args.exploration_noise,
        device=args.device,
    )

    # 6. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"gp_guided_inference_{timestamp}.pt")

    torch.save({
        "latents": results["latents"].cpu(),
        "embeddings": results["embeddings"].cpu(),
        "predicted_error_rates": results["predicted_error_rates"].cpu(),
        "predicted_std": results["predicted_std"].cpu(),
        "guidance_norms": results["guidance_norms"],
        "config": {
            "tfa_checkpoint": args.tfa_checkpoint,
            "hbbops_path": args.hbbops_path,
            "gp_type": args.gp_type,
            "batch_size": args.batch_size,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "ucb_beta": args.ucb_beta,
            "exploration_noise": args.exploration_noise,
            "min_fidelity": args.min_fidelity,
            "latent_dim": latent_dim,
            "n_train": len(latents),
        },
        "train_data": {
            "latents": latents.cpu(),
            "error_rates": error_rates.cpu(),
            "instructions": instructions,
        },
    }, output_path)

    logger.info(f"\nResults saved to: {output_path}")

    # 7. Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Training points: {len(latents)}")
    logger.info(f"Generated samples: {args.batch_size}")
    logger.info(f"Predicted error rates: {results['predicted_error_rates'].min():.3f} - {results['predicted_error_rates'].max():.3f}")
    logger.info(f"Mean guidance norm: {sum(results['guidance_norms']) / len(results['guidance_norms']):.4f}")

    # Show best predictions
    best_idx = results["predicted_error_rates"].argmin()
    logger.info(f"\nBest predicted sample:")
    logger.info(f"  Index: {best_idx}")
    logger.info(f"  Predicted error rate: {results['predicted_error_rates'][best_idx]:.3f}")
    logger.info(f"  Prediction std: {results['predicted_std'][best_idx]:.3f}")


if __name__ == "__main__":
    main()
