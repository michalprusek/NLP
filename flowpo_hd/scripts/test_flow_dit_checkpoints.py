#!/usr/bin/env python3
"""
Test FlowDiT checkpoints: compare generation quality across checkpoints.

Tests:
1. Generation from noise → decode with SONAR
2. Reconstruction of known instructions
3. Velocity magnitude at t=0.9 (manifold proximity indicator)
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flowpo_hd.flow_dit import FlowDiT, integrate_euler
from flowpo_hd.utils import SONARHelper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_flow_dit(checkpoint_path: str, device: str = "cuda") -> FlowDiT:
    """Load FlowDiT model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model = FlowDiT(
        latent_dim=1024,
        hidden_dim=1024,
        num_layers=4,
        time_embed_dim=256,
        mlp_ratio=2.0,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    return model, ckpt


def test_generation(model: FlowDiT, sonar: SONARHelper, n_samples: int = 10,
                   num_steps: int = 50, device: str = "cuda") -> dict:
    """Test generation from noise."""
    results = {}

    with torch.no_grad():
        # Sample noise and integrate
        z_noise = torch.randn(n_samples, 1024, device=device)
        z_data = integrate_euler(model, z_noise, num_steps=num_steps)

        # Stats
        results['z_noise_norm'] = z_noise.norm(dim=-1).mean().item()
        results['z_data_norm'] = z_data.norm(dim=-1).mean().item()

        # Scale to typical SONAR norm (~0.18-0.28)
        target_norm = 0.20
        z_scaled = z_data * (target_norm / z_data.norm(dim=-1, keepdim=True))

        # Decode
        texts = sonar.decode(z_scaled)
        results['generated_texts'] = texts

        # Check for garbage (very short or repetitive)
        valid = 0
        for t in texts:
            if len(t) > 10 and len(set(t.split())) > 3:
                valid += 1
        results['valid_ratio'] = valid / n_samples

    return results


def test_reconstruction(model: FlowDiT, sonar: SONARHelper,
                        test_instructions: list, num_steps: int = 50,
                        device: str = "cuda") -> dict:
    """Test reconstruction of known instructions."""
    results = {}

    with torch.no_grad():
        # Encode original
        z_orig = sonar.encode(test_instructions).to(device)
        orig_norms = z_orig.norm(dim=-1)
        results['orig_norm'] = orig_norms.mean().item()

        # Add small perturbation
        noise = torch.randn_like(z_orig) * 0.01
        z_pert = z_orig + noise

        # Project back using flow (t=0.5 → t=1)
        # Treat perturbed as "noisy" version at t=0.5
        z_proj = integrate_euler(model, z_pert, num_steps=num_steps//2, t_start=0.5, t_end=1.0)

        # Scale back to original norm
        z_proj_scaled = z_proj * (orig_norms.unsqueeze(-1) / z_proj.norm(dim=-1, keepdim=True))

        # Cosine similarity
        cos_sim = F.cosine_similarity(z_orig, z_proj_scaled, dim=-1)
        results['cos_sim_mean'] = cos_sim.mean().item()
        results['cos_sim_std'] = cos_sim.std().item()

        # Decode
        decoded = sonar.decode(z_proj_scaled)
        results['reconstructed'] = list(zip(test_instructions, decoded))

    return results


def test_velocity(model: FlowDiT, sonar: SONARHelper,
                 test_embeddings: torch.Tensor, device: str = "cuda") -> dict:
    """Test velocity magnitude at t=0.9 (manifold proximity)."""
    results = {}

    with torch.no_grad():
        x = test_embeddings.to(device)
        t = torch.full((x.shape[0],), 0.9, device=device)

        v = model(x, t)
        v_norm = v.norm(dim=-1)

        results['velocity_norm_mean'] = v_norm.mean().item()
        results['velocity_norm_std'] = v_norm.std().item()

        # Lower velocity = closer to manifold
        results['interpretation'] = 'Lower is better (closer to manifold)'

    return results


def main():
    parser = argparse.ArgumentParser(description='Test FlowDiT checkpoints')
    parser.add_argument('--checkpoints', nargs='+', default=[
        'flowpo_hd/checkpoints_mega/best.pt',
        'flowpo_hd/checkpoints_mega_aux2/best.pt',
        'flowpo_hd/checkpoints_mega_aux/best.pt',
    ])
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Initialize SONAR
    logger.info("Initializing SONAR...")
    sonar = SONARHelper(device=args.device, normalize=False)

    # Test instructions for reconstruction
    test_instructions = [
        "Let's think step by step and solve this problem carefully.",
        "Break down the problem into smaller steps before solving.",
        "First, identify what the question is asking.",
        "Show your work and explain each step.",
        "Consider all the given information before answering.",
    ]

    # Get test embeddings
    test_embeddings = sonar.encode(test_instructions)

    results_all = {}

    for ckpt_path in args.checkpoints:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {ckpt_path}")
        logger.info(f"{'='*60}")

        if not Path(ckpt_path).exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        # Load model
        model, ckpt_info = load_flow_dit(ckpt_path, args.device)
        logger.info(f"Epoch: {ckpt_info.get('epoch', 'N/A')}, Step: {ckpt_info.get('global_step', 'N/A')}")
        logger.info(f"Val Loss: {ckpt_info.get('best_val_loss', 'N/A')}")

        results = {
            'checkpoint': ckpt_path,
            'epoch': ckpt_info.get('epoch', None),
            'step': ckpt_info.get('global_step', None),
            'val_loss': ckpt_info.get('best_val_loss', None),
        }

        # Test 1: Generation
        logger.info("\n--- Generation Test ---")
        gen_results = test_generation(model, sonar, args.n_samples, args.num_steps, args.device)
        results['generation'] = gen_results
        logger.info(f"Noise norm: {gen_results['z_noise_norm']:.4f}")
        logger.info(f"Data norm: {gen_results['z_data_norm']:.4f}")
        logger.info(f"Valid ratio: {gen_results['valid_ratio']:.2%}")

        logger.info("\nGenerated samples:")
        for i, text in enumerate(gen_results['generated_texts'][:5]):
            logger.info(f"  [{i+1}] {text[:100]}...")

        # Test 2: Reconstruction
        logger.info("\n--- Reconstruction Test ---")
        recon_results = test_reconstruction(model, sonar, test_instructions, args.num_steps, args.device)
        results['reconstruction'] = {
            'cos_sim_mean': recon_results['cos_sim_mean'],
            'cos_sim_std': recon_results['cos_sim_std'],
        }
        logger.info(f"Cosine similarity: {recon_results['cos_sim_mean']:.4f} ± {recon_results['cos_sim_std']:.4f}")

        logger.info("\nReconstructions:")
        for orig, recon in recon_results['reconstructed'][:3]:
            logger.info(f"  Orig:  {orig[:60]}...")
            logger.info(f"  Recon: {recon[:60]}...")
            logger.info("")

        # Test 3: Velocity
        logger.info("--- Velocity Test (t=0.9) ---")
        vel_results = test_velocity(model, sonar, test_embeddings, args.device)
        results['velocity'] = vel_results
        logger.info(f"Velocity norm: {vel_results['velocity_norm_mean']:.4f} ± {vel_results['velocity_norm_std']:.4f}")
        logger.info(f"({vel_results['interpretation']})")

        results_all[ckpt_path] = results

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}\n")

    logger.info("| Checkpoint | Val Loss | Valid Gen | Recon CosSim | Velocity |")
    logger.info("|------------|----------|-----------|--------------|----------|")

    for ckpt_path, r in results_all.items():
        name = Path(ckpt_path).parent.name
        val_loss = r.get('val_loss', float('inf'))
        valid = r.get('generation', {}).get('valid_ratio', 0)
        cos_sim = r.get('reconstruction', {}).get('cos_sim_mean', 0)
        vel = r.get('velocity', {}).get('velocity_norm_mean', float('inf'))
        logger.info(f"| {name:10s} | {val_loss:.4f}   | {valid:.0%}       | {cos_sim:.4f}       | {vel:.4f}   |")

    # Best checkpoint
    best = min(results_all.items(), key=lambda x: x[1].get('val_loss', float('inf')))
    logger.info(f"\nBest checkpoint: {best[0]}")


if __name__ == "__main__":
    main()
