#!/usr/bin/env python3
"""
GP BO with FlowDiT for Candidate Generation.

Strategy:
1. GP operates in 1024D SONAR space
2. FlowDiT generates NEW candidates from noise (not projection!)
3. Small perturbations (1-2%) for local exploration stay valid
4. TuRBO trust region limits exploration to valid regions

Key insight: FlowDiT can generate diverse valid instructions, but cannot project
arbitrary embeddings back to manifold. Use it for sampling, not correction.
"""

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flowpo_hd.beta_gp import BetaHeteroscedasticGP, BetaGPConfig
from flowpo_hd.flow_dit import FlowDiT, integrate_euler
from flowpo_hd.utils import SONARHelper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FlowDiTGenerator:
    """Generate candidates using FlowDiT."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device

        logger.info(f"Loading FlowDiT from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        self.model = FlowDiT(
            latent_dim=1024, hidden_dim=1024, num_layers=4,
            time_embed_dim=256, mlp_ratio=2.0
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device).eval()

        logger.info(f"FlowDiT loaded (step {ckpt.get('global_step')}, val_loss {ckpt.get('best_val_loss', 'N/A'):.4f})")

    @torch.no_grad()
    def generate(self, n_samples: int, target_norm: float = 0.26, num_steps: int = 50) -> torch.Tensor:
        """Generate candidates from noise."""
        # Sample scaled noise
        z = torch.randn(n_samples, 1024, device=self.device) * (target_norm / 32)

        # Flow integration
        x = integrate_euler(self.model, z, num_steps=num_steps)

        # Scale to target norm
        x = x * (target_norm / x.norm(dim=-1, keepdim=True))

        return x


def load_warm_start_data(path: str) -> dict:
    """Load warm start data from space mapping."""
    with open(path) as f:
        data = json.load(f)

    results = data['results']

    return {
        'embeddings': torch.tensor([r['embedding'] for r in results], dtype=torch.float64),
        'accuracies': torch.tensor([r['accuracy'] for r in results], dtype=torch.float64),
        'instructions': [r['instruction'] for r in results],
        'fidelities': torch.full((len(results),), data['config']['n_examples'], dtype=torch.float64),
    }


def evaluate_instruction(instruction: str, llm, gsm8k_examples, n_examples: int = 1319):
    """Evaluate instruction on GSM8K."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # Sample examples
    if n_examples < len(gsm8k_examples):
        indices = np.random.choice(len(gsm8k_examples), n_examples, replace=False)
        examples = [gsm8k_examples[i] for i in indices]
    else:
        examples = gsm8k_examples

    # Build prompts
    prompts = [f"Q: {ex['question']}\n{instruction}\nA:" for ex in examples]

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Score
    correct = 0
    for i, output in enumerate(outputs):
        pred_text = output.outputs[0].text
        numbers = re.findall(r'-?\d+\.?\d*', pred_text.replace(',', ''))
        pred = float(numbers[-1]) if numbers else None

        match = re.search(r'####\s*(-?\d+\.?\d*)', examples[i]['answer'].replace(',', ''))
        gold = float(match.group(1)) if match else None

        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            correct += 1

    accuracy = correct / len(examples)
    return accuracy, 1 - accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm-start', type=str, required=True,
                       help='Path to space mapping results for warm start')
    parser.add_argument('--flowdit-checkpoint', type=str,
                       default='flowpo_hd/checkpoints_flow_dit_aux/best.pt',
                       help='FlowDiT checkpoint for generation')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--eval-fidelity', type=int, default=1319)
    parser.add_argument('--output-dir', type=str, default='flowpo_hd/results')
    parser.add_argument('--use-flowdit-gen', action='store_true',
                       help='Use FlowDiT to generate candidates from noise')
    parser.add_argument('--perturbation-scale', type=float, default=0.02,
                       help='Scale for local perturbations (fraction of norm)')
    parser.add_argument('--gen-ratio', type=float, default=0.3,
                       help='Ratio of FlowDiT-generated vs perturbed candidates')
    parser.add_argument('--skip-llm', action='store_true')
    args = parser.parse_args()

    device = "cuda"

    logger.info("=" * 60)
    logger.info("GP BO with FlowDiT Generation")
    logger.info("=" * 60)

    # Load warm start data
    logger.info(f"Loading warm start from {args.warm_start}...")
    warm = load_warm_start_data(args.warm_start)

    X = warm['embeddings']
    acc = warm['accuracies']
    fid = warm['fidelities']

    target_norm = X.norm(dim=-1).mean().item()
    logger.info(f"Loaded {len(X)} points, target_norm={target_norm:.4f}")

    # Find best
    best_idx = (1 - acc).argmin()
    best_err = (1 - acc)[best_idx].item()
    best_inst = warm['instructions'][best_idx]
    logger.info(f"Best from warm start: err={best_err:.3f}")
    logger.info(f"  {best_inst[:100]}...")

    # Initialize GP
    config = BetaGPConfig(input_dim=1024, trust_region_init=0.1)
    gp = BetaHeteroscedasticGP(config)
    gp.fit(X, acc, fid)
    logger.info("GP fitted")

    # Initialize SONAR
    logger.info("Initializing SONAR...")
    sonar = SONARHelper(device=device, normalize=False)

    # Initialize FlowDiT generator (if enabled)
    flowdit_gen = None
    if args.use_flowdit_gen:
        flowdit_gen = FlowDiTGenerator(args.flowdit_checkpoint, device)

    # Initialize vLLM
    if not args.skip_llm:
        from vllm import LLM
        from datasets import load_dataset

        logger.info("Initializing vLLM...")
        llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization=0.8, max_model_len=4096)

        logger.info("Loading GSM8K...")
        gsm8k = list(load_dataset("openai/gsm8k", "main", split="test"))

    # Bounds from data
    bounds = torch.stack([X.min(dim=0).values, X.max(dim=0).values])

    # Optimization loop
    results = []
    X_train = X.clone()
    acc_train = acc.clone()
    fid_train = fid.clone()

    best_overall = best_err
    best_overall_inst = best_inst

    for i in range(args.iterations):
        logger.info(f"\n--- Iteration {i+1}/{args.iterations} ---")

        # Generate candidate
        if flowdit_gen and np.random.random() < args.gen_ratio:
            # Use FlowDiT to generate from noise
            logger.info("Generating candidate with FlowDiT...")
            candidate = flowdit_gen.generate(1, target_norm=target_norm).to(torch.float64)
        else:
            # Use GP acquisition + small perturbation
            logger.info("Using GP acquisition...")
            candidate, _ = gp.get_candidate_turbo(bounds, n_candidates=64)

            # Add small perturbation to stay valid
            noise = torch.randn_like(candidate) * candidate.norm() * args.perturbation_scale
            candidate = candidate + noise

        # Predict
        mean, std = gp.predict(candidate)
        logger.info(f"GP prediction: err={mean.item():.3f} ± {std.item():.3f}")

        # Decode
        candidate_float = candidate.to(dtype=torch.float32, device=device)
        instruction = sonar.decode(candidate_float)[0]
        logger.info(f"Instruction:\n{instruction}")

        if args.skip_llm:
            true_err = mean.item() + np.random.randn() * 0.1
            true_err = np.clip(true_err, 0, 1)
        else:
            accuracy, true_err = evaluate_instruction(instruction, llm, gsm8k, args.eval_fidelity)
            logger.info(f"Evaluation: acc={accuracy:.3f}, err={true_err:.3f}")

        # Analysis
        sigma_err = (true_err - mean.item()) / max(std.item(), 1e-6)
        logger.info(f"Prediction error: {true_err - mean.item():+.4f} ({sigma_err:+.2f}σ)")

        # Update
        gp.update_turbo_state(true_err)

        if true_err < best_overall:
            best_overall = true_err
            best_overall_inst = instruction
            logger.info(f"NEW BEST: err={best_overall:.3f}")

        # Refit GP
        X_train = torch.cat([X_train, candidate])
        acc_train = torch.cat([acc_train, torch.tensor([1 - true_err], dtype=torch.float64)])
        fid_train = torch.cat([fid_train, torch.tensor([args.eval_fidelity], dtype=torch.float64)])
        gp.fit(X_train, acc_train, fid_train)
        gp._center = candidate.squeeze()

        results.append({
            'iteration': i + 1,
            'instruction': instruction,
            'error_rate': float(true_err),
            'predicted': float(mean.item()),
            'sigma_error': float(sigma_err),
            'used_flowdit': flowdit_gen is not None and np.random.random() < args.gen_ratio,
        })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best error: {best_overall:.4f}")
    logger.info(f"Best instruction:\n{best_overall_inst}")

    # Save
    output_path = Path(args.output_dir) / f"gp_flowdit_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': results,
            'best_error': float(best_overall),
            'best_instruction': best_overall_inst,
        }, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
