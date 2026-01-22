#!/usr/bin/env python3
"""
GP BO with Perturbation-based Candidate Generation.

Key insight: In 1024D SONAR space, arbitrary points decode to garbage.
Solution: Generate candidates as small perturbations of known-good embeddings.

From FINDINGS.md:
- 1% perturbation → CosSim 0.95, perfect preservation
- 2% perturbation → CosSim 0.84, perfect preservation
- 5% perturbation → CosSim 0.53, minor changes, still valid
- 10%+ → semantics break

Strategy:
1. Select parent embedding from training data (acquisition-weighted)
2. Add perturbation (1-3% of norm) in random direction
3. GP predicts on perturbed embedding
4. Evaluate best candidate
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
from flowpo_hd.utils import SONARHelper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def generate_perturbation_candidates(
    X_train: torch.Tensor,
    gp: BetaHeteroscedasticGP,
    n_candidates: int = 64,
    perturbation_scale: float = 0.02,  # 2% of norm
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Generate candidates via perturbation of training embeddings.

    Args:
        X_train: Training embeddings [N, D]
        gp: Fitted GP for acquisition-weighted selection
        n_candidates: Number of candidates to generate
        perturbation_scale: Perturbation magnitude as fraction of embedding norm
        temperature: Temperature for softmax parent selection (lower = more exploitation)

    Returns:
        candidates: [n_candidates, D] perturbed embeddings
    """
    N, D = X_train.shape

    # Get GP predictions for selection weighting
    with torch.no_grad():
        mean, std = gp.predict(X_train)

    # Thompson Sampling: sample from posterior to get scores
    # Lower error rate = better, so negate for selection
    samples = torch.normal(mean, std)
    scores = -samples  # Higher score = better (lower error)

    # Softmax for parent selection probabilities
    probs = F.softmax(scores / temperature, dim=0).numpy()

    # Sample parents with replacement (weighted by acquisition)
    parent_indices = np.random.choice(N, size=n_candidates, p=probs, replace=True)
    parents = X_train[parent_indices]  # [n_candidates, D]

    # Generate perturbations
    # Random direction (normalized) scaled by embedding norm
    directions = torch.randn(n_candidates, D, dtype=torch.float64)
    directions = F.normalize(directions, dim=-1)

    # Scale by parent norm and perturbation scale
    parent_norms = parents.norm(dim=-1, keepdim=True)  # [n_candidates, 1]
    perturbations = directions * parent_norms * perturbation_scale

    # Add perturbation
    candidates = parents + perturbations

    return candidates, parent_indices


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
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--eval-fidelity', type=int, default=1319)
    parser.add_argument('--output-dir', type=str, default='flowpo_hd/results')
    parser.add_argument('--perturbation-scale', type=float, default=0.02,
                       help='Perturbation magnitude (fraction of norm)')
    parser.add_argument('--n-candidates', type=int, default=64,
                       help='Number of candidates per iteration')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for parent selection (lower = more exploitation)')
    parser.add_argument('--skip-llm', action='store_true')
    args = parser.parse_args()

    device = "cuda"

    logger.info("=" * 60)
    logger.info("GP BO with Perturbation-based Candidate Generation")
    logger.info("=" * 60)
    logger.info(f"  perturbation_scale: {args.perturbation_scale}")
    logger.info(f"  n_candidates: {args.n_candidates}")
    logger.info(f"  temperature: {args.temperature}")
    logger.info("=" * 60)

    # Load warm start data
    logger.info(f"Loading warm start from {args.warm_start}...")
    warm = load_warm_start_data(args.warm_start)

    X = warm['embeddings']
    acc = warm['accuracies']
    fid = warm['fidelities']
    instructions = warm['instructions']

    avg_norm = X.norm(dim=-1).mean().item()
    logger.info(f"Loaded {len(X)} points, avg_norm={avg_norm:.4f}")

    # Find best
    best_idx = (1 - acc).argmin()
    best_err = (1 - acc)[best_idx].item()
    best_inst = instructions[best_idx]
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

    # Initialize vLLM
    if not args.skip_llm:
        from vllm import LLM
        from datasets import load_dataset

        logger.info("Initializing vLLM...")
        llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization=0.8, max_model_len=4096)

        logger.info("Loading GSM8K...")
        gsm8k = list(load_dataset("openai/gsm8k", "main", split="test"))

    # Optimization loop
    results = []
    X_train = X.clone()
    acc_train = acc.clone()
    fid_train = fid.clone()
    inst_train = list(instructions)

    best_overall = best_err
    best_overall_inst = best_inst

    for i in range(args.iterations):
        logger.info(f"\n--- Iteration {i+1}/{args.iterations} ---")
        iter_start = time.time()

        # Generate candidates via perturbation
        candidates, parent_indices = generate_perturbation_candidates(
            X_train,
            gp,
            n_candidates=args.n_candidates,
            perturbation_scale=args.perturbation_scale,
            temperature=args.temperature,
        )

        # Get GP predictions for all candidates
        with torch.no_grad():
            means, stds = gp.predict(candidates)

        # Select best by Thompson Sampling
        samples = torch.normal(means, stds)
        best_cand_idx = samples.argmin()

        candidate = candidates[best_cand_idx:best_cand_idx+1]
        parent_idx = parent_indices[best_cand_idx]
        mean = means[best_cand_idx]
        std = stds[best_cand_idx]

        logger.info(f"Parent instruction (idx={parent_idx}):")
        logger.info(f"  {inst_train[parent_idx][:80]}...")
        logger.info(f"GP Prediction: err={mean.item():.3f} ± {std.item():.3f}")

        # Decode
        candidate_float = candidate.to(dtype=torch.float32, device=device)
        instruction = sonar.decode(candidate_float)[0]
        logger.info(f"Generated instruction:")
        logger.info(f"  {instruction}")

        # Check similarity to parent
        parent_emb = X_train[parent_idx:parent_idx+1]
        cos_sim = F.cosine_similarity(candidate.float(), parent_emb.float()).item()
        logger.info(f"CosSim to parent: {cos_sim:.4f}")

        if args.skip_llm:
            true_err = mean.item() + np.random.randn() * 0.1
            true_err = np.clip(true_err, 0, 1)
        else:
            accuracy, true_err = evaluate_instruction(instruction, llm, gsm8k, args.eval_fidelity)
            logger.info(f"Evaluation: acc={accuracy:.3f}, err={true_err:.3f}")

        # Analysis
        sigma_err = (true_err - mean.item()) / max(std.item(), 1e-6)
        logger.info(f"Prediction error: {true_err - mean.item():+.4f} ({sigma_err:+.2f}σ)")

        # Update TuRBO state
        gp.update_turbo_state(true_err)

        if true_err < best_overall:
            best_overall = true_err
            best_overall_inst = instruction
            logger.info(f"*** NEW BEST: err={best_overall:.3f} ***")

        # Update training data
        X_train = torch.cat([X_train, candidate])
        acc_train = torch.cat([acc_train, torch.tensor([1 - true_err], dtype=torch.float64)])
        fid_train = torch.cat([fid_train, torch.tensor([args.eval_fidelity], dtype=torch.float64)])
        inst_train.append(instruction)

        # Refit GP
        gp.fit(X_train, acc_train, fid_train)

        iter_time = time.time() - iter_start
        logger.info(f"Iteration time: {iter_time:.1f}s")

        results.append({
            'iteration': i + 1,
            'instruction': instruction,
            'parent_instruction': inst_train[parent_idx],
            'parent_idx': int(parent_idx),
            'error_rate': float(true_err),
            'predicted': float(mean.item()),
            'predicted_std': float(std.item()),
            'sigma_error': float(sigma_err),
            'cos_sim_to_parent': float(cos_sim),
        })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best error: {best_overall:.4f}")
    logger.info(f"Best instruction:\n{best_overall_inst}")

    # Save
    output_path = Path(args.output_dir) / f"gp_perturbation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(args.output_dir, exist_ok=True)
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
