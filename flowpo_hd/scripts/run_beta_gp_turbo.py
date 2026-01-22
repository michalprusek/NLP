#!/usr/bin/env python3
"""
FlowPO-HD with BetaHeteroscedasticGP + TuRBO.

Uses space mapping results for initial GP training, then runs TuRBO optimization.

Usage:
    uv run python -m flowpo_hd.scripts.run_beta_gp_turbo \
        --mapping-results flowpo_hd/results/space_mapping_100x100.json \
        --iterations 20
"""

# MUST be first - set spawn method before any imports that use CUDA
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from flowpo_hd.beta_gp import BetaHeteroscedasticGP, BetaGPConfig
from flowpo_hd.utils import SONARHelper, FlowDiTHelper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mapping_results(path: str) -> dict:
    """Load space mapping results."""
    with open(path) as f:
        data = json.load(f)

    results = data['results']

    # Convert to tensors
    embeddings = torch.tensor([r['embedding'] for r in results], dtype=torch.float64)
    accuracies = torch.tensor([r['accuracy'] for r in results], dtype=torch.float64)
    error_rates = torch.tensor([r['error_rate'] for r in results], dtype=torch.float64)
    fidelities = torch.full((len(results),), data['config']['n_examples'], dtype=torch.float64)
    instructions = [r['instruction'] for r in results]

    return {
        'embeddings': embeddings,
        'accuracies': accuracies,
        'error_rates': error_rates,
        'fidelities': fidelities,
        'instructions': instructions,
        'config': data['config'],
    }


def evaluate_instruction(instruction: str, llm, gsm8k_examples, n_examples: int = 1319):
    """Evaluate instruction on GSM8K."""
    import re
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
        # Extract last number
        numbers = re.findall(r'-?\d+\.?\d*', pred_text.replace(',', ''))
        pred = float(numbers[-1]) if numbers else None

        # Extract gold
        match = re.search(r'####\s*(-?\d+\.?\d*)', examples[i]['answer'].replace(',', ''))
        gold = float(match.group(1)) if match else None

        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            correct += 1

    accuracy = correct / len(examples)
    return accuracy, 1 - accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping-results', type=str, required=True,
                        help='Path to space mapping results JSON')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--eval-fidelity', type=int, default=1319,
                        help='Number of examples for evaluation')
    parser.add_argument('--output-dir', type=str, default='flowpo_hd/results')
    parser.add_argument('--skip-llm-eval', action='store_true',
                        help='Skip LLM evaluation (for testing)')
    parser.add_argument('--acquisition', type=str, default='ts',
                        choices=['ts', 'ei', 'nei'],
                        help='Acquisition: ts=Thompson Sampling, ei=EI, nei=Noisy EI (best for noisy)')
    parser.add_argument('--n-candidates', type=int, default=32,
                        help='Number of Sobol candidates for acquisition')
    parser.add_argument('--beta-alpha', type=float, default=10.0,
                        help='Beta prior alpha (for smoothing)')
    parser.add_argument('--beta-beta', type=float, default=2.0,
                        help='Beta prior beta (for smoothing)')
    parser.add_argument('--trust-region-init', type=float, default=0.5,
                        help='Initial TuRBO trust region size')
    parser.add_argument('--use-flowdit', action='store_true',
                        help='Use FlowDiT for manifold projection (RECOMMENDED)')
    parser.add_argument('--flowdit-checkpoint', type=str,
                        default='flowpo_hd/checkpoints_mega_aux2/best.pt',
                        help='Path to FlowDiT checkpoint')
    parser.add_argument('--flowdit-steps', type=int, default=20,
                        help='Number of ODE steps for FlowDiT')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FlowPO-HD with BetaGP + TuRBO")
    logger.info("=" * 60)

    # Log ALL parameters
    logger.info("RUN PARAMETERS:")
    logger.info(f"  mapping_results: {args.mapping_results}")
    logger.info(f"  iterations: {args.iterations}")
    logger.info(f"  eval_fidelity: {args.eval_fidelity}")
    logger.info(f"  acquisition: {args.acquisition}")
    logger.info(f"  n_candidates: {args.n_candidates}")
    logger.info(f"  beta_alpha: {args.beta_alpha}")
    logger.info(f"  beta_beta: {args.beta_beta}")
    logger.info(f"  trust_region_init: {args.trust_region_init}")
    logger.info(f"  skip_llm_eval: {args.skip_llm_eval}")
    logger.info(f"  use_flowdit: {args.use_flowdit}")
    if args.use_flowdit:
        logger.info(f"  flowdit_checkpoint: {args.flowdit_checkpoint}")
        logger.info(f"  flowdit_steps: {args.flowdit_steps}")
    logger.info("=" * 60)

    # Load mapping results
    logger.info(f"Loading mapping results from {args.mapping_results}")
    mapping = load_mapping_results(args.mapping_results)

    X = mapping['embeddings']
    acc = mapping['accuracies']
    fid = mapping['fidelities']

    logger.info(f"Loaded {len(X)} mapping points")
    logger.info(f"Error rate range: [{mapping['error_rates'].min():.3f}, {mapping['error_rates'].max():.3f}]")

    # Find best from mapping
    best_idx = mapping['error_rates'].argmin()
    best_err = mapping['error_rates'][best_idx].item()
    best_inst = mapping['instructions'][best_idx]
    logger.info(f"Best from mapping: err={best_err:.3f}")
    logger.info(f"  Best instruction from mapping:\n{best_inst}")

    # Initialize BetaGP
    config = BetaGPConfig(
        input_dim=X.shape[1],
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
        trust_region_init=args.trust_region_init,
    )
    gp = BetaHeteroscedasticGP(config)
    logger.info(f"GP CONFIG: kernel=rbf, ls_prior_loc={config.ls_prior_loc:.2f}, ls_prior_scale={config.ls_prior_scale:.2f}")

    # Fit initial GP
    logger.info("Fitting BetaGP on mapping data...")
    gp.fit(X, acc, fid)

    # Compute bounds from mapping data
    bounds = torch.stack([X.min(dim=0).values, X.max(dim=0).values])

    # Initialize vLLM FIRST (before any CUDA ops to avoid fork issues)
    if not args.skip_llm_eval:
        from vllm import LLM
        from datasets import load_dataset

        logger.info("Initializing vLLM (must be first for CUDA fork)...")
        llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            gpu_memory_utilization=0.8,  # Leave room for SONAR
            max_model_len=4096,
        )

        logger.info("Initializing SONAR decoder...")
        sonar = SONARHelper(device='cuda')

        # Initialize FlowDiT for manifold projection (if enabled)
        flow_dit = None
        if args.use_flowdit:
            logger.info(f"Initializing FlowDiT from {args.flowdit_checkpoint}...")
            flow_dit = FlowDiTHelper(
                checkpoint_path=args.flowdit_checkpoint,
                device='cuda',
                num_steps=args.flowdit_steps,
            )

        logger.info("Loading GSM8K...")
        gsm8k = load_dataset("openai/gsm8k", "main", split="test")
        gsm8k_examples = list(gsm8k)

    # TuRBO optimization loop
    logger.info("=" * 60)
    logger.info(f"Starting TuRBO optimization for {args.iterations} iterations")
    logger.info("=" * 60)

    results = []
    X_train = X.clone()
    acc_train = acc.clone()
    fid_train = fid.clone()

    best_overall_err = best_err
    best_overall_inst = best_inst

    for i in range(args.iterations):
        iter_start = time.time()

        logger.info(f"\n--- Iteration {i+1}/{args.iterations} ---")
        logger.info(f"TuRBO: L={gp._trust_region_length:.3f}, best={gp._best_value:.3f}")

        # Get candidate
        candidate, acq_val = gp.get_candidate_turbo(
            bounds, n_candidates=args.n_candidates, acquisition=args.acquisition
        )

        # Predict
        mean, std = gp.predict(candidate)
        logger.info(f"GP Prediction: err={mean.item():.3f} ± {std.item():.3f}")

        if args.skip_llm_eval:
            # Simulate evaluation
            true_err = mean.item() + np.random.randn() * 0.05
            true_err = np.clip(true_err, 0, 1)
            instruction = f"[Simulated instruction {i+1}]"
        else:
            # Decode candidate
            candidate_float = candidate.to(dtype=torch.float32, device='cuda')

            # Project to manifold if FlowDiT enabled
            if flow_dit is not None:
                logger.info("Projecting candidate through FlowDiT...")
                candidate_float = flow_dit.project_to_manifold(candidate_float)

            instruction = sonar.decode(candidate_float)[0]
            logger.info(f"Generated instruction:\n{instruction}")

            # Evaluate
            accuracy, true_err = evaluate_instruction(
                instruction, llm, gsm8k_examples, args.eval_fidelity
            )
            logger.info(f"Evaluation: acc={accuracy:.3f}, err={true_err:.3f}")

        # Post-evaluation analysis
        pred_err = mean.item()
        sigma_err = (true_err - pred_err) / max(std.item(), 1e-6)
        logger.info(f"Prediction error: {true_err - pred_err:+.4f} ({sigma_err:+.2f}σ)")

        if abs(sigma_err) > 2:
            logger.warning(f"GP was {'over' if sigma_err > 0 else 'under'}confident!")
        else:
            logger.info("GP prediction was well-calibrated")

        # Update TuRBO
        gp.update_turbo_state(true_err)

        # Update best
        if true_err < best_overall_err:
            best_overall_err = true_err
            best_overall_inst = instruction
            logger.info(f"NEW BEST: err={best_overall_err:.3f}")

        # Add to training data and refit
        X_train = torch.cat([X_train, candidate], dim=0)
        acc_train = torch.cat([acc_train, torch.tensor([1 - true_err], dtype=torch.float64)])
        fid_train = torch.cat([fid_train, torch.tensor([args.eval_fidelity], dtype=torch.float64)])

        gp.fit(X_train, acc_train, fid_train)

        # Update TuRBO center
        gp._center = candidate.squeeze()

        # Log result (ensure all values are JSON serializable)
        results.append({
            'iteration': i + 1,
            'instruction': instruction,
            'error_rate': float(true_err),
            'predicted_err': float(pred_err),
            'predicted_std': float(std.item()),
            'sigma_error': float(sigma_err),
            'trust_region': float(gp._trust_region_length),
            'best_so_far': float(best_overall_err),
        })

        iter_time = time.time() - iter_start
        logger.info(f"Iteration time: {iter_time:.1f}s")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best error rate: {best_overall_err:.4f}")
    logger.info(f"Best instruction:\n{best_overall_inst}")

    # Calibration analysis
    sigma_errors = [r['sigma_error'] for r in results]
    logger.info(f"\nCalibration analysis:")
    logger.info(f"  Mean |σ-error|: {np.mean(np.abs(sigma_errors)):.2f}")
    logger.info(f"  Within 1σ: {sum(abs(s) < 1 for s in sigma_errors)}/{len(sigma_errors)}")
    logger.info(f"  Within 2σ: {sum(abs(s) < 2 for s in sigma_errors)}/{len(sigma_errors)}")

    # Save results
    output_path = Path(args.output_dir) / f"beta_gp_turbo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'mapping_results': args.mapping_results,
                'iterations': args.iterations,
                'eval_fidelity': args.eval_fidelity,
                'beta_alpha': config.beta_alpha,
                'beta_beta': config.beta_beta,
            },
            'results': results,
            'best_error_rate': float(best_overall_err),
            'best_instruction': best_overall_inst,
            'calibration': {
                'mean_abs_sigma_error': float(np.mean(np.abs(sigma_errors))),
                'within_1_sigma': int(sum(abs(s) < 1 for s in sigma_errors)),
                'within_2_sigma': int(sum(abs(s) < 2 for s in sigma_errors)),
            }
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
