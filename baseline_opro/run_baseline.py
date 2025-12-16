#!/usr/bin/env python3
"""
Baseline OPRO Experiment

Runs pure OPRO (no HbBoPs) for 4 models independently with the same budget.
Each model gets 50,000 evaluations (200k total / 4 models).

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python baseline_opro/run_baseline.py
"""
import os
import sys
import gc
import json
from pathlib import Path
from datetime import datetime

# Set GPU before any CUDA imports
from baseline_opro.config import GPU_ID
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_opro.config import (
    MODELS,
    TOTAL_BUDGET,
    PER_MODEL_BUDGET,
    MINIBATCH_SIZE,
    NUM_CANDIDATES_PER_ITER,
)


def get_model_short_name(model_name: str) -> str:
    """Get short name for directory/file naming."""
    parts = model_name.split('/')
    name = parts[-1] if len(parts) > 1 else model_name
    return name.lower().replace('-', '_').replace('.', '_')


def run_opro_for_model(
    model_name: str,
    output_dir: Path,
    budget: int,
) -> dict:
    """Run OPRO optimization for a single model."""
    # Import here to allow GPU cleanup between models
    from src.llm_client import create_llm_client
    from src.gsm8k_evaluator import GSM8KEvaluator
    from src.opro import OPRO

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"Budget: {budget:,} evaluations")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM client (force vLLM backend for local GPU)
    print(f"Loading model: {model_name}")
    llm_client = create_llm_client(
        model_name=model_name,
        backend="vllm",
        gpu_memory_utilization=0.85,
    )

    # Initialize evaluator (training set for optimization)
    print("Loading GSM8K dataset...")
    evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split="train",
    )
    print(f"Training set: {len(evaluator)} examples")

    # Initialize OPRO
    optimizer = OPRO(
        task_llm_client=llm_client,
        meta_llm_client=llm_client,  # Same model for meta-optimization
        evaluator=evaluator,
        num_iterations=1000,  # High number - budget will stop us
        num_candidates_per_iter=NUM_CANDIDATES_PER_ITER,
        minibatch_size=MINIBATCH_SIZE,
        total_budget=budget,
    )

    # Run optimization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_prompt, history = optimizer.optimize(verbose=True)

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")

    test_evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split="test",
    )
    print(f"Test set: {len(test_evaluator)} examples")

    # Evaluate best prompt on test set
    test_batch = test_evaluator.get_batch(0, len(test_evaluator))
    test_questions = [ex['question'] for ex in test_batch]
    test_prompts = [f"Question: {q}\n\n{best_prompt}\n\nAnswer:" for q in test_questions]

    print(f"Evaluating best prompt on test set...")
    test_outputs = llm_client.generate_batch(
        test_prompts, temperature=0.0, max_new_tokens=2048
    )

    test_indices = [ex['idx'] for ex in test_batch]
    test_results = test_evaluator.evaluate_batch(test_outputs, test_indices)
    test_accuracy = test_results['accuracy']

    print(f"\nTest accuracy: {test_accuracy:.2%}")

    # Prepare results
    validation_accuracy = max(sp.score for sp in optimizer.scored_prompts) if optimizer.scored_prompts else 0

    results = {
        "model": model_name,
        "timestamp": timestamp,
        "config": {
            "budget": budget,
            "minibatch_size": MINIBATCH_SIZE,
            "num_candidates": NUM_CANDIDATES_PER_ITER,
        },
        "budget_used": optimizer.budget_used,
        "best_prompt": best_prompt,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "history": history,
    }

    # Save results
    result_file = output_dir / f"opro_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    prompt_file = output_dir / f"best_prompt_{timestamp}.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(f"# OPRO Best Prompt\n")
        f.write(f"# Model: {model_name}\n")
        f.write(f"# Validation accuracy: {validation_accuracy:.2%}\n")
        f.write(f"# Test accuracy: {test_accuracy:.2%}\n\n")
        f.write(best_prompt)

    print(f"\nResults saved to: {result_file}")

    # Cleanup to free GPU memory
    del llm_client
    del optimizer
    gc.collect()

    # Force CUDA cleanup
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    return {
        "model": model_name,
        "best_prompt": best_prompt,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "budget_used": optimizer.budget_used if 'optimizer' in dir() else budget,
    }


def main():
    print(f"\n{'='*70}")
    print("BASELINE OPRO EXPERIMENT")
    print(f"{'='*70}")
    print(f"Models: {len(MODELS)}")
    print(f"Total budget: {TOTAL_BUDGET:,}")
    print(f"Per-model budget: {PER_MODEL_BUDGET:,}")
    print(f"Minibatch size: {MINIBATCH_SIZE}")
    print(f"GPU: {GPU_ID}")
    print(f"{'='*70}\n")

    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("baseline_opro/results") / f"run_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run OPRO for each model sequentially
    for i, model_name in enumerate(MODELS):
        print(f"\n[{i+1}/{len(MODELS)}] Starting optimization for {model_name}")

        short_name = get_model_short_name(model_name)
        model_output_dir = base_output_dir / short_name

        result = run_opro_for_model(
            model_name=model_name,
            output_dir=model_output_dir,
            budget=PER_MODEL_BUDGET,
        )

        all_results[model_name] = result

        print(f"\n[{i+1}/{len(MODELS)}] Completed {model_name}")
        print(f"  Validation: {result['validation_accuracy']:.2%}")
        print(f"  Test: {result['test_accuracy']:.2%}")

    # Find best overall
    best_model = max(all_results.keys(), key=lambda k: all_results[k]['test_accuracy'])
    best_result = all_results[best_model]

    # Create summary
    summary = {
        "timestamp": timestamp,
        "total_budget": TOTAL_BUDGET,
        "per_model_budget": PER_MODEL_BUDGET,
        "eval_examples": MINIBATCH_SIZE,
        "results": {
            model: {
                "best_prompt": r["best_prompt"],
                "validation_accuracy": r["validation_accuracy"],
                "test_accuracy": r["test_accuracy"],
                "budget_used": r["budget_used"],
            }
            for model, r in all_results.items()
        },
        "best_overall": {
            "model": best_model,
            "prompt": best_result["best_prompt"],
            "validation_accuracy": best_result["validation_accuracy"],
            "test_accuracy": best_result["test_accuracy"],
        },
    }

    # Save summary
    summary_file = base_output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nResults by model:")
    for model, result in sorted(all_results.items(), key=lambda x: -x[1]['test_accuracy']):
        print(f"  {model}")
        print(f"    Validation: {result['validation_accuracy']:.2%}")
        print(f"    Test:       {result['test_accuracy']:.2%}")

    print(f"\nBest overall: {best_model}")
    print(f"  Test accuracy: {best_result['test_accuracy']:.2%}")
    print(f"  Prompt: {best_result['best_prompt'][:100]}...")

    print(f"\nSummary saved to: {summary_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
