"""Hyperparameter tuning for InvBO decoder.

Grid search over key VAE and GP hyperparameters.

Usage:
    # Quick tuning (subset of parameters)
    uv run python -m generation.invbo_decoder.tune_hyperparams --quick

    # Full grid search (slow, many combinations)
    uv run python -m generation.invbo_decoder.tune_hyperparams --full

    # Custom parameters
    uv run python -m generation.invbo_decoder.tune_hyperparams --vae-betas 0.05 0.1 --vae-epochs 500 1000
"""

import argparse
import json
import itertools
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch

from generation.invbo_decoder.training import InvBOTrainer, TrainingConfig
from generation.invbo_decoder.inference import InvBOInference


@dataclass
class TuningResult:
    """Result of a single hyperparameter configuration."""

    config: Dict[str, Any]
    vae_cosine: float  # VAE reconstruction cosine similarity
    vae_epochs_trained: int  # Actual epochs trained (before early stop)
    gp_mae: float  # GP mean absolute error on training data
    gap_converged: bool  # Whether gap < threshold in inversion
    final_gap: float  # Final gap value
    final_ei: float  # Final expected improvement
    instruction: str  # Generated instruction text
    cosine_similarity: float  # Vec2Text cosine similarity


def run_single_config(
    config: TrainingConfig,
    max_inversion_iters: int = 5,
    gap_threshold: float = 0.1,
    verbose: bool = False,
) -> Optional[TuningResult]:
    """Run training and inference with a single config.

    Args:
        config: Training configuration
        max_inversion_iters: Maximum inversion iterations
        gap_threshold: Gap threshold for convergence
        verbose: Print progress

    Returns:
        TuningResult or None if training fails
    """
    try:
        # Train
        trainer = InvBOTrainer(config)
        gp, decoder = trainer.train(verbose=verbose)

        # Get VAE stats
        vae_cosine = 0.0
        vae_epochs = 0
        if trainer.vae is not None:
            trainer.vae.eval()
            with torch.no_grad():
                # Compute cosine on grid instructions
                cosines = []
                for inst_id in sorted(trainer.error_rates.keys()):
                    emb = trainer.instruction_embeddings[inst_id].unsqueeze(0)
                    recon, _, _, _ = trainer.vae(emb)
                    cos = torch.nn.functional.cosine_similarity(recon, emb).item()
                    cosines.append(cos)
                vae_cosine = sum(cosines) / len(cosines)

        # GP MAE
        gp_errors = []
        for inst_id in sorted(trainer.error_rates.keys()):
            emb = trainer.instruction_embeddings[inst_id]
            pred_mean, _ = gp.predict(emb)
            true_error = trainer.error_rates[inst_id]
            gp_errors.append(abs(pred_mean - true_error))
        gp_mae = sum(gp_errors) / len(gp_errors)

        # Inference
        inference = InvBOInference(
            gp=gp,
            decoder=decoder,
            gtr=trainer.gtr,
            vec2text_steps=50,
            vec2text_beam=4,
        )

        result = inference.optimize_with_inversion(
            method="trust_region",
            n_candidates=500,
            trust_radius=0.3,
            max_inversion_iters=max_inversion_iters,
            gap_threshold=gap_threshold,
            verbose=verbose,
        )

        # Check if gap converged
        # We need to track this during inversion - for now approximate
        gap_converged = result.ei_value > 0 or result.cosine_similarity > 0.9

        return TuningResult(
            config={
                "vae_beta": config.vae_beta,
                "vae_epochs": config.vae_epochs,
                "vae_annealing_epochs": config.vae_annealing_epochs,
                "vae_patience": config.vae_patience,
                "latent_dim": config.latent_dim,
                "gp_epochs": config.gp_epochs,
            },
            vae_cosine=vae_cosine,
            vae_epochs_trained=vae_epochs,
            gp_mae=gp_mae,
            gap_converged=gap_converged,
            final_gap=0.0,  # Would need to track during inversion
            final_ei=result.ei_value,
            instruction=result.instruction_text,
            cosine_similarity=result.cosine_similarity,
        )

    except Exception as e:
        if verbose:
            print(f"  Config failed: {e}")
        return None


def grid_search(
    vae_betas: List[float],
    vae_epochs: List[int],
    vae_annealing_epochs: List[int],
    latent_dims: List[int],
    gp_epochs: List[int],
    vae_patience: int = 100,
    max_inversion_iters: int = 5,
    gap_threshold: float = 0.1,
    results_dir: str = "generation/invbo_decoder/tune_results",
    verbose: bool = True,
) -> List[TuningResult]:
    """Run grid search over hyperparameters.

    Args:
        vae_betas: VAE KL regularization weights to try
        vae_epochs: VAE training epochs to try
        vae_annealing_epochs: KL annealing epochs to try
        latent_dims: Latent dimensions to try
        gp_epochs: GP training epochs to try
        vae_patience: VAE early stopping patience
        max_inversion_iters: Maximum inversion iterations
        gap_threshold: Gap threshold for convergence
        results_dir: Directory to save results
        verbose: Print progress

    Returns:
        List of TuningResults
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_path / f"tune_{timestamp}.jsonl"

    # Generate all combinations
    combinations = list(itertools.product(
        vae_betas, vae_epochs, vae_annealing_epochs, latent_dims, gp_epochs
    ))

    if verbose:
        print(f"=" * 60)
        print(f"Hyperparameter Tuning")
        print(f"=" * 60)
        print(f"Total configurations: {len(combinations)}")
        print(f"Results will be saved to: {log_path}")
        print()

    results = []

    for i, (beta, epochs, annealing, latent_dim, gp_ep) in enumerate(combinations):
        config = TrainingConfig(
            use_vae=True,
            vae_beta=beta,
            vae_epochs=epochs,
            vae_annealing_epochs=annealing,
            vae_patience=vae_patience,
            latent_dim=latent_dim,
            gp_epochs=gp_ep,
        )

        if verbose:
            print(f"[{i + 1}/{len(combinations)}] beta={beta}, epochs={epochs}, "
                  f"annealing={annealing}, latent_dim={latent_dim}, gp_epochs={gp_ep}")

        result = run_single_config(
            config,
            max_inversion_iters=max_inversion_iters,
            gap_threshold=gap_threshold,
            verbose=False,
        )

        if result is not None:
            results.append(result)

            # Log result
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "config": result.config,
                    "vae_cosine": result.vae_cosine,
                    "gp_mae": result.gp_mae,
                    "final_ei": result.final_ei,
                    "cosine_similarity": result.cosine_similarity,
                    "instruction": result.instruction,
                }) + "\n")

            if verbose:
                print(f"  VAE cosine: {result.vae_cosine:.4f}, "
                      f"GP MAE: {result.gp_mae:.4f}, "
                      f"EI: {result.final_ei:.6f}, "
                      f"Cosine: {result.cosine_similarity:.4f}")
        else:
            if verbose:
                print("  FAILED")

    # Save summary
    if results:
        summary_path = results_path / f"summary_{timestamp}.json"

        # Sort by VAE cosine + cosine similarity (composite score)
        results_sorted = sorted(
            results,
            key=lambda r: r.vae_cosine + r.cosine_similarity,
            reverse=True,
        )

        best = results_sorted[0]

        summary = {
            "timestamp": timestamp,
            "total_configs": len(combinations),
            "successful_configs": len(results),
            "best_config": best.config,
            "best_metrics": {
                "vae_cosine": best.vae_cosine,
                "gp_mae": best.gp_mae,
                "final_ei": best.final_ei,
                "cosine_similarity": best.cosine_similarity,
            },
            "best_instruction": best.instruction,
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if verbose:
            print()
            print("=" * 60)
            print("BEST CONFIGURATION")
            print("=" * 60)
            print(f"Config: {best.config}")
            print(f"VAE Cosine: {best.vae_cosine:.4f}")
            print(f"GP MAE: {best.gp_mae:.4f}")
            print(f"EI: {best.final_ei:.6f}")
            print(f"Cosine Similarity: {best.cosine_similarity:.4f}")
            print(f"Instruction: {best.instruction}")
            print(f"\nSummary saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for InvBO decoder"
    )

    # Preset modes
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick tuning with small grid (4 configs)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full grid search (many configs, slow)",
    )

    # Custom parameters
    parser.add_argument(
        "--vae-betas",
        type=float,
        nargs="+",
        default=[0.05, 0.1],
        help="VAE beta values to try",
    )
    parser.add_argument(
        "--vae-epochs",
        type=int,
        nargs="+",
        default=[500, 1000],
        help="VAE epochs to try",
    )
    parser.add_argument(
        "--vae-annealing",
        type=int,
        nargs="+",
        default=[300, 500],
        help="VAE annealing epochs to try",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=[10],
        help="Latent dimensions to try",
    )
    parser.add_argument(
        "--gp-epochs",
        type=int,
        nargs="+",
        default=[1000],
        help="GP epochs to try",
    )
    parser.add_argument(
        "--vae-patience",
        type=int,
        default=100,
        help="VAE early stopping patience",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="generation/invbo_decoder/tune_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Apply presets
    if args.quick:
        vae_betas = [0.05, 0.1]
        vae_epochs = [500]
        vae_annealing = [300]
        latent_dims = [10]
        gp_epochs = [500]
    elif args.full:
        vae_betas = [0.01, 0.05, 0.1, 0.2]
        vae_epochs = [500, 1000, 2000]
        vae_annealing = [200, 500, 800]
        latent_dims = [8, 10, 16]
        gp_epochs = [500, 1000, 2000]
    else:
        vae_betas = args.vae_betas
        vae_epochs = args.vae_epochs
        vae_annealing = args.vae_annealing
        latent_dims = args.latent_dims
        gp_epochs = args.gp_epochs

    grid_search(
        vae_betas=vae_betas,
        vae_epochs=vae_epochs,
        vae_annealing_epochs=vae_annealing,
        latent_dims=latent_dims,
        gp_epochs=gp_epochs,
        vae_patience=args.vae_patience,
        results_dir=args.results_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
