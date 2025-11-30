"""
Evaluate HbBoPs GP predictions against Full Grid ground truth

This script:
1. Runs HbBoPs Hyperband (trains GP on sampled data)
2. Queries GP for all 625 instruction×exemplar pairs
3. Compares predictions with ground truth from full_grid_combined.jsonl
4. Outputs CSV, metrics, and scatter plot visualization

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python hbbops/evaluate_predictions.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --backend vllm \
        --gpu-memory-utilization 0.95
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys
import torch
import gpytorch
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hbbops.hbbops import HbBoPs, Prompt
from hbbops.run_hbbops import (
    GSM8KEvaluator, load_instructions, load_exemplars
)
from src.llm_client import create_llm_client


def get_gp_predictions(hbbops: HbBoPs) -> pd.DataFrame:
    """
    Query trained GP for all 625 instruction×exemplar pairs.

    Returns DataFrame with columns:
        instruction_id, exemplar_id, predicted_error, predicted_accuracy, predicted_std
    """
    if hbbops.gp_model is None:
        raise ValueError("GP model not trained. Run hyperband first.")

    predictions = []

    # Set models to eval mode
    hbbops.gp_model.eval()
    hbbops.likelihood.eval()

    for prompt in hbbops.prompts:
        # Get embeddings
        inst_emb, ex_emb = hbbops.embed_prompt(prompt)

        # Create input tensor
        X_inst = torch.tensor(inst_emb, dtype=torch.float32, device=hbbops.device).unsqueeze(0)
        X_ex = torch.tensor(ex_emb, dtype=torch.float32, device=hbbops.device).unsqueeze(0)
        X = torch.cat([X_inst, X_ex], dim=1)

        # Normalize using training statistics
        X_norm = (X - hbbops.X_mean) / hbbops.X_std

        # Predict
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = hbbops.likelihood(hbbops.gp_model(X_norm))
            mean_norm = pred_dist.mean.item()
            std_norm = pred_dist.stddev.item()

            # Denormalize to original error scale
            mean = mean_norm * hbbops.y_std.item() + hbbops.y_mean.item()
            std = std_norm * hbbops.y_std.item()

        # Clamp to valid range [0, 1]
        mean = max(0.0, min(1.0, mean))

        predictions.append({
            'instruction_id': prompt.instruction_id,
            'exemplar_id': prompt.exemplar_id,
            'predicted_error': mean,
            'predicted_accuracy': 1.0 - mean,
            'predicted_std': std
        })

    return pd.DataFrame(predictions)


def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    """Load ground truth from full_grid_combined.jsonl"""
    records = []
    with open(gt_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    df['gt_accuracy'] = 1.0 - df['error_rate']
    df = df.rename(columns={'error_rate': 'gt_error'})
    return df[['instruction_id', 'exemplar_id', 'gt_error', 'gt_accuracy']]


def compute_metrics(results: pd.DataFrame) -> dict:
    """Compute ranking and prediction metrics"""
    # Rankings
    results['gt_rank'] = results['gt_accuracy'].rank(ascending=False)
    results['pred_rank'] = results['predicted_accuracy'].rank(ascending=False)
    results['rank_diff'] = abs(results['gt_rank'] - results['pred_rank'])

    # Correlation metrics
    kendall_tau, kendall_p = kendalltau(results['gt_accuracy'], results['predicted_accuracy'])
    spearman_r, spearman_p = spearmanr(results['gt_accuracy'], results['predicted_accuracy'])

    # MAE
    mae = abs(results['gt_accuracy'] - results['predicted_accuracy']).mean()

    # Top-k overlap
    top_10_pct = int(len(results) * 0.1)  # 63 for 625
    top_20_pct = int(len(results) * 0.2)  # 125 for 625

    gt_top_10 = set(results.nlargest(top_10_pct, 'gt_accuracy').index)
    pred_top_10 = set(results.nlargest(top_10_pct, 'predicted_accuracy').index)
    top_10_overlap = len(gt_top_10 & pred_top_10) / top_10_pct

    gt_top_20 = set(results.nlargest(top_20_pct, 'gt_accuracy').index)
    pred_top_20 = set(results.nlargest(top_20_pct, 'predicted_accuracy').index)
    top_20_overlap = len(gt_top_20 & pred_top_20) / top_20_pct

    # Best prompt agreement
    gt_best_idx = results['gt_accuracy'].idxmax()
    pred_best_idx = results['predicted_accuracy'].idxmax()
    gt_best = results.loc[gt_best_idx, ['instruction_id', 'exemplar_id']].to_dict()
    pred_best = results.loc[pred_best_idx, ['instruction_id', 'exemplar_id']].to_dict()
    best_match = gt_best == pred_best

    # Rank of GT best in predictions
    gt_best_pred_rank = int(results.loc[gt_best_idx, 'pred_rank'])

    return {
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mae': mae,
        'top_10_overlap': top_10_overlap,
        'top_20_overlap': top_20_overlap,
        'best_match': best_match,
        'gt_best': gt_best,
        'pred_best': pred_best,
        'gt_best_pred_rank': gt_best_pred_rank,
        'mean_rank_diff': results['rank_diff'].mean(),
        'median_rank_diff': results['rank_diff'].median()
    }


def create_scatter_plot(results: pd.DataFrame, metrics: dict, output_path: Path):
    """Create scatter plot of predicted vs GT accuracy"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Mark top 10%
    top_10_pct = int(len(results) * 0.1)
    gt_top_10 = set(results.nlargest(top_10_pct, 'gt_accuracy').index)

    # Colors: blue for regular, red for top-10% GT
    colors = ['red' if i in gt_top_10 else 'blue' for i in results.index]
    alphas = [0.8 if i in gt_top_10 else 0.3 for i in results.index]

    # Plot points
    for i, (_, row) in enumerate(results.iterrows()):
        ax.scatter(row['gt_accuracy'], row['predicted_accuracy'],
                   c=colors[i], alpha=alphas[i], s=30)

    # Diagonal line (perfect prediction)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect prediction')

    # Labels and title
    ax.set_xlabel('Ground Truth Accuracy', fontsize=12)
    ax.set_ylabel('Predicted Accuracy (GP)', fontsize=12)
    ax.set_title('HbBoPs GP Predictions vs Full Grid Ground Truth', fontsize=14)

    # Annotation box with metrics
    textstr = '\n'.join([
        f"Kendall τ: {metrics['kendall_tau']:.3f}",
        f"Spearman r: {metrics['spearman_r']:.3f}",
        f"MAE: {metrics['mae']:.3f}",
        f"Top-10% overlap: {metrics['top_10_overlap']:.1%}",
        f"Top-20% overlap: {metrics['top_20_overlap']:.1%}",
        f"Best match: {'Yes' if metrics['best_match'] else 'No'}",
        f"GT best → pred rank: {metrics['gt_best_pred_rank']}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Legend
    ax.scatter([], [], c='red', alpha=0.8, s=30, label='Top 10% (GT)')
    ax.scatter([], [], c='blue', alpha=0.3, s=30, label='Other')
    ax.legend(loc='lower right')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate HbBoPs GP predictions vs Full Grid')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--backend', type=str, default='vllm',
                        choices=['vllm', 'transformers', 'claude', 'deepinfra', 'auto'],
                        help='Backend for LLM')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.95,
                        help='GPU memory utilization for vLLM')
    parser.add_argument('--bmin', type=int, default=10,
                        help='Minimum validation instances for Hyperband')
    parser.add_argument('--eta', type=float, default=2.0,
                        help='Halving parameter for Hyperband')
    parser.add_argument('--encoder', type=str, default='bert-base-uncased',
                        help='Encoder model for embeddings')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device for computation')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--full-initial-bracket', action='store_true',
                        help='Evaluate ALL prompts in first Hyperband bracket')
    parser.add_argument('--num-instructions', type=int, default=None,
                        help='Number of instructions to sample (default: use all)')
    parser.add_argument('--num-exemplars', type=int, default=None,
                        help='Number of exemplars to sample from train set (default: use all from examples.txt)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')

    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    gt_path = script_dir / "results" / "full_grid_combined.jsonl"

    # Set random seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("Loading data...")

    with open(data_dir / "validation.json", 'r') as f:
        validation_data = json.load(f)
    print(f"  Validation examples: {len(validation_data)}")

    # Load training data for exemplar generation
    with open(data_dir / "train.json", 'r') as f:
        train_data = json.load(f)
    print(f"  Training examples: {len(train_data)}")

    instructions = load_instructions(str(script_dir / "instructions.txt"))

    # Sample instructions if requested
    if args.num_instructions and args.num_instructions < len(instructions):
        instructions = random.sample(instructions, args.num_instructions)
        print(f"  Sampled {len(instructions)} instructions")

    # Generate exemplars from train set if num_exemplars specified
    if args.num_exemplars:
        # Sample random examples from training set
        sampled_train = random.sample(train_data, min(args.num_exemplars, len(train_data)))
        exemplars = []
        for ex in sampled_train:
            # Format as Q&A exemplar
            exemplar = f"Q: {ex['question']}\nA: {ex['answer']}"
            exemplars.append(exemplar)
        print(f"  Generated {len(exemplars)} exemplars from train set")
    else:
        exemplars = load_exemplars(str(script_dir / "examples.txt"))

    print(f"  Instructions: {len(instructions)}")
    print(f"  Exemplars: {len(exemplars)}")
    print(f"  Total prompts: {len(instructions) * len(exemplars)}")

    # Load ground truth (only if not sampling - GT is for original 25x25 grid)
    use_sampling = args.num_instructions or args.num_exemplars
    gt_df = None
    if not use_sampling and gt_path.exists():
        print(f"\nLoading ground truth from {gt_path}...")
        gt_df = load_ground_truth(gt_path)
        print(f"  GT records: {len(gt_df)}")
        print(f"  GT accuracy range: [{gt_df['gt_accuracy'].min():.3f}, {gt_df['gt_accuracy'].max():.3f}]")
    elif use_sampling:
        print("\nSkipping ground truth (sampling mode - GT not available for sampled prompts)")

    # Initialize LLM client
    print(f"\nInitializing LLM client ({args.backend})...")
    llm_client = create_llm_client(
        model_name=args.model,
        backend=args.backend,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # Warmup vLLM
    if hasattr(llm_client, 'warmup'):
        print("Warming up vLLM...")
        llm_client.warmup()

    # Create evaluator
    evaluator = GSM8KEvaluator(llm_client, debug=args.debug)

    # Initialize HbBoPs
    print("\nInitializing HbBoPs...")
    hbbops = HbBoPs(
        instructions=instructions,
        exemplars=exemplars,
        validation_data=validation_data,
        llm_evaluator=evaluator,
        encoder_name=args.encoder,
        bmin=args.bmin,
        eta=args.eta,
        device=args.device,
        full_initial_bracket=args.full_initial_bracket
    )

    # Run Hyperband
    print("\n" + "="*60)
    print("Running Hyperband (this trains the GP model)...")
    print("="*60)
    best_prompt, best_val_error = hbbops.run_hyperband(verbose=True)

    print(f"\nHyperband complete!")
    print(f"  Best prompt: Instruction {best_prompt.instruction_id}, Exemplar {best_prompt.exemplar_id}")
    print(f"  Best validation error: {best_val_error:.4f}")
    print(f"  Design data size: {len(hbbops.design_data)}")

    # Get GP predictions for all pairs
    print("\n" + "="*60)
    print(f"Querying GP for all {len(hbbops.prompts)} pairs...")
    print("="*60)
    pred_df = get_gp_predictions(hbbops)
    print(f"  Predictions obtained: {len(pred_df)}")
    print(f"  Predicted accuracy range: [{pred_df['predicted_accuracy'].min():.3f}, {pred_df['predicted_accuracy'].max():.3f}]")

    # Compare with ground truth (if available)
    if gt_df is not None:
        # Merge with ground truth
        print("\nMerging predictions with ground truth...")
        results = pd.merge(pred_df, gt_df, on=['instruction_id', 'exemplar_id'])
        print(f"  Merged records: {len(results)}")

        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_metrics(results)

        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nCorrelation Metrics:")
        print(f"  Kendall τ:  {metrics['kendall_tau']:.4f} (p={metrics['kendall_p']:.2e})")
        print(f"  Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.2e})")
        print(f"\nPrediction Accuracy:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"\nRanking Agreement:")
        print(f"  Top-10% overlap: {metrics['top_10_overlap']:.1%}")
        print(f"  Top-20% overlap: {metrics['top_20_overlap']:.1%}")
        print(f"  Mean rank difference: {metrics['mean_rank_diff']:.1f}")
        print(f"  Median rank difference: {metrics['median_rank_diff']:.1f}")
        print(f"\nBest Prompt:")
        print(f"  GT best: Instruction {metrics['gt_best']['instruction_id']}, Exemplar {metrics['gt_best']['exemplar_id']}")
        print(f"  Pred best: Instruction {metrics['pred_best']['instruction_id']}, Exemplar {metrics['pred_best']['exemplar_id']}")
        print(f"  Match: {'Yes' if metrics['best_match'] else 'No'}")
        print(f"  GT best → predicted rank: {metrics['gt_best_pred_rank']}")

        # Save outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_path = output_dir / f"prediction_comparison_{timestamp}.csv"
        results.to_csv(csv_path, index=False)
        print(f"\nCSV saved: {csv_path}")

        # Metrics JSON
        metrics_path = output_dir / f"prediction_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                **metrics,
                'model': args.model,
                'bmin': args.bmin,
                'eta': args.eta,
                'design_data_size': len(hbbops.design_data),
                'timestamp': timestamp
            }, f, indent=2, default=str)
        print(f"Metrics saved: {metrics_path}")

        # Scatter plot
        plot_path = output_dir / f"prediction_scatter_{timestamp}.png"
        create_scatter_plot(results, metrics, plot_path)

        # Print top-10 comparison
        print("\n" + "="*60)
        print("TOP 10 COMPARISON")
        print("="*60)
        print("\nGround Truth Top 10:")
        gt_top10 = results.nlargest(10, 'gt_accuracy')[['instruction_id', 'exemplar_id', 'gt_accuracy', 'predicted_accuracy', 'pred_rank']]
        print(gt_top10.to_string(index=False))

        print("\nPredicted Top 10:")
        pred_top10 = results.nlargest(10, 'predicted_accuracy')[['instruction_id', 'exemplar_id', 'gt_accuracy', 'predicted_accuracy', 'gt_rank']]
        pred_top10 = pred_top10.rename(columns={'gt_rank': 'gt_rank'})
        print(pred_top10.to_string(index=False))

    else:
        # Sampling mode - just show predictions and best prompt
        print("\n" + "="*60)
        print("RESULTS (Sampling Mode)")
        print("="*60)

        # Sort by predicted accuracy
        pred_df_sorted = pred_df.sort_values('predicted_accuracy', ascending=False)
        print(f"\nTop 5 predicted prompts:")
        for idx, row in pred_df_sorted.head(5).iterrows():
            print(f"  Inst {row['instruction_id']:2d}, Ex {row['exemplar_id']:2d}: "
                  f"acc={row['predicted_accuracy']:.3f} ± {row['predicted_std']:.3f}")

        print(f"\nBest prompt from Hyperband:")
        print(f"  Instruction {best_prompt.instruction_id}:")
        print(f"    {hbbops.instructions[best_prompt.instruction_id][:100]}...")
        print(f"  Exemplar {best_prompt.exemplar_id}:")
        print(f"    {hbbops.exemplars[best_prompt.exemplar_id][:100]}...")
        print(f"  Validation error: {best_val_error:.4f}")
        print(f"  Accuracy: {1 - best_val_error:.2%}")

        # Save predictions CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"predictions_sampled_{timestamp}.csv"
        pred_df.to_csv(csv_path, index=False)
        print(f"\nPredictions saved: {csv_path}")


if __name__ == "__main__":
    main()
