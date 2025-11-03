"""Hyperparameter tuning for Claudette binary classifier using Optuna."""

import argparse
import optuna
import torch
import torch.distributed as dist
from pathlib import Path
import json

from .config import Config
from .data_loader import (
    load_dataset, create_splits, create_dataloaders, get_class_weights
)
from .encoder import LegalBERTEncoder
from .model import DeepResidualMLP
from .train import train
from .evaluate import evaluate


def objective(trial: optuna.Trial, config: Config, device: torch.device) -> float:
    """Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        config: Base configuration
        device: PyTorch device

    Returns:
        Validation F1 score (metric to maximize)
    """
    # Hyperparameters to tune
    config.learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    config.encoder_lr = trial.suggest_float('encoder_lr', 5e-6, 5e-5, log=True)
    config.dropout = trial.suggest_float('dropout', 0.2, 0.6)
    config.batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    # Hidden dimensions
    hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [512, 768, 1024])
    hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [256, 384, 512])
    hidden_dim_3 = trial.suggest_categorical('hidden_dim_3', [128, 192, 256])
    config.hidden_dims = [hidden_dim_1, hidden_dim_2, hidden_dim_3]

    config.num_residual_blocks = trial.suggest_int('num_residual_blocks', 2, 5)

    # Focal loss parameters
    config.focal_alpha = trial.suggest_float('focal_alpha', 0.2, 0.4)
    config.focal_gamma = trial.suggest_float('focal_gamma', 1.5, 2.5)

    # Load dataset
    texts, labels = load_dataset(config.dataset_path)

    # Create splits
    train_dataset, val_dataset, _ = create_splits(
        texts, labels,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed
    )

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        train_dataset, val_dataset, val_dataset,  # Use val for test (not needed)
        batch_size=config.batch_size,
        use_oversampling=config.use_oversampling,
        use_distributed=False
    )

    # Get class weights
    class_weights = get_class_weights(train_dataset.labels)

    # Initialize models
    encoder = LegalBERTEncoder(
        model_name=config.encoder_name,
        freeze_encoder=False  # Train encoder during tuning
    ).to(device)

    embedding_dim = encoder.get_embedding_dim()
    classifier = DeepResidualMLP(
        input_dim=embedding_dim,
        hidden_dims=config.hidden_dims,
        num_residual_blocks=config.num_residual_blocks,
        dropout=config.dropout
    ).to(device)

    # Train (reduced epochs for faster tuning)
    config.num_epochs = 20  # Fewer epochs during tuning
    config.save_best_model = False  # Don't save during tuning

    train_losses, val_metrics = train(
        encoder, classifier,
        train_loader, val_loader,
        config, device,
        class_weights=class_weights,
        rank=0,
        is_distributed=False
    )

    # Return best validation F1
    best_f1 = max([m['f1'] for m in val_metrics])

    return best_f1


def main():
    """Main hyperparameter tuning function."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Claudette classifier")
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--study-name', type=str, default='claudette_tuning',
                       help='Name for Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for distributed tuning (optional)')

    args = parser.parse_args()

    # Get device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("Claudette Binary Classifier - Hyperparameter Tuning")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Study name: {args.study_name}")
    print()

    # Create base config
    config = Config()
    config.use_focal_loss = True
    config.use_oversampling = True
    config.use_class_weights = True

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # Maximize F1 score
        storage=args.storage,
        load_if_exists=True
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, device),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    # Print results
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Complete!")
    print("=" * 60)
    print(f"\nBest trial:")
    print(f"  Value (F1): {study.best_trial.value:.4f}")
    print(f"  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save results
    output_dir = Path("results/claudette_classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'best_params': study.best_trial.params,
        'best_value': study.best_trial.value,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params
            }
            for trial in study.trials
        ]
    }

    output_path = output_dir / 'hyperparameter_tuning_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print importance
    print("\n" + "=" * 60)
    print("Hyperparameter Importance:")
    print("=" * 60)
    importance = optuna.importance.get_param_importances(study)
    for param, imp in importance.items():
        print(f"  {param}: {imp:.4f}")


if __name__ == '__main__':
    main()
