"""Main entry point for training Claudette binary classifier."""

import argparse
import json
import torch
from pathlib import Path

from .config import Config
from .data_loader import (
    load_dataset, create_splits, create_dataloaders, get_class_weights
)
from .encoder import LegalBERTEncoder
from .model import DeepResidualMLP
from .train import train
from .evaluate import evaluate_detailed


def get_device(device_str: str = "auto") -> torch.device:
    """Get PyTorch device.

    Args:
        device_str: Device string (auto, cuda, mps, cpu)

    Returns:
        PyTorch device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    return device


def save_results(train_losses: list, val_metrics: list, test_metrics: dict, config: Config):
    """Save training results to JSON.

    Args:
        train_losses: Training losses per epoch
        val_metrics: Validation metrics per epoch
        test_metrics: Final test metrics
        config: Training configuration
    """
    results = {
        'config': {
            'encoder_name': config.encoder_name,
            'hidden_dims': config.hidden_dims,
            'num_residual_blocks': config.num_residual_blocks,
            'dropout': config.dropout,
            'learning_rate': config.learning_rate,
            'encoder_lr': config.encoder_lr,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'use_class_weights': config.use_class_weights,
            'use_focal_loss': config.use_focal_loss,
            'use_oversampling': config.use_oversampling,
            'focal_alpha': config.focal_alpha,
            'focal_gamma': config.focal_gamma,
        },
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }

    output_path = config.output_dir / 'training_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Claudette binary classifier")
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for classifier')
    parser.add_argument('--encoder-lr', type=float, default=1e-5, help='Learning rate for encoder')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--no-focal-loss', action='store_true', help='Disable focal loss')
    parser.add_argument('--no-oversampling', action='store_true', help='Disable oversampling')
    parser.add_argument('--no-class-weights', action='store_true', help='Disable class weights')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden dimensions for MLP')
    parser.add_argument('--num-residual-blocks', type=int, default=3,
                       help='Number of residual blocks per layer')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--train-encoder', action='store_true',
                       help='Train encoder weights (by default encoder is frozen)')

    args = parser.parse_args()

    # Create config
    config = Config()
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.encoder_lr = args.encoder_lr
    config.num_epochs = args.epochs
    config.device = args.device
    config.use_focal_loss = not args.no_focal_loss
    config.use_oversampling = not args.no_oversampling
    config.use_class_weights = not args.no_class_weights
    config.hidden_dims = args.hidden_dims
    config.num_residual_blocks = args.num_residual_blocks
    config.dropout = args.dropout

    print("=" * 60)
    print("Claudette Binary Classifier Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Encoder: {config.encoder_name}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Residual blocks: {config.num_residual_blocks}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate (classifier): {config.learning_rate}")
    print(f"  Learning rate (encoder): {config.encoder_lr}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Focal loss: {config.use_focal_loss}")
    print(f"  Oversampling: {config.use_oversampling}")
    print(f"  Class weights: {config.use_class_weights}")
    print(f"  Train encoder: {args.train_encoder}")
    print(f"  Freeze encoder: {not args.train_encoder}")
    print()

    # Load dataset
    print("Loading dataset...")
    texts, labels = load_dataset(config.dataset_path)

    # Create splits
    train_dataset, val_dataset, test_dataset = create_splits(
        texts, labels,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.batch_size,
        use_oversampling=config.use_oversampling
    )

    # Get class weights
    class_weights = get_class_weights(train_dataset.labels)

    # Get device
    device = get_device(config.device)
    print(f"\nUsing device: {device}")

    # Initialize models
    print(f"\nInitializing models...")
    encoder = LegalBERTEncoder(
        model_name=config.encoder_name,
        freeze_encoder=not args.train_encoder  # By default, encoder is frozen
    ).to(device)

    embedding_dim = encoder.get_embedding_dim()
    classifier = DeepResidualMLP(
        input_dim=embedding_dim,
        hidden_dims=config.hidden_dims,
        num_residual_blocks=config.num_residual_blocks,
        dropout=config.dropout
    ).to(device)

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Total trainable parameters: {encoder_params + classifier_params:,}")

    # Train
    train_losses, val_metrics = train(
        encoder, classifier,
        train_loader, val_loader,
        config, device,
        class_weights=class_weights
    )

    # Load best model for final evaluation
    if config.save_best_model:
        print("\nLoading best model for final evaluation...")
        checkpoint = torch.load(config.output_dir / 'best_model.pt')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")

    # Final evaluation on test set
    test_metrics = evaluate_detailed(
        encoder, classifier, test_loader, device, split_name="Test"
    )

    # Save results
    save_results(train_losses, val_metrics, test_metrics, config)

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
