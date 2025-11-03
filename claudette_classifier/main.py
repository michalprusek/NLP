"""Main entry point for training Claudette binary classifier."""

import argparse
import json
import torch
import torch.distributed as dist
from pathlib import Path
import os

from .config import Config
from .data_loader import (
    load_dataset, create_splits, create_dataloaders, get_class_weights
)
from .encoder import LegalBERTEncoder
from .model import DeepResidualMLP
from .train import train
from .evaluate import evaluate_detailed


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process."""
    return rank == 0


def get_device(device_str: str = "auto", local_rank: int = 0) -> torch.device:
    """Get PyTorch device.

    Args:
        device_str: Device string (auto, cuda, mps, cpu)
        local_rank: Local rank for distributed training

    Returns:
        PyTorch device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif device_str == "cuda":
        device = torch.device(f"cuda:{local_rank}")
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
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()

    parser = argparse.ArgumentParser(description="Train Claudette binary classifier")
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
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

    # Only print from main process
    if is_main_process(rank):
        print("=" * 60)
        print("Claudette Binary Classifier Training")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Encoder: {config.encoder_name}")
        print(f"  Hidden dims: {config.hidden_dims}")
        print(f"  Residual blocks: {config.num_residual_blocks}")
        print(f"  Dropout: {config.dropout}")
        print(f"  Batch size per GPU: {config.batch_size}")
        if is_distributed:
            print(f"  Total batch size: {config.batch_size * world_size}")
            print(f"  World size: {world_size}")
        print(f"  Learning rate (classifier): {config.learning_rate}")
        print(f"  Learning rate (encoder): {config.encoder_lr}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Focal loss: {config.use_focal_loss}")
        print(f"  Oversampling: {config.use_oversampling}")
        print(f"  Class weights: {config.use_class_weights}")
        print(f"  Train encoder: {args.train_encoder}")
        print(f"  Freeze encoder: {not args.train_encoder}")
        print()

    # Load dataset (only print from main process)
    if is_main_process(rank):
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

    # Create dataloaders (with distributed sampler if needed)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.batch_size,
        use_oversampling=config.use_oversampling and not is_distributed,  # Disable oversampling with DDP
        use_distributed=is_distributed,
        world_size=world_size,
        rank=rank
    )

    # Get class weights
    class_weights = get_class_weights(train_dataset.labels)

    # Get device
    device = get_device(config.device, local_rank)
    if is_main_process(rank):
        print(f"\nUsing device: {device}")

    # Initialize models
    if is_main_process(rank):
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

    # Wrap models in DDP if distributed
    if is_distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=not args.train_encoder  # Only if encoder is frozen
        )
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank
        )

    # Count parameters (only print from main process)
    if is_main_process(rank):
        # Unwrap DDP if needed to count parameters
        encoder_model = encoder.module if is_distributed else encoder
        classifier_model = classifier.module if is_distributed else classifier

        encoder_params = sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in classifier_model.parameters())
        print(f"Encoder parameters: {encoder_params:,}")
        print(f"Classifier parameters: {classifier_params:,}")
        print(f"Total trainable parameters: {encoder_params + classifier_params:,}")

    # Train
    train_losses, val_metrics = train(
        encoder, classifier,
        train_loader, val_loader,
        config, device,
        class_weights=class_weights,
        rank=rank,
        is_distributed=is_distributed
    )

    # Load best model for final evaluation (only on main process)
    if config.save_best_model and is_main_process(rank):
        print("\nLoading best model for final evaluation...")
        checkpoint = torch.load(config.output_dir / 'best_model.pt')

        # Unwrap DDP if needed before loading state dict
        encoder_model = encoder.module if is_distributed else encoder
        classifier_model = classifier.module if is_distributed else classifier

        encoder_model.load_state_dict(checkpoint['encoder_state_dict'])
        classifier_model.load_state_dict(checkpoint['classifier_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")

    # Synchronize before final evaluation
    if is_distributed:
        dist.barrier()

    # Final evaluation on test set (only on main process)
    if is_main_process(rank):
        test_metrics = evaluate_detailed(
            encoder, classifier, test_loader, device, split_name="Test"
        )

        # Save results
        save_results(train_losses, val_metrics, test_metrics, config)

        print("\nTraining completed!")

    # Cleanup distributed training
    if is_distributed:
        cleanup_distributed()


if __name__ == '__main__':
    main()
