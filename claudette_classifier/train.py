"""Training loop with validation and early stopping."""

import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .encoder import LegalBERTEncoder
from .model import DeepResidualMLP
from .loss import get_loss_function
from .evaluate import evaluate


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric: float) -> bool:
        """Check if training should stop.

        Args:
            val_metric: Validation metric to monitor (higher is better)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0

        return self.early_stop


def train_epoch(
    encoder: LegalBERTEncoder,
    classifier: DeepResidualMLP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0
) -> float:
    """Train for one epoch.

    Args:
        encoder: Legal-BERT encoder
        classifier: MLP classifier
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        gradient_clip: Gradient clipping value

    Returns:
        Average training loss
    """
    encoder.train()
    classifier.train()

    total_loss = 0.0
    num_batches = 0

    for texts, labels in tqdm(train_loader, desc="Training", leave=False):
        labels = labels.to(device)

        # Forward pass
        embeddings = encoder(texts, device)
        logits = classifier(embeddings)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (only on trainable parameters)
        trainable_params = [p for p in encoder.parameters() if p.requires_grad]
        trainable_params.extend(classifier.parameters())
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train(
    encoder: LegalBERTEncoder,
    classifier: DeepResidualMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    rank: int = 0,
    is_distributed: bool = False
) -> Tuple[list[float], list[dict]]:
    """Train the model with validation and early stopping.

    Args:
        encoder: Legal-BERT encoder (possibly wrapped in DDP)
        classifier: MLP classifier (possibly wrapped in DDP)
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        class_weights: Class weights for loss function
        rank: Process rank for distributed training
        is_distributed: Whether using distributed training

    Returns:
        Tuple of (train_losses, val_metrics_history)
    """
    # Setup loss function
    criterion = get_loss_function(
        use_focal_loss=config.use_focal_loss,
        use_class_weights=config.use_class_weights,
        class_weights=class_weights,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma
    )

    # Unwrap DDP if needed
    encoder_model = encoder.module if hasattr(encoder, 'module') else encoder
    classifier_model = classifier.module if hasattr(classifier, 'module') else classifier

    # Setup optimizer with different learning rates for encoder and classifier
    # Only include parameters that require gradients
    optimizer_params = []

    # Add encoder parameters only if they require gradients (not frozen)
    encoder_trainable_params = [p for p in encoder_model.parameters() if p.requires_grad]
    if encoder_trainable_params:
        optimizer_params.append({
            'params': encoder_trainable_params,
            'lr': config.encoder_lr
        })

    # Classifier is always trainable
    optimizer_params.append({
        'params': classifier_model.parameters(),
        'lr': config.learning_rate
    })

    optimizer = torch.optim.AdamW(optimizer_params)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    # Training history
    train_losses = []
    val_metrics_history = []
    best_val_f1 = 0.0

    # Only print from main process
    if rank == 0:
        print(f"\nStarting training for {config.num_epochs} epochs...")
        print(f"Device: {device}")
        print(f"Encoder LR: {config.encoder_lr}, Classifier LR: {config.learning_rate}")

    for epoch in range(config.num_epochs):
        start_time = time.time()

        # Set epoch for DistributedSampler to ensure proper shuffling
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Train one epoch
        train_loss = train_epoch(
            encoder, classifier, train_loader, optimizer,
            criterion, device, config.gradient_clip
        )
        train_losses.append(train_loss)

        # Validate
        val_metrics = evaluate(encoder, classifier, val_loader, device)
        val_metrics_history.append(val_metrics)

        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])

        epoch_time = time.time() - start_time

        # Print progress (only from main process)
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{config.num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val - Acc: {val_metrics['accuracy']:.4f}, "
                  f"P: {val_metrics['precision']:.4f}, "
                  f"R: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"AUROC: {val_metrics['auroc']:.4f}")

        # Save best model (only from main process)
        if config.save_best_model and val_metrics['f1'] > best_val_f1 and rank == 0:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder_model.state_dict(),
                'classifier_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, config.output_dir / 'best_model.pt')
            print(f"Saved best model (F1: {best_val_f1:.4f})")

        # Early stopping check
        if early_stopping(val_metrics['f1']):
            if rank == 0:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    return train_losses, val_metrics_history
