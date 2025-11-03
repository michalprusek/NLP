"""Evaluation metrics for binary classification."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .encoder import LegalBERTEncoder
from .model import DeepResidualMLP


def evaluate(
    encoder: LegalBERTEncoder,
    classifier: DeepResidualMLP,
    data_loader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate model on a dataset.

    Args:
        encoder: Legal-BERT encoder
        classifier: MLP classifier
        data_loader: Data loader
        device: Device to evaluate on

    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1, auroc)
    """
    encoder.eval()
    classifier.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for texts, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            labels = labels.to(device)

            # Forward pass
            embeddings = encoder(texts, device)
            logits = classifier(embeddings)

            # Get predictions and probabilities
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1': f1_score(all_labels, all_predictions, zero_division=0),
        'auroc': roc_auc_score(all_labels, all_probabilities)
    }

    return metrics


def evaluate_detailed(
    encoder: LegalBERTEncoder,
    classifier: DeepResidualMLP,
    data_loader: DataLoader,
    device: torch.device,
    split_name: str = "Test"
) -> dict:
    """Evaluate model with detailed metrics and confusion matrix.

    Args:
        encoder: Legal-BERT encoder
        classifier: MLP classifier
        data_loader: Data loader
        device: Device to evaluate on
        split_name: Name of the split being evaluated

    Returns:
        Dictionary of metrics with additional details
    """
    encoder.eval()
    classifier.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for texts, labels in tqdm(data_loader, desc=f"Evaluating {split_name}", leave=False):
            labels = labels.to(device)

            # Forward pass
            embeddings = encoder(texts, device)
            logits = classifier(embeddings)

            # Get predictions and probabilities
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auroc = roc_auc_score(all_labels, all_probabilities)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    # Print results
    print(f"\n{'=' * 60}")
    print(f"{split_name} Set Evaluation")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUROC:     {auroc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Fair  Unfair")
    print(f"Actual Fair    {tn:4d}  {fp:4d}")
    print(f"       Unfair  {fn:4d}  {tp:4d}")
    print(f"\nClassification Report:")
    print(classification_report(
        all_labels, all_predictions,
        target_names=['Fair', 'Unfair'],
        zero_division=0
    ))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'num_samples': len(all_labels),
        'num_positive': int(all_labels.sum()),
        'num_negative': int(len(all_labels) - all_labels.sum())
    }

    return metrics
