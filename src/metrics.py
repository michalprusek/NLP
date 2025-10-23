"""
Multi-label classification metrics for Claudette ToS evaluation.

This module provides comprehensive metrics computation including:
- Micro/Macro/Weighted F1 scores
- Per-class Precision/Recall/F1
- Confusion Matrix
- Hamming Loss
- Class distribution statistics

Designed to be compatible with the Claudette evaluator and follow
scikit-learn conventions where applicable.
"""
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np
from collections import defaultdict


# Label mapping for Claudette dataset
CLAUDETTE_LABELS = {
    0: "Limitation of liability",
    1: "Unilateral termination",
    2: "Unilateral change",
    3: "Arbitration",
    4: "Content removal",
    5: "Choice of law",
    6: "Other",
    7: "Contract by using",
    8: "Jurisdiction",
}


def compute_multilabel_metrics(
    y_true: List[Set[int]],
    y_pred: List[Set[int]],
    num_classes: int = 9,
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive multi-label classification metrics.

    Args:
        y_true: List of ground truth label sets
        y_pred: List of predicted label sets
        num_classes: Number of possible classes (default: 9 for Claudette)
        label_names: Optional mapping from label index to name

    Returns:
        Dictionary containing:
        - micro_f1: Micro-averaged F1 score
        - macro_f1: Macro-averaged F1 score
        - weighted_f1: Weighted F1 score (by support)
        - micro_precision: Micro-averaged precision
        - micro_recall: Micro-averaged recall
        - macro_precision: Macro-averaged precision
        - macro_recall: Macro-averaged recall
        - hamming_loss: Hamming loss (fraction of wrong labels)
        - subset_accuracy: Exact match accuracy
        - per_class: Dict with per-class metrics
        - confusion_matrix: Confusion matrix (num_classes x num_classes)
        - support: Number of true instances per class
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

    if len(y_true) == 0:
        return _empty_metrics(num_classes, label_names)

    label_names = label_names or CLAUDETTE_LABELS

    # Convert sets to binary indicator format for efficient computation
    y_true_binary = _sets_to_binary(y_true, num_classes)
    y_pred_binary = _sets_to_binary(y_pred, num_classes)

    # Compute per-sample metrics (for hamming loss, subset accuracy)
    subset_matches = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    subset_accuracy = subset_matches / len(y_true)

    # Hamming loss: average fraction of wrong labels per sample
    hamming_loss = _compute_hamming_loss(y_true_binary, y_pred_binary)

    # Compute micro-averaged metrics (aggregate all TP, FP, FN across classes)
    micro_metrics = _compute_micro_metrics(y_true_binary, y_pred_binary)

    # Compute per-class metrics
    per_class_metrics = _compute_per_class_metrics(
        y_true_binary, y_pred_binary, num_classes, label_names
    )

    # Compute macro-averaged metrics (unweighted average across classes)
    macro_metrics = _compute_macro_metrics(per_class_metrics)

    # Compute weighted metrics (weighted by class support)
    weighted_metrics = _compute_weighted_metrics(per_class_metrics)

    # Compute confusion matrix
    confusion_matrix = _compute_confusion_matrix(y_true, y_pred, num_classes)

    # Compute class support (number of true instances per class)
    support = _compute_support(y_true_binary)

    return {
        # Micro-averaged (aggregate all classes)
        'micro_f1': micro_metrics['f1'],
        'micro_precision': micro_metrics['precision'],
        'micro_recall': micro_metrics['recall'],

        # Macro-averaged (unweighted average)
        'macro_f1': macro_metrics['f1'],
        'macro_precision': macro_metrics['precision'],
        'macro_recall': macro_metrics['recall'],

        # Weighted (by support)
        'weighted_f1': weighted_metrics['f1'],
        'weighted_precision': weighted_metrics['precision'],
        'weighted_recall': weighted_metrics['recall'],

        # Overall metrics
        'hamming_loss': hamming_loss,
        'subset_accuracy': subset_accuracy,

        # Per-class breakdown
        'per_class': per_class_metrics,

        # Confusion matrix and support
        'confusion_matrix': confusion_matrix.tolist(),
        'support': support.tolist(),

        # Metadata
        'num_samples': len(y_true),
        'num_classes': num_classes,
    }


def _empty_metrics(num_classes: int, label_names: Dict[int, str]) -> Dict[str, Any]:
    """Return empty metrics structure for edge case of zero samples."""
    return {
        'micro_f1': 0.0,
        'micro_precision': 0.0,
        'micro_recall': 0.0,
        'macro_f1': 0.0,
        'macro_precision': 0.0,
        'macro_recall': 0.0,
        'weighted_f1': 0.0,
        'weighted_precision': 0.0,
        'weighted_recall': 0.0,
        'hamming_loss': 0.0,
        'subset_accuracy': 0.0,
        'per_class': {},
        'confusion_matrix': [[0] * num_classes for _ in range(num_classes)],
        'support': [0] * num_classes,
        'num_samples': 0,
        'num_classes': num_classes,
    }


def _sets_to_binary(label_sets: List[Set[int]], num_classes: int) -> np.ndarray:
    """
    Convert list of label sets to binary indicator matrix.

    Args:
        label_sets: List of sets of label indices
        num_classes: Number of classes

    Returns:
        Binary matrix of shape (num_samples, num_classes)
    """
    binary = np.zeros((len(label_sets), num_classes), dtype=np.int32)
    for i, labels in enumerate(label_sets):
        for label in labels:
            if 0 <= label < num_classes:
                binary[i, label] = 1
    return binary


def _compute_hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Hamming loss: fraction of wrong labels.

    Hamming loss = (# wrong labels) / (# samples * # classes)
    """
    return float(np.mean(y_true != y_pred))


def _compute_micro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute micro-averaged precision, recall, F1.

    Micro-averaging: aggregate TP, FP, FN across all classes, then compute metrics.
    """
    # True positives: both true and pred are 1
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # False positives: pred is 1, true is 0
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # False negatives: true is 1, pred is 0
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
    }


def _compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    label_names: Dict[int, str],
) -> Dict[int, Dict[str, float]]:
    """
    Compute precision, recall, F1 for each class independently.

    Returns dict mapping class index to metrics dict.
    """
    per_class = {}

    for class_idx in range(num_classes):
        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred[:, class_idx]

        tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
        fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
        fn = np.sum((y_true_class == 1) & (y_pred_class == 0))
        tn = np.sum((y_true_class == 0) & (y_pred_class == 0))

        support = int(np.sum(y_true_class))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[class_idx] = {
            'name': label_names.get(class_idx, f"Class {class_idx}"),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': support,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        }

    return per_class


def _compute_macro_metrics(per_class_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute macro-averaged metrics: unweighted average across classes.

    Macro-averaging treats all classes equally, regardless of support.
    """
    if not per_class_metrics:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    precisions = [m['precision'] for m in per_class_metrics.values()]
    recalls = [m['recall'] for m in per_class_metrics.values()]
    f1s = [m['f1'] for m in per_class_metrics.values()]

    return {
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'f1': float(np.mean(f1s)),
    }


def _compute_weighted_metrics(per_class_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute weighted metrics: average weighted by class support.

    Weighted averaging gives more weight to classes with more instances.
    """
    if not per_class_metrics:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    total_support = sum(m['support'] for m in per_class_metrics.values())

    if total_support == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    weighted_precision = sum(
        m['precision'] * m['support'] for m in per_class_metrics.values()
    ) / total_support

    weighted_recall = sum(
        m['recall'] * m['support'] for m in per_class_metrics.values()
    ) / total_support

    weighted_f1 = sum(
        m['f1'] * m['support'] for m in per_class_metrics.values()
    ) / total_support

    return {
        'precision': float(weighted_precision),
        'recall': float(weighted_recall),
        'f1': float(weighted_f1),
    }


def _compute_confusion_matrix(
    y_true: List[Set[int]],
    y_pred: List[Set[int]],
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix for multi-label classification.

    For multi-label, we count co-occurrences: if true label is i and predicted j,
    increment matrix[i][j]. Multiple true/pred labels contribute multiple counts.

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    for true_labels, pred_labels in zip(y_true, y_pred):
        if not true_labels and not pred_labels:
            # Both empty - no confusion to record
            continue
        elif not true_labels:
            # False positives: no true labels, but predicted something
            for pred_label in pred_labels:
                # Record as confusion from "no label" to predicted label
                # We'll use a special convention: skip these for now
                pass
        elif not pred_labels:
            # False negatives: true labels exist, but predicted nothing
            # Skip for now
            pass
        else:
            # Both have labels - record confusion
            for true_label in true_labels:
                for pred_label in pred_labels:
                    if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
                        matrix[true_label][pred_label] += 1

    return matrix


def _compute_support(y_true: np.ndarray) -> np.ndarray:
    """Compute support (number of true instances) per class."""
    return np.sum(y_true, axis=0)


def format_metrics_table(
    metrics: Dict[str, Any],
    title: str = "Multi-label Classification Metrics",
    show_per_class: bool = True,
    show_confusion_matrix: bool = False,
) -> str:
    """
    Format metrics as a readable text table for console output.

    Args:
        metrics: Metrics dictionary from compute_multilabel_metrics()
        title: Title for the output
        show_per_class: Include per-class breakdown
        show_confusion_matrix: Include confusion matrix

    Returns:
        Formatted string ready for printing
    """
    lines = []
    width = 80

    # Title
    lines.append("=" * width)
    lines.append(title.center(width))
    lines.append("=" * width)
    lines.append("")

    # Overall metrics
    lines.append("Overall Metrics:")
    lines.append("-" * width)
    lines.append(f"  Subset Accuracy (Exact Match): {metrics['subset_accuracy']:.1%}")
    lines.append(f"  Hamming Loss:                   {metrics['hamming_loss']:.4f}")
    lines.append("")

    # Micro/Macro/Weighted metrics
    lines.append("Aggregated Metrics:")
    lines.append("-" * width)
    lines.append(f"  {'Averaging':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    lines.append(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    lines.append(f"  {'Micro':<15} {metrics['micro_precision']:>11.1%} {metrics['micro_recall']:>11.1%} {metrics['micro_f1']:>11.1%}")
    lines.append(f"  {'Macro':<15} {metrics['macro_precision']:>11.1%} {metrics['macro_recall']:>11.1%} {metrics['macro_f1']:>11.1%}")
    lines.append(f"  {'Weighted':<15} {metrics['weighted_precision']:>11.1%} {metrics['weighted_recall']:>11.1%} {metrics['weighted_f1']:>11.1%}")
    lines.append("")

    # Per-class metrics
    if show_per_class and metrics.get('per_class'):
        lines.append("Per-Class Metrics:")
        lines.append("-" * width)
        lines.append(f"  {'Class':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        lines.append(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

        for class_idx in sorted(metrics['per_class'].keys()):
            m = metrics['per_class'][class_idx]
            name = m['name'][:28]  # Truncate long names
            lines.append(
                f"  {class_idx}. {name:<27} "
                f"{m['precision']:>11.1%} "
                f"{m['recall']:>11.1%} "
                f"{m['f1']:>11.1%} "
                f"{m['support']:>9}"
            )
        lines.append("")

    # Confusion matrix (optional, can be large)
    if show_confusion_matrix and metrics.get('confusion_matrix'):
        lines.append("Confusion Matrix:")
        lines.append("-" * width)
        lines.append("  (Row: True label, Column: Predicted label)")
        cm = np.array(metrics['confusion_matrix'])

        # Header
        header = "     " + "".join(f"{i:>5}" for i in range(len(cm)))
        lines.append(header)

        # Rows
        for i, row in enumerate(cm):
            row_str = f"  {i:>2} " + "".join(f"{val:>5}" for val in row)
            lines.append(row_str)
        lines.append("")

    lines.append("=" * width)

    return "\n".join(lines)


def format_metrics_compact(metrics: Dict[str, Any]) -> str:
    """
    Format key metrics in a compact single-line format for iteration logs.

    Args:
        metrics: Metrics dictionary from compute_multilabel_metrics()

    Returns:
        Compact string like "Acc: 85.2% | Micro-F1: 87.3% | Macro-F1: 82.1% | Hamming: 0.034"
    """
    return (
        f"Acc: {metrics['subset_accuracy']:.1%} | "
        f"Micro-F1: {metrics['micro_f1']:.1%} | "
        f"Macro-F1: {metrics['macro_f1']:.1%} | "
        f"Hamming: {metrics['hamming_loss']:.3f}"
    )
