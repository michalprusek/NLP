"""
Claudette Binary (ToS clause binary classification) evaluator

Evaluates binary classification for Terms of Service fairness analysis:
- Fair (0): No unfair clauses detected
- Unfair (1): At least one unfair clause type present
"""
import re
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from src.claudette_evaluator import get_ground_truth_labels
from src.metrics import compute_multilabel_metrics


# Binary label mapping
BINARY_LABELS = {
    0: "Fair",
    1: "Unfair",
}


def extract_binary_label_from_output(text: str, verbose: bool = False) -> Set[int]:
    """
    Extract binary prediction from model output.

    Supports multiple output formats:
    - "LABEL: 0" or "LABEL: 1"
    - "CLASSIFICATION: FAIR" or "CLASSIFICATION: UNFAIR"
    - "fair" or "unfair" (case-insensitive)
    - Raw numbers: "0" or "1"

    Returns set with single element: {0} for fair, {1} for unfair.
    """
    if not text or not text.strip():
        return {0}  # Default to fair

    text = text.strip()
    text_lower = text.lower()

    # Strategy 1: Explicit label markers with numbers
    patterns = [
        r'label\s*:\s*([01])',
        r'classification\s*:\s*([01])',
        r'answer\s*:\s*([01])',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            label = int(match.group(1))
            if verbose:
                print(f"  Extracted via pattern '{pattern}': {label}")
            return {label}

    # Strategy 2: Text keywords for UNFAIR
    unfair_patterns = [
        r'\bunfair\b',
        r'classification\s*:\s*unfair',
        r'label\s*:\s*unfair',
    ]

    for pattern in unfair_patterns:
        if re.search(pattern, text_lower):
            if verbose:
                print(f"  Extracted 'unfair' via pattern '{pattern}'")
            return {1}

    # Strategy 3: Text keywords for FAIR
    fair_patterns = [
        r'\bfair\b',
        r'classification\s*:\s*fair',
        r'label\s*:\s*fair',
    ]

    for pattern in fair_patterns:
        if re.search(pattern, text_lower):
            if verbose:
                print(f"  Extracted 'fair' via pattern '{pattern}'")
            return {0}

    # Strategy 4: Look for "unfair" anywhere in text (more permissive)
    if 'unfair' in text_lower:
        if verbose:
            print(f"  Found 'unfair' in text")
        return {1}

    # Strategy 5: Look for "fair" anywhere in text (but NOT "unfair")
    # Check for "unfair" first to avoid false positives
    if 'unfair' not in text_lower and 'fair' in text_lower:
        if verbose:
            print(f"  Found 'fair' (without 'unfair') in text")
        return {0}

    # Strategy 6: Fallback - extract numbers from LAST portion only (avoid picking up numbers in reasoning)
    # Check last 50 chars to focus on final answer, not explanation
    last_portion = text[-50:] if len(text) > 50 else text
    nums = re.findall(r'\b([01])\b', last_portion)
    if nums:
        label = int(nums[-1])  # Take last occurrence
        if verbose:
            print(f"  Extracted last number from end: {label}")
        return {label}

    # Strategy 7: Check for presence of unfair keywords as last resort
    # If model mentions unfair terms but didn't give clear label, lean towards unfair
    unfair_keywords = [
        r'liabilit', r'terminat', r'arbitrat', r'unilateral',
        r'removal', r'jurisdiction', r'sole\s+discretion',
        r'without\s+notice', r'at\s+any\s+time'
    ]
    for keyword in unfair_keywords:
        if re.search(keyword, text_lower):
            if verbose:
                print(f"  Found unfair keyword '{keyword}', inferring UNFAIR")
            return {1}

    # Default: If truly ambiguous with NO signals, conservatively mark as FAIR
    # But this should be rare now with better extraction above
    if verbose:
        print(f"  No clear signals found, defaulting to fair (0)")
    return {0}


def get_binary_ground_truth(example: Dict[str, Any]) -> Set[int]:
    """
    Convert multi-label ground truth to binary classification.

    Returns {1} if ANY unfair label present (any of the 9 boolean fields is True),
    {0} otherwise (all boolean fields are False = fair clause).
    """
    # Reuse existing function to get multi-label ground truth
    labels = get_ground_truth_labels(example)

    # Binary: any unfair label → unfair (1), no labels → fair (0)
    return {1} if len(labels) > 0 else {0}


class ClaudetteBinaryEvaluator:
    """Evaluator for Claudette binary classification (fair vs unfair)"""

    # Task metadata (for template selection)
    task_type = "classification"
    task_name = "claudette_binary"  # For prompt template loading

    def __init__(
        self,
        dataset_path: str = "datasets/tos_local",
        split: str = "test",
        debug: bool = False
    ):
        """
        Initialize Claudette binary evaluator.

        Args:
            dataset_path: Path to Claudette dataset (supports both Arrow and JSON formats)
            split: 'train', 'validation', or 'test'
            debug: Enable verbose logging
        """
        dataset_dir = Path(dataset_path)

        # Detect format: check if directory contains .json files
        json_files = list(dataset_dir.glob("*.json"))
        has_json = len([f for f in json_files if f.stem in ['train', 'validation', 'test']]) > 0

        if has_json:
            # Load JSON dataset using wrapper
            from src.json_dataset_wrapper import JSONDatasetDict
            ds = JSONDatasetDict.load_from_disk(dataset_path)
        else:
            # Load HuggingFace Arrow format
            from datasets import load_from_disk
            ds = load_from_disk(dataset_path)

        if split not in ds:
            # Handle split mismatch - use available split
            available_splits = list(ds.keys())
            print(f"Warning: Split '{split}' not found. Available: {available_splits}")
            if 'test' in available_splits:
                split = 'test'
            elif available_splits:
                split = available_splits[0]
            else:
                raise ValueError(f"No splits found in dataset at {dataset_path}")

        self.dataset = ds[split]
        self.split = split
        self.debug = debug

        print(f"Loaded Claudette Binary {split} split: {len(self.dataset)} examples")

    def evaluate_batch(
        self,
        outputs: List[str],
        indices: List[int],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate batch of model outputs with binary classification metrics.

        Args:
            outputs: List of model outputs
            indices: List of example indices
            verbose: Print detailed output

        Returns:
            Dictionary with:
            - accuracy: Overall accuracy
            - precision: Precision for unfair class
            - recall: Recall for unfair class
            - f1: F1 score for unfair class
            - tp, fp, tn, fn: Confusion matrix components
            - micro_f1, macro_f1: Aggregated F1 scores
            - per_class: Metrics for each class (fair vs unfair)
            - details: Per-example results
            - failed_extractions: Count of failed extractions
        """
        if len(outputs) != len(indices):
            raise ValueError("outputs and indices must have same length")

        # Collect all predictions and ground truths as binary sets
        y_true = []
        y_pred = []
        details = []
        failed_extractions = 0

        for i, (output, idx) in enumerate(zip(outputs, indices)):
            example = self.dataset[idx]

            # Extract predictions and ground truth (both as sets with single element)
            pred_labels = extract_binary_label_from_output(output, verbose=(verbose and i < 3))
            true_labels = get_binary_ground_truth(example)

            y_true.append(true_labels)
            y_pred.append(pred_labels)

            # Check if extraction failed (shouldn't happen with default to {0})
            if len(pred_labels) == 0:
                failed_extractions += 1

            # Compute per-example correctness
            is_correct = pred_labels == true_labels

            # Debug output
            if (verbose or self.debug) and i < 3:
                print(f"\n--- Example {i+1} ---")
                text = example.get('sentence', '')
                print(f"Text: {text[:150]}...")
                print(f"Output: {output[:200]}...")
                true_label = list(true_labels)[0] if true_labels else 0
                pred_label = list(pred_labels)[0] if pred_labels else 0
                print(f"True: {true_label} ({BINARY_LABELS[true_label]}) | "
                      f"Pred: {pred_label} ({BINARY_LABELS[pred_label]}) | "
                      f"Match: {is_correct}")

            details.append({
                'idx': idx,
                'text': example.get('sentence', ''),
                'ground_truth': list(true_labels)[0] if true_labels else 0,
                'predicted': list(pred_labels)[0] if pred_labels else 0,
                'correct': is_correct,
                'output': output,
            })

        # Compute comprehensive metrics using existing multi-label function with num_classes=2!
        # This is the key insight: binary classification is just multi-label with 2 classes
        binary_metrics = compute_multilabel_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=2,
            label_names=BINARY_LABELS,
        )

        # Extract binary-specific metrics
        total = len(outputs)
        correct = int(binary_metrics['subset_accuracy'] * total)

        # Get confusion matrix components from per_class metrics
        # Class 1 (Unfair) is the positive class
        unfair_metrics = binary_metrics['per_class'][1]
        tp = unfair_metrics['tp']
        fp = unfair_metrics['fp']
        fn = unfair_metrics['fn']
        tn = unfair_metrics['tn']

        # Build result dict with backward compatibility + binary metrics
        result = {
            # Primary metrics (used by optimizers)
            'accuracy': binary_metrics['subset_accuracy'],
            'correct': correct,
            'total': total,
            'failed_extractions': failed_extractions,

            # Binary classification metrics (for unfair class as positive)
            'precision': unfair_metrics['precision'],
            'recall': unfair_metrics['recall'],
            'f1': unfair_metrics['f1'],

            # Confusion matrix components
            'tp': tp,  # True positives (correctly identified unfair)
            'fp': fp,  # False positives (fair predicted as unfair)
            'tn': tn,  # True negatives (correctly identified fair)
            'fn': fn,  # False negatives (unfair predicted as fair)

            # Comprehensive metrics (from multilabel computation)
            'micro_f1': binary_metrics['micro_f1'],
            'micro_precision': binary_metrics['micro_precision'],
            'micro_recall': binary_metrics['micro_recall'],

            'macro_f1': binary_metrics['macro_f1'],
            'macro_precision': binary_metrics['macro_precision'],
            'macro_recall': binary_metrics['macro_recall'],

            'weighted_f1': binary_metrics['weighted_f1'],
            'weighted_precision': binary_metrics['weighted_precision'],
            'weighted_recall': binary_metrics['weighted_recall'],

            'per_class': binary_metrics['per_class'],
            'confusion_matrix': binary_metrics['confusion_matrix'],
            'support': binary_metrics['support'],

            # Per-example details
            'details': details,
        }

        return result

    def get_batch(self, start_idx: int, batch_size: int) -> List[Dict[str, Any]]:
        """
        Get batch of examples.

        Returns examples in format compatible with main.py:
        {'idx': int, 'question': str, 'answer': str}

        Note: 'question' is aliased to 'sentence' field, 'answer' to binary label
        """
        end_idx = min(start_idx + batch_size, len(self.dataset))
        return [
            {
                'idx': i,
                'question': self.dataset[i]['sentence'],  # Alias for compatibility
                'answer': str(list(get_binary_ground_truth(self.dataset[i]))[0]),  # Binary label as string
            }
            for i in range(start_idx, end_idx)
        ]

    def get_batch_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Get batch of examples by specific indices.

        Returns examples in format compatible with main.py:
        {'idx': int, 'question': str, 'answer': str}
        """
        return [
            {
                'idx': i,
                'question': self.dataset[i]['sentence'],  # Alias for compatibility
                'answer': str(list(get_binary_ground_truth(self.dataset[i]))[0]),  # Binary label as string
            }
            for i in indices
        ]

    def __len__(self) -> int:
        return len(self.dataset)
