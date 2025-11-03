"""
Claudette Binary (ToS clause binary classification) evaluator - simplified version
"""
import re
from typing import Dict, List, Any, Set
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
    Extract binary prediction from model output - simplified version.

    Returns set with single element: {0} for fair, {1} for unfair.
    """
    if not text or not text.strip():
        if verbose:
            print(f"  Empty output, defaulting to fair (0)")
        return {0}

    text = text.strip()
    text_lower = text.lower()

    # Strategy 1: Explicit structured labels (highest priority)
    structured_patterns = [
        (r'(?:label|classification|answer|prediction)\s*:\s*([01])', 'structured number'),
        (r'(?:label|classification|answer|prediction)\s*:\s*unfair', 'structured unfair'),
        (r'(?:label|classification|answer|prediction)\s*:\s*fair', 'structured fair'),
    ]

    for pattern, desc in structured_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if match.lastindex and match.group(1) in ['0', '1']:
                label = int(match.group(1))
            else:
                label = 1 if 'unfair' in match.group(0) else 0

            if verbose:
                print(f"  Extracted via {desc}: {label}")
            return {label}

    # Strategy 2: Look for clear statements in last portion (conclusion)
    conclusion_text = text_lower[-150:] if len(text_lower) > 150 else text_lower

    # Check for "unfair" keyword
    if re.search(r'\bunfair\b', conclusion_text):
        if verbose:
            print(f"  Found 'unfair' in conclusion")
        return {1}

    # Check for "fair" keyword (not part of "unfair")
    if re.search(r'\bfair\b', conclusion_text) and 'unfair' not in conclusion_text:
        if verbose:
            print(f"  Found 'fair' in conclusion")
        return {0}

    # Strategy 3: Look for numeric labels in last portion
    last_portion = text_lower[-50:] if len(text_lower) > 50 else text_lower
    nums = re.findall(r'\b([01])\b', last_portion)
    if nums:
        label = int(nums[-1])
        if verbose:
            print(f"  Extracted last number: {label}")
        return {label}

    # Final fallback: default to fair
    if verbose:
        print(f"  No clear label found, defaulting to fair (0)")

    return {0}


def get_binary_ground_truth(example: Dict[str, Any]) -> Set[int]:
    """
    Convert multi-label ground truth to binary classification.

    Returns {1} if ANY unfair label present, {0} otherwise.
    """
    labels = get_ground_truth_labels(example)
    return {1} if len(labels) > 0 else {0}


class ClaudetteBinaryEvaluator:
    """Evaluator for Claudette binary classification (fair vs unfair)"""

    # Task metadata
    task_type = "classification"
    task_name = "claudette_binary"

    def __init__(
        self,
        dataset_path: str = "datasets/tos_local",
        split: str = "test",
        debug: bool = False
    ):
        """
        Initialize Claudette binary evaluator.
        """
        dataset_dir = Path(dataset_path)

        # Detect format
        json_files = list(dataset_dir.glob("*.json"))
        has_json = len([f for f in json_files if f.stem in ['train', 'validation', 'test']]) > 0

        if has_json:
            from src.json_dataset_wrapper import JSONDatasetDict
            ds = JSONDatasetDict.load_from_disk(dataset_path)
        else:
            from datasets import load_from_disk
            ds = load_from_disk(dataset_path)

        if split not in ds:
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
        """
        if len(outputs) != len(indices):
            raise ValueError("outputs and indices must have same length")

        # Collect predictions and ground truths
        y_true = []
        y_pred = []
        details = []
        failed_extractions = 0

        for i, (output, idx) in enumerate(zip(outputs, indices)):
            example = self.dataset[idx]

            # Extract predictions and ground truth
            pred_labels = extract_binary_label_from_output(output, verbose=(verbose and i < 3))
            true_labels = get_binary_ground_truth(example)

            y_true.append(true_labels)
            y_pred.append(pred_labels)

            if len(pred_labels) == 0:
                failed_extractions += 1

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

        # Compute comprehensive metrics
        binary_metrics = compute_multilabel_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=2,
            label_names=BINARY_LABELS,
        )

        total = len(outputs)
        correct = int(binary_metrics['subset_accuracy'] * total)

        # Extract confusion matrix components
        unfair_metrics = binary_metrics['per_class'][1]
        tp = unfair_metrics['tp']
        fp = unfair_metrics['fp']
        fn = unfair_metrics['fn']
        tn = unfair_metrics['tn']

        result = {
            # Primary metrics
            'accuracy': binary_metrics['subset_accuracy'],
            'correct': correct,
            'total': total,
            'failed_extractions': failed_extractions,

            # Binary classification metrics
            'precision': unfair_metrics['precision'],
            'recall': unfair_metrics['recall'],
            'f1': unfair_metrics['f1'],

            # Confusion matrix
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,

            # Comprehensive metrics
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

            'details': details,
        }

        return result

    def get_batch(self, start_idx: int, batch_size: int) -> List[Dict[str, Any]]:
        """
        Get batch of examples.
        """
        end_idx = min(start_idx + batch_size, len(self.dataset))
        return [
            {
                'idx': i,
                'question': self.dataset[i]['sentence'],
                'answer': str(list(get_binary_ground_truth(self.dataset[i]))[0]),
            }
            for i in range(start_idx, end_idx)
        ]

    def get_batch_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Get batch of examples by specific indices.
        """
        return [
            {
                'idx': i,
                'question': self.dataset[i]['sentence'],
                'answer': str(list(get_binary_ground_truth(self.dataset[i]))[0]),
            }
            for i in indices
        ]

    def __len__(self) -> int:
        return len(self.dataset)
