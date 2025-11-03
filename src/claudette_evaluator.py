"""
Claudette (ToS clause classification) evaluator - simplified version
"""
import re
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from src.metrics import compute_multilabel_metrics


# Label mapping from metadata.json
LABEL_MAP = {
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

# Reverse mapping for text → number
TEXT_TO_IDX = {name.lower(): idx for idx, name in LABEL_MAP.items()}


def extract_labels_from_output(text: str, verbose: bool = False) -> Set[int]:
    """
    Extract predicted labels from model output - simplified version.

    Supports:
    - Numeric format: "LABEL: 0, 4, 7" or "Labels: [0, 4, 7]"
    - NONE indicators for fair clauses
    """
    if not text or not text.strip():
        return set()

    text = text.strip()
    labels = set()

    # Strategy 1: Look for explicit numeric labels
    patterns = [
        r'final_labels?\s*:\s*\[?([0-9,\s]+)\]?',
        r'labels?\s*:\s*\[?([0-9,\s]+)\]?',
        r'categories?\s*:\s*\[?([0-9,\s]+)\]?',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            content = match.group(1)
            # Extract all numbers 0-8
            nums = re.findall(r'\b([0-8])\b', content)
            for n in nums:
                labels.add(int(n))

            if labels and verbose:
                print(f"  Extracted via pattern '{pattern}': {sorted(labels)}")

            if labels:
                return labels

    # Strategy 2: Check for NONE indicators (for 90% neutral clauses)
    none_patterns = [
        r'\bNONE\b',
        r'\bno\s+labels?\b',
        r'\bfair\b',
    ]
    for pattern in none_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            if verbose:
                print(f"  Extracted NONE via pattern '{pattern}'")
            return set()

    # Strategy 3: Fallback - extract numbers from last portion (last 50 chars)
    last_portion = text[-50:] if len(text) > 50 else text
    nums = re.findall(r'\b([0-8])\b', last_portion)
    for n in nums:
        labels.add(int(n))

    if verbose and labels:
        print(f"  Extracted from last portion: {sorted(labels)}")

    return labels


def get_ground_truth_labels(example: Dict[str, Any]) -> Set[int]:
    """
    Extract ground truth labels from Claudette dataset example.
    """
    field_to_idx = {
        'ltd': 0,
        'ter': 1,
        'ch': 2,
        'a': 3,
        'cr': 4,
        'law': 5,
        'pinc': 6,
        'use': 7,
        'j': 8,
    }

    labels = set()
    for field, idx in field_to_idx.items():
        if example.get(field, False):
            labels.add(idx)

    return labels


def compute_metrics(pred_labels: Set[int], true_labels: Set[int]) -> Dict[str, float]:
    """
    Compute metrics for multi-label classification.
    """
    exact_match = pred_labels == true_labels

    if len(pred_labels) == 0:
        precision = 1.0 if len(true_labels) == 0 else 0.0
    else:
        precision = len(pred_labels & true_labels) / len(pred_labels)

    if len(true_labels) == 0:
        recall = 1.0 if len(pred_labels) == 0 else 0.0
    else:
        recall = len(pred_labels & true_labels) / len(true_labels)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        'exact_match': 1.0 if exact_match else 0.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


class ClaudetteEvaluator:
    """Evaluator for Claudette (ToS clause classification) dataset"""

    # Task metadata (for template selection)
    task_type = "classification"
    task_name = "claudette"

    def __init__(
        self,
        dataset_path: str = "datasets/tos_local",
        split: str = "test",
        debug: bool = False
    ):
        """
        Initialize Claudette evaluator.
        """
        dataset_dir = Path(dataset_path)

        # Detect format: check if directory contains .json files
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

        print(f"Loaded Claudette {split} split: {len(self.dataset)} examples")

    def evaluate_batch(
        self,
        outputs: List[str],
        indices: List[int],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate batch of model outputs with comprehensive multi-label metrics.
        """
        if len(outputs) != len(indices):
            raise ValueError("outputs and indices must have same length")

        # Collect all predictions and ground truths
        y_true = []
        y_pred = []
        details = []
        failed_extractions = 0

        for i, (output, idx) in enumerate(zip(outputs, indices)):
            example = self.dataset[idx]

            # Extract predictions and ground truth
            pred_labels = extract_labels_from_output(output, verbose=(verbose and i < 3))
            true_labels = get_ground_truth_labels(example)

            y_true.append(true_labels)
            y_pred.append(pred_labels)

            # Compute per-example metrics
            per_example_metrics = compute_metrics(pred_labels, true_labels)
            is_correct = per_example_metrics['exact_match'] == 1.0

            if not pred_labels and true_labels:
                failed_extractions += 1

            # Debug output
            if (verbose or self.debug) and i < 3:
                print(f"\n--- Example {i+1} ---")
                text = example.get('sentence', '')
                print(f"Text: {text[:150]}...")
                print(f"Output: {output[:200]}...")
                print(f"True: {sorted(true_labels)} | Pred: {sorted(pred_labels)} | Match: {is_correct}")
                print(f"Metrics: P={per_example_metrics['precision']:.2f} R={per_example_metrics['recall']:.2f} F1={per_example_metrics['f1']:.2f}")

            details.append({
                'idx': idx,
                'text': example.get('sentence', ''),
                'ground_truth': sorted(true_labels),
                'predicted': sorted(pred_labels),
                'correct': is_correct,
                'output': output,
                'metrics': per_example_metrics,
            })

        # Compute comprehensive multi-label metrics
        multilabel_metrics = compute_multilabel_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=9,
            label_names=LABEL_MAP,
        )

        total = len(outputs)
        correct = int(multilabel_metrics['subset_accuracy'] * total)

        result = {
            # Backward compatible fields
            'accuracy': multilabel_metrics['subset_accuracy'],
            'correct': correct,
            'total': total,
            'failed_extractions': failed_extractions,

            # Legacy averaged metrics
            'avg_precision': sum(d['metrics']['precision'] for d in details) / total if total > 0 else 0.0,
            'avg_recall': sum(d['metrics']['recall'] for d in details) / total if total > 0 else 0.0,
            'avg_f1': sum(d['metrics']['f1'] for d in details) / total if total > 0 else 0.0,

            # Comprehensive metrics
            'micro_f1': multilabel_metrics['micro_f1'],
            'micro_precision': multilabel_metrics['micro_precision'],
            'micro_recall': multilabel_metrics['micro_recall'],

            'macro_f1': multilabel_metrics['macro_f1'],
            'macro_precision': multilabel_metrics['macro_precision'],
            'macro_recall': multilabel_metrics['macro_recall'],

            'weighted_f1': multilabel_metrics['weighted_f1'],
            'weighted_precision': multilabel_metrics['weighted_precision'],
            'weighted_recall': multilabel_metrics['weighted_recall'],

            'hamming_loss': multilabel_metrics['hamming_loss'],

            'per_class': multilabel_metrics['per_class'],
            'confusion_matrix': multilabel_metrics['confusion_matrix'],
            'support': multilabel_metrics['support'],

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
                'answer': str(sorted(get_ground_truth_labels(self.dataset[i]))),
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
                'answer': str(sorted(get_ground_truth_labels(self.dataset[i]))),
            }
            for i in indices
        ]

    def __len__(self) -> int:
        return len(self.dataset)
