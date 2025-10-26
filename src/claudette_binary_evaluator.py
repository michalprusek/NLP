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
    Extract binary prediction from model output with robust handling of:
    - Negations ("not unfair", "isn't fair")
    - Confidence expressions ("likely fair", "probably unfair")
    - Multiple output formats (structured labels, natural language)

    Prioritizes explicit labels over keyword matching to avoid false positives.

    Returns set with single element: {0} for fair, {1} for unfair.
    """
    if not text or not text.strip():
        if verbose:
            print(f"  Empty output, defaulting to fair (0)")
        return {0}  # Default to fair for empty output

    text = text.strip()
    text_lower = text.lower()

    # Strategy 1: Explicit structured labels (highest priority)
    # These patterns look for clear label markers
    structured_patterns = [
        (r'(?:label|classification|answer|prediction)\s*:\s*([01])', 'structured number'),
        (r'(?:label|classification|answer|prediction)\s*:\s*unfair', 'structured unfair'),
        (r'(?:label|classification|answer|prediction)\s*:\s*fair', 'structured fair'),
        (r'final\s+(?:answer|label|classification)\s*:\s*([01])', 'final answer number'),
        (r'final\s+(?:answer|label|classification)\s*:\s*unfair', 'final answer unfair'),
        (r'final\s+(?:answer|label|classification)\s*:\s*fair', 'final answer fair'),
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

    # Strategy 2: Check for negations BEFORE keyword matching
    # This prevents "not unfair" from being classified as unfair
    negation_unfair_patterns = [
        r'\b(?:not|isn\'t|is\s+not|aren\'t|are\s+not|no)\s+(?:\w+\s+){0,3}unfair',
        r'\b(?:fails?\s+to\s+be|doesn\'t\s+seem)\s+unfair',
        r'unfair\s+(?:is|would\s+be)\s+(?:in)?correct',
    ]

    for pattern in negation_unfair_patterns:
        if re.search(pattern, text_lower):
            if verbose:
                print(f"  Found negation of unfair via '{pattern}', returning fair (0)")
            return {0}

    negation_fair_patterns = [
        r'\b(?:not|isn\'t|is\s+not|aren\'t|are\s+not|no)\s+(?:\w+\s+){0,3}fair',
        r'\b(?:fails?\s+to\s+be|doesn\'t\s+seem)\s+fair',
        r'fair\s+(?:is|would\s+be)\s+(?:in)?correct',
    ]

    for pattern in negation_fair_patterns:
        if re.search(pattern, text_lower):
            # Check if this is "not fair" in context of unfairness
            if verbose:
                print(f"  Found negation of fair via '{pattern}', returning unfair (1)")
            return {1}

    # Strategy 3: Look for clear affirmative statements (with optional confidence markers)
    # Prioritize end of text (last 200 chars) where conclusion usually is
    conclusion_text = text_lower[-200:] if len(text_lower) > 200 else text_lower

    affirmative_unfair = [
        r'\b(?:is|seems?|appears?|likely|probably|definitely|clearly)\s+unfair\b',
        r'\bunfair\s+(?:clause|term|provision)\b',
        r'\bclassif(?:y|ied)\s+as\s+unfair\b',
        r'\bconclusion\s*:\s*unfair\b',
    ]

    for pattern in affirmative_unfair:
        if re.search(pattern, conclusion_text):
            if verbose:
                print(f"  Found affirmative unfair statement: '{pattern}'")
            return {1}

    affirmative_fair = [
        r'\b(?:is|seems?|appears?|likely|probably|definitely|clearly)\s+fair\b',
        r'\bfair\s+(?:clause|term|provision)\b',
        r'\bclassif(?:y|ied)\s+as\s+fair\b',
        r'\bconclusion\s*:\s*fair\b',
    ]

    for pattern in affirmative_fair:
        if re.search(pattern, conclusion_text):
            if verbose:
                print(f"  Found affirmative fair statement: '{pattern}'")
            return {0}

    # Strategy 4: Fallback to isolated keyword in conclusion
    # Only if no negation or affirmative pattern matched
    # Search in last 100 chars only to focus on final decision
    final_portion = text_lower[-100:] if len(text_lower) > 100 else text_lower

    # Look for standalone "unfair" word (not preceded by negation)
    if re.search(r'(?<!\bnot\s)(?<!\bisn\'t\s)\bunfair\b', final_portion):
        if verbose:
            print(f"  Found standalone 'unfair' in final portion")
        return {1}

    # Look for standalone "fair" word (not preceded by negation or followed by "unfair")
    if re.search(r'(?<!\bnot\s)(?<!\bisn\'t\s)\bfair\b(?!\s*\w*unfair)', final_portion):
        # Double-check it's not "unfair"
        if 'unfair' not in final_portion:
            if verbose:
                print(f"  Found standalone 'fair' in final portion")
            return {0}

    # Strategy 5: Look for numeric labels in last portion
    nums = re.findall(r'\b([01])\b', final_portion)
    if nums:
        label = int(nums[-1])  # Take last occurrence
        if verbose:
            print(f"  Extracted last number from final portion: {label}")
        return {label}

    # Strategy 6: If output is very short (< 20 chars), do simple keyword check
    if len(text_lower) < 20:
        if 'unfair' in text_lower and 'fair' not in text_lower.replace('unfair', ''):
            if verbose:
                print(f"  Short output contains only 'unfair'")
            return {1}
        if 'fair' in text_lower and 'unfair' not in text_lower:
            if verbose:
                print(f"  Short output contains only 'fair'")
            return {0}
        if '1' in text_lower:
            if verbose:
                print(f"  Short output contains '1'")
            return {1}
        if '0' in text_lower:
            if verbose:
                print(f"  Short output contains '0'")
            return {0}

    # Final fallback: warn and default to fair
    # NOTE: This might indicate the model didn't provide a clear answer
    if verbose:
        print(f"  WARNING: No clear label found in output, defaulting to fair (0)")
        print(f"  Output preview: {text[:200]}...")

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
