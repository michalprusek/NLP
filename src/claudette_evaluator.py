"""
Claudette (ToS clause classification) evaluator

Evaluates multi-label classification for Terms of Service fairness analysis.
"""
import re
from typing import Dict, List, Any, Optional, Set
from datasets import load_from_disk


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
    Extract predicted labels from model output.

    Supports both numeric (0-8) and text formats:
    - "LABEL: 0, 4, 7" or "Labels: [0, 4, 7]"
    - "Categories: Arbitration, Limitation of liability"
    - "final_labels: 0, 3"

    Returns set of numeric labels.
    """
    if not text or not text.strip():
        return set()

    text = text.strip()
    labels = set()

    # Strategy 1: Look for explicit label markers with numbers
    patterns = [
        r'final_labels?\s*:\s*\[?([0-9,\s]+)\]?',
        r'labels?\s*:\s*\[?([0-9,\s]+)\]?',
        r'categories?\s*:\s*\[?([0-9,\s]+)\]?',
        r'classification\s*:\s*\[?([0-9,\s]+)\]?',
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

    # Strategy 2: Look for explicit label markers with text
    text_patterns = [
        r'final_labels?\s*:\s*\[?([^\]]+)\]?',
        r'labels?\s*:\s*\[?([^\]]+)\]?',
        r'categories?\s*:\s*\[?([^\]]+)\]?',
    ]

    for pattern in text_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            content = match.group(1).lower()
            # Check each label name
            for label_text, idx in TEXT_TO_IDX.items():
                if label_text in content:
                    labels.add(idx)

            if labels and verbose:
                print(f"  Extracted text labels: {sorted(labels)}")

            if labels:
                return labels

    # Strategy 3: Search for label mentions in full text
    text_lower = text.lower()
    for label_text, idx in TEXT_TO_IDX.items():
        # Use word boundaries for multi-word labels
        pattern = r'\b' + re.escape(label_text) + r'\b'
        if re.search(pattern, text_lower):
            labels.add(idx)

    # Strategy 4: Fallback - extract any numbers 0-8 mentioned
    if not labels:
        nums = re.findall(r'\b([0-8])\b', text)
        for n in nums:
            labels.add(int(n))

    if verbose and labels:
        print(f"  Final extracted labels: {sorted(labels)}")

    return labels


def get_ground_truth_labels(example: Dict[str, Any]) -> Set[int]:
    """
    Extract ground truth labels from Claudette dataset example.

    The dataset has boolean fields for each category:
    - ltd (0): Limitation of liability
    - ter (1): Unilateral termination
    - ch (2): Unilateral change
    - a (3): Arbitration
    - cr (4): Content removal
    - law (5): Choice of law
    - pinc (6): Other
    - use (7): Contract by using
    - j (8): Jurisdiction
    """
    # Map boolean fields to label indices
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

    For Claudette, it's actually single-label classification (one label per example),
    but we handle it as multi-label for generality.
    """
    # Exact match (all labels correct)
    exact_match = pred_labels == true_labels

    # Precision, Recall, F1
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

    def __init__(
        self,
        dataset_path: str = "datasets/claudette",
        split: str = "test",
        debug: bool = False
    ):
        """
        Initialize Claudette evaluator.

        Args:
            dataset_path: Path to Claudette dataset
            split: 'train', 'validation', or 'test'
            debug: Enable verbose logging
        """
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

        print(f"Loaded Claudette {split} split: {len(self.dataset)} examples")

    def evaluate_batch(
        self,
        outputs: List[str],
        indices: List[int],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate batch of model outputs.

        Args:
            outputs: List of model outputs
            indices: List of example indices
            verbose: Print detailed output

        Returns:
            Dictionary with accuracy, metrics, and details
        """
        if len(outputs) != len(indices):
            raise ValueError("outputs and indices must have same length")

        correct = 0
        details = []
        failed_extractions = 0

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for i, (output, idx) in enumerate(zip(outputs, indices)):
            example = self.dataset[idx]

            # Extract predictions and ground truth
            pred_labels = extract_labels_from_output(output, verbose=(verbose and i < 3))
            true_labels = get_ground_truth_labels(example)

            # Compute metrics
            metrics = compute_metrics(pred_labels, true_labels)
            is_correct = metrics['exact_match'] == 1.0

            if not pred_labels:
                failed_extractions += 1

            if is_correct:
                correct += 1

            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']

            # Debug output
            if (verbose or self.debug) and i < 3:
                print(f"\n--- Example {i+1} ---")
                # Get sentence from dataset
                text = example.get('sentence', '')
                print(f"Text: {text[:150]}...")
                print(f"Output: {output[:200]}...")
                print(f"True: {sorted(true_labels)} | Pred: {sorted(pred_labels)} | Match: {is_correct}")
                print(f"Metrics: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

            details.append({
                'idx': idx,
                'text': example.get('sentence', ''),
                'ground_truth': sorted(true_labels),
                'predicted': sorted(pred_labels),
                'correct': is_correct,
                'output': output,
                'metrics': metrics,
            })

        total = len(outputs)

        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
            'failed_extractions': failed_extractions,
            'avg_precision': total_precision / total if total > 0 else 0.0,
            'avg_recall': total_recall / total if total > 0 else 0.0,
            'avg_f1': total_f1 / total if total > 0 else 0.0,
            'details': details,
        }

    def get_batch(self, start_idx: int, batch_size: int) -> List[Dict[str, Any]]:
        """
        Get batch of examples.

        Returns examples in format compatible with main.py:
        {'idx': int, 'question': str, 'answer': str}

        Note: 'question' is aliased to 'sentence' field, 'answer' to labels
        """
        end_idx = min(start_idx + batch_size, len(self.dataset))
        return [
            {
                'idx': i,
                'question': self.dataset[i]['sentence'],  # Alias for compatibility
                'answer': str(sorted(get_ground_truth_labels(self.dataset[i]))),  # Convert labels to string
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
                'answer': str(sorted(get_ground_truth_labels(self.dataset[i]))),  # Convert labels to string
            }
            for i in indices
        ]

    def __len__(self) -> int:
        return len(self.dataset)
