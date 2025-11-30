"""GSM8K evaluator with simplified answer extraction"""
import re
from typing import Dict, List, Any, Optional
from datasets import load_from_disk


# Simple number pattern - matches integers and decimals (including negative)
NUMBER_PATTERN = r'[-+]?\d+(?:[.,]\d+)?'


def compare_numbers(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """Compare two numbers with tolerance."""
    if predicted == ground_truth:
        return True

    try:
        # Remove commas (thousands separators) and normalize
        pred_clean = predicted.replace(',', '')
        gt_clean = ground_truth.replace(',', '')
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)
        return abs(pred_num - gt_num) <= tolerance
    except (ValueError, TypeError):
        return False


def extract_answer(text: str) -> Optional[str]:
    """Extract last number from model output."""
    if not text:
        return None

    # Find all numbers and return the last one
    numbers = re.findall(NUMBER_PATTERN, text)
    return numbers[-1] if numbers else None


def extract_ground_truth(answer_text: str) -> str:
    """Extract last number from GSM8K answer."""
    if not answer_text:
        raise ValueError("Empty ground truth text")

    numbers = re.findall(NUMBER_PATTERN, answer_text)
    if numbers:
        return numbers[-1]

    raise ValueError(f"Could not extract ground truth from: {answer_text!r}")


class GSM8KEvaluator:
    """Evaluator for GSM8K dataset"""

    def __init__(self, dataset_path: str = "datasets/gsm8k", split: str = "test", debug: bool = False):
        '''
        load_from_disk je z HuggingFace datasets knihovny
        https://huggingface.co/docs/datasets/loading

        FORMAT:
          datasets/gsm8k/
            ├── train/
            │   ├── data-00000-of-00001.arrow
            │   └── state.json
            ├── test/
            │   ├── data-00000-of-00001.arrow
            │   └── state.json
            └── dataset_dict.json
        '''
        ds = load_from_disk(dataset_path)
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in dataset at {dataset_path}")
        self.dataset = ds[split]
        self.split = split
        self.debug = debug

    def evaluate_batch(self, outputs: List[str], indices: List[int]) -> Dict[str, Any]:
        """Evaluate batch of outputs against ground truth."""
        if len(outputs) != len(indices):
            raise ValueError("outputs and indices must have the same length")

        correct = 0
        details = []

        for output, idx in zip(outputs, indices):
            example = self.dataset[idx]
            gt = extract_ground_truth(example['answer'])
            pred = extract_answer(output)

            is_correct = pred is not None and compare_numbers(pred, gt)
            if is_correct:
                correct += 1

            details.append({
                'idx': idx,
                'question': example['question'],
                'ground_truth': gt,
                'predicted': pred,
                'correct': is_correct,
                'output': output
            })

        total = len(outputs)
        return {
            'accuracy': correct / total if total else 0.0,
            'correct': correct,
            'total': total,
            'details': details,
        }

    def get_batch(self, start_idx: int, batch_size: int) -> List[Dict[str, Any]]:
        end_idx = min(start_idx + batch_size, len(self.dataset))
        return [
            {'idx': i, 'question': self.dataset[i]['question'], 'answer': self.dataset[i]['answer']}
            for i in range(start_idx, end_idx)
        ]

    def __len__(self) -> int:
        return len(self.dataset)
