"""GSM8K evaluator with simplified answer extraction"""
import re
from typing import Dict, List, Any, Optional
from datasets import load_from_disk


# Simple number pattern - matches integers and decimals
NUMBER_PATTERN = r'[-+]?\d+(?:[.,]\d+)?'


def normalize_number(text: str) -> Optional[str]:
    """
    Normalize numeric string to standard format.
    Handles commas as thousands separators.
    """
    if not text:
        return None

    s = text.strip()

    # Remove commas (thousands separators)
    s = s.replace(',', '')

    # Normalize decimal point (convert comma to period)
    if '.' not in s and ',' in text:
        s = text.replace(',', '.')

    # Parse as float and convert to string
    try:
        num = float(s)
        # If it's an integer, return as integer string
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return None


def compare_numbers(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """
    Compare two numbers with tolerance.
    """
    if predicted == ground_truth:
        return True

    try:
        pred_num = float(predicted)
        gt_num = float(ground_truth)
        return abs(pred_num - gt_num) <= tolerance
    except (ValueError, TypeError):
        return False


def extract_answer(text: str, verbose: bool = False) -> Optional[str]:
    """
    Extract final number from model output.
    Simply returns the last number found in text.
    """
    if not text:
        return None

    text = text.strip()

    # Find all numbers and return the last one
    numbers = re.findall(NUMBER_PATTERN, text)
    if numbers:
        result = normalize_number(numbers[-1])
        if verbose and result:
            print(f"  Found last number: {result}")
        return result

    if verbose:
        print("  FAILED to extract answer")
    return None


def extract_ground_truth(answer_text: str) -> str:
    """
    Extract ground truth from GSM8K answer field.
    Expected format: "#### NUMBER"
    """
    if not answer_text:
        raise ValueError("Empty ground truth text")

    # Look for #### NUMBER
    match = re.search(rf'####\s*({NUMBER_PATTERN})', answer_text)
    if match:
        result = normalize_number(match.group(1))
        if result is not None:
            return result

    # Fallback: last number
    numbers = re.findall(NUMBER_PATTERN, answer_text)
    if numbers:
        result = normalize_number(numbers[-1])
        if result is not None:
            return result

    raise ValueError(f"Could not extract ground truth from: {answer_text!r}")


class GSM8KEvaluator:
    """Evaluator for GSM8K dataset with simplified extraction"""

    def __init__(self, dataset_path: str = "datasets/gsm8k", split: str = "test", debug: bool = False):
        """
        Args:
            dataset_path: Path to saved HF dataset
            split: 'train' or 'test'
            debug: Verbose logs
        """
        ds = load_from_disk(dataset_path)
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in dataset at {dataset_path}")
        self.dataset = ds[split]
        self.split = split
        self.debug = debug

    def evaluate_batch(self, outputs: List[str], indices: List[int], verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate batch of outputs against ground truth.
        """
        if len(outputs) != len(indices):
            raise ValueError("outputs and indices must have the same length")

        correct = 0
        details = []
        failed_extractions = 0

        for i, (output, idx) in enumerate(zip(outputs, indices)):
            example = self.dataset[idx]
            gt = extract_ground_truth(example['answer'])
            pred = extract_answer(output, verbose=False)

            is_correct = (pred is not None and compare_numbers(pred, gt))
            if pred is None:
                failed_extractions += 1
            if is_correct:
                correct += 1

            if (verbose or self.debug) and i < 3:
                print(f"\n--- Example {i+1} ---")
                question_preview = example['question'][:120].replace('\n', ' ')
                output_preview = output[:200].replace('\n', ' ')
                print(f"Q: {question_preview}")
                print(f"Out: {output_preview}")
                print(f"GT: {gt} | Pred: {pred} | Correct: {is_correct}")

            details.append({
                'idx': idx,
                'question': example['question'],
                'ground_truth': gt,
                'predicted': pred,
                'correct': is_correct,
                'output': output
            })

        total = len(outputs)
        if failed_extractions > 0 and (verbose or self.debug):
            print(f"\n⚠️  Failed to extract answer: {failed_extractions}/{total} ({failed_extractions/total*100:.1f}%)")

        return {
            'accuracy': correct / total if total else 0.0,
            'correct': correct,
            'total': total,
            'details': details,
            'failed_extractions': failed_extractions
        }

    def get_batch(self, start_idx: int, batch_size: int) -> List[Dict[str, Any]]:
        end_idx = min(start_idx + batch_size, len(self.dataset))
        return [
            {'idx': i, 'question': self.dataset[i]['question'], 'answer': self.dataset[i]['answer']}
            for i in range(start_idx, end_idx)
        ]

    def __len__(self) -> int:
        return len(self.dataset)
