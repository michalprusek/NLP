"""GSM8K evaluator matching lm-evaluation-harness standard.

Reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/
"""
import re
from typing import Dict, List, Any, Optional
from datasets import load_from_disk


# Pattern 1: Strict - extract number after #### marker (standard GSM8K format)
STRICT_PATTERN = r'####\s*(-?[\d,]+\.?\d*)'

# Pattern 2: LaTeX boxed format (common in math models like Qwen)
BOXED_PATTERN = r'\\boxed\{(-?[\d,]+\.?\d*)\}'

# Pattern 3: Flexible - extract numeric values (fallback)
FLEXIBLE_PATTERN = r'(-?\$?[\d,]+\.?\d*)'


def normalize_answer(answer: str) -> str:
    """Normalize answer by removing commas, dollar signs, trailing periods.

    Matches lm-eval-harness regexes_to_ignore: [",", "\\$", "\\.(?!\\d)"]
    """
    if answer is None:
        return None
    # Remove commas (thousands separators)
    answer = answer.replace(',', '')
    # Remove dollar signs
    answer = answer.replace('$', '')
    # Remove trailing period (but not decimal point)
    answer = re.sub(r'\.(?!\d)', '', answer)
    return answer.strip()


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from model output using robust methodology.

    Priority order (matches Qwen output formats):
    1. #### (number) - standard GSM8K format
    2. \\boxed{number} - LaTeX format (common in Qwen math outputs)
    3. Last number in text - fallback for "The answer is X"
    """
    if not text:
        return None

    # Method 1: Try to find #### marker (strict GSM8K format)
    match = re.search(STRICT_PATTERN, text)
    if match:
        return normalize_answer(match.group(1))

    # Method 2: Try to find \boxed{} (LaTeX format, common in Qwen)
    match = re.search(BOXED_PATTERN, text)
    if match:
        return normalize_answer(match.group(1))

    # Method 3: Fallback - extract all numbers, take the last one
    # This handles "The answer is 42" format
    numbers = re.findall(FLEXIBLE_PATTERN, text)
    # Filter out empty matches and pure punctuation
    numbers = [n for n in numbers if n and re.search(r'\d', n)]
    if numbers:
        return normalize_answer(numbers[-1])

    return None


def extract_ground_truth(answer_text: str) -> str:
    """Extract ground truth from GSM8K answer field.

    GSM8K format: "reasoning steps... #### ANSWER"
    """
    if not answer_text:
        raise ValueError("Empty ground truth text")

    # Standard GSM8K format: answer after ####
    match = re.search(STRICT_PATTERN, answer_text)
    if match:
        return normalize_answer(match.group(1))

    # Fallback: last number in text
    numbers = re.findall(FLEXIBLE_PATTERN, answer_text)
    numbers = [n for n in numbers if n and re.search(r'\d', n)]
    if numbers:
        return normalize_answer(numbers[-1])

    raise ValueError(f"Could not extract ground truth from: {answer_text!r}")


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """Compare answers using exact string match after normalization.

    Matches lm-eval-harness exact_match metric.
    """
    if predicted is None or ground_truth is None:
        return False

    # Normalize both
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact string match (standard)
    if pred_norm == gt_norm:
        return True

    # Numeric comparison for floating point tolerance
    try:
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)
        # Use small tolerance for float comparison
        return abs(pred_num - gt_num) < 1e-5
    except (ValueError, TypeError):
        return False


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
        """Evaluate batch of outputs against ground truth.

        Uses lm-eval-harness standard:
        1. Extract answer after #### or last number
        2. Normalize (remove $, commas, trailing periods)
        3. Exact string match
        """
        if len(outputs) != len(indices):
            raise ValueError("outputs and indices must have the same length")

        correct = 0
        details = []

        for output, idx in zip(outputs, indices):
            example = self.dataset[idx]
            gt = extract_ground_truth(example['answer'])
            pred = extract_answer(output)

            is_correct = compare_answers(pred, gt)
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
