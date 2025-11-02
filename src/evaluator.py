"""GSM8K evaluator with answer extraction and scoring (robust)"""
import re
from typing import Dict, List, Any, Optional
from decimal import Decimal, InvalidOperation, localcontext, Context
from datasets import load_from_disk

# Unicode spaces often used as thousands separators
_THIN_SPACES = "\u00A0\u2009\u202F"  # nbsp, thin space, narrow nbsp

# Regex for capturing numbers in text (allows thousands separators and %, no trailing period)
NUMBER_PATTERN = r"""
[-+]?
(?:
    \d{1,3}(?:[,\s'""" + _THIN_SPACES + r"""]\d{3})+  # 1,234 or 1 234 or 1'234
    |
    \d+                                             # or simple integer
)
(?:[.,]\d+)?                                        # optional decimal part
(?:[eE][-+]?\d+)?                                   # optional scientific notation
%?                                                 # optional percentage
"""

_NUM_RE = re.compile(NUMBER_PATTERN, re.VERBOSE)

# Compact version of NUMBER_PATTERN for use in dynamic regex (without VERBOSE)
_NUM_COMPACT = r'[-+]?(?:\d{1,3}(?:[,\s\'' + _THIN_SPACES + r']\d{3})+|\d+)(?:[.,]\d+)?(?:[eE][-+]?\d+)?%?'

# Precompiled patterns for extract_answer and extract_ground_truth
_GROUND_TRUTH_PATTERN = re.compile(r'####\s*(' + _NUM_COMPACT + r')')
_FINAL_ANSWER_PATTERN = re.compile(r'final_answer\s*:\s*(' + _NUM_COMPACT + r')', re.IGNORECASE)
_HASH_PATTERN = re.compile(r'####\s*(' + _NUM_COMPACT + r')')
_BOXED_PATTERN = re.compile(r'\\boxed\{\s*(' + _NUM_COMPACT + r')\s*\}')
_ANSWER_IS_PATTERN = re.compile(r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(' + _NUM_COMPACT + r')', re.IGNORECASE)
_ANSWER_COLON_PATTERN = re.compile(r'(?:final\s+)?answer\s*:\s*(' + _NUM_COMPACT + r')', re.IGNORECASE)
_THEREFORE_PATTERN = re.compile(r'therefore,?\s*(?:the\s+answer\s+is\s*)?(' + _NUM_COMPACT + r')', re.IGNORECASE)
_SO_THUS_PATTERN = re.compile(r'(?:so|thus),?\s*(?:the\s+answer\s+is\s*)?(' + _NUM_COMPACT + r')', re.IGNORECASE)
_RESULT_PATTERN = re.compile(r'result\s*:?\s*(' + _NUM_COMPACT + r')', re.IGNORECASE)
_EQUALS_PATTERN = re.compile(r'=\s*(' + _NUM_COMPACT + r')\s*(?:$|\n)')

# Units/currencies to remove at end of token (not in middle of number)
_UNIT_SUFFIX_RE = re.compile(
    r'(?:km|m|cm|mm|kg|g|mg|lb|oz|l|ml|h|min|s|usd|eur|czk|gbp|jpy|cny|k|°c|°f|k)$',
    re.IGNORECASE
)

_CURRENCY_CHARS = r"$€£¥₹₽₩₺₫₪฿₴₦"

def _clean_grouping_and_decimal(s: str) -> str:
    """
    Proper handling of comma/period: distinguishes decimal vs. thousands separators.
    """
    s = s.strip()
    # Negative in parentheses: (123) -> -123
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]

    # Remove currencies scattered in string
    s = re.sub('[' + re.escape(_CURRENCY_CHARS) + ']', '', s)

    # Remove thin spaces and regular spaces used as thousands separators
    for ch in _THIN_SPACES + " ":
        s = s.replace(ch, '')

    # Apostrophes as thousands separators
    s = s.replace("'", "")

    # Heuristic for commas vs. periods:
    if ',' in s and '.' in s:
        # Take the last one as decimal separator, remove the other
        last_dot = s.rfind('.')
        last_comma = s.rfind(',')
        if last_comma > last_dot:
            # comma is decimal, periods are thousands
            s = s.replace('.', '')
            s = s.replace(',', '.')
        else:
            # period is decimal, commas are thousands
            s = s.replace(',', '')
    elif ',' in s:
        # Only commas: decide based on thousands pattern
        if re.fullmatch(r'\d{1,3}(?:,\d{3})+(?:,\d+)?', s):
            # Looks like thousands → remove commas
            s = s.replace(',', '')
        else:
            # Treat comma as decimal
            s = s.replace(',', '.')
    # Otherwise only periods → nothing

    # Remove trailing units (keep % for later conversion)
    # E.g. "1200m" → "1200"
    while True:
        m = _UNIT_SUFFIX_RE.search(s)
        if m:
            # Only if it's truly at the end
            if m.end() == len(s):
                s = s[:m.start()]
                s = s.strip()
                continue
        break

    return s.strip()

def normalize_number(text: str) -> Optional[str]:
    """
    Normalizes numeric string to deterministic representation.
    - safely via Decimal (no float)
    - correct interpretation of comma/period/thousands separators
    - support for percentage and scientific notation
    """
    if not text:
        return None

    s = text.strip()
    is_percentage = '%' in s
    # remove all percentages (conversion done below)
    s = s.replace('%', '')
    s = _clean_grouping_and_decimal(s)

    # empty after cleaning?
    if not s or s in ['+', '-', '.', '+.', '-.']:
        return None

    # Safe parsing
    try:
        with localcontext(Context(prec=50)):  # high precision
            d = Decimal(s)
            if is_percentage:
                d = d / Decimal(100)

            # If exactly integral, return integer
            try:
                _ = d.to_integral_exact()  # raises InvalidOperation if not exact integer
                return str(d.quantize(Decimal(1)))
            except InvalidOperation:
                # Decimal: canonical string without unnecessary zeros and without scientific notation
                s_fixed = format(d.normalize(), 'f')
                # remove trailing zeros and period
                if '.' in s_fixed:
                    s_fixed = s_fixed.rstrip('0').rstrip('.')
                return s_fixed if s_fixed else '0'
    except Exception:
        return None

def _is_int_str(s: str) -> bool:
    return bool(re.fullmatch(r'[-+]?\d+', s))

def compare_numbers(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """
    Strict comparison:
      - If both integers → must be textually identical (after normalization).
      - Otherwise numerically with small absolute tolerance.
    """
    if predicted == ground_truth:
        return True

    # integer vs. integer
    if _is_int_str(predicted) and _is_int_str(ground_truth):
        return predicted == ground_truth

    # numerical comparison with tolerance
    try:
        with localcontext(Context(prec=50)):
            pd = Decimal(predicted)
            gd = Decimal(ground_truth)
            diff = abs(pd - gd)
            return diff <= Decimal(str(tolerance))
    except Exception:
        return False

def extract_answer(text: str, verbose: bool = False) -> Optional[str]:
    """
    Extract final number from model output (priority):
    1) final_answer: <num>
    2) #### <num>
    3) \\boxed{<num>}
    4) common phrases (the answer is, result:, therefore ...)
    5) last number in text (fallback)
    """
    if not text:
        return None

    if verbose:
        print(f"Extracting from: {text[:200]}...")

    t = text.strip()

    # 1) explicit final_answer
    m = _FINAL_ANSWER_PATTERN.search(t)
    if m:
        res = normalize_number(m.group(1))
        if verbose and res:
            print(f"  Found via final_answer: {res}")
        if res is not None:
            return res

    # 2) #### NUMBER
    m = _HASH_PATTERN.search(t)
    if m:
        res = normalize_number(m.group(1))
        if verbose and res:
            print(f"  Found via ####: {res}")
        if res is not None:
            return res

    # 3) \boxed{NUMBER}
    m = _BOXED_PATTERN.search(t)
    if m:
        res = normalize_number(m.group(1))
        if verbose and res:
            print(f"  Found via \\boxed{{}}: {res}")
        if res is not None:
            return res

    # 4) Other common patterns (case-insensitive)
    patterns_to_try = [
        (_ANSWER_IS_PATTERN, "answer is"),
        (_ANSWER_COLON_PATTERN, "answer:"),
        (_THEREFORE_PATTERN, "therefore"),
        (_SO_THUS_PATTERN, "so/thus"),
        (_RESULT_PATTERN, "result:"),
        (_EQUALS_PATTERN, "="),
    ]

    for pattern, name in patterns_to_try:
        m = pattern.search(t)
        if m:
            res = normalize_number(m.group(1))
            if verbose and res:
                print(f"  Found via {name}: {res}")
            if res is not None:
                return res

    # 5) Last number in text (fallback)
    nums = _NUM_RE.findall(t)
    if nums:
        res = normalize_number(nums[-1])
        if verbose and res:
            print(f"  Found last number: {res}")
        if res is not None:
            return res

    if verbose:
        print("  FAILED to extract answer!")
    return None

def extract_ground_truth(answer_text: str) -> str:
    """
    Ground truth from 'answer' field (GSM8K standardly has '#### NUMBER').
    If missing, falls back to last number (fallback).
    """
    if not answer_text:
        raise ValueError("Empty ground truth text")

    m = _GROUND_TRUTH_PATTERN.search(answer_text)
    if m:
        res = normalize_number(m.group(1))
        if res is not None:
            return res

    # Fallback (should not be needed often)
    nums = _NUM_RE.findall(answer_text)
    if nums:
        res = normalize_number(nums[-1])
        if res is not None:
            return res

    raise ValueError(f"Could not extract ground truth from: {answer_text!r}")

class GSM8KEvaluator:
    """Evaluator for GSM8K dataset (strict EM on numbers with decimal tolerance)"""

    def __init__(self, dataset_path: str = "datasets/gsm8k", split: str = "test", debug: bool = False):
        """
        Args:
            dataset_path: Path to saved HF dataset (load_from_disk)
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
        Evaluate batch of outputs against ground truth (strict match).
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

            is_corr = (pred is not None and compare_numbers(pred, gt))
            if pred is None:
                failed_extractions += 1
            if is_corr:
                correct += 1

            if (verbose or self.debug) and i < 3:
                print(f"\n--- Example {i+1} ---")
                question_preview = example['question'][:120].replace('\n', ' ')
                output_preview = output[:200].replace('\n', ' ')
                print(f"Q: {question_preview}")
                print(f"Out: {output_preview}")
                print(f"GT: {gt} | Pred: {pred} | Correct: {is_corr}")

            details.append({
                'idx': idx,
                'question': example['question'],
                'ground_truth': gt,
                'predicted': pred,
                'correct': is_corr,
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

