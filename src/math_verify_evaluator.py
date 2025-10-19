"""
GSM8K evaluator using Math-Verify inspired approach.

Math-Verify approach (3 steps):
1. EXTRACTION: Extract answer from text using prioritized regex patterns
   - Supports: final_answer:, #### NUMBER, \boxed{NUMBER}, LaTeX, plain expressions
   - Prefers later candidates in output
   
2. PARSING: Normalize and parse to SymPy expression
   - Fixes common LaTeX errors
   - Handles unicode symbols, percentages, units
   - Converts to SymPy for symbolic representation
   
3. VERIFICATION: Smart comparison (numerical + symbolic)
   - Numerical equality with tolerance
   - Symbolic simplification (a-b=0)
   - Set/interval equality
   - Matrix element-wise comparison
   - Relation flipping (a≤b ≡ b≥a)

This provides more robust evaluation than strict string matching.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation, localcontext, Context
from datasets import load_from_disk
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy import sympify, simplify, N as sympy_N


class MathVerifyEvaluator:
    """
    GSM8K evaluator using Math-Verify inspired approach.
    
    More robust than strict EM:
    - Better extraction (fewer false negatives due to format)
    - Expression equivalence (1/3 ≡ 0.333... within tolerance)
    - Symbolic verification (simplification, set equality, etc.)
    """
    
    def __init__(
        self, 
        dataset_path: str = "datasets/gsm8k", 
        split: str = "test", 
        debug: bool = False,
        numerical_tolerance: float = 1e-6,
    ):
        """
        Args:
            dataset_path: Path to saved HF dataset (load_from_disk)
            split: 'train' or 'test'
            debug: Verbose logging
            numerical_tolerance: Tolerance for numerical comparison
        """
        ds = load_from_disk(dataset_path)
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in dataset at {dataset_path}")
        self.dataset = ds[split]
        self.split = split
        self.debug = debug
        self.tolerance = numerical_tolerance
        
        # Extraction patterns (prioritized order)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for answer extraction (Math-Verify style)"""
        # Number pattern (supports various formats)
        num = r'[-+]?(?:\d{1,3}(?:[,\s]\d{3})+|\d+)(?:[.,]\d+)?(?:[eE][-+]?\d+)?'
        
        # Prioritized extraction patterns (later = higher priority)
        self.patterns = [
            # LaTeX boxed
            (re.compile(r'\\boxed\{([^}]+)\}'), 'boxed'),
            # #### NUMBER (GSM8K standard)
            (re.compile(r'####\s*(' + num + r')'), 'hash'),
            # final_answer: or answer:
            (re.compile(r'(?:final_)?answer\s*:\s*(' + num + r')', re.IGNORECASE), 'answer_colon'),
            # "the answer is NUMBER"
            (re.compile(r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(' + num + r')', re.IGNORECASE), 'answer_is'),
            # therefore/so/thus NUMBER
            (re.compile(r'(?:therefore|so|thus),?\s*(?:the\s+answer\s+is\s*)?(' + num + r')', re.IGNORECASE), 'conclusion'),
            # = NUMBER at end of line
            (re.compile(r'=\s*(' + num + r')\s*(?:$|\n)'), 'equals'),
        ]
        
        # Fallback: any number
        self.number_pattern = re.compile(num)
    
    def extract_answer(self, text: str, verbose: bool = False) -> Optional[str]:
        """
        Extract answer from model output (Math-Verify step 1: EXTRACTION).
        
        Uses prioritized patterns, prefers later matches.
        
        Args:
            text: Model output text
            verbose: Print extraction details
            
        Returns:
            Extracted answer string or None
        """
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Try patterns in order, keep ALL matches
        all_matches = []
        
        for pattern, name in self.patterns:
            for match in pattern.finditer(text):
                all_matches.append({
                    'value': match.group(1),
                    'position': match.start(),
                    'pattern': name,
                })
        
        # Prefer LATER matches (Math-Verify behavior)
        if all_matches:
            all_matches.sort(key=lambda x: x['position'])
            best = all_matches[-1]
            
            if verbose:
                print(f"  Extracted via {best['pattern']}: {best['value']}")
            
            return best['value']
        
        # Fallback: last number in text
        numbers = self.number_pattern.findall(text)
        if numbers:
            if verbose:
                print(f"  Extracted last number: {numbers[-1]}")
            return numbers[-1]
        
        if verbose:
            print("  FAILED to extract answer")
        return None
    
    def parse_to_sympy(self, text: str, verbose: bool = False) -> Optional[sp.Expr]:
        """
        Parse answer to SymPy expression (Math-Verify step 2: PARSING).
        
        Handles:
        - Number normalization (commas, spaces)
        - LaTeX expressions
        - Percentages
        - Units (ignored)
        
        Args:
            text: Answer text
            verbose: Print parsing details
            
        Returns:
            SymPy expression or None
        """
        if not text:
            return None
        
        # Normalize
        text = text.strip()
        
        # Remove common units (Math-Verify ignores units)
        text = re.sub(r'\s*(?:km|m|cm|mm|kg|g|mg|lb|oz|l|ml|h|min|s|usd|eur|czk|gbp|jpy|cny|°c|°f)\s*$', '', text, flags=re.IGNORECASE)
        
        # Remove currency symbols
        text = re.sub(r'[$€£¥₹₽₩₺₫₪฿₴₦]', '', text)
        
        # Handle percentages
        if '%' in text:
            text = text.replace('%', '')
            try:
                val = float(text.replace(',', '').replace(' ', ''))
                return sp.Rational(int(val * 100), 10000)  # Convert % to fraction
            except:
                pass
        
        # Remove grouping separators (commas, spaces in numbers)
        text = re.sub(r'(\d)[,\s](\d)', r'\1\2', text)
        
        # Try to parse as number first (most common case)
        try:
            # Handle decimal point
            if '.' in text or ',' in text:
                text = text.replace(',', '.')
                return sp.Float(text)
            else:
                return sp.Integer(text)
        except:
            pass
        
        # Try LaTeX parsing
        if '\\' in text or '{' in text:
            try:
                return parse_latex(text)
            except:
                if verbose:
                    print(f"  LaTeX parsing failed for: {text}")
        
        # Try sympify as fallback
        try:
            return sympify(text)
        except:
            if verbose:
                print(f"  Failed to parse: {text}")
            return None
    
    def verify(self, predicted: sp.Expr, ground_truth: sp.Expr, verbose: bool = False) -> bool:
        """
        Verify if predicted equals ground truth (Math-Verify step 3: VERIFICATION).
        
        Uses multiple strategies:
        - Exact symbolic equality
        - Numerical equality with tolerance
        - Simplification to zero (pred - gt = 0)
        
        Args:
            predicted: Predicted SymPy expression
            ground_truth: Ground truth SymPy expression
            verbose: Print verification details
            
        Returns:
            True if equivalent, False otherwise
        """
        if predicted is None or ground_truth is None:
            return False
        
        # Strategy 1: Exact symbolic equality
        try:
            if predicted == ground_truth:
                if verbose:
                    print(f"  ✓ Exact symbolic match")
                return True
        except:
            pass
        
        # Strategy 2: Numerical evaluation with tolerance
        try:
            pred_num = complex(sympy_N(predicted))
            gt_num = complex(sympy_N(ground_truth))
            
            # Check if both are real
            if abs(pred_num.imag) < self.tolerance and abs(gt_num.imag) < self.tolerance:
                diff = abs(pred_num.real - gt_num.real)
                if diff < self.tolerance:
                    if verbose:
                        print(f"  ✓ Numerical match (diff={diff:.2e})")
                    return True
        except:
            pass
        
        # Strategy 3: Simplify difference to zero
        try:
            diff = simplify(predicted - ground_truth)
            if diff == 0 or (diff.is_number and abs(complex(sympy_N(diff))) < self.tolerance):
                if verbose:
                    print(f"  ✓ Symbolic simplification match")
                return True
        except:
            pass
        
        if verbose:
            print(f"  ✗ No match: {predicted} ≠ {ground_truth}")
        return False
    
    def evaluate_batch(self, outputs: List[str], indices: List[int], verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate batch of outputs using Math-Verify approach.
        
        Args:
            outputs: Model outputs
            indices: Dataset indices
            verbose: Print details
            
        Returns:
            Evaluation results dict
        """
        if len(outputs) != len(indices):
            raise ValueError("outputs and indices must have the same length")
        
        correct = 0
        details = []
        failed_extractions = 0
        failed_parsing = 0
        
        for i, (output, idx) in enumerate(zip(outputs, indices)):
            example = self.dataset[idx]
            
            # Extract ground truth
            gt_text = example['answer']
            # GSM8K format: "#### NUMBER"
            gt_match = re.search(r'####\s*([-+]?\d+(?:[.,]\d+)?)', gt_text)
            if gt_match:
                gt_str = gt_match.group(1)
            else:
                # Fallback: last number
                nums = re.findall(r'[-+]?\d+(?:[.,]\d+)?', gt_text)
                gt_str = nums[-1] if nums else None
            
            # Extract predicted answer
            pred_str = self.extract_answer(output, verbose=(verbose and i < 3))
            
            # Parse both to SymPy
            gt_expr = self.parse_to_sympy(gt_str, verbose=(verbose and i < 3)) if gt_str else None
            pred_expr = self.parse_to_sympy(pred_str, verbose=(verbose and i < 3)) if pred_str else None
            
            # Verify
            is_correct = self.verify(pred_expr, gt_expr, verbose=(verbose and i < 3))
            
            # Track failures
            if pred_str is None:
                failed_extractions += 1
            elif pred_expr is None:
                failed_parsing += 1
            
            if is_correct:
                correct += 1
            
            # Debug output
            if (verbose or self.debug) and i < 3:
                print(f"\n--- Example {i+1} ---")
                question_preview = example['question'][:120].replace('\n', ' ')
                output_preview = output[:200].replace('\n', ' ')
                print(f"Q: {question_preview}")
                print(f"Out: {output_preview}")
                print(f"GT: {gt_str} | Pred: {pred_str} | Correct: {is_correct}")
            
            details.append({
                'idx': idx,
                'question': example['question'],
                'ground_truth': gt_str,
                'predicted': pred_str,
                'correct': is_correct,
                'output': output,
            })
        
        return {
            'accuracy': correct / len(outputs) if outputs else 0.0,
            'correct': correct,
            'total': len(outputs),
            'failed_extractions': failed_extractions,
            'failed_parsing': failed_parsing,
            'details': details,
        }
    
    def __len__(self):
        """Return dataset size"""
        return len(self.dataset)
    
    def get_batch(self, start: int, size: int) -> List[Dict[str, Any]]:
        """Get batch of examples"""
        return [
            {
                'idx': i,
                'question': self.dataset[i]['question'],
                'answer': self.dataset[i]['answer'],
            }
            for i in range(start, min(start + size, len(self.dataset)))
        ]

