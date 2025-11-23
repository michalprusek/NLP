"""
Main script to run HbBoPs on GSM8K

Usage:
    cd hbbops
    uv run python run_hbbops.py --model Qwen/Qwen2.5-7B-Instruct
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import re
import numpy as np
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hbbops.hbbops import HbBoPs, Prompt
from src.llm_client import create_llm_client


def extract_answer(text: str) -> float:
    """
    Extract numerical answer from LLM output

    Tries multiple patterns:
    1. final_answer: NUMBER
    2. #### NUMBER
    3. \\boxed{NUMBER}
    4. Last number in text
    """
    patterns = [
        (r'final_answer:\s*(-?\d+\.?\d*)', re.IGNORECASE),
        (r'####\s*(-?\d+\.?\d*)', 0),
        (r'\\boxed\{(-?\d+\.?\d*)\}', 0)
    ]

    for pattern, flags in patterns:
        match = re.search(pattern, text, flags) if flags else re.search(pattern, text)
        if match:
            return float(match.group(1))

    # Fallback: last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return float(numbers[-1]) if numbers else None


def exact_match_with_tolerance(pred: float, gold: float, tolerance: float = 1e-4) -> bool:
    """Check if prediction matches gold answer within tolerance"""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < tolerance


class GSM8KEvaluator:
    """Evaluator for GSM8K prompts"""

    def __init__(self, llm_client, debug: bool = False):
        self.llm_client = llm_client
        self.debug = debug

    def _extract_gold_answer(self, answer_str: str) -> Optional[float]:
        """Extract numerical answer from gold answer string"""
        match = re.search(r'####\s*(-?\d+\.?\d*)', answer_str)
        return float(match.group(1)) if match else None

    def __call__(self, prompt: Prompt, validation_data: list) -> float:
        """
        Evaluate prompt on validation data

        Args:
            prompt: Prompt object
            validation_data: List of dicts with 'question' and 'answer'

        Returns:
            Validation error (fraction of incorrect answers)
        """
        errors = 0
        total = len(validation_data)

        # Prepare all prompts
        full_prompts = []
        for example in validation_data:
            question = example['question']
            full_prompts.append(f"Question: {question}\n\n{str(prompt)}\n\nAnswer:")

        # Batch generation
        try:
            responses = self.llm_client.generate_batch(full_prompts, max_tokens=1024)
        except Exception as e:
            if self.debug:
                print(f"Warning: LLM batch generation failed: {e}")
            return 1.0  # Return 100% error on failure

        # Evaluate responses
        for i, example in enumerate(validation_data):
            gold_answer_str = example['answer']
            gold_answer = self._extract_gold_answer(gold_answer_str)
            
            if gold_answer is None:
                if self.debug:
                    print(f"Warning: Could not extract gold answer from: {gold_answer_str}")
                errors += 1
                continue

            response = responses[i]
            
            # Extract predicted answer
            try:
                pred_answer = extract_answer(response)
            except Exception as e:
                if self.debug:
                    print(f"Warning: Failed to extract answer from response: {e}")
                errors += 1
                continue

            # Check if correct
            if not exact_match_with_tolerance(pred_answer, gold_answer):
                errors += 1

            if self.debug:
                print(f"\nQuestion: {example['question']}")
                print(f"Gold: {gold_answer}")
                print(f"Predicted: {pred_answer}")
                print(f"Correct: {exact_match_with_tolerance(pred_answer, gold_answer)}")
                print(f"Response: {response[:200]}...")

        return errors / total


def load_instructions(file_path: str) -> list:
    """Load instructions from TXT file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    return [
        re.sub(r'^\d+\.\s*', '', line.strip())
        for line in lines
        if line.strip() and not line.startswith('#') and line[0].isdigit()
    ]


def load_exemplars(file_path: str) -> list:
    """Load exemplars from TXT file"""
    with open(file_path, 'r') as f:
        content = f.read()

    exemplars = []
    for block in content.split('=' * 80):
        if not block.strip():
            continue

        examples = _parse_qa_examples(block)
        if examples:
            exemplars.append('\n\n'.join(examples))

    return exemplars


def _parse_qa_examples(block: str) -> list:
    """Parse Q&A examples from a text block"""
    lines = [l for l in block.split('\n') if not l.startswith('#')]
    examples = []
    current_q = None

    for line in lines:
        line = line.strip()
        if line.startswith('Q:'):
            current_q = line[2:].strip()
        elif line.startswith('A:') and current_q:
            examples.append(f"Q: {current_q}\nA: {line[2:].strip()}")
            current_q = None

    return examples


def save_results(
    output_dir: Path, args, instructions: list, exemplars: list,
    hbbops: HbBoPs, best_prompt: Prompt, best_val_error: float, test_error: float
) -> None:
    """Save optimization results to JSON and TXT files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "method": "HbBoPs",
        "model": args.model,
        "backend": args.backend,
        "config": {
            "bmin": args.bmin,
            "eta": args.eta,
            "encoder": args.encoder,
            "num_instructions": len(instructions),
            "num_exemplars": len(exemplars),
            "num_prompts": len(hbbops.prompts)
        },
        "best_prompt": {
            "instruction_id": best_prompt.instruction_id,
            "exemplar_id": best_prompt.exemplar_id,
            "instruction": best_prompt.instruction,
            "exemplar": best_prompt.exemplar
        },
        "validation_error": best_val_error,
        "test_error": test_error,
        "design_data_size": len(hbbops.design_data),
        "num_evaluations": len(hbbops.evaluation_cache)
    }

    # Save JSON
    json_path = output_dir / f"hbbops_{timestamp}.json"
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved:")
        print(f"  {json_path}")
    except Exception as e:
        print(f"\nWarning: Failed to save JSON results: {e}")

    # Save TXT
    txt_path = output_dir / f"hbbops_{timestamp}.txt"
    try:
        with open(txt_path, 'w') as f:
            f.write("HbBoPs Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Backend: {args.backend}\n\n")
            f.write(f"Validation error: {best_val_error:.4f} ({best_val_error * 100:.2f}%)\n")
            f.write(f"Test error: {test_error:.4f} ({test_error * 100:.2f}%)\n\n")
            f.write("Best Prompt:\n")
            f.write("-" * 80 + "\n")
            f.write(str(best_prompt))
        print(f"  {txt_path}")
    except Exception as e:
        print(f"Warning: Failed to save TXT results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run HbBoPs on GSM8K')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--backend', type=str, default='vllm',
                        choices=['vllm', 'transformers', 'claude'],
                        help='Backend for LLM')
    parser.add_argument('--bmin', type=int, default=10,
                        help='Minimum validation instances for Hyperband')
    parser.add_argument('--eta', type=float, default=2.0,
                        help='Halving parameter for Hyperband')
    parser.add_argument('--encoder', type=str, default='bert-base-uncased',
                        help='Encoder model for embeddings')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device for computation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    try:
        with open(data_dir / "validation.json", 'r') as f:
            validation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: validation.json not found in {data_dir}")
        print("Please run setup script first to create data splits.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in validation.json: {e}")
        return

    try:
        with open(data_dir / "test.json", 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: test.json not found in {data_dir}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in test.json: {e}")
        return

    print(f"Validation examples: {len(validation_data)}")
    print(f"Test examples: {len(test_data)}")

    # Load instructions and exemplars
    print("\nLoading instructions and exemplars...")
    try:
        instructions = load_instructions(str(script_dir / "instructions.txt"))
        if not instructions:
            print("Error: No instructions found in instructions.txt")
            return
    except FileNotFoundError:
        print(f"Error: instructions.txt not found in {script_dir}")
        return
    except Exception as e:
        print(f"Error loading instructions: {e}")
        return

    try:
        exemplars = load_exemplars(str(script_dir / "examples.txt"))
        if not exemplars:
            print("Error: No exemplars found in examples.txt")
            return
    except FileNotFoundError:
        print(f"Error: examples.txt not found in {script_dir}")
        return
    except Exception as e:
        print(f"Error loading exemplars: {e}")
        return

    print(f"Instructions: {len(instructions)}")
    print(f"Exemplars: {len(exemplars)}")

    # Initialize LLM client
    print(f"\nInitializing LLM client ({args.backend})...")
    try:
        llm_client = create_llm_client(
            model_name=args.model,
            backend=args.backend,
            device=args.device
        )
    except Exception as e:
        print(f"Error: Failed to initialize LLM client: {e}")
        return

    # Create evaluator
    evaluator = GSM8KEvaluator(llm_client, debug=args.debug)

    # Initialize HbBoPs
    print("\nInitializing HbBoPs...")
    try:
        hbbops = HbBoPs(
            instructions=instructions,
            exemplars=exemplars,
            validation_data=validation_data,
            llm_evaluator=evaluator,
            encoder_name=args.encoder,
            bmin=args.bmin,
            eta=args.eta,
            device=args.device
        )
    except Exception as e:
        print(f"Error: Failed to initialize HbBoPs: {e}")
        return

    # Run optimization
    best_prompt, best_val_error = hbbops.run_hyperband(verbose=True)

    # Evaluate on test set
    print("\nEvaluating best prompt on test set...")
    test_error = evaluator(best_prompt, test_data)

    print(f"\nFinal Results:")
    print(f"  Validation error: {best_val_error:.4f} ({best_val_error * 100:.2f}%)")
    print(f"  Test error: {test_error:.4f} ({test_error * 100:.2f}%)")
    print(f"\nBest Prompt:")
    print(f"  Instruction ID: {best_prompt.instruction_id}")
    print(f"  Exemplar ID: {best_prompt.exemplar_id}")
    print(f"\nInstruction:")
    print(f"  {best_prompt.instruction}")
    print(f"\nExemplar (first 200 chars):")
    print(f"  {best_prompt.exemplar[:200]}...")

    # Save results
    save_results(
        output_dir, args, instructions, exemplars, hbbops,
        best_prompt, best_val_error, test_error
    )


if __name__ == "__main__":
    main()
