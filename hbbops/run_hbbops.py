"""
Run HbBoPs on GSM8K

Usage:
    cd hbbops
    uv run python run_hbbops.py --model Qwen/Qwen2.5-7B-Instruct
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import re
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hbbops.hbbops import HbBoPs, Prompt
from src.llm_client import create_llm_client


# Simple number pattern - matches integers and decimals (including negative)
NUMBER_PATTERN = r'[-+]?\d+(?:[.,]\d+)?'


def extract_answer(text: str) -> str | None:
    """Extract last number from model output."""
    if not text:
        return None
    numbers = re.findall(NUMBER_PATTERN, text)
    return numbers[-1] if numbers else None


def compare_numbers(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """Compare two numbers with tolerance."""
    if predicted == ground_truth:
        return True
    try:
        pred_clean = predicted.replace(',', '')
        gt_clean = ground_truth.replace(',', '')
        return abs(float(pred_clean) - float(gt_clean)) <= tolerance
    except (ValueError, TypeError):
        return False


class GSM8KEvaluator:
    """Evaluator for GSM8K prompts"""

    def __init__(self, llm_client, debug: bool = False):
        self.llm_client = llm_client
        self.debug = debug

    def __call__(self, prompt: Prompt, validation_data: list) -> float:
        """Evaluate prompt on validation data, returns error rate"""
        errors = 0
        prompts = [f"Question: {ex['question']}\n\n{str(prompt)}\n\nAnswer:" for ex in validation_data]

        try:
            responses = self.llm_client.generate_batch(prompts, max_tokens=1024)
        except Exception as e:
            if self.debug:
                print(f"LLM error: {e}")
            return 1.0

        for i, ex in enumerate(validation_data):
            gold = re.findall(NUMBER_PATTERN, ex['answer'])
            gold = gold[-1] if gold else None

            pred = extract_answer(responses[i])

            if gold is None or pred is None or not compare_numbers(pred, gold):
                errors += 1

            if self.debug:
                print(f"Q: {ex['question'][:50]}... Gold: {gold}, Pred: {pred}")

        return errors / len(validation_data)


def load_instructions(file_path: str) -> list:
    """Load instructions from TXT file"""
    with open(file_path, 'r') as f:
        return [re.sub(r'^\d+\.\s*', '', line.strip())
                for line in f if line.strip() and not line.startswith('#') and line[0].isdigit()]


def load_exemplars(file_path: str) -> list:
    """Load exemplars from TXT file"""
    with open(file_path, 'r') as f:
        content = f.read()

    exemplars = []
    for block in content.split('=' * 80):
        if not block.strip():
            continue
        lines = [l for l in block.split('\n') if not l.startswith('#')]
        examples, current_q = [], None
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                current_q = line[2:].strip()
            elif line.startswith('A:') and current_q:
                examples.append(f"Q: {current_q}\nA: {line[2:].strip()}")
                current_q = None
        if examples:
            exemplars.append('\n\n'.join(examples))
    return exemplars


def main():
    parser = argparse.ArgumentParser(description='Run HbBoPs on GSM8K')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--backend', type=str, default='vllm', choices=['vllm', 'transformers', 'claude'])
    parser.add_argument('--bmin', type=int, default=10, help='Min validation instances (default: 10)')
    parser.add_argument('--eta', type=float, default=2.0, help='Halving parameter (default: 2.0)')
    parser.add_argument('--encoder', type=str, default='bert-base-uncased')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    with open(script_dir / "data/validation.json") as f:
        validation_data = json.load(f)
    with open(script_dir / "data/test.json") as f:
        test_data = json.load(f)

    print(f"Validation: {len(validation_data)}, Test: {len(test_data)}")

    # Load prompts
    instructions = load_instructions(str(script_dir / "instructions.txt"))
    exemplars = load_exemplars(str(script_dir / "examples.txt"))
    print(f"Instructions: {len(instructions)}, Exemplars: {len(exemplars)}")

    # Initialize
    print(f"\nInitializing LLM ({args.backend})...")
    llm_client = create_llm_client(args.model, args.backend, args.device)
    evaluator = GSM8KEvaluator(llm_client, args.debug)

    print("\nInitializing HbBoPs...")
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

    # Run optimization
    best_prompt, best_val_error = hbbops.run_hyperband()

    # Evaluate on test
    print("\nEvaluating on test set...")
    test_error = evaluator(best_prompt, test_data)

    # Results
    print(f"\n{'=' * 60}")
    print(f"Validation error: {best_val_error:.4f} ({best_val_error * 100:.2f}%)")
    print(f"Test error: {test_error:.4f} ({test_error * 100:.2f}%)")
    print(f"Best prompt: instruction={best_prompt.instruction_id}, exemplar={best_prompt.exemplar_id}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "method": "HbBoPs",
        "model": args.model,
        "config": {"bmin": args.bmin, "eta": args.eta},
        "best_prompt": {
            "instruction_id": best_prompt.instruction_id,
            "exemplar_id": best_prompt.exemplar_id,
            "instruction": best_prompt.instruction,
            "exemplar": best_prompt.exemplar
        },
        "validation_error": best_val_error,
        "test_error": test_error,
        "num_evaluations": len(hbbops.evaluation_cache)
    }

    with open(output_dir / f"hbbops_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/hbbops_{timestamp}.json")


if __name__ == "__main__":
    main()
