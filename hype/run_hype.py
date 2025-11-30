#!/usr/bin/env python3
"""
HYPE - Hyperband Prompt Evolution

CLI runner for evolutionary prompt optimization on GSM8K.

Usage:
    # Basic run with local model
    uv run python hype/run_hype.py --model Qwen/Qwen2.5-7B-Instruct

    # With Claude for meta-optimization
    uv run python hype/run_hype.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --meta-model sonnet \
        --generations 5

    # Quick test
    uv run python hype/run_hype.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --generations 2 \
        --bmin 5
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Optional

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hype.evolution import HYPE, HYPEConfig
from src.llm_client import create_llm_client


def extract_answer(text: str) -> Optional[float]:
    """
    Extract numerical answer from LLM output.
    Tries multiple patterns: final_answer, ####, \\boxed, last number.
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

    # Fallback: last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return float(numbers[-1]) if numbers else None


def exact_match_with_tolerance(pred: float, gold: float, tolerance: float = 1e-4) -> bool:
    """Check if prediction matches gold within tolerance"""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < tolerance


class GSM8KEvaluator:
    """Evaluator for GSM8K math problems"""

    def __init__(self, llm_client, debug: bool = False):
        self.llm_client = llm_client
        self.debug = debug

    def _extract_gold_answer(self, answer_str: str) -> Optional[float]:
        """Extract numerical answer from gold answer string"""
        match = re.search(r'####\s*(-?\d+\.?\d*)', answer_str)
        return float(match.group(1)) if match else None

    def __call__(self, prompt, validation_data: list) -> float:
        """Evaluate prompt, return error rate (fraction incorrect)"""
        errors = 0
        total = len(validation_data)

        # Prepare prompts
        full_prompts = [
            f"Question: {ex['question']}\n\n{str(prompt)}\n\nAnswer:"
            for ex in validation_data
        ]

        # Batch generation
        try:
            responses = self.llm_client.generate_batch(full_prompts, max_tokens=1024)
        except Exception as e:
            if self.debug:
                print(f"Warning: Batch generation failed: {e}")
            return 1.0

        # Evaluate
        for i, ex in enumerate(validation_data):
            gold = self._extract_gold_answer(ex['answer'])
            if gold is None:
                errors += 1
                continue

            try:
                pred = extract_answer(responses[i])
                if not exact_match_with_tolerance(pred, gold):
                    errors += 1
            except:
                errors += 1

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


def resolve_model_alias(model: str) -> str:
    """Resolve model aliases to full names"""
    aliases = {
        'haiku': 'claude-haiku-4-5-20251001',
        'sonnet': 'claude-sonnet-4-5-20251022',
    }
    return aliases.get(model, model)


def main():
    parser = argparse.ArgumentParser(
        description='HYPE - Hyperband Prompt Evolution on GSM8K',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  uv run python hype/run_hype.py --model Qwen/Qwen2.5-7B-Instruct

  # With Claude meta-optimizer
  uv run python hype/run_hype.py --model Qwen/Qwen2.5-7B-Instruct --meta-model sonnet

  # Quick test (2 generations, small budget)
  uv run python hype/run_hype.py --model Qwen/Qwen2.5-7B-Instruct --generations 2 --bmin 5
        """
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Task model for evaluation')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['vllm', 'transformers', 'claude', 'deepinfra', 'auto'],
                        help='Backend for task model (auto detects API models)')
    parser.add_argument('--meta-model', type=str, default=None,
                        help='Meta-optimizer model (for generation). Aliases: haiku, sonnet')
    parser.add_argument('--meta-backend', type=str, default='auto',
                        help='Backend for meta-model')

    # Evolution arguments
    parser.add_argument('--generations', type=int, default=5,
                        help='Number of evolution generations')
    parser.add_argument('--new-instructions', type=int, default=3,
                        help='New instructions per generation (Method A)')
    parser.add_argument('--new-exemplars', type=int, default=3,
                        help='New exemplars per generation (Method C)')
    parser.add_argument('--no-recombination', action='store_true',
                        help='Disable Method B (recombination)')

    # HbBoPs arguments
    parser.add_argument('--bmin', type=int, default=10,
                        help='Minimum budget (validation instances)')
    parser.add_argument('--eta', type=float, default=2.0,
                        help='Hyperband halving parameter')
    parser.add_argument('--encoder', type=str, default='bert-base-uncased',
                        help='BERT encoder for embeddings')

    # Pool management
    parser.add_argument('--max-instructions', type=int, default=50,
                        help='Maximum instruction pool size')
    parser.add_argument('--max-exemplars', type=int, default=50,
                        help='Maximum exemplar pool size')

    # Data arguments
    parser.add_argument('--instructions-file', type=str, default=None,
                        help='Path to instructions file (default: hbbops/instructions.txt)')
    parser.add_argument('--exemplars-file', type=str, default=None,
                        help='Path to exemplars file (default: hbbops/examples.txt)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory (default: hbbops/data)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='hype/results',
                        help='Output directory for results')

    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu, mps)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                        help='GPU IDs to use (e.g., "0,1")')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90,
                        help='GPU memory utilization for vLLM (0.0-1.0)')

    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    hbbops_dir = base_dir / 'hbbops'
    data_dir = Path(args.data_dir) if args.data_dir else hbbops_dir / 'data'

    instructions_file = args.instructions_file or str(hbbops_dir / 'instructions.txt')
    exemplars_file = args.exemplars_file or str(hbbops_dir / 'examples.txt')

    # Load data
    print("Loading data...")
    instructions = load_instructions(instructions_file)
    exemplars = load_exemplars(exemplars_file)

    print(f"  Instructions: {len(instructions)}")
    print(f"  Exemplars: {len(exemplars)}")

    # Load validation data
    with open(data_dir / 'validation.json', 'r') as f:
        validation_data = json.load(f)
    print(f"  Validation samples: {len(validation_data)}")

    # Load training data (for bootstrap)
    training_data = validation_data  # Fallback
    train_path = data_dir / 'train.json'
    if train_path.exists():
        with open(train_path, 'r') as f:
            training_data = json.load(f)
        print(f"  Training samples: {len(training_data)}")

    # Set GPU if specified
    if args.gpu_ids:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Create task LLM client
    print(f"\nInitializing task model: {args.model}")
    llm_client = create_llm_client(
        model_name=args.model,
        backend=args.backend,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # Warmup vLLM
    if hasattr(llm_client, 'warmup'):
        print("Warming up vLLM...")
        llm_client.warmup()

    # Create meta LLM client (if different)
    meta_llm_client = llm_client
    if args.meta_model:
        meta_model = resolve_model_alias(args.meta_model)
        print(f"Initializing meta-model: {meta_model}")
        meta_llm_client = create_llm_client(
            model_name=meta_model,
            backend=args.meta_backend
        )

    # Create evaluator
    evaluator = GSM8KEvaluator(llm_client, debug=args.debug)

    # Create config
    config = HYPEConfig(
        num_generations=args.generations,
        num_new_instructions=args.new_instructions,
        num_new_exemplars=args.new_exemplars,
        use_recombination=not args.no_recombination,
        bmin=args.bmin,
        eta=args.eta,
        encoder_name=args.encoder,
        max_instructions=args.max_instructions,
        max_exemplars=args.max_exemplars,
        output_dir=args.output_dir,
    )

    # Create HYPE
    hype = HYPE(
        instructions=instructions,
        exemplars=exemplars,
        validation_data=validation_data,
        llm_client=llm_client,
        meta_llm_client=meta_llm_client,
        training_data=training_data,
        config=config,
        device=args.device,
        verbose=not args.quiet,
    )

    # Set model name for checkpointing
    hype._model_name = args.model

    # Run evolution
    best_prompt, best_error = hype.evolve(evaluator)

    # Save results
    output_dir = Path(args.output_dir)
    json_path, txt_path = hype.save_results(output_dir, model_name=args.model)

    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  TXT: {txt_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("HYPE FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Accuracy: {1 - best_error:.2%}")
    print(f"Best Error: {best_error:.4f}")

    if best_prompt:
        print(f"\nBest Instruction (ID {best_prompt.instruction_id}):")
        print(f"  {best_prompt.instruction[:100]}...")
        print(f"\nBest Exemplar (ID {best_prompt.exemplar_id}):")
        print(f"  {best_prompt.exemplar[:100]}...")


if __name__ == '__main__':
    main()
