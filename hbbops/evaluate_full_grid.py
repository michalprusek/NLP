import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import create_llm_client
from hbbops.hbbops import Prompt
from hbbops.run_hbbops import (
    load_instructions,
    load_exemplars,
    GSM8KEvaluator
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate all prompts on full validation set")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--backend", type=str, default="vllm", help="LLM backend")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory (relative to script)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts for testing")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir

    # Setup output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"full_grid_{timestamp}.jsonl"
    
    print(f"Saving results to {output_file}")

    # Load data
    print("Loading data...")
    try:
        with open(data_dir / "validation.json", 'r') as f:
            validation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: validation.json not found in {data_dir}")
        return

    instructions = load_instructions(str(script_dir / "instructions.txt"))
    exemplars = load_exemplars(str(script_dir / "examples.txt"))
    
    print(f"Loaded {len(instructions)} instructions, {len(exemplars)} exemplars")
    print(f"Loaded {len(validation_data)} validation examples")

    # Generate prompts
    prompts = [
        Prompt(instruction, exemplar, i_idx, e_idx)
        for i_idx, instruction in enumerate(instructions)
        for e_idx, exemplar in enumerate(exemplars)
    ]
    
    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Limiting to {args.limit} prompts for testing")

    print(f"Evaluating {len(prompts)} prompts...")

    # Init LLM
    llm_client = create_llm_client(
        model_name=args.model,
        backend=args.backend,
        device="auto"
    )
    evaluator = GSM8KEvaluator(llm_client)

    # Evaluation loop
    results = []
    with open(output_file, "w") as f:
        for i, prompt in enumerate(tqdm(prompts)):
            error_rate = evaluator(prompt, validation_data)
            
            result = {
                "prompt_idx": i,
                "instruction_id": prompt.instruction_id,
                "exemplar_id": prompt.exemplar_id,
                "error_rate": error_rate,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save incrementally
            f.write(json.dumps(result) + "\n")
            f.flush()
            results.append(result)

    print("Evaluation complete!")

if __name__ == "__main__":
    main()
