import argparse
import json
import sys
import os
import re
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
    GSM8KEvaluator,
    extract_answer,
    exact_match_with_tolerance
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate all prompts on full validation set")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--backend", type=str, default="vllm", help="LLM backend")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory (relative to script)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts for testing")
    parser.add_argument("--samples", type=int, default=None, help="Limit validation samples per prompt")
    parser.add_argument("--mega-batch", type=int, default=0,
                        help="Process multiple prompts together in mega-batches (0=disabled, try 4-8 for speedup)")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="GPU memory utilization (default: 0.90, max 0.95)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start prompt index for data parallelism (default: 0)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End prompt index (exclusive) for data parallelism (default: all)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir

    # Setup output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add GPU suffix for data parallel runs
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    output_file = output_dir / f"full_grid_{timestamp}_gpu{gpu_id}.jsonl"
    
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

    # Limit validation samples if specified
    if args.samples:
        validation_data = validation_data[:args.samples]
        print(f"Using {args.samples} validation samples per prompt")

    # Generate prompts
    prompts = [
        Prompt(instruction, exemplar, i_idx, e_idx)
        for i_idx, instruction in enumerate(instructions)
        for e_idx, exemplar in enumerate(exemplars)
    ]
    
    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Limiting to {args.limit} prompts for testing")

    # Apply data parallelism slicing (for multi-GPU runs)
    if args.start_idx > 0 or args.end_idx is not None:
        end_idx = args.end_idx if args.end_idx else len(prompts)
        prompts = prompts[args.start_idx:end_idx]
        print(f"Data parallel slice: prompts {args.start_idx} to {end_idx}")

    print(f"Evaluating {len(prompts)} prompts...")

    # Init LLM
    llm_client = create_llm_client(
        model_name=args.model,
        backend=args.backend,
        device="auto",
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # Explicit warmup with representative prompts (critical for vLLM performance!)
    if hasattr(llm_client, 'warmup'):
        # Create a sample prompt similar to actual evaluation prompts
        sample_prompt = prompts[0]
        sample_question = validation_data[0]['question']
        warmup_prompt = f"Question: {sample_question}\n\n{str(sample_prompt)}\n\nAnswer:"
        print("Running explicit warmup with representative prompt...")
        llm_client.warmup([warmup_prompt])

    evaluator = GSM8KEvaluator(llm_client)

    # Evaluation loop - optionally use mega-batching for better GPU utilization
    results = []

    if args.mega_batch > 0:
        # MEGA-BATCH MODE: Process multiple prompts together
        # This reduces kernel launch overhead and improves GPU utilization
        print(f"Using mega-batch mode with batch size {args.mega_batch}")

        with open(output_file, "w") as f:
            for batch_start in tqdm(range(0, len(prompts), args.mega_batch)):
                batch_prompts = prompts[batch_start:batch_start + args.mega_batch]

                # Prepare ALL full prompts for this mega-batch
                all_full_prompts = []
                prompt_indices = []  # Track which prompt each full_prompt belongs to
                for prompt_idx, prompt in enumerate(batch_prompts):
                    for example in validation_data:
                        question = example['question']
                        all_full_prompts.append(f"Question: {question}\n\n{str(prompt)}\n\nAnswer:")
                        prompt_indices.append(prompt_idx)

                # Single batch generation for all prompts!
                try:
                    responses = llm_client.generate_batch(all_full_prompts, max_tokens=1024)
                except Exception as e:
                    print(f"Warning: Mega-batch generation failed: {e}")
                    # Fall back to individual evaluation
                    for i, prompt in enumerate(batch_prompts):
                        error_rate = evaluator(prompt, validation_data)
                        result = {
                            "prompt_idx": batch_start + i,
                            "instruction_id": prompt.instruction_id,
                            "exemplar_id": prompt.exemplar_id,
                            "error_rate": error_rate,
                            "timestamp": datetime.now().isoformat()
                        }
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        results.append(result)
                    continue

                # Calculate error rates for each prompt in the batch
                val_size = len(validation_data)
                for i, prompt in enumerate(batch_prompts):
                    start_idx = i * val_size
                    end_idx = start_idx + val_size
                    prompt_responses = responses[start_idx:end_idx]

                    # Calculate errors
                    errors = 0
                    for j, example in enumerate(validation_data):
                        gold_answer_str = example['answer']
                        match = re.search(r'####\s*(-?\d+\.?\d*)', gold_answer_str)
                        gold_answer = float(match.group(1)) if match else None

                        if gold_answer is None:
                            errors += 1
                            continue

                        response = prompt_responses[j]
                        try:
                            pred_answer = extract_answer(response)
                        except:
                            errors += 1
                            continue

                        if not exact_match_with_tolerance(pred_answer, gold_answer):
                            errors += 1

                    error_rate = errors / val_size
                    result = {
                        "prompt_idx": batch_start + i,
                        "instruction_id": prompt.instruction_id,
                        "exemplar_id": prompt.exemplar_id,
                        "error_rate": error_rate,
                        "timestamp": datetime.now().isoformat()
                    }
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    results.append(result)

    else:
        # STANDARD MODE: One prompt at a time
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
