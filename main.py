#!/usr/bin/env python3
"""
Main entry point for prompt optimization and evaluation.

Usage:
    # Optimize prompts
    python main.py --mode optimize --method protegi --model Qwen/Qwen2.5-7B-Instruct
    python main.py --mode optimize --method opro --model Qwen/Qwen2.5-7B-Instruct --backend vllm

    # Evaluate a prompt
    python main.py --mode evaluate --model Qwen/Qwen2.5-7B-Instruct --prompt "Solve step by step."
    python main.py --mode evaluate --model Qwen/Qwen2.5-7B-Instruct --prompt-file results/best_prompt.txt
    python main.py --mode evaluate --task claudette --prompt "Classify this ToS clause."
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.llm_client import create_llm_client
from src.gsm8k_evaluator import GSM8KEvaluator
from src.claudette_evaluator import ClaudetteEvaluator
from src.claudette_binary_evaluator import ClaudetteBinaryEvaluator
from src.protegi import ProTeGi
from src.opro import OPRO


def main():
    parser = argparse.ArgumentParser(description="Prompt optimization and evaluation for GSM8K and Claudette")

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["optimize", "evaluate"],
        help="Mode: optimize (run ProTeGi/OPRO) or evaluate (test a prompt)",
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "claudette", "claudette_binary"],
        help="Task: gsm8k (math problems), claudette (ToS multi-label), or claudette_binary (ToS binary)",
    )

    # Method selection (only for optimize mode)
    parser.add_argument(
        "--method",
        type=str,
        choices=["protegi", "opro"],
        help="Optimization method to use (required for --mode optimize)",
    )

    # Model selection - Task model (being optimized)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Task model name - the model being optimized (e.g., Qwen/Qwen2.5-7B-Instruct, Equall/Saul-7B-Instruct-v1)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["transformers", "vllm", "claude", "openai", "auto"],
        help="LLM backend for task model (auto detects Claude and OpenAI models)",
    )

    # Meta-optimizer model (for gradient generation, prompt editing)
    parser.add_argument(
        "--meta-model",
        type=str,
        default=None,
        help="Meta-optimizer model for gradient generation and prompt editing. If not specified, uses --model. Examples: claude-3-5-sonnet-20241022, claude-3-haiku-20240307",
    )

    parser.add_argument(
        "--meta-backend",
        type=str,
        default="auto",
        choices=["transformers", "vllm", "claude", "openai", "auto"],
        help="Backend for meta-optimizer model (auto detects Claude and OpenAI models)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run on (auto detects best available)",
    )

    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model data type (auto: bfloat16 for MPS, float16 for CUDA, float32 for CPU)",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (vLLM only). Use 2 for dual GPU setup.",
    )

    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1",
        help="Comma-separated GPU IDs to use (e.g., '0,1' for GPU 0 and 1)",
    )

    # Dataset settings
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset (defaults: datasets/gsm8k for GSM8K, tommasobonomo/sem_eval_2023_task_4 for Claudette)",
    )

    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use for training/optimization",
    )

    parser.add_argument(
        "--val-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use for validation",
    )

    # Optimization settings
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of optimization iterations",
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=261,
        help="Number of examples per evaluation",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size for ProTeGi",
    )

    parser.add_argument(
        "--num-candidates",
        type=int,
        default=8,
        help="Number of candidates per iteration for OPRO (paper uses 8)",
    )

    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Initial prompt (defaults to task-specific prompt from src/prompts/<task>/initial.txt)",
    )

    # Evaluate mode arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to evaluate (required for --mode evaluate)",
    )

    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File containing prompt to evaluate (alternative to --prompt)",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (None = all)",
    )

    # Evaluator selection (GSM8K only uses standard evaluator now)
    parser.add_argument(
        "--evaluator",
        type=str,
        default="standard",
        choices=["standard"],
        help="Evaluator to use: standard (exact match with numerical tolerance)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (shows model outputs and extraction details)",
    )

    args = parser.parse_args()

    # Validate mode-specific arguments
    if args.mode == "optimize":
        if args.method is None:
            parser.error("--method is required for --mode optimize")
    elif args.mode == "evaluate":
        if args.prompt is None and args.prompt_file is None:
            parser.error("Either --prompt or --prompt-file is required for --mode evaluate")
        if args.prompt and args.prompt_file:
            parser.error("Cannot specify both --prompt and --prompt-file")

    # Set task-specific defaults
    if args.dataset_path is None:
        if args.task == "gsm8k":
            args.dataset_path = "datasets/gsm8k"
        elif args.task == "claudette":
            args.dataset_path = "datasets/tos_local"
        elif args.task == "claudette_binary":
            args.dataset_path = "datasets/tos_local"  # Same dataset, binary labels

    # Load prompt for evaluate mode
    if args.mode == "evaluate":
        if args.prompt_file:
            prompt_path = Path(args.prompt_file)
            if not prompt_path.exists():
                parser.error(f"Prompt file not found: {args.prompt_file}")
            args.prompt = prompt_path.read_text(encoding='utf-8').strip()
            print(f"Loaded prompt from: {args.prompt_file}")

    # Load initial prompt from file if provided (optimize mode)
    if args.mode == "optimize":
        if args.initial_prompt is None:
            prompt_file = Path(__file__).parent / "src" / "prompts" / args.task / "initial.txt"
            if prompt_file.exists():
                args.initial_prompt = prompt_file.read_text(encoding='utf-8').strip()
            # Otherwise leave as None - optimizer will use its own defaults

    # Set GPU visibility
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Print configuration
    print("\n" + "="*80)
    if args.mode == "optimize":
        print("PROMPT OPTIMIZATION CONFIGURATION")
        print("="*80)
        print(f"Mode: OPTIMIZE")
        print(f"Task: {args.task.upper()}")
        print(f"Method: {args.method.upper()}")
        print(f"Model: {args.model}")
        print(f"Backend: {args.backend}")
        print(f"GPUs: {args.gpu_ids}")
        if args.backend == "vllm" and args.tensor_parallel_size > 1:
            print(f"Tensor Parallel Size: {args.tensor_parallel_size} (using {args.tensor_parallel_size} GPUs)")
        print(f"Iterations: {args.iterations}")
        print(f"Minibatch Size: {args.minibatch_size}")
    else:  # evaluate mode
        print("PROMPT EVALUATION CONFIGURATION")
        print("="*80)
        print(f"Mode: EVALUATE")
        print(f"Task: {args.task.upper()}")
        print(f"Model: {args.model}")
        print(f"Backend: {args.backend}")
        print(f"GPUs: {args.gpu_ids}")
        if args.backend == "vllm" and args.tensor_parallel_size > 1:
            print(f"Tensor Parallel Size: {args.tensor_parallel_size} (using {args.tensor_parallel_size} GPUs)")
        if args.max_examples:
            print(f"Max Examples: {args.max_examples}")
        print(f"\nPrompt to evaluate:")
        print(f"  {args.prompt[:100]}..." if len(args.prompt) > 100 else f"  {args.prompt}")
    print("="*80 + "\n")

    # Set meta-model defaults (only for optimize mode)
    if args.mode == "optimize":
        if args.meta_model is None:
            args.meta_model = args.model
            args.meta_backend = args.backend
            print(f"\nMeta-optimizer model not specified, using task model: {args.model}")
        else:
            print(f"\nUsing separate meta-optimizer model: {args.meta_model}")

    # Initialize Task LLM client (the model being optimized)
    print(f"\nInitializing TASK model ({args.backend}):")
    print(f"  Model: {args.model}")
    

    # Prepare Task LLM client kwargs
    task_llm_kwargs = {
        "model_name": args.model,
        "backend": args.backend,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
    }

    # Add tensor parallelism for vLLM
    if args.backend == "vllm":
        task_llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size

    task_llm_client = create_llm_client(**task_llm_kwargs)

    # Initialize Meta-optimizer LLM client (only for optimize mode)
    if args.mode == "optimize":
        print(f"\nInitializing META-OPTIMIZER model ({args.meta_backend}):")
        print(f"  Model: {args.meta_model}")

        # Prepare Meta LLM client kwargs
        meta_llm_kwargs = {
            "model_name": args.meta_model,
            "backend": args.meta_backend,
        }

        # Only add device/dtype for non-API backends (Claude and OpenAI use API)
        if args.meta_backend not in ["claude", "openai", "auto"] or ("claude" not in args.meta_model.lower() and "gpt" not in args.meta_model.lower()):
            meta_llm_kwargs["device"] = args.device
            meta_llm_kwargs["torch_dtype"] = args.torch_dtype

        # Add tensor parallelism for vLLM (not for Claude or OpenAI)
        if args.meta_backend == "vllm":
            meta_llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size

        # Create meta-optimizer client (can be same as task client if same model)
        if args.meta_model == args.model and args.meta_backend == args.backend:
            meta_llm_client = task_llm_client
            print("  (Using same client as task model)")
        else:
            meta_llm_client = create_llm_client(**meta_llm_kwargs)

    # Initialize evaluators
    print(f"Loading {args.task.upper()} dataset from {args.dataset_path}")
    if args.mode == "optimize":
        print(f"  Training split: {args.train_split}")
        print(f"  Validation split: {args.val_split}")
    else:  # evaluate mode
        print(f"  Evaluation split: {args.val_split}")

    # Select evaluator class based on task
    if args.task == "claudette":
        EvaluatorClass = ClaudetteEvaluator
        print(f"  Evaluator: ClaudetteEvaluator (ToS classification)")
    elif args.task == "claudette_binary":
        EvaluatorClass = ClaudetteBinaryEvaluator
        print(f"  Evaluator: ClaudetteBinaryEvaluator (Binary ToS classification: fair vs unfair)")
    else:  # gsm8k
        EvaluatorClass = GSM8KEvaluator
        print(f"  Evaluator: GSM8KEvaluator (exact match with numerical tolerance)")

    # Create evaluator instances
    if args.mode == "optimize":
        train_evaluator = EvaluatorClass(
            dataset_path=args.dataset_path,
            split=args.train_split,
            debug=args.debug,
        )
        val_evaluator = EvaluatorClass(
            dataset_path=args.dataset_path,
            split=args.val_split,
            debug=args.debug,
        )
        print(f"  Training dataset size: {len(train_evaluator)} examples")
        print(f"  Validation dataset size: {len(val_evaluator)} examples\n")
    else:  # evaluate mode
        val_evaluator = EvaluatorClass(
            dataset_path=args.dataset_path,
            split=args.val_split,
            debug=args.debug,
        )
        print(f"  Dataset size: {len(val_evaluator)} examples\n")

    if args.debug:
        print("🐛 DEBUG MODE ENABLED - Will show model outputs and extraction details\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set task description based on task (used by optimizers)
    if args.task == "claudette":
        task_description = "Classify Terms of Service clauses as FAIR (LABELS: NONE) or UNFAIR (one or more categories 0-5, 7-8). 90% are FAIR. Categories: 0=Limitation of liability, 1=Unilateral termination, 2=Unilateral change, 3=Arbitration, 4=Content removal, 5=Choice of law, 7=Contract by using, 8=Jurisdiction."
    elif args.task == "claudette_binary":
        task_description = "Classify Terms of Service clauses as FAIR (0) or UNFAIR (1). Fair clauses are neutral, unfair clauses contain problematic terms."
    else:
        task_description = "Solve math word problems step by step and provide the final numerical answer."

    # ========================================================================
    # OPTIMIZE MODE: Run ProTeGi or OPRO
    # ========================================================================
    if args.mode == "optimize":
        if args.method == "protegi":
            print(f"Running ProTeGi optimization on {args.train_split} split...")
            optimizer = ProTeGi(
                task_llm_client=task_llm_client,
                meta_llm_client=meta_llm_client,
                evaluator=train_evaluator,
                beam_size=args.beam_size,
                num_iterations=args.iterations,
                minibatch_size=args.minibatch_size,
                task_description=task_description,
            )
            best_prompt, history = optimizer.optimize(
                initial_prompt=args.initial_prompt,
                verbose=not args.quiet,
            )

            # Save results
            output_file = output_dir / f"protegi_{args.task}_{timestamp}.json"

        elif args.method == "opro":
            print(f"Running OPRO optimization on {args.train_split} split...")
            optimizer = OPRO(
                task_llm_client=task_llm_client,
                meta_llm_client=meta_llm_client,
                evaluator=train_evaluator,
                num_iterations=args.iterations,
                num_candidates_per_iter=args.num_candidates,
                minibatch_size=args.minibatch_size,
                task_description=task_description,
            )
            # Pass initial prompts if provided, otherwise None (optimizer uses defaults)
            initial_prompts_arg = [args.initial_prompt] if args.initial_prompt is not None else None
            best_prompt, history = optimizer.optimize(
                initial_prompts=initial_prompts_arg,
                verbose=not args.quiet,
            )

            # Save results
            output_file = output_dir / f"opro_{args.task}_{timestamp}.json"

        # Convert history to JSON-serializable format (handle numpy types)
        def make_json_serializable(obj):
            """Convert numpy types to Python types"""
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            return obj

        # Extract initial prompts for JSON
        if args.method == "opro" and args.initial_prompt is None:
            # Get seed prompts from history (iteration == -1)
            seed_prompts = [h['prompt'] for h in history if h.get('iteration') == -1]
            initial_prompts_for_json = seed_prompts if seed_prompts else None
        else:
            initial_prompts_for_json = args.initial_prompt

        # Save results to JSON
        results = {
            "task": args.task,
            "method": args.method,
            "task_model": args.model,
            "task_backend": args.backend,
            "meta_model": args.meta_model,
            "meta_backend": args.meta_backend,
            "timestamp": timestamp,
            "config": {
                "iterations": args.iterations,
                "minibatch_size": args.minibatch_size,
                "beam_size": args.beam_size if args.method == "protegi" else None,
                "num_candidates": args.num_candidates if args.method == "opro" else None,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "train_size": len(train_evaluator),
                "val_size": len(val_evaluator),
                "initial_prompt": initial_prompts_for_json,  # Can be string, list, or None
            },
            "best_prompt": best_prompt,
            "history": make_json_serializable(history),
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save best prompt to text file
        prompt_file = output_file.with_suffix('.txt')
        with open(prompt_file, "w", encoding='utf-8') as f:
            f.write(f"# {args.method.upper()} Optimized Prompt\n")
            f.write(f"# Model: {args.model}\n")
            f.write(f"# Timestamp: {timestamp}\n")

            # Extract initial prompts from history for OPRO
            if args.method == "opro" and args.initial_prompt is None:
                # Get seed prompts from history (iteration == -1)
                seed_prompts = [h['prompt'] for h in history if h.get('iteration') == -1]
                if seed_prompts:
                    f.write(f"# Initial prompts (OPRO seeds):\n")
                    for i, prompt in enumerate(seed_prompts, 1):
                        prompt_display = prompt if prompt else "(empty prompt)"
                        f.write(f"#   {i}. {prompt_display}\n")
                else:
                    f.write(f"# Initial prompt: {args.initial_prompt}\n")
            else:
                f.write(f"# Initial prompt: {args.initial_prompt}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("OPTIMIZED PROMPT:\n")
            f.write("="*80 + "\n\n")
            f.write(best_prompt)
            f.write("\n")

        print(f"\nResults saved to: {output_file}")
        print(f"Best prompt saved to: {prompt_file}")

    # ========================================================================
    # EVALUATE MODE: Just set best_prompt and skip to validation
    # ========================================================================
    else:  # evaluate mode
        print(f"\nEvaluating prompt on {args.val_split} split...")
        best_prompt = args.prompt  # Use the provided prompt

    # ========================================================================
    # FINAL VALIDATION: Common for both optimize and evaluate modes
    # ========================================================================
    # Final validation on FULL test split (or subset if max_examples specified)
    print(f"\n{'='*80}")
    print(f"Final Validation on {args.val_split.upper()} Split")
    print(f"{'='*80}\n")

    # Evaluate on validation set (full or subset based on max_examples)
    total_size = len(val_evaluator)
    val_size = min(args.max_examples, total_size) if args.max_examples else total_size

    if args.max_examples and args.max_examples < total_size:
        print(f"Evaluating on {val_size}/{total_size} examples from {args.val_split} split...")
    else:
        print(f"Evaluating on {val_size} examples from {args.val_split} split...")

    # Get examples
    batch = val_evaluator.get_batch(0, val_size)
    # Handle both 'question' (GSM8K) and 'text' (Claudette) fields
    questions = [example.get('question', example.get('text', '')) for example in batch]
    prompts = [f"{best_prompt}\n\nQuestion: {q}\nAnswer:" for q in questions]

    # Generate all outputs in one batch (vLLM handles this efficiently)
    # Use temperature=0.0 for deterministic, reproducible validation results
    outputs = task_llm_client.generate_batch(prompts, temperature=0.0)

    # Evaluate
    indices = [example['idx'] for example in batch]
    val_results = val_evaluator.evaluate_batch(outputs, indices)

    # Display results - check if Claudette task for comprehensive metrics
    if args.task == "claudette":
        print(f"\nValidation Results:")
        print(f"  Subset Accuracy: {val_results['accuracy']:.1%} ({val_results['correct']}/{val_results['total']})")
        print(f"\n  Micro F1:    {val_results.get('micro_f1', 0.0):.1%}")
        print(f"  Macro F1:    {val_results.get('macro_f1', 0.0):.1%}")
        print(f"  Weighted F1: {val_results.get('weighted_f1', 0.0):.1%}")
        print(f"  Hamming Loss: {val_results.get('hamming_loss', 0.0):.4f}")
    elif args.task == "claudette_binary":
        print(f"\nValidation Results:")
        print(f"  Accuracy:  {val_results['accuracy']:.1%} ({val_results['correct']}/{val_results['total']})")
        print(f"  Macro-F1:  {val_results.get('macro_f1', 0.0):.1%} (avg of Fair-F1 and Unfair-F1)")
        print(f"  Precision: {val_results.get('precision', 0.0):.1%} (Unfair class)")
        print(f"  Recall:    {val_results.get('recall', 0.0):.1%} (Unfair class)")
        print(f"  F1-Unfair: {val_results.get('f1', 0.0):.1%}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {val_results.get('tp', 0):>4}  FP: {val_results.get('fp', 0):>4}")
        print(f"    FN: {val_results.get('fn', 0):>4}  TN: {val_results.get('tn', 0):>4}")
    else:
        print(f"\nValidation Accuracy: {val_results['accuracy']:.1%}")
        print(f"Correct: {val_results['correct']}/{val_results['total']}")

    # Save validation results (only for optimize mode)
    if args.mode == "optimize":
        # Update results with validation evaluation
        validation_dict = {
            "split": args.val_split,
            "accuracy": val_results['accuracy'],
            "correct": val_results['correct'],
            "total": val_results['total'],
        }

        # Add comprehensive metrics for Claudette
        if args.task == "claudette":
            validation_dict.update({
                "micro_f1": val_results.get('micro_f1', 0.0),
                "micro_precision": val_results.get('micro_precision', 0.0),
                "micro_recall": val_results.get('micro_recall', 0.0),
                "macro_f1": val_results.get('macro_f1', 0.0),
                "macro_precision": val_results.get('macro_precision', 0.0),
                "macro_recall": val_results.get('macro_recall', 0.0),
                "weighted_f1": val_results.get('weighted_f1', 0.0),
                "weighted_precision": val_results.get('weighted_precision', 0.0),
                "weighted_recall": val_results.get('weighted_recall', 0.0),
                "hamming_loss": val_results.get('hamming_loss', 0.0),
                "per_class": val_results.get('per_class', {}),
                "confusion_matrix": val_results.get('confusion_matrix', []),
                "support": val_results.get('support', []),
            })
        elif args.task == "claudette_binary":
            validation_dict.update({
                "precision": val_results.get('precision', 0.0),
                "recall": val_results.get('recall', 0.0),
                "f1": val_results.get('f1', 0.0),
                "tp": val_results.get('tp', 0),
                "fp": val_results.get('fp', 0),
                "tn": val_results.get('tn', 0),
                "fn": val_results.get('fn', 0),
                "micro_f1": val_results.get('micro_f1', 0.0),
                "macro_f1": val_results.get('macro_f1', 0.0),
                "per_class": val_results.get('per_class', {}),
            })

        results["validation"] = validation_dict

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Update prompt file with validation accuracy
        with open(prompt_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"VALIDATION ({args.val_split} split, {val_size} examples):\n")

            if args.task == "claudette":
                f.write(f"Subset Accuracy: {val_results['accuracy']:.1%}\n")
                f.write(f"Correct: {val_results['correct']}/{val_results['total']}\n\n")
                f.write(f"Comprehensive Metrics:\n")
                f.write(f"  Micro F1:         {val_results.get('micro_f1', 0.0):.1%}\n")
                f.write(f"  Micro Precision:  {val_results.get('micro_precision', 0.0):.1%}\n")
                f.write(f"  Micro Recall:     {val_results.get('micro_recall', 0.0):.1%}\n\n")
                f.write(f"  Macro F1:         {val_results.get('macro_f1', 0.0):.1%}\n")
                f.write(f"  Macro Precision:  {val_results.get('macro_precision', 0.0):.1%}\n")
                f.write(f"  Macro Recall:     {val_results.get('macro_recall', 0.0):.1%}\n\n")
                f.write(f"  Weighted F1:      {val_results.get('weighted_f1', 0.0):.1%}\n")
                f.write(f"  Weighted Precision: {val_results.get('weighted_precision', 0.0):.1%}\n")
                f.write(f"  Weighted Recall:  {val_results.get('weighted_recall', 0.0):.1%}\n\n")
                f.write(f"  Hamming Loss:     {val_results.get('hamming_loss', 0.0):.4f}\n")
            elif args.task == "claudette_binary":
                f.write(f"Accuracy:  {val_results['accuracy']:.1%}\n")
                f.write(f"Correct:   {val_results['correct']}/{val_results['total']}\n\n")
                f.write(f"Binary Classification Metrics:\n")
                f.write(f"  Macro-F1:  {val_results.get('macro_f1', 0.0):.1%} (average of Fair-F1 and Unfair-F1)\n")
                f.write(f"  Precision: {val_results.get('precision', 0.0):.1%} (Unfair class)\n")
                f.write(f"  Recall:    {val_results.get('recall', 0.0):.1%} (Unfair class)\n")
                f.write(f"  F1-Unfair: {val_results.get('f1', 0.0):.1%}\n\n")
                f.write(f"Confusion Matrix:\n")
                f.write(f"  TP: {val_results.get('tp', 0):>4}  FP: {val_results.get('fp', 0):>4}\n")
                f.write(f"  FN: {val_results.get('fn', 0):>4}  TN: {val_results.get('tn', 0):>4}\n")
            else:
                f.write(f"Accuracy: {val_results['accuracy']:.1%}\n")
                f.write(f"Correct: {val_results['correct']}/{val_results['total']}\n")

            f.write(f"{'='*80}\n")


    if __name__ == "__main__":
        main()
