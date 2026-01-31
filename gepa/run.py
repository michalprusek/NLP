#!/usr/bin/env python3
"""
Run GEPA prompt optimization for GSM8K.

Usage:
    python -m gepa.run --model qwen --backend vllm --budget 150000
"""
import argparse

from shared.llm_client import create_llm_client
from shared.gsm8k_evaluator import GSM8KEvaluator
from gepa.gepa import GEPAOptimizer


# Initial seed prompts (diverse starting points)
INITIAL_PROMPTS = [
    # Simple direct approach
    "Think step by step and solve this math problem. Show your work clearly.",

    # Structured approach
    """To solve this problem:
1. Identify the key information and numbers
2. Determine what is being asked
3. Set up the calculation
4. Solve step by step
5. State the final answer""",

    # Chain of thought
    "Let's work through this problem carefully. First, understand what we're given. Then, figure out what we need to find. Finally, calculate the answer showing each step.",

    # Best from ProTeGi (reference)
    """1. Extract Data: List out all important numbers and their contextual information.
2. Assign Variables: Use letters or symbols to denote unknown values.
3. Break Down Steps: Segment the problem into smaller parts.
4. Detailed Process: Outline the logical progression with computations.
5. Conclude: Provide the final answer.""",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run GEPA optimization for GSM8K")

    # Model settings
    parser.add_argument("--model", type=str, default="qwen",
                       help="Task model (qwen, llama, or full model name)")
    parser.add_argument("--meta-model", type=str, default=None,
                       help="Meta/reflection model (default: same as task model)")
    parser.add_argument("--backend", type=str, default="vllm",
                       choices=["vllm", "openai", "deepinfra", "auto"],
                       help="LLM backend")

    # GEPA parameters
    parser.add_argument("--budget", type=int, default=150000,
                       help="Total task LLM call budget")
    parser.add_argument("--minibatch-size", type=int, default=64,
                       help="Examples per evaluation")
    parser.add_argument("--pareto-size", type=int, default=10,
                       help="Max Pareto front size")
    parser.add_argument("--mutations", type=int, default=4,
                       help="Mutations per iteration")
    parser.add_argument("--exploit-prob", type=float, default=0.8,
                       help="Probability of exploiting Pareto front")

    # vLLM settings
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallelism for vLLM")

    # Output
    parser.add_argument("--output-dir", type=str, default="gepa/results",
                       help="Output directory for results")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GEPA: Genetic-Pareto Prompt Optimization")
    print("=" * 60)
    print(f"Task model: {args.model}")
    print(f"Meta model: {args.meta_model or args.model}")
    print(f"Backend: {args.backend}")
    print(f"Budget: {args.budget}")
    print(f"Minibatch size: {args.minibatch_size}")
    print("=" * 60)

    # Create LLM clients
    print("\nInitializing LLM clients...")

    task_llm = create_llm_client(
        args.model,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    if args.meta_model and args.meta_model != args.model:
        meta_llm = create_llm_client(
            args.meta_model,
            backend=args.backend,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    else:
        meta_llm = task_llm

    # Create evaluator
    print("Loading GSM8K dataset...")
    evaluator = GSM8KEvaluator()

    # Create GEPA optimizer
    optimizer = GEPAOptimizer(
        task_llm_client=task_llm,
        evaluator=evaluator,
        meta_llm_client=meta_llm,
        pareto_max_size=args.pareto_size,
        mutations_per_iteration=args.mutations,
        exploit_probability=args.exploit_prob,
        minibatch_size=args.minibatch_size,
        total_budget=args.budget,
    )

    # Run optimization
    print("\nStarting GEPA optimization...")
    best = optimizer.optimize(INITIAL_PROMPTS, verbose=True)

    # Evaluate on test set
    test_accuracy = optimizer.evaluate_on_test_set(best.prompt, verbose=True)

    # Save results
    json_path, txt_path = optimizer.save_results(args.output_dir)
    print(f"\nResults saved to: {json_path}")
    print(f"Best prompt saved to: {txt_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best prompt:\n{best.prompt}")
    print(f"\nValidation accuracy: {best.accuracy:.2%}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Task LLM budget used: {optimizer.budget_used}")
    print(f"Meta LLM calls: {optimizer.meta_calls}")
    print("=" * 60)


if __name__ == "__main__":
    main()
