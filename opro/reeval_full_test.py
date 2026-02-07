"""Re-evaluate OPRO prompts on the full GSM8K test set (1319 examples).

Original OPRO benchmark used eval_set_size=261. This script re-scores
all 100 prompts on the full test set for fair comparison with ProTeGi
and RieLBO-GSM8K.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m opro.reeval_full_test
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    input_path = Path("opro/results/benchmark_100prompts_sonnet.json")
    output_path = Path("opro/results/benchmark_100prompts_sonnet_full1319.json")

    with open(input_path) as f:
        data = json.load(f)

    prompts = data["evaluated_prompts"]
    logger.info(f"Loaded {len(prompts)} prompts from {input_path}")

    # Load evaluator with full test set
    from shared.gsm8k_evaluator import GSM8KEvaluator
    evaluator = GSM8KEvaluator(dataset_path="datasets/gsm8k", split="test")
    all_indices = list(range(len(evaluator)))
    questions = [evaluator.dataset[i]["question"] for i in all_indices]
    logger.info(f"Eval set: {len(questions)} test examples")

    # Load vLLM
    from shared.llm_client import create_llm_client
    llm_client = create_llm_client("qwen", backend="vllm")

    # Re-evaluate each prompt
    results = []
    best_score = 0.0
    best_prompt = ""

    for i, entry in enumerate(prompts):
        prompt = entry["prompt"]
        old_score = entry["score"]

        formatted = [f"Q: {q}\n{prompt}\nA:" for q in questions]

        outputs = llm_client.generate_batch(
            formatted, temperature=0.0, max_new_tokens=512
        )

        eval_results = evaluator.evaluate_batch(outputs, all_indices)
        new_score = eval_results["accuracy"]

        if new_score > best_score:
            best_score = new_score
            best_prompt = prompt

        results.append({
            "prompt": prompt,
            "score": new_score,
            "old_score_261": old_score,
            "correct": eval_results["correct"],
            "total": eval_results["total"],
        })

        logger.info(
            f"Prompt {i+1}/{len(prompts)}: "
            f"{new_score:.4f} ({eval_results['correct']}/{eval_results['total']}) "
            f"[was {old_score:.4f} on 261] | best={best_score:.4f}"
        )

    # Save
    out_data = {
        "method": "opro",
        "model": data.get("model", "Qwen/Qwen2.5-7B-Instruct"),
        "eval_size": len(questions),
        "config": data.get("config", {}),
        "evaluated_prompts": results,
        "best_prompt": best_prompt,
        "best_score": best_score,
        "total_evaluated": len(results),
    }
    out_data["config"]["eval_size_original"] = 261
    out_data["config"]["eval_size_reeval"] = len(questions)

    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)

    logger.info(f"\nDone. Best: {best_score:.4f}")
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
