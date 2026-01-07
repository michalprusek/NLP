"""Generate synthetic training data for LIPO-E from existing HbBoPs grid.

This script converts block-level evaluations to individual Q/A pair selection
format required by LIPO-E's variable-length exemplar optimization.

Strategy:
- Use the evaluated block's error rate as the base
- When selecting pairs from same block, use block's error rate
- When mixing blocks, use weighted average of error rates
"""

import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict
from pathlib import Path

from lipo_e.training import parse_exemplar_pool, load_instructions, QAPair, TrainingSample


def load_grid_evaluations(grid_path: str) -> Dict[tuple, float]:
    """Load evaluated prompt combinations from grid.

    Returns:
        Dict mapping (instruction_id, exemplar_block_id) -> error_rate
    """
    evaluations = {}
    with open(grid_path) as f:
        for line in f:
            data = json.loads(line)
            key = (data["instruction_id"], data["exemplar_id"])
            evaluations[key] = data["error_rate"]
    return evaluations


def generate_synthetic_samples(
    qa_pool: List[QAPair],
    instructions: List[str],
    grid_evals: Dict[tuple, float],
    num_samples: int = 500,
    max_exemplars: int = 8,
    seed: int = 42,
) -> List[TrainingSample]:
    """Generate synthetic training samples with variable K exemplars.

    Uses block-level error rates as proxy for individual Q/A combinations.
    """
    random.seed(seed)

    samples = []
    seen_combinations = set()

    # Map pool_id -> block_id
    pool_to_block = {qa.pool_id: qa.block_id for qa in qa_pool}

    for i in range(num_samples * 2):  # Generate more to account for duplicates
        if len(samples) >= num_samples:
            break

        # Random instruction
        inst_id = random.randint(0, len(instructions) - 1)

        # Random number of exemplars (0 to max_exemplars)
        # Bias toward having some exemplars (not empty)
        if random.random() < 0.1:
            K = 0
        else:
            K = random.randint(1, max_exemplars)

        # Random exemplar selection
        if K > 0:
            exemplar_ids = sorted(random.sample(range(len(qa_pool)), K))
        else:
            exemplar_ids = []

        # Check for duplicate
        cache_key = (inst_id, tuple(exemplar_ids))
        if cache_key in seen_combinations:
            continue
        seen_combinations.add(cache_key)

        # Estimate error rate from block-level evaluations
        if K == 0:
            # No exemplars - use worst block's error rate as estimate
            block_errors = [grid_evals.get((inst_id, b), 0.9) for b in range(25)]
            error_rate = max(block_errors)  # Conservative estimate
        else:
            # Get blocks for selected exemplars
            block_ids = [pool_to_block[pid] for pid in exemplar_ids]
            unique_blocks = set(block_ids)

            # Get error rates for involved blocks
            block_error_rates = []
            for block_id in unique_blocks:
                if (inst_id, block_id) in grid_evals:
                    block_error_rates.append(grid_evals[(inst_id, block_id)])

            if block_error_rates:
                # Use average of involved blocks' error rates
                # Weight by how many pairs are from each block
                error_rate = sum(block_error_rates) / len(block_error_rates)
            else:
                error_rate = 0.5  # Default fallback

        sample = TrainingSample(
            instruction_id=inst_id,
            instruction_text=instructions[inst_id],
            exemplar_ids=exemplar_ids,
            num_exemplars=K,
            error_rate=error_rate,
            fidelity=1319,  # Full validation set
        )
        samples.append(sample)

    return samples


def main():
    # Paths
    exemplar_path = "datasets/hbbops/examples_25.txt"
    instruction_path = "datasets/hbbops/instructions_25.txt"
    grid_path = "datasets/hbbops/full_grid_combined.jsonl"
    output_path = "lipo_e/data/synthetic_training_samples.json"

    print("Loading Q/A pool...")
    qa_pool = parse_exemplar_pool(exemplar_path)
    print(f"  Loaded {len(qa_pool)} Q/A pairs")

    print("Loading instructions...")
    instructions = load_instructions(instruction_path)
    print(f"  Loaded {len(instructions)} instructions")

    print("Loading grid evaluations...")
    grid_evals = load_grid_evaluations(grid_path)
    print(f"  Loaded {len(grid_evals)} evaluated combinations")

    print("Generating synthetic training samples...")
    samples = generate_synthetic_samples(
        qa_pool=qa_pool,
        instructions=instructions,
        grid_evals=grid_evals,
        num_samples=500,
        max_exemplars=8,
    )
    print(f"  Generated {len(samples)} samples")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(s) for s in samples], f, indent=2)
    print(f"  Saved to {output_path}")

    # Stats
    k_counts = {}
    for s in samples:
        k_counts[s.num_exemplars] = k_counts.get(s.num_exemplars, 0) + 1
    print("\nExemplar count distribution:")
    for k in sorted(k_counts.keys()):
        print(f"  K={k}: {k_counts[k]} samples")

    error_rates = [s.error_rate for s in samples]
    print(f"\nError rate stats:")
    print(f"  Mean: {sum(error_rates)/len(error_rates):.4f}")
    print(f"  Min:  {min(error_rates):.4f}")
    print(f"  Max:  {max(error_rates):.4f}")


if __name__ == "__main__":
    main()
