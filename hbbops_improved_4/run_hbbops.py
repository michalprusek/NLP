"""
Run HbBoPs Improved 4 on GSM8K

Features:
- Heteroscedastic noise (Wilson score)
- Output warping (logit + Delta method)
- Multi-fidelity GP (product kernel)
- Top 75% fidelity filtering (excludes bottom 25%)
- Model persistence (--save-gp)

Usage:
    cd hbbops_improved_4
    uv run python run_hbbops.py --model Qwen/Qwen2.5-7B-Instruct --save-gp
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import re
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from hbbops_improved_4.hbbops import HbBoPs, Prompt
from src.llm_client import create_llm_client


class TeeLogger:
    """Write to both console and file simultaneously."""

    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# Simple number pattern - matches integers and decimals (including negative)
NUMBER_PATTERN = r'[-+]?\d+(?:[.,]\d+)?'


def extract_answer(text: str) -> Optional[str]:
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
    parser = argparse.ArgumentParser(description='Run HbBoPs Improved 4 on GSM8K')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--backend', type=str, default='vllm', choices=['vllm', 'transformers', 'claude'])
    parser.add_argument('--bmin', type=int, default=10, help='Min validation instances (default: 10)')
    parser.add_argument('--eta', type=float, default=2.0, help='Halving parameter (default: 2.0)')
    parser.add_argument('--encoder', type=str, default='bert-base-uncased')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--instructions', type=str, default='instructions.txt', help='Instructions file')
    parser.add_argument('--exemplars', type=str, default='examples.txt', help='Exemplars file')
    parser.add_argument('--use-test-set', action='store_true', help='Use test set for HbBoPs (for comparison with full grid)')
    parser.add_argument('--ground-truth', type=str, default=None, help='Path to ground truth JSONL for comparison')
    # New arguments for improved version
    parser.add_argument('--save-gp', action='store_true', help='Save GP model at the end of optimization')
    parser.add_argument('--logit-epsilon', type=float, default=0.001, help='Epsilon for logit clipping (default: 0.001)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Setup logging - save complete output to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"hbbops_improved4_{timestamp}.log"
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    print(f"=" * 70)
    print(f"HbBoPs IMPROVED 4 Run Started at {datetime.now().isoformat()}")
    print(f"=" * 70)
    print(f"\nImprovements:")
    print(f"  - Heteroscedastic noise (Wilson score variance)")
    print(f"  - Output warping (logit + Delta method)")
    print(f"  - Multi-fidelity GP (product kernel K_deep × K_fidelity)")
    print(f"  - Top 75% fidelity filtering (excludes bottom 25%)")
    print(f"  - Model persistence (--save-gp)")
    print(f"\nLog file: {log_file}")
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  bmin: {args.bmin}")
    print(f"  eta: {args.eta}")
    print(f"  Logit epsilon: {args.logit_epsilon}")
    print(f"  Instructions: {args.instructions}")
    print(f"  Exemplars: {args.exemplars}")
    print(f"  Use test set: {args.use_test_set}")
    print(f"  Save GP: {args.save_gp}")
    print(f"  Ground truth: {args.ground_truth}")
    print()

    # Load data - try both possible locations
    print("Loading data...")
    data_dir = script_dir / "data"
    if not data_dir.exists():
        # Try parent's hbbops/data directory
        data_dir = script_dir.parent / "hbbops" / "data"
    if not data_dir.exists():
        # Try parent's hbbops_improved_2/data directory
        data_dir = script_dir.parent / "hbbops_improved_2" / "data"

    with open(data_dir / "validation.json") as f:
        validation_data = json.load(f)
    with open(data_dir / "test.json") as f:
        test_data = json.load(f)

    # Use test set for HbBoPs if requested (for comparison with full grid evaluation)
    if args.use_test_set:
        print("Using TEST set for HbBoPs optimization (for ground truth comparison)")
        hbbops_data = test_data
    else:
        hbbops_data = validation_data

    print(f"Validation: {len(validation_data)}, Test: {len(test_data)}")
    print(f"HbBoPs will use: {len(hbbops_data)} instances ({'test' if args.use_test_set else 'validation'})")

    # Load prompts - try multiple locations
    instructions_path = script_dir / args.instructions
    if not instructions_path.exists():
        instructions_path = script_dir.parent / "hbbops" / args.instructions
    if not instructions_path.exists():
        instructions_path = script_dir.parent / "hbbops_improved_2" / args.instructions

    exemplars_path = script_dir / args.exemplars
    if not exemplars_path.exists():
        exemplars_path = script_dir.parent / "hbbops" / args.exemplars
    if not exemplars_path.exists():
        exemplars_path = script_dir.parent / "hbbops_improved_2" / args.exemplars

    instructions = load_instructions(str(instructions_path))
    exemplars = load_exemplars(str(exemplars_path))
    print(f"Instructions: {len(instructions)}, Exemplars: {len(exemplars)}")

    # Initialize
    print(f"\nInitializing LLM ({args.backend})...")
    llm_client = create_llm_client(args.model, args.backend)
    evaluator = GSM8KEvaluator(llm_client, args.debug)

    print("\nInitializing HbBoPs Improved 4...")
    hbbops = HbBoPs(
        instructions=instructions,
        exemplars=exemplars,
        validation_data=hbbops_data,
        llm_evaluator=evaluator,
        encoder_name=args.encoder,
        bmin=args.bmin,
        eta=args.eta,
        device=args.device,
        logit_epsilon=args.logit_epsilon,
        use_wilson_score=True
    )

    # Run optimization
    # Pass output_dir only if save-gp is requested
    save_output_dir = str(output_dir) if args.save_gp else None
    best_prompt, best_val_error = hbbops.run_hyperband(output_dir=save_output_dir)

    # Evaluate on test
    print("\nEvaluating on test set...")
    test_error = evaluator(best_prompt, test_data)

    # Results
    print(f"\n{'=' * 60}")
    print(f"Validation error: {best_val_error:.4f} ({best_val_error * 100:.2f}%)")
    print(f"Test error: {test_error:.4f} ({test_error * 100:.2f}%)")
    print(f"Best prompt: instruction={best_prompt.instruction_id}, exemplar={best_prompt.exemplar_id}")

    # Save (using same timestamp as log file)
    results = {
        "method": "HbBoPs_Improved_4",
        "model": args.model,
        "config": {
            "bmin": args.bmin,
            "eta": args.eta,
            "logit_epsilon": args.logit_epsilon,
            "improvements": [
                "heteroscedastic_noise_wilson",
                "output_warping_logit_delta",
                "multi_fidelity_gp_product_kernel",
                "top_75pct_fidelity_filtering"
            ]
        },
        "best_prompt": {
            "instruction_id": best_prompt.instruction_id,
            "exemplar_id": best_prompt.exemplar_id,
            "instruction": best_prompt.instruction,
            "exemplar": best_prompt.exemplar
        },
        "validation_error": best_val_error,
        "test_error": test_error,
        "num_evaluations": len(hbbops.evaluation_cache),
        "fidelity_levels": hbbops.fidelity_levels,
        "num_design_data_points": len(hbbops.design_data)
    }

    with open(output_dir / f"hbbops_improved4_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/hbbops_improved4_{timestamp}.json")

    if args.save_gp:
        print(f"GP model saved to {output_dir}/gp_model_final.pt")

    # Compare with ground truth if provided
    if args.ground_truth:
        compare_with_ground_truth(hbbops, args.ground_truth, instructions, exemplars,
                                  output_dir, timestamp)

    # Finish logging
    print(f"\n{'=' * 70}")
    print(f"HbBoPs Improved 4 Run Finished at {datetime.now().isoformat()}")
    print(f"Full log saved to: {log_file}")
    print(f"{'=' * 70}")

    # Restore stdout and close logger
    sys.stdout = tee_logger.terminal
    sys.stderr = tee_logger.terminal
    tee_logger.close()


def compare_with_ground_truth(hbbops, ground_truth_path: str, instructions: list, exemplars: list,
                              output_dir: Path = None, timestamp: str = None):
    """Compare HbBoPs accuracies with ground truth.

    Key features:
    - Outputs ALL prompts evaluated at ANY fidelity (using their max fidelity)
    - Calculates Spearman on ALL evaluated prompts vs GT
    - Reports LLM call counts (sum of fidelities)
    """
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH GROUND TRUTH")
    print(f"{'=' * 70}")

    # Load ground truth
    gt_data = {}
    with open(ground_truth_path, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gt_data[(r['instruction_id'], r['exemplar_id'])] = r['error_rate']

    # Detect grid configuration and set ID mapping
    num_inst, num_ex = len(instructions), len(exemplars)
    total_prompts = num_inst * num_ex

    if num_inst == 25 and num_ex == 25:
        print("Using FULL GRID mode (25x25, no ID mapping)")
        instruction_mapping = list(range(25))
        exemplar_mapping = list(range(25))
    elif num_inst == 10 and num_ex == 25:
        print("Using UNIFORM mode (10x25, percentile-based instruction mapping)")
        instruction_mapping = [8, 20, 15, 7, 14, 11, 3, 22, 18, 1]
        exemplar_mapping = list(range(25))
    elif num_inst == 5 and num_ex == 5:
        print("Using SELECTED SUBSET mode (5x5)")
        instruction_mapping = [8, 9, 17, 2, 1]
        exemplar_mapping = [11, 7, 13, 14, 20]
    else:
        print(f"Using CUSTOM mode ({num_inst}x{num_ex}, identity mapping)")
        instruction_mapping = list(range(num_inst))
        exemplar_mapping = list(range(num_ex))

    # Get all fidelity levels from cache
    fidelities = sorted(set(f for (_, _, f) in hbbops.evaluation_cache.keys()))
    max_fidelity = max(fidelities) if fidelities else 0

    print(f"Grid size: {num_inst} x {num_ex} = {total_prompts} prompts")
    print(f"Fidelity levels: {len(fidelities)} ({min(fidelities) if fidelities else 0} to {max_fidelity})")

    # === LLM CALL COUNTING ===
    hbbops_llm_calls = sum(fidelity for (_, _, fidelity) in hbbops.evaluation_cache.keys())
    gt_llm_calls = total_prompts * 1319
    efficiency_ratio = gt_llm_calls / hbbops_llm_calls if hbbops_llm_calls > 0 else 0

    print(f"\n{'=' * 70}")
    print("LLM CALL STATISTICS")
    print(f"{'=' * 70}")
    print(f"HbBoPs LLM calls: {hbbops_llm_calls:,}")
    print(f"GT LLM calls:     {gt_llm_calls:,}")
    print(f"Efficiency ratio: {efficiency_ratio:.1f}x fewer calls")

    # === BUILD ALL EVALUATED PROMPTS (using max fidelity for each) ===
    prompt_max_fidelity = {}
    for (inst_id, ex_id, fidelity), error in hbbops.evaluation_cache.items():
        key = (inst_id, ex_id)
        if key not in prompt_max_fidelity or fidelity > prompt_max_fidelity[key][0]:
            prompt_max_fidelity[key] = (fidelity, error)

    # Build comparison list for ALL evaluated prompts
    all_comparisons = []
    total_diff = 0

    for (inst_id, ex_id), (max_fid, error) in prompt_max_fidelity.items():
        orig_inst = instruction_mapping[inst_id] if inst_id < len(instruction_mapping) else inst_id
        orig_ex = exemplar_mapping[ex_id] if ex_id < len(exemplar_mapping) else ex_id

        gt_error = gt_data.get((orig_inst, orig_ex), None)
        if gt_error is not None:
            diff = (error - gt_error) * 100
            total_diff += abs(diff)
            all_comparisons.append({
                'sel_inst': inst_id,
                'sel_ex': ex_id,
                'orig_inst': orig_inst,
                'orig_ex': orig_ex,
                'max_fidelity': max_fid,
                'hbbops_error': error,
                'gt_error': gt_error,
                'diff_pp': diff
            })

    all_comparisons.sort(key=lambda x: x['gt_error'])
    count = len(all_comparisons)
    coverage = count / total_prompts if total_prompts > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"ALL EVALUATED PROMPTS ({count}/{total_prompts} = {coverage*100:.1f}% coverage)")
    print(f"{'=' * 70}")
    print(f"{'sel_i':>5} {'sel_e':>5} {'orig_i':>6} {'orig_e':>6} {'fid':>5} {'HbBoPs':>8} {'GT':>8} {'Diff':>8}")
    print("-" * 70)

    for c in all_comparisons:
        print(f"{c['sel_inst']:>5} {c['sel_ex']:>5} {c['orig_inst']:>6} {c['orig_ex']:>6} "
              f"{c['max_fidelity']:>5} {c['hbbops_error']*100:>7.2f}% {c['gt_error']*100:>7.2f}% {c['diff_pp']:>+7.2f}pp")

    print("-" * 70)
    mean_abs_diff = total_diff / count if count > 0 else 0
    print(f"Mean absolute difference: {mean_abs_diff:.2f}pp")

    # === SPEARMAN CORRELATION ===
    if count > 1:
        hbbops_ranks = {(c['sel_inst'], c['sel_ex']): i for i, c in enumerate(sorted(all_comparisons, key=lambda x: x['hbbops_error']))}
        gt_ranks = {(c['sel_inst'], c['sel_ex']): i for i, c in enumerate(sorted(all_comparisons, key=lambda x: x['gt_error']))}

        n = count
        d_squared_sum = sum((hbbops_ranks[(c['sel_inst'], c['sel_ex'])] - gt_ranks[(c['sel_inst'], c['sel_ex'])])**2 for c in all_comparisons)
        spearman = 1 - (6 * d_squared_sum) / (n * (n**2 - 1)) if n > 1 else 0
    else:
        spearman = 0

    print(f"Spearman rank correlation (on {count} prompts): {spearman:.4f}")

    # === BEST PROMPT COMPARISON ===
    if all_comparisons:
        hbbops_best = min(all_comparisons, key=lambda x: x['hbbops_error'])
        gt_best = min(all_comparisons, key=lambda x: x['gt_error'])
        same_selection = (hbbops_best['sel_inst'], hbbops_best['sel_ex']) == (gt_best['sel_inst'], gt_best['sel_ex'])

        print(f"\nHbBoPs best: inst={hbbops_best['sel_inst']}, ex={hbbops_best['sel_ex']} "
              f"(error={hbbops_best['hbbops_error']*100:.2f}%, fid={hbbops_best['max_fidelity']})")
        print(f"GT best:     inst={gt_best['sel_inst']}, ex={gt_best['sel_ex']} (error={gt_best['gt_error']*100:.2f}%)")
        print(f"Same selection: {'YES' if same_selection else 'NO'}")
    else:
        hbbops_best = gt_best = None
        same_selection = False

    # === FIDELITY ANALYSIS ===
    print(f"\n{'=' * 70}")
    print("FIDELITY LEVEL ANALYSIS")
    print(f"{'=' * 70}")

    fidelity_analysis = []
    for fid in fidelities:
        fid_errors = [(inst_id, ex_id, err) for (inst_id, ex_id, f), err in hbbops.evaluation_cache.items() if f == fid]
        if fid_errors:
            best_at_fid = min(fid_errors, key=lambda x: x[2])
            print(f"Fidelity {fid:>4}: {len(fid_errors):>3} prompts, "
                  f"best=({best_at_fid[0]},{best_at_fid[1]}) error={best_at_fid[2]*100:.2f}%")
            fidelity_analysis.append({
                "fidelity": fid,
                "num_prompts": len(fid_errors),
                "best_inst": best_at_fid[0],
                "best_ex": best_at_fid[1],
                "best_error": best_at_fid[2]
            })

    # === SAVE RESULTS ===
    if output_dir and timestamp:
        comparison_results = {
            "ground_truth_path": ground_truth_path,
            "instruction_mapping": instruction_mapping,
            "exemplar_mapping": exemplar_mapping,
            "total_prompts_in_grid": total_prompts,

            "llm_calls": {
                "hbbops_total": hbbops_llm_calls,
                "gt_total": gt_llm_calls,
                "efficiency_ratio": efficiency_ratio
            },

            "all_evaluated_prompts": all_comparisons,

            "metrics": {
                "num_prompts_evaluated": count,
                "coverage": coverage,
                "spearman_all": spearman,
                "mean_abs_diff_pp": mean_abs_diff,
                "same_best_selection": same_selection
            },

            "hbbops_best": {
                "inst": hbbops_best['sel_inst'] if hbbops_best else None,
                "ex": hbbops_best['sel_ex'] if hbbops_best else None,
                "error": hbbops_best['hbbops_error'] if hbbops_best else None,
                "fidelity": hbbops_best['max_fidelity'] if hbbops_best else None
            },
            "gt_best": {
                "inst": gt_best['sel_inst'] if gt_best else None,
                "ex": gt_best['sel_ex'] if gt_best else None,
                "error": gt_best['gt_error'] if gt_best else None
            },

            "fidelity_levels": fidelities,
            "fidelity_analysis": fidelity_analysis
        }
        comparison_file = output_dir / f"hbbops_improved4_gt_comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\nComparison saved to {comparison_file}")


if __name__ == "__main__":
    main()
