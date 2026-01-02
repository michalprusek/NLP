"""HbBoPs results extraction and saving utilities.

Provides functions to:
- Parse HbBoPs evaluation results from log files
- Save results to JSON format
- Load results from JSON format
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime


def parse_hbbops_log(log_path: str) -> Dict[int, Dict]:
    """Parse HbBoPs evaluation results from log file.

    Extracts prompt evaluations at all fidelity levels and returns
    the highest fidelity evaluation for each prompt.

    Args:
        log_path: Path to the log file

    Returns:
        Dict mapping prompt index to evaluation result:
        {
            prompt_idx: {
                "error_rate": float,
                "accuracy": float,
                "fidelity": int
            }
        }
    """
    # Pattern: [N/M] Prompt IDX: error=E at fidelity=F
    # or: Prompt IDX: error=E at fidelity=F
    pattern = re.compile(
        r"Prompt (\d+): error=([\d.]+) at fidelity=(\d+)"
    )

    results: Dict[int, Dict] = {}

    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    prompt_idx = int(match.group(1))
                    error_rate = float(match.group(2))
                    fidelity = int(match.group(3))

                    # Keep only highest fidelity for each prompt
                    if prompt_idx not in results or fidelity > results[prompt_idx]["fidelity"]:
                        results[prompt_idx] = {
                            "error_rate": error_rate,
                            "accuracy": 1.0 - error_rate,
                            "fidelity": fidelity
                        }
    except FileNotFoundError:
        raise FileNotFoundError(
            f"HbBoPs log file not found: {log_path}\n"
            f"Ensure the log file exists or run HbBoPs first."
        )

    if not results:
        raise ValueError(
            f"No evaluation results found in log file: {log_path}\n"
            f"Expected lines matching pattern: 'Prompt N: error=X.XX at fidelity=Y'\n"
            f"The file may be empty or have unexpected format."
        )

    return results


def save_hbbops_results(
    results: Dict[int, Dict],
    output_path: str,
    source_log: Optional[str] = None,
    max_fidelity: int = 1319,
    instructions: Optional[list] = None,
) -> None:
    """Save HbBoPs evaluation results to JSON file.

    Args:
        results: Dict from parse_hbbops_log or from Hyperband design_data
        output_path: Path to save JSON file
        source_log: Path to source log file (for metadata)
        max_fidelity: Maximum possible fidelity (validation set size)
        instructions: Optional list of instruction strings (indexed by prompt_idx)
    """
    output = {
        "metadata": {
            "source_log": source_log,
            "max_fidelity": max_fidelity,
            "num_evaluated": len(results),
            "created_at": datetime.now().isoformat(),
        },
        "results": {}
    }

    for idx, data in results.items():
        result_entry = {
            "error_rate": data["error_rate"],
            "accuracy": data["accuracy"],
            "fidelity": data["fidelity"]
        }
        if instructions and idx < len(instructions):
            result_entry["instruction"] = instructions[idx]
        output["results"][str(idx)] = result_entry

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved HbBoPs results to {output_path}")
    print(f"  Evaluated prompts: {len(results)}")

    # Print fidelity distribution
    fidelities = [d["fidelity"] for d in results.values()]
    if fidelities:
        max_f = max(fidelities)
        full_fidelity_count = sum(1 for f in fidelities if f == max_f)
        print(f"  Max fidelity reached: {max_f}")
        print(f"  Prompts at max fidelity: {full_fidelity_count}")


def load_hbbops_results(json_path: str) -> Tuple[Dict[int, Dict], Dict]:
    """Load HbBoPs results from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Tuple of (results dict, metadata dict)

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        ValueError: If the JSON is invalid or missing required keys
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"HbBoPs results file not found: {json_path}\n"
            f"Run HbBoPs or use --hyperband-evals-path to specify correct path."
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in HbBoPs results file: {json_path}\n"
            f"JSON error: {e}"
        )

    if "results" not in data:
        raise ValueError(
            f"Missing 'results' key in HbBoPs file: {json_path}\n"
            f"Expected format: {{'metadata': ..., 'results': {{...}}}}\n"
            f"Found keys: {list(data.keys())}"
        )

    results = {}
    for idx_str, result in data["results"].items():
        results[int(idx_str)] = result

    return results, data.get("metadata", {})


def extract_from_hyperband(hyperband) -> Dict[int, Dict]:
    """Extract results from Hyperband object's design_data.

    Args:
        hyperband: Hyperband instance with design_data attribute

    Returns:
        Dict mapping prompt index to evaluation result
    """
    results: Dict[int, Dict] = {}

    for inst_id, _embedding, error, fidelity in hyperband.design_data:
        if inst_id not in results or fidelity > results[inst_id]["fidelity"]:
            results[inst_id] = {
                "error_rate": error,
                "accuracy": 1.0 - error,
                "fidelity": fidelity
            }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract HbBoPs results from log file")
    parser.add_argument("log_path", help="Path to log file")
    parser.add_argument(
        "--output", "-o",
        default="lipo/data/hbbops_results.json",
        help="Output JSON path (default: lipo/data/hbbops_results.json)"
    )
    parser.add_argument(
        "--instructions",
        default="lipo/data/ape_instructions.json",
        help="Path to instructions JSON to include instruction text"
    )
    parser.add_argument(
        "--max-fidelity",
        type=int,
        default=1319,
        help="Maximum fidelity (validation set size)"
    )

    args = parser.parse_args()

    # Parse log
    print(f"Parsing {args.log_path}...")
    results = parse_hbbops_log(args.log_path)

    # Load instructions if available
    instructions = None
    if Path(args.instructions).exists():
        with open(args.instructions, 'r') as f:
            data = json.load(f)
            instructions = data.get("instructions", [])
        print(f"Loaded {len(instructions)} instructions from {args.instructions}")

    # Save results
    save_hbbops_results(
        results=results,
        output_path=args.output,
        source_log=args.log_path,
        max_fidelity=args.max_fidelity,
        instructions=instructions,
    )
