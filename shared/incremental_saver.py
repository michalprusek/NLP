"""
Incremental Prompt Saver for benchmarking experiments.

Saves evaluated prompts incrementally to JSON after each evaluation,
enabling recovery from crashes and real-time monitoring.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class IncrementalPromptSaver:
    """
    Saves evaluated prompts incrementally to JSON.

    Each prompt evaluation is saved immediately, allowing:
    - Recovery from crashes (all previous evaluations preserved)
    - Real-time monitoring of experiment progress
    - Exact stopping at max_prompts limit
    """

    def __init__(
        self,
        output_path: str,
        method: str,
        model: str,
        config: Dict[str, Any],
    ):
        """
        Initialize the incremental saver.

        Args:
            output_path: Path to JSON output file
            method: Name of the optimization method (opro, protegi, gepa)
            model: Model name/alias used for evaluation
            config: Configuration dict with method-specific parameters
        """
        self.output_path = Path(output_path)
        self.data = {
            "method": method,
            "model": model,
            "started_at": datetime.now().isoformat(),
            "config": config,
            "evaluated_prompts": [],
            "completed_at": None,
            "best_prompt": None,
            "best_score": None,
            "total_evaluated": 0,
        }

        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write initial file
        self._write_file()

    @property
    def count(self) -> int:
        """Return the number of evaluated prompts saved so far."""
        return len(self.data["evaluated_prompts"])

    def save_prompt(
        self,
        prompt: str,
        score: float,
        iteration: int,
        method_specific: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a single evaluated prompt.

        Args:
            prompt: The prompt text that was evaluated
            score: Evaluation score (accuracy)
            iteration: Optimization iteration number
            method_specific: Optional method-specific metadata (e.g., candidate_idx for OPRO)
        """
        entry = {
            "eval_id": self.count + 1,
            "iteration": iteration,
            "prompt": prompt,
            "score": score,
            "timestamp": datetime.now().isoformat(),
        }

        if method_specific:
            entry["method_specific"] = method_specific

        self.data["evaluated_prompts"].append(entry)
        self.data["total_evaluated"] = self.count

        # Update best if this is better
        if self.data["best_score"] is None or score > self.data["best_score"]:
            self.data["best_prompt"] = prompt
            self.data["best_score"] = score

        # Write after each prompt (atomic operation)
        self._write_file()

    def finalize(self, best_prompt: str, best_score: float) -> None:
        """
        Finalize the experiment with final best prompt and score.

        Args:
            best_prompt: The best prompt found during optimization
            best_score: Score of the best prompt
        """
        self.data["completed_at"] = datetime.now().isoformat()
        self.data["best_prompt"] = best_prompt
        self.data["best_score"] = best_score
        self.data["total_evaluated"] = self.count
        self._write_file()

    def _write_file(self) -> None:
        """Write data to JSON file (overwrites existing)."""
        # Write to temp file first, then rename for atomicity
        temp_path = self.output_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            # Atomic rename
            temp_path.rename(self.output_path)
        except Exception:
            # Fallback: direct write
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
