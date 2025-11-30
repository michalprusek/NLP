"""
Component Scoring System for HYPE

Computes two key metrics:
- S_I (Instruction Robustness Score): How well an instruction performs across exemplars
- S_E (Exemplar Synergy Score): How consistently an exemplar performs across instructions
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

from hype.data_types import EvaluationRecord, ComponentScore


class ComponentScorer:
    """
    Compute component-level scores from Hyperband evaluation history.

    Key insight: Weight scores by budget (fidelity level) because
    higher-budget evaluations are more reliable.
    """

    def __init__(self):
        self.records: List[EvaluationRecord] = []
        self._by_instruction: Dict[int, List[EvaluationRecord]] = defaultdict(list)
        self._by_exemplar: Dict[int, List[EvaluationRecord]] = defaultdict(list)
        self._by_pair: Dict[Tuple[int, int], List[EvaluationRecord]] = defaultdict(list)

        # Cached scores (invalidated on new records)
        self._instruction_scores: Dict[int, ComponentScore] = {}
        self._exemplar_scores: Dict[int, ComponentScore] = {}
        self._cache_valid = False

    def add_record(self, record: EvaluationRecord) -> None:
        """Add a single evaluation record"""
        self.records.append(record)
        self._by_instruction[record.instruction_id].append(record)
        self._by_exemplar[record.exemplar_id].append(record)
        self._by_pair[(record.instruction_id, record.exemplar_id)].append(record)
        self._cache_valid = False

    def add_records(self, records: List[EvaluationRecord]) -> None:
        """Add multiple evaluation records"""
        for record in records:
            self.add_record(record)

    def _ensure_cache(self) -> None:
        """Recompute scores if cache is invalid"""
        if self._cache_valid:
            return

        self._instruction_scores = {}
        self._exemplar_scores = {}

        # Compute instruction scores
        for inst_id in self._by_instruction:
            self._instruction_scores[inst_id] = self._compute_instruction_score(inst_id)

        # Compute exemplar scores
        for ex_id in self._by_exemplar:
            self._exemplar_scores[ex_id] = self._compute_exemplar_score(ex_id)

        self._cache_valid = True

    def _compute_instruction_score(self, instruction_id: int) -> ComponentScore:
        """
        Compute S_I: Instruction Robustness Score

        Formula: S_I = sum(budget * accuracy) / sum(budget)

        Higher budget evaluations contribute more weight because they are
        more reliable estimates of true performance.
        """
        records = self._by_instruction[instruction_id]
        if not records:
            return ComponentScore(instruction_id, 0.0, 0.0, 0, 0)

        weighted_sum = 0.0
        weight_total = 0.0
        accuracies = []
        max_budget = 0

        for r in records:
            accuracy = r.accuracy
            weight = r.budget  # Use fidelity as weight
            weighted_sum += weight * accuracy
            weight_total += weight
            accuracies.append(accuracy)
            max_budget = max(max_budget, r.budget)

        score = weighted_sum / weight_total if weight_total > 0 else 0.0
        variance = float(np.var(accuracies)) if len(accuracies) > 1 else 0.0

        return ComponentScore(
            component_id=instruction_id,
            score=score,
            variance=variance,
            num_evaluations=len(records),
            max_budget_seen=max_budget
        )

    def _compute_exemplar_score(self, exemplar_id: int) -> ComponentScore:
        """
        Compute S_E: Exemplar Synergy Score

        We compute mean accuracy and variance across instructions.
        Low variance indicates the exemplar works consistently (synergistic).
        High variance indicates it's specialized to certain instructions.

        The score combines mean and inverse variance:
        S_E = mean_accuracy * (1 / (1 + sqrt(variance)))
        """
        records = self._by_exemplar[exemplar_id]
        if not records:
            return ComponentScore(exemplar_id, 0.0, 0.0, 0, 0)

        # Group by instruction to get per-instruction performance
        by_instruction: Dict[int, List[float]] = defaultdict(list)
        max_budget = 0

        for r in records:
            by_instruction[r.instruction_id].append(r.accuracy)
            max_budget = max(max_budget, r.budget)

        # Compute mean accuracy per instruction
        instruction_means = [np.mean(accs) for accs in by_instruction.values()]

        if len(instruction_means) < 2:
            # Not enough data to compute variance
            mean_acc = float(np.mean(instruction_means)) if instruction_means else 0.0
            return ComponentScore(
                component_id=exemplar_id,
                score=mean_acc,
                variance=0.0,
                num_evaluations=len(records),
                max_budget_seen=max_budget
            )

        mean_acc = float(np.mean(instruction_means))
        variance = float(np.var(instruction_means))

        # Synergy score: high accuracy + low variance
        synergy_factor = 1.0 / (1.0 + np.sqrt(variance))
        score = mean_acc * synergy_factor

        return ComponentScore(
            component_id=exemplar_id,
            score=score,
            variance=variance,
            num_evaluations=len(records),
            max_budget_seen=max_budget
        )

    def get_instruction_score(self, instruction_id: int) -> Optional[ComponentScore]:
        """Get score for a specific instruction"""
        self._ensure_cache()
        return self._instruction_scores.get(instruction_id)

    def get_exemplar_score(self, exemplar_id: int) -> Optional[ComponentScore]:
        """Get score for a specific exemplar"""
        self._ensure_cache()
        return self._exemplar_scores.get(exemplar_id)

    def get_all_instruction_scores(self) -> Dict[int, ComponentScore]:
        """Get scores for all instructions"""
        self._ensure_cache()
        return self._instruction_scores.copy()

    def get_all_exemplar_scores(self) -> Dict[int, ComponentScore]:
        """Get scores for all exemplars"""
        self._ensure_cache()
        return self._exemplar_scores.copy()

    def get_top_instructions(self, n: int = 5) -> List[Tuple[int, ComponentScore]]:
        """Get top-N instructions by S_I score"""
        self._ensure_cache()
        sorted_scores = sorted(
            self._instruction_scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        return sorted_scores[:n]

    def get_top_exemplars(self, n: int = 5) -> List[Tuple[int, ComponentScore]]:
        """Get top-N exemplars by S_E score"""
        self._ensure_cache()
        sorted_scores = sorted(
            self._exemplar_scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        return sorted_scores[:n]

    def get_stable_exemplars(self, n: int = 5) -> List[Tuple[int, ComponentScore]]:
        """
        Get exemplars with lowest variance (most stable across instructions).
        Secondary sort by mean accuracy.
        """
        self._ensure_cache()
        sorted_scores = sorted(
            self._exemplar_scores.items(),
            key=lambda x: (x[1].variance, -x[1].score)  # Low variance, high score
        )
        return sorted_scores[:n]

    def get_weak_instructions(self, n: int = 5) -> List[Tuple[int, ComponentScore]]:
        """Get bottom-N instructions for improvement via semantic gradient"""
        self._ensure_cache()
        sorted_scores = sorted(
            self._instruction_scores.items(),
            key=lambda x: x[1].score
        )
        return sorted_scores[:n]

    def get_survival_depth(self, instruction_id: int) -> int:
        """
        Get maximum budget (fidelity) this instruction survived to.
        Higher = instruction survived more Hyperband rounds.
        """
        records = self._by_instruction[instruction_id]
        return max((r.budget for r in records), default=0)

    def get_pair_performance(
        self, instruction_id: int, exemplar_id: int
    ) -> Optional[float]:
        """Get best accuracy for a specific instruction-exemplar pair"""
        records = self._by_pair.get((instruction_id, exemplar_id), [])
        if not records:
            return None
        # Return accuracy at highest budget
        best_record = max(records, key=lambda r: r.budget)
        return best_record.accuracy

    def get_untried_pairs(
        self,
        instruction_ids: List[int],
        exemplar_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """Get pairs that haven't been evaluated yet"""
        untried = []
        for inst_id in instruction_ids:
            for ex_id in exemplar_ids:
                if (inst_id, ex_id) not in self._by_pair:
                    untried.append((inst_id, ex_id))
        return untried

    def summary(self) -> Dict:
        """Get summary statistics"""
        self._ensure_cache()

        inst_scores = [s.score for s in self._instruction_scores.values()]
        ex_scores = [s.score for s in self._exemplar_scores.values()]

        return {
            "num_records": len(self.records),
            "num_instructions": len(self._instruction_scores),
            "num_exemplars": len(self._exemplar_scores),
            "instruction_scores": {
                "mean": float(np.mean(inst_scores)) if inst_scores else 0.0,
                "std": float(np.std(inst_scores)) if inst_scores else 0.0,
                "min": float(np.min(inst_scores)) if inst_scores else 0.0,
                "max": float(np.max(inst_scores)) if inst_scores else 0.0,
            },
            "exemplar_scores": {
                "mean": float(np.mean(ex_scores)) if ex_scores else 0.0,
                "std": float(np.std(ex_scores)) if ex_scores else 0.0,
                "min": float(np.min(ex_scores)) if ex_scores else 0.0,
                "max": float(np.max(ex_scores)) if ex_scores else 0.0,
            }
        }
