"""
Per-model GP-based candidate selection.

This module implements independent candidate selection for each model
using GP predictions with UCB (Upper Confidence Bound) scoring.

Algorithm:
1. Get GP predictions (mean, std) for all candidates on all models
2. For each model, rank candidates by UCB score: mean - kappa * std
   (Lower is better since we predict error rates)
3. Select top-k per model
4. Return union of all selected candidates (up to max_union)
"""
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_model_hybrid.config import (
    MultiModelHybridConfig,
    MultiModelHybridCandidate,
    PerModelCandidateScore,
)
from multi_model_optimizer.multi_output_gp import MultiOutputGPTrainer


class PerModelSelector:
    """
    Selects top-k candidates independently for each model using GP.

    This allows each model to "vote" for its best candidates,
    ensuring we explore candidates that may be particularly good
    for specific models even if aggregated score is not optimal.

    Example:
        >>> selector = PerModelSelector(gp_trainer, config)
        >>> union_candidates, per_model_info = selector.select_candidates(candidates)
        >>> # union_candidates: up to 30 unique candidates
        >>> # per_model_info: {'model_A': [idx1, idx2, ...], 'model_B': [...]}
    """

    def __init__(
        self,
        gp_trainer: MultiOutputGPTrainer,
        config: MultiModelHybridConfig,
    ):
        """
        Initialize per-model selector.

        Args:
            gp_trainer: Trained multi-output GP with ICM kernel
            config: Configuration with per-model selection params
        """
        self.gp_trainer = gp_trainer
        self.config = config
        self.model_names = config.target_models

    def select_candidates(
        self,
        candidates: List[MultiModelHybridCandidate],
    ) -> Tuple[List[MultiModelHybridCandidate], Dict[str, List[int]]]:
        """
        Select candidates using per-model GP predictions.

        Args:
            candidates: All candidates to consider

        Returns:
            Tuple of:
                - Selected candidates (union of per-model top-k)
                - Dict mapping model_name -> list of selected candidate indices
        """
        if not candidates:
            return [], {}

        if self.gp_trainer.gp_params is None:
            # GP not trained: return random subset
            random.shuffle(candidates)
            max_n = self.config.per_model_selection.max_union_candidates
            return candidates[:max_n], {}

        # Compute GP predictions for all candidates
        self._compute_gp_predictions(candidates)

        # Select top-k per model
        per_model_selections: Dict[str, List[int]] = {}
        selected_indices: Set[int] = set()

        for model_name in self.model_names:
            # Sort by UCB score (lower is better for error rate)
            model_scores = [
                (i, c.gp_predictions[model_name].ucb_score)
                for i, c in enumerate(candidates)
                if model_name in c.gp_predictions
            ]
            model_scores.sort(key=lambda x: x[1])

            # Take top-k
            top_k = self.config.per_model_selection.top_k_per_model
            top_indices = [idx for idx, _ in model_scores[:top_k]]

            per_model_selections[model_name] = top_indices
            selected_indices.update(top_indices)

            # Update rank info and selected_by_models
            for rank, idx in enumerate(top_indices):
                candidates[idx].gp_predictions[model_name].rank_in_model = rank + 1
                if model_name not in candidates[idx].selected_by_models:
                    candidates[idx].selected_by_models.append(model_name)

        # Build union list
        union_candidates = [candidates[i] for i in sorted(selected_indices)]

        # Limit to max_union if needed (prioritize multi-model selected)
        max_union = self.config.per_model_selection.max_union_candidates
        if len(union_candidates) > max_union:
            # Sort by number of models that selected this candidate (descending)
            union_candidates.sort(
                key=lambda c: -len(c.selected_by_models)
            )
            union_candidates = union_candidates[:max_union]

        return union_candidates, per_model_selections

    def _compute_gp_predictions(
        self,
        candidates: List[MultiModelHybridCandidate],
    ) -> None:
        """
        Compute GP predictions for all candidates on all models.

        Updates candidates in-place with gp_predictions dict.

        Args:
            candidates: List of candidates to score
        """
        n = len(candidates)
        inst_embs = np.array([c.instruction_embedding for c in candidates])
        ex_embs = np.array([c.exemplar_embedding for c in candidates])

        # Get predictions from multi-output GP
        # Returns: Dict[model_name, np.ndarray] for means and stds
        means, stds = self.gp_trainer.predict(inst_embs, ex_embs)

        kappa = self.config.per_model_selection.ucb_kappa

        for i, candidate in enumerate(candidates):
            candidate.gp_predictions = {}

            for model_name in self.model_names:
                mean_error = float(means[model_name][i])
                std_error = float(stds[model_name][i])

                # UCB for minimization: lower bound of error
                # We want to explore candidates that MIGHT have low error
                # UCB = mean - kappa * std gives optimistic estimate
                ucb_score = mean_error - kappa * std_error

                candidate.gp_predictions[model_name] = PerModelCandidateScore(
                    instruction_id=candidate.instruction_id,
                    exemplar_id=candidate.exemplar_id,
                    model_name=model_name,
                    gp_mean=mean_error,
                    gp_std=std_error,
                    ucb_score=ucb_score,
                    rank_in_model=-1,  # Set later during selection
                )

    def get_selection_summary(
        self,
        candidates: List[MultiModelHybridCandidate],
        per_model_selections: Dict[str, List[int]],
    ) -> str:
        """
        Generate human-readable summary of selection.

        Args:
            candidates: All candidates
            per_model_selections: Per-model selection indices

        Returns:
            Summary string
        """
        lines = ["Per-Model GP Selection Summary:"]

        for model_name, indices in per_model_selections.items():
            model_short = model_name.split("/")[-1]
            lines.append(f"\n  {model_short}:")

            for rank, idx in enumerate(indices[:5]):  # Show top 5
                c = candidates[idx]
                score = c.gp_predictions[model_name]
                lines.append(
                    f"    {rank+1}. inst={c.instruction_id}, ex={c.exemplar_id}, "
                    f"mean={score.gp_mean:.4f}, std={score.gp_std:.4f}, "
                    f"ucb={score.ucb_score:.4f}"
                )

        # Count overlap
        all_selected = set()
        for indices in per_model_selections.values():
            all_selected.update(indices)

        overlap_counts = {}
        for idx in all_selected:
            count = sum(1 for indices in per_model_selections.values() if idx in indices)
            overlap_counts[count] = overlap_counts.get(count, 0) + 1

        lines.append(f"\n  Union size: {len(all_selected)}")
        lines.append(f"  Overlap distribution:")
        for count in sorted(overlap_counts.keys(), reverse=True):
            lines.append(f"    Selected by {count} models: {overlap_counts[count]} candidates")

        return "\n".join(lines)
