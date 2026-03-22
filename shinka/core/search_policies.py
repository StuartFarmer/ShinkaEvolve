"""
Search-policy services used by context sampling.

These classes consume repository state and computed archive state. They do not
own persistence.
"""

from __future__ import annotations

import logging
import random
from typing import Any, List, Optional, Sequence

import numpy as np

from shinka.database.program import Program
from shinka.controllers.program_controller import ProgramController

logger = logging.getLogger(__name__)


def _sample_with_powerlaw(items: Sequence[Program], alpha: float = 1.0) -> Program:
    if not items:
        raise ValueError("Empty items list for power-law sampling")

    probs = np.array([(i + 1) ** (-alpha) for i in range(len(items))], dtype=float)
    if float(np.sum(probs)) == 0.0:
        probs = np.ones(len(items), dtype=float)
    probs = probs / np.sum(probs)
    return items[int(np.random.choice(len(items), p=probs))]


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        exp_neg_x = np.exp(-x)
        return float(1.0 / (1.0 + exp_neg_x))
    exp_x = np.exp(x)
    return float(exp_x / (1.0 + exp_x))


def _sort_programs_by_score(programs: Sequence[Program]) -> List[Program]:
    def score(program: Program) -> float:
        if program.combined_score is not None:
            return float(program.combined_score)
        if program.public_metrics:
            return sum(program.public_metrics.values()) / len(program.public_metrics)
        return -float("inf")

    return sorted(
        programs,
        key=lambda program: (score(program), float(program.timestamp)),
        reverse=True,
    )


class ParentSelector:
    """Repository-backed parent selection policy."""

    def __init__(
        self,
        *,
        parent_selection_strategy: str = "weighted",
        exploitation_alpha: float = 1.0,
        parent_selection_lambda: float = 10.0,
        num_beams: int = 5,
    ):
        self.parent_selection_strategy = parent_selection_strategy
        self.exploitation_alpha = exploitation_alpha
        self.parent_selection_lambda = parent_selection_lambda
        self.num_beams = num_beams

    def has_correct_programs(
        self,
        repository: ProgramController,
        *,
        island_idx: Optional[int] = None,
    ) -> bool:
        return bool(repository.list_correct(island_idx=island_idx))

    def get_incorrect_program_for_fix(
        self,
        repository: ProgramController,
        *,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        incorrect = repository.list_incorrect(island_idx=island_idx)
        if not incorrect:
            return None
        return random.choice(incorrect)

    def select_with_fix_mode(
        self,
        repository: ProgramController,
        archive_programs: Sequence[Program],
        *,
        island_idx: Optional[int] = None,
    ) -> tuple[Program, bool]:
        if not self.has_correct_programs(repository, island_idx=island_idx):
            incorrect = self.get_incorrect_program_for_fix(
                repository,
                island_idx=island_idx,
            )
            if incorrect is not None:
                return incorrect, True
            raise ValueError("Repository empty - no programs to sample or fix.")
        return self.select(repository, archive_programs, island_idx=island_idx), False

    def select(
        self,
        repository: ProgramController,
        archive_programs: Sequence[Program],
        *,
        island_idx: Optional[int] = None,
    ) -> Program:
        strategy_name = self.parent_selection_strategy

        if strategy_name == "power_law":
            parent = self._select_power_law(repository, archive_programs, island_idx)
        elif strategy_name == "weighted":
            parent = self._select_weighted(repository, archive_programs, island_idx)
        elif strategy_name == "beam_search":
            parent = self._select_beam_search(repository, island_idx)
        elif strategy_name == "best_of_n":
            parent = self._select_best_of_n(repository, island_idx)
        elif strategy_name == "winner_take_all":
            parent = self._select_winner_take_all(repository, island_idx)
        elif strategy_name == "sequential":
            parent = self._select_sequential(repository, island_idx)
        else:
            raise ValueError(f"Unknown parent selection strategy: {strategy_name}")

        if parent is not None:
            return parent

        fallback = repository.get_best(island_idx=island_idx)
        if fallback is not None:
            return fallback

        fallback = repository.get_most_recent(island_idx=island_idx)
        if fallback is not None:
            return fallback

        raise ValueError("Repository empty or parent sampling failed.")

    def _archive_candidates(
        self,
        archive_programs: Sequence[Program],
        island_idx: Optional[int],
    ) -> List[Program]:
        candidates = [program for program in archive_programs if program.correct]
        if island_idx is not None:
            candidates = [program for program in candidates if program.island_idx == island_idx]
        return _sort_programs_by_score(candidates)

    def _correct_candidates(
        self,
        repository: ProgramController,
        island_idx: Optional[int],
    ) -> List[Program]:
        return _sort_programs_by_score(repository.list_correct(island_idx=island_idx))

    def _select_power_law(
        self,
        repository: ProgramController,
        archive_programs: Sequence[Program],
        island_idx: Optional[int],
    ) -> Optional[Program]:
        candidates = self._archive_candidates(archive_programs, island_idx)
        if candidates:
            return _sample_with_powerlaw(candidates, self.exploitation_alpha)

        candidates = self._correct_candidates(repository, island_idx)
        if candidates:
            return _sample_with_powerlaw(candidates, self.exploitation_alpha)

        return repository.get_best(island_idx=island_idx)

    def _select_weighted(
        self,
        repository: ProgramController,
        archive_programs: Sequence[Program],
        island_idx: Optional[int],
    ) -> Optional[Program]:
        candidates = self._archive_candidates(archive_programs, island_idx)
        if not candidates:
            return repository.get_best(island_idx=island_idx)

        scores = [float(program.combined_score or 0.0) for program in candidates]
        alpha_0 = float(np.median(scores)) if scores else 0.0
        score_deviations = [abs(score - alpha_0) for score in scores]
        mad = float(np.median(score_deviations)) if score_deviations else 1.0
        scale_factor = max(mad, 1e-6)
        lambda_ = float(self.parent_selection_lambda)

        weights = []
        for program in candidates:
            normalized_diff = (float(program.combined_score or 0.0) - alpha_0) / scale_factor
            s_i = _stable_sigmoid(lambda_ * normalized_diff)
            h_i = 1.0 / (1.0 + float(program.children_count or 0))
            weights.append(s_i * h_i)

        weights_arr = np.array(weights, dtype=float)
        if float(np.sum(weights_arr)) == 0.0:
            weights_arr = np.ones(len(candidates), dtype=float) / len(candidates)
        else:
            weights_arr = weights_arr / np.sum(weights_arr)

        idx = int(np.random.choice(len(candidates), p=weights_arr))
        return candidates[idx]

    def _select_beam_search(
        self,
        repository: ProgramController,
        island_idx: Optional[int],
    ) -> Optional[Program]:
        num_beams = int(self.num_beams)
        beam_parent_id = repository.get_metadata("beam_search_parent_id")

        if beam_parent_id:
            beam_parent = repository.get(beam_parent_id)
            if beam_parent is not None and (
                island_idx is None or beam_parent.island_idx == island_idx
            ):
                children_count = repository.get_children_count(beam_parent.id)
                if children_count < num_beams:
                    return beam_parent

        best_program = repository.get_best(island_idx=island_idx)
        if best_program is not None:
            if not repository.read_only:
                repository.set_metadata("beam_search_parent_id", best_program.id)
            return best_program
        return None

    def _select_best_of_n(
        self,
        repository: ProgramController,
        island_idx: Optional[int],
    ) -> Optional[Program]:
        programs = repository.list_by_island(island_idx, correct_only=True) if island_idx is not None else repository.list_correct()
        generation_zero = [
            program for program in programs if program.generation == 0 and program.correct
        ]
        generation_zero = sorted(
            generation_zero,
            key=lambda program: (program.generation, program.timestamp, program.id),
        )
        if generation_zero:
            return generation_zero[0]
        return repository.get_earliest(correct_only=True, island_idx=island_idx)

    def _select_winner_take_all(
        self,
        repository: ProgramController,
        island_idx: Optional[int],
    ) -> Optional[Program]:
        best = repository.get_best(island_idx=island_idx)
        if best is not None:
            return best
        return repository.get_most_recent(correct_only=True, island_idx=island_idx)

    def _select_sequential(
        self,
        repository: ProgramController,
        island_idx: Optional[int],
    ) -> Optional[Program]:
        program = repository.get_most_recent(correct_only=True, island_idx=island_idx)
        if program is not None:
            return program
        return repository.get_most_recent(island_idx=island_idx)


class InspirationSelector:
    """Repository-backed inspiration selection policy."""

    def __init__(
        self,
        *,
        enforce_island_separation: bool = False,
        elite_selection_ratio: float = 0.0,
    ):
        self.enforce_island_separation = enforce_island_separation
        self.elite_selection_ratio = elite_selection_ratio

    def select_archive(
        self,
        repository: ProgramController,
        parent: Program,
        archive_programs: Sequence[Program],
        *,
        n: int,
    ) -> List[Program]:
        if n <= 0:
            return []

        enforce_separation = self.enforce_island_separation
        parent_island_idx = parent.island_idx
        inspirations: List[Program] = []
        selected_ids = {parent.id}

        candidate_archive = [program for program in archive_programs if program.correct]
        if enforce_separation and parent_island_idx is not None:
            candidate_archive = [
                program
                for program in candidate_archive
                if program.island_idx == parent_island_idx
            ]

        best_program = repository.get_best()
        if (
            best_program is not None
            and best_program.correct
            and best_program.id not in selected_ids
            and (
                not enforce_separation
                or best_program.island_idx == parent_island_idx
            )
        ):
            inspirations.append(best_program)
            selected_ids.add(best_program.id)

        num_elites = max(0, int(n * self.elite_selection_ratio))
        for program in _sort_programs_by_score(candidate_archive):
            if len(inspirations) >= n or len(inspirations) >= num_elites + (1 if best_program else 0):
                break
            if program.id in selected_ids:
                continue
            inspirations.append(program)
            selected_ids.add(program.id)

        remaining_candidates = [
            program
            for program in candidate_archive
            if program.id not in selected_ids
        ]
        random.shuffle(remaining_candidates)
        for program in remaining_candidates:
            if len(inspirations) >= n:
                break
            inspirations.append(program)
            selected_ids.add(program.id)

        if len(inspirations) < n and not enforce_separation:
            global_candidates = [
                program
                for program in _sort_programs_by_score(archive_programs)
                if program.correct and program.id not in selected_ids
            ]
            random.shuffle(global_candidates)
            for program in global_candidates:
                if len(inspirations) >= n:
                    break
                inspirations.append(program)
                selected_ids.add(program.id)

        return inspirations[:n]

    def select_top_k(
        self,
        parent: Program,
        archive_programs: Sequence[Program],
        *,
        excluded_programs: Sequence[Program],
        k: int,
    ) -> List[Program]:
        if k <= 0:
            return []

        enforce_separation = self.enforce_island_separation
        parent_island_idx = parent.island_idx
        excluded_ids = {parent.id}
        excluded_ids.update(program.id for program in excluded_programs)

        candidates = [program for program in archive_programs if program.correct]
        if enforce_separation:
            if parent_island_idx is None:
                return []
            candidates = [
                program for program in candidates if program.island_idx == parent_island_idx
            ]

        ranked = [
            program
            for program in _sort_programs_by_score(candidates)
            if program.id not in excluded_ids
        ]
        return ranked[:k]
