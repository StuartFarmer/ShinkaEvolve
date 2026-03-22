"""
Search-policy services used by context sampling.

These classes consume database state and computed archive state. They do not
own persistence.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Sequence

import numpy as np

from shinka.database import program_reads, run_state_ops
from shinka.database.connection import Database
from shinka.database.program import Program

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
    """Database-backed parent selection policy."""

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
        db: Database,
        *,
        island_idx: Optional[int] = None,
    ) -> bool:
        with db.session() as session:
            return bool(program_reads.list_correct(session, island_idx=island_idx))

    def get_incorrect_program_for_fix(
        self,
        db: Database,
        *,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        with db.session() as session:
            incorrect = program_reads.list_incorrect(session, island_idx=island_idx)
        if not incorrect:
            return None
        return random.choice(incorrect)

    def select_with_fix_mode(
        self,
        db: Database,
        archive_programs: Sequence[Program],
        *,
        island_idx: Optional[int] = None,
    ) -> tuple[Program, bool]:
        if not self.has_correct_programs(db, island_idx=island_idx):
            incorrect = self.get_incorrect_program_for_fix(db, island_idx=island_idx)
            if incorrect is not None:
                return incorrect, True
            raise ValueError("No programs available to sample or fix.")
        return self.select(db, archive_programs, island_idx=island_idx), False

    def select(
        self,
        db: Database,
        archive_programs: Sequence[Program],
        *,
        island_idx: Optional[int] = None,
    ) -> Program:
        strategy_name = self.parent_selection_strategy

        if strategy_name == "power_law":
            parent = self._select_power_law(db, archive_programs, island_idx)
        elif strategy_name == "weighted":
            parent = self._select_weighted(db, archive_programs, island_idx)
        elif strategy_name == "beam_search":
            parent = self._select_beam_search(db, island_idx)
        elif strategy_name == "best_of_n":
            parent = self._select_best_of_n(db, island_idx)
        elif strategy_name == "winner_take_all":
            parent = self._select_winner_take_all(db, island_idx)
        elif strategy_name == "sequential":
            parent = self._select_sequential(db, island_idx)
        else:
            raise ValueError(f"Unknown parent selection strategy: {strategy_name}")

        if parent is not None:
            return parent

        with db.session() as session:
            fallback = program_reads.get_best(session, island_idx=island_idx)
        if fallback is not None:
            return fallback

        with db.session() as session:
            fallback = program_reads.get_most_recent(session, island_idx=island_idx)
        if fallback is not None:
            return fallback

        raise ValueError("Database is empty or parent sampling failed.")

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
        db: Database,
        island_idx: Optional[int],
    ) -> List[Program]:
        with db.session() as session:
            return _sort_programs_by_score(
                program_reads.list_correct(session, island_idx=island_idx)
            )

    def _select_power_law(
        self,
        db: Database,
        archive_programs: Sequence[Program],
        island_idx: Optional[int],
    ) -> Optional[Program]:
        candidates = self._archive_candidates(archive_programs, island_idx)
        if candidates:
            return _sample_with_powerlaw(candidates, self.exploitation_alpha)

        candidates = self._correct_candidates(db, island_idx)
        if candidates:
            return _sample_with_powerlaw(candidates, self.exploitation_alpha)

        with db.session() as session:
            return program_reads.get_best(session, island_idx=island_idx)

    def _select_weighted(
        self,
        db: Database,
        archive_programs: Sequence[Program],
        island_idx: Optional[int],
    ) -> Optional[Program]:
        candidates = self._archive_candidates(archive_programs, island_idx)
        if not candidates:
            with db.session() as session:
                return program_reads.get_best(session, island_idx=island_idx)

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
        db: Database,
        island_idx: Optional[int],
    ) -> Optional[Program]:
        num_beams = int(self.num_beams)
        with db.session_scope() as session:
            beam_parent_id = run_state_ops.get(session, "beam_search_parent_id")

            if beam_parent_id:
                beam_parent = program_reads.get(session, beam_parent_id)
                if beam_parent is not None and (
                    island_idx is None or beam_parent.island_idx == island_idx
                ):
                    children_count = program_reads.get_children_count(session, beam_parent.id)
                    if children_count < num_beams:
                        return beam_parent

            best_program = program_reads.get_best(session, island_idx=island_idx)
            if best_program is not None:
                if not db.read_only:
                    run_state_ops.set(session, "beam_search_parent_id", best_program.id)
                return best_program
        return None

    def _select_best_of_n(
        self,
        db: Database,
        island_idx: Optional[int],
    ) -> Optional[Program]:
        with db.session() as session:
            candidates = (
                program_reads.list_by_island(session, island_idx, correct_only=True)
                if island_idx is not None
                else program_reads.list_correct(session)
            )
            generation_zero = [
                program for program in candidates if program.generation == 0 and program.correct
            ]
            generation_zero = sorted(
                generation_zero,
                key=lambda program: (program.generation, program.timestamp, program.id),
            )
            if generation_zero:
                return generation_zero[0]
            return program_reads.get_earliest(session, correct_only=True, island_idx=island_idx)

    def _select_winner_take_all(
        self,
        db: Database,
        island_idx: Optional[int],
    ) -> Optional[Program]:
        with db.session() as session:
            best = program_reads.get_best(session, island_idx=island_idx)
            if best is not None:
                return best
            return program_reads.get_most_recent(
                session,
                correct_only=True,
                island_idx=island_idx,
            )

    def _select_sequential(
        self,
        db: Database,
        island_idx: Optional[int],
    ) -> Optional[Program]:
        with db.session() as session:
            program = program_reads.get_most_recent(
                session,
                correct_only=True,
                island_idx=island_idx,
            )
            if program is not None:
                return program
            return program_reads.get_most_recent(session, island_idx=island_idx)


def sort_programs_by_score(programs: Sequence[Program]) -> List[Program]:
    return _sort_programs_by_score(programs)
