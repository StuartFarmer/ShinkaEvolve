"""
Computed archive policy.

The archive is no longer treated as persisted source-of-truth state for search
behavior. Instead, it is a small derived subset of correct programs that can be
recomputed cheaply from the repository on demand.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from .program import Program

logger = logging.getLogger(__name__)


class ArchivePolicy(ABC):
    """Pure archive selection policy over already-persisted programs."""

    def __init__(
        self,
        archive_size: int,
        archive_criteria: Optional[Dict[str, float]] = None,
    ):
        self.archive_size = archive_size
        self.archive_criteria = archive_criteria or {"combined_score": 1.0}

    @abstractmethod
    def compute(self, programs: List[Program]) -> List[Program]:
        """Return the current computed archive, ordered best-first."""

    def pick_random(self, programs: List[Program]) -> Optional[Program]:
        archive = self.compute(programs)
        if not archive:
            return None
        return random.choice(archive)

    def _prepare_programs(self, programs: List[Program]) -> List[Program]:
        return sorted(
            [program for program in programs if program.correct],
            key=lambda p: (p.generation, p.timestamp, p.id),
        )

    def _sort_archive(self, programs: List[Program]) -> List[Program]:
        return sorted(
            programs,
            key=self._sort_key,
            reverse=True,
        )

    def _sort_key(self, program: Program) -> tuple[float, float]:
        score = (
            float(program.combined_score)
            if program.combined_score is not None
            else self._average_public_metric(program)
        )
        return score, float(program.timestamp)

    def _average_public_metric(self, program: Program) -> float:
        if not program.public_metrics:
            return -float("inf")
        return sum(program.public_metrics.values()) / len(program.public_metrics)

    def _get_criterion_value(self, program: Program, criterion: str) -> float:
        if criterion == "combined_score":
            return float(program.combined_score or 0.0)

        if criterion in (program.public_metrics or {}):
            value = program.public_metrics[criterion]
            return float(value) if value is not None else 0.0

        metrics = (program.metadata or {}).get("code_analysis_metrics", {})

        if criterion == "complexity":
            return float(program.complexity or 0.0)
        if criterion == "loc":
            return float(metrics.get("lines_of_code", len(program.code.splitlines())))
        if criterion == "cyclomatic":
            return float(metrics.get("cyclomatic_complexity", 1.0))
        if criterion == "maintainability":
            return float(metrics.get("maintainability_index", 100.0))
        if criterion == "nesting":
            return float(metrics.get("max_nesting_depth", 1))

        logger.warning("Unknown archive criterion: %s", criterion)
        return 0.0

    def _compute_archive_score_ranked(
        self, program: Program, archive_programs: List[Program]
    ) -> float:
        criteria: Dict[str, float] = self.archive_criteria

        if not archive_programs:
            primary_criterion = next(iter(criteria.keys()), "combined_score")
            primary_weight = criteria.get(primary_criterion, 1.0)
            value = self._get_criterion_value(program, primary_criterion)
            return value if primary_weight > 0 else -value

        all_programs = archive_programs + [program]

        score = 0.0
        for criterion, weight in criteria.items():
            values = [self._get_criterion_value(p, criterion) for p in all_programs]
            program_value = values[-1]

            if weight > 0:
                rank = sum(1 for v in values if v < program_value) / len(values)
            else:
                rank = sum(1 for v in values if v > program_value) / len(values)
                weight = abs(weight)

            score += weight * rank

        return score

    def _is_better(
        self,
        program1: Program,
        program2: Program,
        archive_programs: Optional[List[Program]] = None,
    ) -> bool:
        if program1.correct and not program2.correct:
            return True
        if program2.correct and not program1.correct:
            return False

        criteria = self.archive_criteria
        use_ranked = archive_programs is not None and len(criteria) > 1

        if use_ranked:
            context = [
                p for p in archive_programs if p.id not in (program1.id, program2.id)
            ]
            s1 = self._compute_archive_score_ranked(program1, context)
            s2 = self._compute_archive_score_ranked(program2, context)
            if s1 != s2:
                return s1 > s2
        else:
            s1 = program1.combined_score
            s2 = program2.combined_score

            if s1 is not None and s2 is not None:
                if s1 != s2:
                    return s1 > s2
            elif s1 is not None:
                return True
            elif s2 is not None:
                return False

            avg1 = self._average_public_metric(program1)
            avg2 = self._average_public_metric(program2)
            if avg1 != avg2:
                return avg1 > avg2

        return program1.timestamp > program2.timestamp


class FitnessArchivePolicy(ArchivePolicy):
    """Archive policy that retains the best programs under the configured criteria."""

    def compute(self, programs: List[Program]) -> List[Program]:
        if self.archive_size <= 0:
            return []

        archive: List[Program] = []
        for program in self._prepare_programs(programs):
            if len(archive) < self.archive_size:
                archive.append(program)
                continue

            criteria = self.archive_criteria
            if len(criteria) > 1:
                worst = min(
                    archive,
                    key=lambda p: self._compute_archive_score_ranked(p, archive),
                )
            else:
                worst = archive[0]
                for candidate in archive[1:]:
                    if self._is_better(worst, candidate):
                        worst = candidate

            if self._is_better(program, worst, archive):
                archive = [p for p in archive if p.id != worst.id] + [program]

        return self._sort_archive(archive)


class CrowdingArchivePolicy(ArchivePolicy):
    """Archive policy that preserves diversity by replacing local neighbors."""

    def compute(self, programs: List[Program]) -> List[Program]:
        if self.archive_size <= 0:
            return []

        archive: List[Program] = []
        fitness_fallback = FitnessArchivePolicy(
            archive_size=self.archive_size,
            archive_criteria=self.archive_criteria,
        )
        for program in self._prepare_programs(programs):
            if len(archive) < self.archive_size:
                archive.append(program)
                continue

            if not program.embedding:
                archive = fitness_fallback.compute(archive + [program])
                continue

            most_similar = self._find_most_similar(program.embedding, archive)
            if most_similar is None:
                archive = fitness_fallback.compute(archive + [program])
                continue

            if self._is_better(program, most_similar, archive):
                archive = [p for p in archive if p.id != most_similar.id] + [program]

        return self._sort_archive(archive)

    def _find_most_similar(
        self,
        embedding: List[float],
        archive_programs: List[Program],
    ) -> Optional[Program]:
        best_similarity = -float("inf")
        most_similar: Optional[Program] = None

        embedding_arr = np.array(embedding)
        embedding_norm = np.linalg.norm(embedding_arr)
        if embedding_norm < 1e-8:
            return None

        for program in archive_programs:
            if not program.embedding:
                continue
            program_embedding = np.array(program.embedding)
            program_norm = np.linalg.norm(program_embedding)
            if program_norm < 1e-8:
                continue

            similarity = float(
                np.dot(embedding_arr, program_embedding) / (embedding_norm * program_norm)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                most_similar = program

        return most_similar


def create_archive_policy(
    archive_selection_strategy: str = "fitness",
    archive_size: int = 40,
    archive_criteria: Optional[Dict[str, float]] = None,
) -> ArchivePolicy:
    """Factory for the configured computed archive policy."""
    criteria = archive_criteria or {"combined_score": 1.0}
    strategy = archive_selection_strategy
    if strategy == "crowding":
        return CrowdingArchivePolicy(
            archive_size=archive_size,
            archive_criteria=criteria,
        )
    return FitnessArchivePolicy(
        archive_size=archive_size,
        archive_criteria=criteria,
    )
