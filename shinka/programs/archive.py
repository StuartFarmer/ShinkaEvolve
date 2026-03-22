"""
Computed archive selection functions.

The archive is a derived subset of correct programs. Callers choose a function,
pass in programs, and get back an ordered archive.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Dict, List, Optional, TypeAlias

import numpy as np

from .model import Program

logger = logging.getLogger(__name__)

ArchivePolicy: TypeAlias = Callable[[List[Program]], List[Program]]


def _prepare_programs(programs: List[Program]) -> List[Program]:
    return sorted(
        [program for program in programs if program.correct],
        key=lambda program: (program.generation, program.timestamp, program.id),
    )


def _average_public_metric(program: Program) -> float:
    if not program.public_metrics:
        return -float("inf")
    return sum(program.public_metrics.values()) / len(program.public_metrics)


def _sort_key(program: Program) -> tuple[float, float]:
    score = (
        float(program.combined_score)
        if program.combined_score is not None
        else _average_public_metric(program)
    )
    return score, float(program.timestamp)


def _sort_archive(programs: List[Program]) -> List[Program]:
    return sorted(programs, key=_sort_key, reverse=True)


def _get_criterion_value(program: Program, criterion: str) -> float:
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
    program: Program,
    archive_programs: List[Program],
    *,
    archive_criteria: Dict[str, float],
) -> float:
    if not archive_programs:
        primary_criterion = next(iter(archive_criteria.keys()), "combined_score")
        primary_weight = archive_criteria.get(primary_criterion, 1.0)
        value = _get_criterion_value(program, primary_criterion)
        return value if primary_weight > 0 else -value

    all_programs = archive_programs + [program]
    score = 0.0
    for criterion, weight in archive_criteria.items():
        values = [_get_criterion_value(item, criterion) for item in all_programs]
        program_value = values[-1]
        if weight > 0:
            rank = sum(1 for value in values if value < program_value) / len(values)
        else:
            rank = sum(1 for value in values if value > program_value) / len(values)
            weight = abs(weight)
        score += weight * rank
    return score


def _is_better(
    program1: Program,
    program2: Program,
    *,
    archive_criteria: Dict[str, float],
    archive_programs: Optional[List[Program]] = None,
) -> bool:
    if program1.correct and not program2.correct:
        return True
    if program2.correct and not program1.correct:
        return False

    use_ranked = archive_programs is not None and len(archive_criteria) > 1
    if use_ranked:
        context = [
            program
            for program in archive_programs
            if program.id not in (program1.id, program2.id)
        ]
        score1 = _compute_archive_score_ranked(
            program1,
            context,
            archive_criteria=archive_criteria,
        )
        score2 = _compute_archive_score_ranked(
            program2,
            context,
            archive_criteria=archive_criteria,
        )
        if score1 != score2:
            return score1 > score2
    else:
        score1 = program1.combined_score
        score2 = program2.combined_score
        if score1 is not None and score2 is not None:
            if score1 != score2:
                return score1 > score2
        elif score1 is not None:
            return True
        elif score2 is not None:
            return False

        avg1 = _average_public_metric(program1)
        avg2 = _average_public_metric(program2)
        if avg1 != avg2:
            return avg1 > avg2

    return program1.timestamp > program2.timestamp


def compute_fitness_archive(
    programs: List[Program],
    *,
    archive_size: int = 40,
    archive_criteria: Optional[Dict[str, float]] = None,
) -> List[Program]:
    if archive_size <= 0:
        return []

    criteria = archive_criteria or {"combined_score": 1.0}
    archive: List[Program] = []
    for program in _prepare_programs(programs):
        if len(archive) < archive_size:
            archive.append(program)
            continue

        if len(criteria) > 1:
            worst = min(
                archive,
                key=lambda candidate: _compute_archive_score_ranked(
                    candidate,
                    archive,
                    archive_criteria=criteria,
                ),
            )
        else:
            worst = archive[0]
            for candidate in archive[1:]:
                if _is_better(
                    worst,
                    candidate,
                    archive_criteria=criteria,
                ):
                    worst = candidate

        if _is_better(
            program,
            worst,
            archive_criteria=criteria,
            archive_programs=archive,
        ):
            archive = [candidate for candidate in archive if candidate.id != worst.id]
            archive.append(program)

    return _sort_archive(archive)


def _find_most_similar(
    embedding: List[float],
    archive_programs: List[Program],
) -> Optional[Program]:
    embedding_arr = np.array(embedding)
    embedding_norm = np.linalg.norm(embedding_arr)
    if embedding_norm < 1e-8:
        return None

    best_similarity = -float("inf")
    most_similar: Optional[Program] = None
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


def compute_crowding_archive(
    programs: List[Program],
    *,
    archive_size: int = 40,
    archive_criteria: Optional[Dict[str, float]] = None,
) -> List[Program]:
    if archive_size <= 0:
        return []

    criteria = archive_criteria or {"combined_score": 1.0}
    archive: List[Program] = []
    for program in _prepare_programs(programs):
        if len(archive) < archive_size:
            archive.append(program)
            continue

        if not program.embedding:
            archive = compute_fitness_archive(
                archive + [program],
                archive_size=archive_size,
                archive_criteria=criteria,
            )
            continue

        most_similar = _find_most_similar(program.embedding, archive)
        if most_similar is None:
            archive = compute_fitness_archive(
                archive + [program],
                archive_size=archive_size,
                archive_criteria=criteria,
            )
            continue

        if _is_better(
            program,
            most_similar,
            archive_criteria=criteria,
            archive_programs=archive,
        ):
            archive = [
                candidate for candidate in archive if candidate.id != most_similar.id
            ]
            archive.append(program)

    return _sort_archive(archive)


def pick_random_archive_program(
    programs: List[Program],
    archive_policy: ArchivePolicy,
) -> Optional[Program]:
    archive = archive_policy(programs)
    if not archive:
        return None
    return random.choice(archive)


def create_archive_policy(
    archive_selection_strategy: str = "fitness",
    archive_size: int = 40,
    archive_criteria: Optional[Dict[str, float]] = None,
) -> ArchivePolicy:
    criteria = archive_criteria or {"combined_score": 1.0}
    strategy = archive_selection_strategy

    if strategy == "crowding":
        return lambda programs: compute_crowding_archive(
            programs,
            archive_size=archive_size,
            archive_criteria=criteria,
        )

    return lambda programs: compute_fitness_archive(
        programs,
        archive_size=archive_size,
        archive_criteria=criteria,
    )


def FitnessArchivePolicy(
    archive_size: int = 40,
    archive_criteria: Optional[Dict[str, float]] = None,
) -> ArchivePolicy:
    return create_archive_policy(
        archive_selection_strategy="fitness",
        archive_size=archive_size,
        archive_criteria=archive_criteria,
    )


def CrowdingArchivePolicy(
    archive_size: int = 40,
    archive_criteria: Optional[Dict[str, float]] = None,
) -> ArchivePolicy:
    return create_archive_policy(
        archive_selection_strategy="crowding",
        archive_size=archive_size,
        archive_criteria=archive_criteria,
    )
