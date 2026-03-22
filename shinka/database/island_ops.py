from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from .archive_policy import pick_random_archive_program
from . import program_reads, program_writes
from .models import ProgramEvaluationRecord, ProgramRecord
from .program import Program
from .types import Island

logger = logging.getLogger(__name__)


def get_program_island(session: Session, program_id: str) -> Optional[int]:
    return session.scalar(select(ProgramRecord.island_idx).where(ProgramRecord.id == program_id))


def get_program_count(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(ProgramRecord)) or 0)


def get_max_island_index(session: Session) -> int:
    value = session.scalar(select(func.max(ProgramRecord.island_idx)))
    return int(value) if value is not None else -1


def get_next_island_index(session: Session, *, num_islands: int) -> int:
    return max(get_max_island_index(session) + 1, num_islands)


def list_islands(session: Session, *, num_islands: int) -> List[Island]:
    max_idx = get_max_island_index(session)
    upper_bound = max(max_idx, num_islands - 1)
    if upper_bound < 0:
        return []

    rows = session.execute(
        select(
            ProgramRecord.island_idx.label("island_idx"),
            func.count().label("total_programs"),
            func.sum(case((ProgramEvaluationRecord.correct.is_(True), 1), else_=0)).label(
                "correct_programs"
            ),
            func.max(
                case(
                    (
                        ProgramEvaluationRecord.correct.is_(True),
                        ProgramEvaluationRecord.combined_score,
                    ),
                    else_=None,
                )
            ).label("best_score"),
        )
        .select_from(ProgramRecord)
        .outerjoin(ProgramEvaluationRecord, ProgramEvaluationRecord.program_id == ProgramRecord.id)
        .where(ProgramRecord.island_idx.is_not(None))
        .group_by(ProgramRecord.island_idx)
    ).all()
    rows_by_island = {int(row.island_idx): row for row in rows}

    islands: List[Island] = []
    for island_idx in range(upper_bound + 1):
        row = rows_by_island.get(island_idx)
        if row is None:
            islands.append(Island(island_idx=island_idx))
            continue

        best_program_id = None
        if row.best_score is not None:
            best_program_id = session.scalar(
                select(ProgramRecord.id)
                .join(ProgramEvaluationRecord, ProgramEvaluationRecord.program_id == ProgramRecord.id)
                .where(
                    ProgramRecord.island_idx == island_idx,
                    ProgramEvaluationRecord.correct.is_(True),
                )
                .order_by(
                    ProgramEvaluationRecord.combined_score.desc(),
                    ProgramRecord.timestamp.asc(),
                    ProgramRecord.id.asc(),
                )
                .limit(1)
            )

        islands.append(
            Island(
                island_idx=island_idx,
                total_programs=int(row.total_programs or 0),
                correct_programs=int(row.correct_programs or 0),
                best_program_id=str(best_program_id) if best_program_id else None,
                best_score=float(row.best_score) if row.best_score is not None else 0.0,
            )
        )
    return islands


def list_initialized_islands(session: Session, *, num_islands: int) -> List[Island]:
    return [island for island in list_islands(session, num_islands=num_islands) if island.initialized]


def list_initialized_island_ids(session: Session, *, num_islands: int) -> List[int]:
    return [island.island_idx for island in list_initialized_islands(session, num_islands=num_islands)]


def are_all_islands_initialized(session: Session, *, num_islands: int) -> bool:
    if num_islands <= 0:
        return True
    return len(list_initialized_island_ids(session, num_islands=num_islands)) >= num_islands


def get_island_populations(session: Session, *, num_islands: int) -> Dict[int, int]:
    if num_islands <= 0:
        return {}
    return {
        island.island_idx: island.total_programs
        for island in list_islands(session, num_islands=num_islands)
    }


def format_populations(session: Session, *, num_islands: int) -> str:
    populations = get_island_populations(session, num_islands=num_islands)
    if not populations:
        return f"0 programs in {num_islands} islands"
    parts = []
    for island_idx, count in sorted(populations.items()):
        island_color = f"color({30 + island_idx % 220})"
        parts.append(f"[{island_color}]I{island_idx}: {count}[/{island_color}]")
    return " | ".join(parts)


def assign_program(session: Session, program: Any, *, num_islands: int) -> None:
    if program.island_idx is not None:
        return
    if num_islands <= 0:
        program.island_idx = 0
        return
    if get_program_count(session) == 0:
        program.island_idx = 0
        if program.metadata is None:
            program.metadata = {}
        program.metadata["_needs_island_copies"] = True
        return
    if program.parent_id:
        parent_island = get_program_island(session, program.parent_id)
        if parent_island is not None:
            program.island_idx = parent_island
            return
    initialized = set(list_initialized_island_ids(session, num_islands=num_islands))
    uninitialized = [idx for idx in range(num_islands) if idx not in initialized]
    if uninitialized:
        program.island_idx = min(uninitialized)
        return
    program.island_idx = random.randint(0, num_islands - 1)


def copy_program_to_islands(
    session: Session,
    program: Program,
    *,
    num_islands: int,
) -> List[str]:
    if num_islands <= 1:
        return []
    created_ids: List[str] = []
    for island_idx in range(1, num_islands):
        created_ids.append(
            program_writes.insert_program_copy_from_object(
                session,
                program=program,
                island_idx=island_idx,
                metadata_updates={
                    "_is_island_copy": True,
                    "_original_program_id": program.id,
                },
                clear_copy_flag=True,
            )
        )
    return created_ids


def perform_migration(
    session: Session,
    *,
    num_islands: int,
    migration_rate: float,
    island_elitism: bool,
    current_generation: int,
) -> bool:
    if num_islands < 2 or migration_rate <= 0:
        return False

    migrated = 0
    migrated_ids: set[str] = set()
    for source_idx in range(num_islands):
        island_size = program_reads.count_by_island(session, source_idx)
        if island_size <= 1:
            continue
        num_migrants = max(1, int(island_size * migration_rate))
        dest_islands = [idx for idx in range(num_islands) if idx != source_idx]
        if not dest_islands:
            continue
        migrants = program_writes.list_migrant_ids(
            session,
            source_idx=source_idx,
            num_migrants=num_migrants,
            island_elitism=island_elitism,
        )
        for migrant_id in migrants:
            if migrant_id in migrated_ids:
                continue
            migrated_ids.add(migrant_id)
            program_writes.migrate_program(
                session,
                migrant_id=migrant_id,
                source_idx=source_idx,
                dest_idx=random.choice(dest_islands),
                current_generation=current_generation,
            )
            migrated += 1
    return migrated > 0


def spawn_island(
    session: Session,
    archive_policy,
    *,
    num_islands: int,
    strategy: str,
    subtree_size: int,
) -> bool:
    source = _select_spawn_source_row(session, archive_policy, strategy=strategy)
    if source is None:
        return False
    new_island_idx = get_next_island_index(session, num_islands=num_islands)
    subtree = _list_spawn_subtree_rows(session, source, max_size=subtree_size)
    old_to_new_id: Dict[str, str] = {}
    for idx, record in enumerate(subtree):
        is_root = idx == 0
        old_parent_id = record.get("parent_id")
        if is_root:
            new_parent_id = None
        elif old_parent_id and old_parent_id in old_to_new_id:
            new_parent_id = old_to_new_id[old_parent_id]
        else:
            new_parent_id = None
        old_to_new_id[record["id"]] = program_writes.insert_program_copy_from_row(
            session,
            source_program=record,
            new_island_idx=new_island_idx,
            new_parent_id=new_parent_id,
            strategy=strategy,
            is_root=is_root,
        )
    logger.info(
        "Spawned island %s from strategy %s using %s programs",
        new_island_idx,
        strategy,
        len(subtree),
    )
    return True


def _select_spawn_source_row(
    session: Session,
    archive_policy,
    *,
    strategy: str,
) -> Optional[Dict]:
    if strategy == "initial":
        return program_reads.get_initial_program_row(session)
    if strategy == "best":
        return program_reads.get_best_program_row(session)
    if strategy == "archive_random":
        program = pick_random_archive_program(
            program_reads.list_correct(session),
            archive_policy,
        )
        return None if program is None else program.to_dict()
    return program_reads.get_initial_program_row(session)


def _list_spawn_subtree_rows(
    session: Session,
    root_program: Dict,
    *,
    max_size: int,
) -> List[Dict]:
    if max_size <= 1:
        return [root_program]
    collected = [root_program]
    queue = [root_program]
    remaining = max_size - 1
    while queue and remaining > 0:
        current = queue.pop(0)
        children = program_reads.get_correct_child_rows(session, current["id"], limit=remaining)
        for child in children:
            if remaining <= 0:
                break
            collected.append(child)
            queue.append(child)
            remaining -= 1
    return collected
