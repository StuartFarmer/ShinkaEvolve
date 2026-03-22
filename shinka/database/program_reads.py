from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import program_hydration
from .models import ProgramEvaluationRecord, ProgramRecord
from .program import Program
from .types import Island, ProgramCountSnapshot


def _program_query(
    *,
    correct_only: Optional[bool] = None,
    island_idx: Optional[int] = None,
    generation: Optional[int] = None,
    parent_id: Optional[str] = None,
):
    query = select(ProgramRecord).outerjoin(
        ProgramEvaluationRecord,
        ProgramEvaluationRecord.program_id == ProgramRecord.id,
    )
    if correct_only is True:
        query = query.where(ProgramEvaluationRecord.correct.is_(True))
    elif correct_only is False:
        query = query.where(
            (ProgramEvaluationRecord.correct.is_(False))
            | (ProgramEvaluationRecord.correct.is_(None))
        )
    if island_idx is not None:
        query = query.where(ProgramRecord.island_idx == island_idx)
    if generation is not None:
        query = query.where(ProgramRecord.generation == generation)
    if parent_id is not None:
        query = query.where(ProgramRecord.parent_id == parent_id)
    return query


def _list_programs(session: Session, query) -> List[Program]:
    records = session.execute(query).scalars().all()
    program_ids = [record.id for record in records]
    inspiration_index = program_hydration.build_inspiration_index(session, program_ids)
    evaluation_index = program_hydration.build_evaluation_index(session, program_ids)
    proposal_index = program_hydration.build_proposal_index(session, program_ids)
    embedding_index = program_hydration.build_embedding_index(session, program_ids)
    projection_index = program_hydration.build_projection_index(session, program_ids)
    return [
        p
        for p in (
            program_hydration.record_to_program(
                session,
                record,
                inspiration_index=inspiration_index,
                evaluation_index=evaluation_index,
                proposal_index=proposal_index,
                embedding_index=embedding_index,
                projection_index=projection_index,
            )
            for record in records
        )
        if p is not None
    ]


def get(session: Session, program_id: str) -> Optional[Program]:
    record = session.get(ProgramRecord, program_id)
    return program_hydration.record_to_program(session, record)


def get_many(session: Session, program_ids: List[str]) -> List[Program]:
    if not program_ids:
        return []
    records = list(
        session.execute(select(ProgramRecord).where(ProgramRecord.id.in_(program_ids))).scalars()
    )
    record_ids = [record.id for record in records]
    inspiration_index = program_hydration.build_inspiration_index(session, record_ids)
    evaluation_index = program_hydration.build_evaluation_index(session, record_ids)
    proposal_index = program_hydration.build_proposal_index(session, record_ids)
    embedding_index = program_hydration.build_embedding_index(session, record_ids)
    projection_index = program_hydration.build_projection_index(session, record_ids)
    by_id = {
        record.id: program_hydration.record_to_program(
            session,
            record,
            inspiration_index=inspiration_index,
            evaluation_index=evaluation_index,
            proposal_index=proposal_index,
            embedding_index=embedding_index,
            projection_index=projection_index,
        )
        for record in records
    }
    return [by_id[program_id] for program_id in program_ids if by_id.get(program_id) is not None]


def get_children_count(session: Session, program_id: str) -> int:
    record = session.get(ProgramRecord, program_id)
    return int(record.children_count) if record else 0


def count_by_island(session: Session, island_idx: int) -> int:
    return int(
        session.scalar(
            select(func.count()).select_from(ProgramRecord).where(
                ProgramRecord.island_idx == island_idx
            )
        )
        or 0
    )


def get_initial_program_row(session: Session) -> Optional[dict[str, Any]]:
    record = session.scalar(
        select(ProgramRecord)
        .where(ProgramRecord.generation == 0, ProgramRecord.parent_id.is_(None))
        .order_by(ProgramRecord.timestamp.asc())
        .limit(1)
    )
    program = None if record is None else get(session, record.id)
    return None if program is None else program.to_dict()


def get_best_program_row(session: Session) -> Optional[dict[str, Any]]:
    record = session.scalar(
        select(ProgramRecord)
        .join(ProgramEvaluationRecord, ProgramEvaluationRecord.program_id == ProgramRecord.id)
        .where(ProgramEvaluationRecord.correct.is_(True))
        .order_by(ProgramEvaluationRecord.combined_score.desc())
        .limit(1)
    )
    program = None if record is None else get(session, record.id)
    return None if program is None else program.to_dict()


def get_correct_child_rows(
    session: Session,
    parent_id: str,
    *,
    limit: Optional[int] = None,
) -> List[dict[str, Any]]:
    query = _program_query(correct_only=True, parent_id=parent_id).order_by(
        ProgramEvaluationRecord.combined_score.desc()
    )
    if limit is not None:
        query = query.limit(limit)
    records = session.execute(query).scalars().all()
    inspiration_index = program_hydration.build_inspiration_index(
        session, [record.id for record in records]
    )
    programs = [
        program_hydration.record_to_program(
            session,
            record,
            inspiration_index=inspiration_index,
        )
        for record in records
    ]
    return [program.to_dict() for program in programs if program is not None]


def list_all(session: Session) -> List[Program]:
    return _list_programs(
        session,
        _program_query().order_by(
            ProgramRecord.generation.asc(),
            ProgramRecord.timestamp.asc(),
            ProgramRecord.id.asc(),
        ),
    )


def list_correct(session: Session, *, island_idx: Optional[int] = None) -> List[Program]:
    return _list_programs(
        session,
        _program_query(correct_only=True, island_idx=island_idx).order_by(
            ProgramRecord.generation.asc(),
            ProgramRecord.timestamp.asc(),
            ProgramRecord.id.asc(),
        ),
    )


def list_incorrect(session: Session, *, island_idx: Optional[int] = None) -> List[Program]:
    return _list_programs(
        session,
        _program_query(correct_only=False, island_idx=island_idx).order_by(
            ProgramRecord.generation.asc(),
            ProgramRecord.timestamp.asc(),
            ProgramRecord.id.asc(),
        ),
    )


def list_by_island(
    session: Session,
    island_idx: int,
    *,
    correct_only: bool = False,
) -> List[Program]:
    return _list_programs(
        session,
        _program_query(
            correct_only=True if correct_only else None,
            island_idx=island_idx,
        ).order_by(
            ProgramRecord.generation.asc(),
            ProgramRecord.timestamp.asc(),
            ProgramRecord.id.asc(),
        ),
    )


def get_island_program_counts(
    session: Session,
    island_indices: Sequence[int],
) -> Dict[int, int]:
    if not island_indices:
        return {}
    counts = {int(idx): 0 for idx in island_indices}
    rows = session.execute(
        select(ProgramRecord.island_idx, func.count())
        .join(ProgramEvaluationRecord, ProgramEvaluationRecord.program_id == ProgramRecord.id)
        .where(
            ProgramRecord.island_idx.in_(list(island_indices)),
            ProgramEvaluationRecord.correct.is_(True),
        )
        .group_by(ProgramRecord.island_idx)
    ).all()
    for island_idx, count in rows:
        counts[int(island_idx)] = int(count)
    return counts


def get_island_best_scores(
    session: Session,
    island_indices: Sequence[int],
) -> Dict[int, float]:
    if not island_indices:
        return {}
    rows = session.execute(
        select(
            ProgramRecord.island_idx,
            func.max(ProgramEvaluationRecord.combined_score),
        )
        .join(ProgramEvaluationRecord, ProgramEvaluationRecord.program_id == ProgramRecord.id)
        .where(
            ProgramRecord.island_idx.in_(list(island_indices)),
            ProgramEvaluationRecord.correct.is_(True),
        )
        .group_by(ProgramRecord.island_idx)
    ).all()
    return {
        int(island_idx): float(score) if score is not None else 0.0
        for island_idx, score in rows
    }


def get_earliest(
    session: Session,
    *,
    correct_only: bool = False,
    island_idx: Optional[int] = None,
) -> Optional[Program]:
    programs = _list_programs(
        session,
        _program_query(
            correct_only=True if correct_only else None,
            island_idx=island_idx,
        )
        .order_by(ProgramRecord.generation.asc(), ProgramRecord.timestamp.asc(), ProgramRecord.id.asc())
        .limit(1),
    )
    return programs[0] if programs else None


def get_most_recent(
    session: Session,
    *,
    correct_only: bool = False,
    island_idx: Optional[int] = None,
) -> Optional[Program]:
    programs = _list_programs(
        session,
        _program_query(
            correct_only=True if correct_only else None,
            island_idx=island_idx,
        )
        .order_by(ProgramRecord.generation.desc(), ProgramRecord.timestamp.desc(), ProgramRecord.id.desc())
        .limit(1),
    )
    return programs[0] if programs else None


def list_by_generation(session: Session, generation: int) -> List[Program]:
    return _list_programs(
        session,
        _program_query(generation=generation).order_by(
            ProgramRecord.generation.asc(),
            ProgramRecord.timestamp.asc(),
            ProgramRecord.id.asc(),
        ),
    )


def list_top(
    session: Session,
    *,
    n: int = 10,
    metric: Optional[str] = "combined_score",
    correct_only: bool = False,
    island_idx: Optional[int] = None,
) -> List[Program]:
    base = _program_query(correct_only=True if correct_only else None, island_idx=island_idx)
    if metric == "combined_score":
        return _list_programs(
            session,
            base.where(ProgramEvaluationRecord.combined_score.is_not(None))
            .order_by(ProgramEvaluationRecord.combined_score.desc())
            .limit(n),
        )
    if metric == "timestamp":
        return _list_programs(session, base.order_by(ProgramRecord.timestamp.desc()).limit(n))
    programs = _list_programs(session, base)
    if not programs:
        return []
    if metric:
        ranked = sorted(
            [p for p in programs if p.public_metrics and metric in p.public_metrics],
            key=lambda p: p.public_metrics.get(metric, -float("inf")),
            reverse=True,
        )
    else:
        ranked = sorted(
            [p for p in programs if p.public_metrics],
            key=lambda p: (
                sum(p.public_metrics.values()) / len(p.public_metrics)
                if p.public_metrics
                else -float("inf")
            ),
            reverse=True,
        )
    return ranked[:n]


def get_summaries(session: Session) -> List[dict[str, Any]]:
    records = session.execute(select(ProgramRecord)).scalars().all()
    program_ids = [record.id for record in records]
    inspiration_index = program_hydration.build_inspiration_index(session, program_ids)
    evaluation_index = program_hydration.build_evaluation_index(session, program_ids)
    proposal_index = program_hydration.build_proposal_index(session, program_ids)
    projection_index = program_hydration.build_projection_index(session, program_ids)
    return [
        program_hydration.record_to_summary(
            session,
            record,
            inspiration_index=inspiration_index,
            evaluation_index=evaluation_index,
            proposal_index=proposal_index,
            projection_index=projection_index,
        )
        for record in records
    ]


def get_count_snapshot(session: Session) -> ProgramCountSnapshot:
    count, max_timestamp = session.execute(
        select(func.count(ProgramRecord.id), func.max(ProgramRecord.timestamp))
    ).one()
    return ProgramCountSnapshot(count=int(count or 0), max_timestamp=max_timestamp)


def get_best(
    session: Session,
    metric: Optional[str] = None,
    *,
    island_idx: Optional[int] = None,
) -> Optional[Program]:
    programs = list_correct(session, island_idx=island_idx)
    if not programs:
        return None
    if metric:
        eligible = [p for p in programs if p.public_metrics and metric in p.public_metrics]
        ranked = sorted(
            eligible,
            key=lambda p: p.public_metrics.get(metric, -float("inf")),
            reverse=True,
        )
    elif any(p.combined_score is not None for p in programs):
        ranked = sorted(
            [p for p in programs if p.combined_score is not None],
            key=lambda p: p.combined_score or -float("inf"),
            reverse=True,
        )
    else:
        ranked = sorted(
            [p for p in programs if p.public_metrics],
            key=lambda p: (
                sum(p.public_metrics.values()) / len(p.public_metrics)
                if p.public_metrics
                else -float("inf")
            ),
            reverse=True,
        )
    return ranked[0] if ranked else None


def get_ancestry(session: Session, program_id: str, *, max_ancestors: int = 10) -> List[Program]:
    ancestors: List[Program] = []
    current = get(session, program_id)
    for _ in range(max_ancestors):
        if current is None or not current.parent_id:
            break
        current = get(session, current.parent_id)
        if current is None:
            break
        ancestors.append(current)
    ancestors.reverse()
    return ancestors
