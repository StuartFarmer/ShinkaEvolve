from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import ProgramRecord, RunStateRecord
from .types import RunMetadataSnapshot

SUPPORTED_KEYS = {
    "last_iteration",
    "best_program_id",
    "beam_search_parent_id",
    "best_score_generation",
    "best_score_ever",
    "initial_program_count_adjustment",
}


def _get_or_create_record(session: Session) -> RunStateRecord:
    record = session.get(RunStateRecord, "default")
    if record is None:
        record = RunStateRecord(id="default")
        session.add(record)
        session.flush()
    return record


def get(session: Session, key: str, default: Optional[str] = None) -> Optional[str]:
    if key not in SUPPORTED_KEYS:
        return default
    record = session.get(RunStateRecord, "default")
    if record is None:
        return default
    value = getattr(record, key)
    return default if value is None else str(value)


def set(session: Session, key: str, value: Optional[str]) -> None:
    if key not in SUPPORTED_KEYS:
        return
    record = _get_or_create_record(session)
    if key in {"last_iteration", "best_score_generation", "initial_program_count_adjustment"}:
        parsed_value = 0 if value is None else int(value)
    elif key == "best_score_ever":
        parsed_value = None if value is None else float(value)
    else:
        parsed_value = value
    setattr(record, key, parsed_value)


def load_snapshot(session: Session, *, read_only: bool) -> RunMetadataSnapshot:
    record = session.get(RunStateRecord, "default")
    if record is None:
        if read_only:
            adjustment = int(
                session.scalar(
                    select(func.count()).select_from(ProgramRecord).where(
                        ProgramRecord.generation == 0,
                        ProgramRecord.parent_id.is_(None),
                    )
                )
                or 0
            )
            adjustment = max(adjustment - 1, 0)
            return RunMetadataSnapshot(initial_program_count_adjustment=adjustment)
        record = _get_or_create_record(session)

    if record.initial_program_count_adjustment == 0:
        initial_root_count = int(
            session.scalar(
                select(func.count()).select_from(ProgramRecord).where(
                    ProgramRecord.generation == 0,
                    ProgramRecord.parent_id.is_(None),
                )
            )
            or 0
        )
        computed_adjustment = max(initial_root_count - 1, 0)
        if computed_adjustment != record.initial_program_count_adjustment and not read_only:
            record.initial_program_count_adjustment = computed_adjustment

    return RunMetadataSnapshot(
        last_iteration=int(record.last_iteration or 0),
        best_program_id=record.best_program_id,
        beam_search_parent_id=record.beam_search_parent_id,
        best_score_generation=int(record.best_score_generation or 0),
        best_score_ever=record.best_score_ever,
        initial_program_count_adjustment=int(record.initial_program_count_adjustment or 0),
    )
