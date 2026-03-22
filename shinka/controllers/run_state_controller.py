from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from shinka.database.connector import DatabaseConnector
from shinka.database.models import RunStateRecord, ProgramRecord
from .types import RunMetadataSnapshot


class RunStateController:
    """Typed controller for global run state stored in ``run_state``."""

    SUPPORTED_KEYS = {
        "last_iteration",
        "best_program_id",
        "beam_search_parent_id",
        "best_score_generation",
        "best_score_ever",
        "initial_program_count_adjustment",
    }

    def __init__(self, connector: DatabaseConnector) -> None:
        self.connector = connector
        self._session_factory = connector.SessionLocal
        self.read_only = connector.read_only

    @contextmanager
    def _managed_session(self, session: Session | None = None):
        if session is not None:
            yield session
            return
        managed = self._session_factory()
        try:
            yield managed
        finally:
            managed.close()

    def _get_or_create_record(self, session: Session) -> RunStateRecord:
        record = session.get(RunStateRecord, "default")
        if record is None:
            record = RunStateRecord(id="default")
            session.add(record)
            session.flush()
        return record

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        if key not in self.SUPPORTED_KEYS:
            return default
        with self._managed_session() as session:
            record = session.get(RunStateRecord, "default")
            if record is None:
                return default
            value = getattr(record, key)
        return default if value is None else str(value)

    def set(self, key: str, value: Optional[str]) -> None:
        if key not in self.SUPPORTED_KEYS:
            return
        if self.read_only:
            raise PermissionError("Cannot update run state in read-only mode.")

        with self._managed_session() as session:
            record = self._get_or_create_record(session)
            if key in {"last_iteration", "best_score_generation", "initial_program_count_adjustment"}:
                parsed_value = 0 if value is None else int(value)
            elif key == "best_score_ever":
                parsed_value = None if value is None else float(value)
            else:
                parsed_value = value
            setattr(record, key, parsed_value)
            session.commit()

    def load_snapshot(self) -> RunMetadataSnapshot:
        with self._managed_session() as session:
            record = session.get(RunStateRecord, "default")
            if record is None:
                if self.read_only:
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
                    return RunMetadataSnapshot(
                        initial_program_count_adjustment=adjustment
                    )
                record = self._get_or_create_record(session)
                session.commit()

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
                if (
                    computed_adjustment != record.initial_program_count_adjustment
                    and not self.read_only
                ):
                    record.initial_program_count_adjustment = computed_adjustment
                    session.commit()

            return RunMetadataSnapshot(
                last_iteration=int(record.last_iteration or 0),
                best_program_id=record.best_program_id,
                beam_search_parent_id=record.beam_search_parent_id,
                best_score_generation=int(record.best_score_generation or 0),
                best_score_ever=record.best_score_ever,
                initial_program_count_adjustment=int(
                    record.initial_program_count_adjustment or 0
                ),
            )
