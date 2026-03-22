from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from shinka.database.connector import DatabaseConnector
from shinka.database.models import ProgramEvaluationRecord, ProgramRecord


class ProgramController:
    """Program-centric interaction layer over ORM models."""

    def __init__(self, connector: DatabaseConnector) -> None:
        self.connector = connector
        self._session_factory = connector.SessionLocal

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

    @staticmethod
    def _row_dict(record: ProgramRecord) -> Dict[str, Any]:
        return {
            column.name: getattr(record, column.name)
            for column in ProgramRecord.__table__.columns
        }

    def get_initial_program_row(self) -> Optional[Dict[str, Any]]:
        with self._managed_session() as session:
            record = session.scalar(
                select(ProgramRecord)
                .where(
                    ProgramRecord.generation == 0,
                    ProgramRecord.parent_id.is_(None),
                )
                .order_by(ProgramRecord.timestamp.asc())
                .limit(1)
            )
        return None if record is None else self._row_dict(record)

    def get_best_program_row(self) -> Optional[Dict[str, Any]]:
        with self._managed_session() as session:
            record = session.scalar(
                select(ProgramRecord)
                .join(
                    ProgramEvaluationRecord,
                    ProgramEvaluationRecord.program_id == ProgramRecord.id,
                )
                .where(ProgramEvaluationRecord.correct.is_(True))
                .order_by(ProgramEvaluationRecord.combined_score.desc())
                .limit(1)
            )
        return None if record is None else self._row_dict(record)
