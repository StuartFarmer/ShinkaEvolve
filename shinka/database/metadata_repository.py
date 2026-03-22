from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import MetadataRecord, ProgramRecord


@dataclass(frozen=True)
class RunMetadataSnapshot:
    last_iteration: int = 0
    best_program_id: Optional[str] = None
    beam_search_parent_id: Optional[str] = None
    best_score_generation: int = 0
    best_score_ever: Optional[float] = None
    initial_program_count_adjustment: int = 0


class MetadataRepository:
    """
    Persistence boundary for run-level metadata stored alongside programs.

    This repository owns simple key/value metadata and any typed reconstruction
    of that metadata into runtime state objects.
    """

    def __init__(
        self,
        *,
        session_factory,
        read_only: bool = False,
    ) -> None:
        self._session_factory = session_factory
        self.read_only = read_only

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

    def ensure_schema(self) -> None:
        return

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._managed_session() as session:
            record = session.get(MetadataRecord, key)
        if not record:
            return default
        value = record.value
        return default if value is None else str(value)

    def set(self, key: str, value: Optional[str]) -> None:
        if self.read_only:
            raise PermissionError("Cannot update metadata in read-only mode.")
        with self._managed_session() as session:
            record = session.get(MetadataRecord, key)
            if record is None:
                session.add(MetadataRecord(key=key, value=value))
            else:
                record.value = value
            session.commit()

    def load_snapshot(self) -> RunMetadataSnapshot:
        last_iteration_raw = self.get("last_iteration")
        if last_iteration_raw is None and not self.read_only:
            self.set("last_iteration", "0")

        best_program_id_raw = self.get("best_program_id")
        if best_program_id_raw in [None, "None"] and not self.read_only:
            self.set("best_program_id", None)

        beam_parent_raw = self.get("beam_search_parent_id")
        if beam_parent_raw in [None, "None"] and not self.read_only:
            self.set("beam_search_parent_id", None)

        best_score_generation_raw = self.get("best_score_generation")
        best_score_ever_raw = self.get("best_score_ever")
        adjustment_raw = self.get("initial_program_count_adjustment")

        if adjustment_raw is None:
            with self._managed_session() as session:
                initial_root_count = int(
                    session.scalar(
                        select(func.count()).select_from(ProgramRecord).where(
                            ProgramRecord.generation == 0,
                            ProgramRecord.parent_id.is_(None),
                        )
                    )
                    or 0
                )
            adjustment = max(initial_root_count - 1, 0)
            if not self.read_only:
                self.set("initial_program_count_adjustment", str(adjustment))
        else:
            adjustment = int(adjustment_raw)

        return RunMetadataSnapshot(
            last_iteration=int(last_iteration_raw) if last_iteration_raw is not None else 0,
            best_program_id=None
            if best_program_id_raw in [None, "None"]
            else str(best_program_id_raw),
            beam_search_parent_id=None
            if beam_parent_raw in [None, "None"]
            else str(beam_parent_raw),
            best_score_generation=int(best_score_generation_raw)
            if best_score_generation_raw is not None
            else 0,
            best_score_ever=float(best_score_ever_raw)
            if best_score_ever_raw is not None
            else None,
            initial_program_count_adjustment=adjustment,
        )
