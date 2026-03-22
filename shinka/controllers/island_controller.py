from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from shinka.database.connector import DatabaseConnector
from shinka.database.island_repository import Island
from shinka.database.models import ProgramEvaluationRecord, ProgramRecord


class IslandController:
    """Island-scoped computed view/controller over persisted programs."""

    def __init__(self, connector: DatabaseConnector) -> None:
        self.connector = connector
        self._session_factory = connector.SessionLocal
        self.num_islands = connector.num_islands

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

    def get_program_island(self, program_id: str) -> Optional[int]:
        with self._managed_session() as session:
            return session.scalar(
                select(ProgramRecord.island_idx).where(ProgramRecord.id == program_id)
            )

    def get_program_count(self) -> int:
        with self._managed_session() as session:
            return int(session.scalar(select(func.count()).select_from(ProgramRecord)) or 0)

    def get_max_island_index(self) -> int:
        with self._managed_session() as session:
            value = session.scalar(select(func.max(ProgramRecord.island_idx)))
        return int(value) if value is not None else -1

    def get_next_island_index(self) -> int:
        return max(self.get_max_island_index() + 1, self.num_islands)

    def list_islands(self) -> List[Island]:
        max_idx = self.get_max_island_index()
        upper_bound = max(max_idx, self.num_islands - 1)
        if upper_bound < 0:
            return []

        with self._managed_session() as session:
            rows = session.execute(
                select(
                    ProgramRecord.island_idx.label("island_idx"),
                    func.count().label("total_programs"),
                    func.sum(
                        case(
                            (ProgramEvaluationRecord.correct.is_(True), 1),
                            else_=0,
                        )
                    ).label("correct_programs"),
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
                .outerjoin(
                    ProgramEvaluationRecord,
                    ProgramEvaluationRecord.program_id == ProgramRecord.id,
                )
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
                        .join(
                            ProgramEvaluationRecord,
                            ProgramEvaluationRecord.program_id == ProgramRecord.id,
                        )
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

    def list_initialized_islands(self) -> List[Island]:
        return [island for island in self.list_islands() if island.initialized]

    def list_initialized_island_ids(self) -> List[int]:
        return [island.island_idx for island in self.list_initialized_islands()]

    def are_all_islands_initialized(self) -> bool:
        if self.num_islands <= 0:
            return True
        return len(self.list_initialized_island_ids()) >= self.num_islands

    def get_island_populations(self) -> Dict[int, int]:
        if self.num_islands <= 0:
            return {}
        return {island.island_idx: island.total_programs for island in self.list_islands()}
