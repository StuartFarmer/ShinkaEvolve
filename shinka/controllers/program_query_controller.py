from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from shinka.database.connector import DatabaseConnector
from shinka.database.models import ProgramEvaluationRecord, ProgramRecord
from shinka.database.program import Program

from .island_controller import IslandController
from .program_hydration_controller import ProgramHydrationController
from .types import Island, ProgramCountSnapshot


class ProgramQueryController:
    """Read/query layer for programs and computed program views."""

    def __init__(self, connector: DatabaseConnector) -> None:
        self.connector = connector
        self._session_factory = connector.SessionLocal
        self.hydration = ProgramHydrationController(connector)
        self.islands = IslandController(connector)

    def _session(self) -> Session:
        return self._session_factory()

    def _program_query(
        self,
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

    def _list_programs(self, query) -> List[Program]:
        with self._session() as session:
            records = session.execute(query).scalars().all()
        program_ids = [record.id for record in records]
        inspiration_index = self.hydration.build_inspiration_index(program_ids)
        evaluation_index = self.hydration.build_evaluation_index(program_ids)
        proposal_index = self.hydration.build_proposal_index(program_ids)
        embedding_index = self.hydration.build_embedding_index(program_ids)
        projection_index = self.hydration.build_projection_index(program_ids)
        return [
            p
            for p in (
                self.hydration.record_to_program(
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

    def get(self, program_id: str) -> Optional[Program]:
        with self._session() as session:
            record = session.get(ProgramRecord, program_id)
        return self.hydration.record_to_program(record)

    def get_many(self, program_ids: List[str]) -> List[Program]:
        if not program_ids:
            return []
        with self._session() as session:
            records = list(
                session.execute(
                    select(ProgramRecord).where(ProgramRecord.id.in_(program_ids))
                ).scalars()
            )
        record_ids = [record.id for record in records]
        inspiration_index = self.hydration.build_inspiration_index(record_ids)
        evaluation_index = self.hydration.build_evaluation_index(record_ids)
        proposal_index = self.hydration.build_proposal_index(record_ids)
        embedding_index = self.hydration.build_embedding_index(record_ids)
        projection_index = self.hydration.build_projection_index(record_ids)
        by_id = {
            record.id: self.hydration.record_to_program(
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

    def get_children_count(self, program_id: str) -> int:
        with self._session() as session:
            record = session.get(ProgramRecord, program_id)
            return int(record.children_count) if record else 0

    def get_program_count(self) -> int:
        return self.islands.get_program_count()

    def count_by_island(self, island_idx: int) -> int:
        with self._session() as session:
            return int(
                session.scalar(
                    select(func.count()).select_from(ProgramRecord).where(
                        ProgramRecord.island_idx == island_idx
                    )
                )
                or 0
            )

    def get_initial_program_row(self) -> Optional[dict[str, Any]]:
        with self._session() as session:
            record = session.scalar(
                select(ProgramRecord)
                .where(
                    ProgramRecord.generation == 0,
                    ProgramRecord.parent_id.is_(None),
                )
                .order_by(ProgramRecord.timestamp.asc())
                .limit(1)
            )
        return None if record is None else self.get(record.id).to_dict()

    def get_best_program_row(self) -> Optional[dict[str, Any]]:
        with self._session() as session:
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
        return None if record is None else self.get(record.id).to_dict()

    def get_correct_child_rows(
        self,
        parent_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[dict[str, Any]]:
        query = (
            self._program_query(correct_only=True, parent_id=parent_id)
            .order_by(ProgramEvaluationRecord.combined_score.desc())
        )
        if limit is not None:
            query = query.limit(limit)
        with self._session() as session:
            records = session.execute(query).scalars().all()
        inspiration_index = self.hydration.build_inspiration_index([record.id for record in records])
        programs = [
            self.hydration.record_to_program(record, inspiration_index=inspiration_index)
            for record in records
        ]
        return [program.to_dict() for program in programs if program is not None]

    def get_best(
        self,
        metric: Optional[str] = None,
        *,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        programs = self.list_correct(island_idx=island_idx)
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

    def get_ancestry(self, program_id: str, *, max_ancestors: int = 10) -> List[Program]:
        ancestors: List[Program] = []
        current = self.get(program_id)
        for _ in range(max_ancestors):
            if current is None or not current.parent_id:
                break
            current = self.get(current.parent_id)
            if current is None:
                break
            ancestors.append(current)
        ancestors.reverse()
        return ancestors

    def list_all(self) -> List[Program]:
        return self._list_programs(
            self._program_query().order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_correct(self, *, island_idx: Optional[int] = None) -> List[Program]:
        return self._list_programs(
            self._program_query(correct_only=True, island_idx=island_idx).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_incorrect(self, *, island_idx: Optional[int] = None) -> List[Program]:
        return self._list_programs(
            self._program_query(correct_only=False, island_idx=island_idx).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_by_island(
        self,
        island_idx: int,
        *,
        correct_only: bool = False,
    ) -> List[Program]:
        return self._list_programs(
            self._program_query(
                correct_only=True if correct_only else None,
                island_idx=island_idx,
            ).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_initialized_islands(self) -> List[Island]:
        return self.islands.list_initialized_islands()

    def list_initialized_island_ids(self) -> List[int]:
        return self.islands.list_initialized_island_ids()

    def list_islands(self) -> List[Island]:
        return self.islands.list_islands()

    def get_island_program_counts(
        self,
        island_indices: Sequence[int],
    ) -> Dict[int, int]:
        if not island_indices:
            return {}
        counts = {int(idx): 0 for idx in island_indices}
        with self._session() as session:
            rows = session.execute(
                select(ProgramRecord.island_idx, func.count())
                .join(
                    ProgramEvaluationRecord,
                    ProgramEvaluationRecord.program_id == ProgramRecord.id,
                )
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
        self,
        island_indices: Sequence[int],
    ) -> Dict[int, float]:
        if not island_indices:
            return {}
        with self._session() as session:
            rows = session.execute(
                select(
                    ProgramRecord.island_idx,
                    func.max(ProgramEvaluationRecord.combined_score),
                )
                .join(
                    ProgramEvaluationRecord,
                    ProgramEvaluationRecord.program_id == ProgramRecord.id,
                )
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
        self,
        *,
        correct_only: bool = False,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        programs = self._list_programs(
            self._program_query(
                correct_only=True if correct_only else None,
                island_idx=island_idx,
            ).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            ).limit(1)
        )
        return programs[0] if programs else None

    def get_most_recent(
        self,
        *,
        correct_only: bool = False,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        programs = self._list_programs(
            self._program_query(
                correct_only=True if correct_only else None,
                island_idx=island_idx,
            ).order_by(
                ProgramRecord.generation.desc(),
                ProgramRecord.timestamp.desc(),
                ProgramRecord.id.desc(),
            ).limit(1)
        )
        return programs[0] if programs else None

    def list_by_generation(self, generation: int) -> List[Program]:
        return self._list_programs(
            self._program_query(generation=generation).order_by(
                ProgramRecord.generation.asc(),
                ProgramRecord.timestamp.asc(),
                ProgramRecord.id.asc(),
            )
        )

    def list_top(
        self,
        *,
        n: int = 10,
        metric: Optional[str] = "combined_score",
        correct_only: bool = False,
        island_idx: Optional[int] = None,
    ) -> List[Program]:
        base = self._program_query(
            correct_only=True if correct_only else None,
            island_idx=island_idx,
        )
        if metric == "combined_score":
            return self._list_programs(
                base.where(ProgramEvaluationRecord.combined_score.is_not(None))
                .order_by(ProgramEvaluationRecord.combined_score.desc())
                .limit(n)
            )
        if metric == "timestamp":
            return self._list_programs(base.order_by(ProgramRecord.timestamp.desc()).limit(n))
        programs = self._list_programs(base)
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

    def get_summaries(self) -> List[dict[str, Any]]:
        with self._session() as session:
            records = session.execute(select(ProgramRecord)).scalars().all()
        program_ids = [record.id for record in records]
        inspiration_index = self.hydration.build_inspiration_index(program_ids)
        evaluation_index = self.hydration.build_evaluation_index(program_ids)
        proposal_index = self.hydration.build_proposal_index(program_ids)
        projection_index = self.hydration.build_projection_index(program_ids)
        return [
            self.hydration.record_to_summary(
                record,
                inspiration_index=inspiration_index,
                evaluation_index=evaluation_index,
                proposal_index=proposal_index,
                projection_index=projection_index,
            )
            for record in records
        ]

    def get_count_snapshot(self) -> ProgramCountSnapshot:
        with self._session() as session:
            count, max_timestamp = session.execute(
                select(func.count(ProgramRecord.id), func.max(ProgramRecord.timestamp))
            ).one()
        return ProgramCountSnapshot(count=int(count or 0), max_timestamp=max_timestamp)
