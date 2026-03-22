from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from shinka.database.connector import DatabaseConnector
from shinka.database.models import ProgramInspirationRecord
from .types import InspirationUse


class InspirationController:
    """Controller for normalized program inspiration edges."""

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

    def replace_for_child(
        self,
        child_program_id: str,
        inspirations: List[InspirationUse],
        *,
        session: Session | None = None,
    ) -> None:
        if self.read_only:
            raise PermissionError("Cannot replace inspiration edges in read-only mode.")
        with self._managed_session(session) as active_session:
            active_session.execute(
                delete(ProgramInspirationRecord).where(
                    ProgramInspirationRecord.child_program_id == child_program_id
                )
            )
            for inspiration in inspirations:
                active_session.add(
                    ProgramInspirationRecord(
                        id=str(uuid.uuid4()),
                        child_program_id=inspiration.child_program_id,
                        source_program_id=inspiration.source_program_id,
                        role=inspiration.role,
                        order_index=inspiration.order_index,
                        weight=inspiration.weight,
                        edge_metadata=dict(inspiration.metadata or {}),
                    )
                )
            if session is None:
                active_session.commit()

    def list_for_child(self, child_program_id: str) -> List[InspirationUse]:
        return self.list_for_children([child_program_id]).get(child_program_id, [])

    def list_for_children(
        self,
        child_program_ids: Iterable[str],
    ) -> Dict[str, List[InspirationUse]]:
        child_ids = [child_id for child_id in child_program_ids if child_id]
        if not child_ids:
            return {}

        with self._managed_session() as session:
            records = session.execute(
                select(ProgramInspirationRecord)
                .where(ProgramInspirationRecord.child_program_id.in_(child_ids))
                .order_by(
                    ProgramInspirationRecord.child_program_id.asc(),
                    ProgramInspirationRecord.role.asc(),
                    ProgramInspirationRecord.order_index.asc(),
                )
            ).scalars().all()

        grouped: Dict[str, List[InspirationUse]] = {child_id: [] for child_id in child_ids}
        for record in records:
            grouped.setdefault(record.child_program_id, []).append(
                InspirationUse(
                    child_program_id=record.child_program_id,
                    source_program_id=record.source_program_id,
                    role=record.role,
                    order_index=record.order_index,
                    weight=record.weight,
                    metadata=dict(record.edge_metadata or {}),
                )
            )
        return grouped

    def list_sources_for_child(
        self,
        child_program_id: str,
        *,
        role: Optional[str] = None,
    ) -> List[str]:
        inspirations = self.list_for_child(child_program_id)
        if role is not None:
            inspirations = [insp for insp in inspirations if insp.role == role]
        return [insp.source_program_id for insp in inspirations]

    def list_children_for_source(
        self,
        source_program_id: str,
        *,
        role: Optional[str] = None,
    ) -> List[str]:
        with self._managed_session() as session:
            query = select(ProgramInspirationRecord.child_program_id).where(
                ProgramInspirationRecord.source_program_id == source_program_id
            )
            if role is not None:
                query = query.where(ProgramInspirationRecord.role == role)
            rows = session.execute(query).all()
        return [str(child_program_id) for (child_program_id,) in rows]

    def count_usage_by_source(
        self,
        source_program_id: str,
        *,
        role: Optional[str] = None,
    ) -> int:
        with self._managed_session() as session:
            query = select(func.count()).select_from(ProgramInspirationRecord).where(
                ProgramInspirationRecord.source_program_id == source_program_id
            )
            if role is not None:
                query = query.where(ProgramInspirationRecord.role == role)
            return int(session.scalar(query) or 0)

    def count_usage_by_role(self, role: str) -> int:
        with self._managed_session() as session:
            query = select(func.count()).select_from(ProgramInspirationRecord).where(
                ProgramInspirationRecord.role == role
            )
            return int(session.scalar(query) or 0)
