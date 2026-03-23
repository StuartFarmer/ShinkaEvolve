from __future__ import annotations

import uuid
from typing import Dict, Iterable, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .models import ProgramInspirationRecord
from .types import InspirationUse


def list_for_children(
    session: Session,
    child_program_ids: Iterable[str],
) -> Dict[str, List[InspirationUse]]:
    child_ids = [child_id for child_id in child_program_ids if child_id]
    if not child_ids:
        return {}

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


def list_for_child(session: Session, child_program_id: str) -> List[InspirationUse]:
    return list_for_children(session, [child_program_id]).get(child_program_id, [])

