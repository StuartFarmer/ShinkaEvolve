from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from .models import MetadataRecord


def get(session: Session, key: str, default: Optional[str] = None) -> Optional[str]:
    record = session.get(MetadataRecord, key)
    if record is None or record.value is None:
        return default
    return str(record.value)


def set(
    session: Session,
    key: str,
    value: Optional[str],
) -> None:
    record = session.get(MetadataRecord, key)
    if record is None:
        session.add(MetadataRecord(key=key, value=value))
    else:
        record.value = value
