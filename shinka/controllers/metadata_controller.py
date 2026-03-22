from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from sqlalchemy.orm import Session

from shinka.database.connection import DatabaseConnection
from shinka.database.models import MetadataRecord


class MetadataController:
    """Controller for generic key/value metadata stored in ``metadata_store``."""

    def __init__(self, connection: DatabaseConnection) -> None:
        self.connection = connection
        self._session_factory = connection.SessionLocal
        self.read_only = connection.read_only

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

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._managed_session() as session:
            record = session.get(MetadataRecord, key)
        if record is None or record.value is None:
            return default
        return str(record.value)

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
