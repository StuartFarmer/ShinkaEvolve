from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Shared database connector for ORM-backed controllers and repositories."""

    def __init__(
        self,
        *,
        db_path: str | None,
        num_islands: int = 2,
        read_only: bool = False,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
    ) -> None:
        self.db_path = db_path
        self.num_islands = num_islands
        self.read_only = read_only
        self.conn = conn
        self.cursor = cursor
        self.engine = create_engine(
            "sqlite://",
            creator=lambda: self.conn,
            poolclass=StaticPool,
            future=True,
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            future=True,
        )

    @classmethod
    def open(
        cls,
        *,
        db_path: str | None = None,
        num_islands: int = 2,
        read_only: bool = False,
    ) -> "DatabaseConnector":
        conn = cls._connect(db_path=db_path, read_only=read_only)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        return cls(
            db_path=db_path,
            num_islands=num_islands,
            read_only=read_only,
            conn=conn,
            cursor=cursor,
        )

    @staticmethod
    def _connect(
        *,
        db_path: str | None,
        read_only: bool,
    ) -> sqlite3.Connection:
        if db_path:
            db_file = Path(db_path).resolve()
            if not read_only:
                db_wal_file = Path(f"{db_file}-wal")
                db_shm_file = Path(f"{db_file}-shm")
                if (
                    db_file.exists()
                    and db_file.stat().st_size == 0
                    and (db_wal_file.exists() or db_shm_file.exists())
                ):
                    logger.warning(
                        "Database file %s is empty but WAL/SHM files exist. "
                        "Removing WAL/SHM files to attempt recovery.",
                        db_file,
                    )
                    if db_wal_file.exists():
                        db_wal_file.unlink()
                    if db_shm_file.exists():
                        db_shm_file.unlink()
                db_file.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(db_file), timeout=30.0)
                logger.debug("Connected to SQLite database: %s", db_file)
                return conn

            if not db_file.exists():
                raise FileNotFoundError(
                    f"Database file not found for read-only connection: {db_file}"
                )
            db_uri = f"file:{db_file}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True, timeout=30.0)
            logger.debug("Connected to SQLite database in read-only mode: %s", db_file)
            return conn

        if read_only:
            raise ValueError("Read-only connector requires db_path")
        logger.info("Initialized in-memory SQLite database.")
        return sqlite3.connect(":memory:")

    def close(self) -> None:
        self.conn.close()
