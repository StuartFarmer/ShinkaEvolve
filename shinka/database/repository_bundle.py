from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dbase import DatabaseConfig
    from .repository import ProgramRepository

logger = logging.getLogger(__name__)


@dataclass
class RepositoryBundle:
    """
    Storage/bootstrap bundle for repository-backed runtime state.

    This owns:
    - opening the sqlite connection
    - handling read-only vs read-write mode
    - lightweight startup recovery checks
    - constructing repository objects on top of that connection

    It is the replacement path for the connection/bootstrap logic that still
    lives inside `ProgramDatabase.__init__`.
    """

    config: DatabaseConfig
    read_only: bool
    conn: sqlite3.Connection
    cursor: sqlite3.Cursor
    programs: "ProgramRepository"

    @classmethod
    def open(
        cls,
        config: "DatabaseConfig",
        *,
        read_only: bool = False,
    ) -> "RepositoryBundle":
        from .repository import ProgramRepository

        conn = cls._connect(config=config, read_only=read_only)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        programs = ProgramRepository.from_existing_connection(
            config=config,
            conn=conn,
            cursor=cursor,
            read_only=read_only,
            ensure_schema=not read_only,
        )
        return cls(
            config=config,
            read_only=read_only,
            conn=conn,
            cursor=cursor,
            programs=programs,
        )

    @staticmethod
    def _connect(
        *,
        config: "DatabaseConfig",
        read_only: bool,
    ) -> sqlite3.Connection:
        db_path_str = config.db_path

        if db_path_str:
            db_file = Path(db_path_str).resolve()
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
            raise ValueError("Read-only bundle requires config.db_path")
        logger.info("Initialized in-memory SQLite database.")
        return sqlite3.connect(":memory:")

    @property
    def metadata(self):
        return self.programs.metadata_repo

    @property
    def islands(self):
        return self.programs.island_repo

    def close(self) -> None:
        self.programs.close()
