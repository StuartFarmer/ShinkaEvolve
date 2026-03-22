from __future__ import annotations

from shinka.database.connection import DatabaseConnection
from shinka.database.models import Base

from .embedding_controller import EmbeddingController
from .island_controller import IslandController
from .inspiration_controller import InspirationController
from .metadata_controller import MetadataController
from .program_controller import ProgramController
from .run_state_controller import RunStateController


class DatabaseController:
    """Master controller facade for model-scoped controllers."""

    @classmethod
    def open(
        cls,
        *,
        db_path: str | None = None,
        num_islands: int = 2,
        read_only: bool = False,
    ) -> "DatabaseController":
        return cls(
            DatabaseConnection.open(
                db_path=db_path,
                num_islands=num_islands,
                read_only=read_only,
            )
        )

    def __init__(self, connection: DatabaseConnection) -> None:
        self.connection = connection
        self.db_path = connection.db_path
        self.num_islands = connection.num_islands
        self.read_only = connection.read_only
        self.conn = connection.conn
        self.cursor = connection.cursor
        self.engine = connection.engine
        self.SessionLocal = connection.SessionLocal
        self._bootstrap()
        self.programs = ProgramController(connection)
        self.metadata = MetadataController(connection)
        self.inspirations = InspirationController(connection)
        self.embeddings = EmbeddingController(connection)
        self.islands = IslandController(connection)
        self.run_state = RunStateController(connection)
        if not self.read_only:
            self.run_state.load_snapshot()

    def _bootstrap(self) -> None:
        self.cursor.execute("PRAGMA busy_timeout = 30000;")
        self.cursor.execute("PRAGMA foreign_keys = ON;")
        if self.read_only:
            return
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA wal_autocheckpoint = 1000;")
        self.cursor.execute("PRAGMA synchronous = NORMAL;")
        self.cursor.execute("PRAGMA cache_size = -64000;")
        self.cursor.execute("PRAGMA temp_store = MEMORY;")
        Base.metadata.create_all(self.engine)
        self.conn.commit()

    def close(self) -> None:
        self.connection.close()
