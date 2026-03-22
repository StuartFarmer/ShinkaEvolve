from __future__ import annotations

from shinka.database.connector import DatabaseConnector
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
            DatabaseConnector.open(
                db_path=db_path,
                num_islands=num_islands,
                read_only=read_only,
            )
        )

    def __init__(self, connector: DatabaseConnector) -> None:
        self.connector = connector
        self.db_path = connector.db_path
        self.num_islands = connector.num_islands
        self.read_only = connector.read_only
        self.conn = connector.conn
        self.cursor = connector.cursor
        self.engine = connector.engine
        self.SessionLocal = connector.SessionLocal
        self._bootstrap()
        self.programs = ProgramController(self)
        self.metadata = MetadataController(self)
        self.inspirations = InspirationController(self)
        self.embeddings = EmbeddingController(self)
        self.islands = IslandController(self)
        self.run_state = RunStateController(self)
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
        self.connector.close()
