from __future__ import annotations

from shinka.database.models import Base
from shinka.database.connector import DatabaseConnector

from .embedding_controller import EmbeddingController
from .island_controller import IslandController
from .inspiration_controller import InspirationController
from .metadata_controller import MetadataController
from .program_controller import ProgramController
from .run_state_controller import RunStateController


class DatabaseController:
    """Master controller facade for model-scoped controllers."""

    def __init__(self, connector: DatabaseConnector) -> None:
        self.connector = connector
        self._bootstrap()
        self.programs = ProgramController(connector)
        self.metadata = MetadataController(connector)
        self.inspirations = InspirationController(connector)
        self.embeddings = EmbeddingController(connector)
        self.islands = IslandController(connector)
        self.run_state = RunStateController(connector)
        if not connector.read_only:
            self.run_state.load_snapshot()

    def _bootstrap(self) -> None:
        self.connector.cursor.execute("PRAGMA busy_timeout = 30000;")
        self.connector.cursor.execute("PRAGMA foreign_keys = ON;")
        if self.connector.read_only:
            return
        self.connector.cursor.execute("PRAGMA journal_mode = WAL;")
        self.connector.cursor.execute("PRAGMA wal_autocheckpoint = 1000;")
        self.connector.cursor.execute("PRAGMA synchronous = NORMAL;")
        self.connector.cursor.execute("PRAGMA cache_size = -64000;")
        self.connector.cursor.execute("PRAGMA temp_store = MEMORY;")
        Base.metadata.create_all(self.connector.engine)
        self.connector.conn.commit()

    def close(self) -> None:
        self.connector.close()
